#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <unistd.h>
#include <string.h>

#define RECOVERY_TAG 99
#define MAP_TAG 100       // Tag for sending global id mapping info
#define MAX_PROCS 100     // Maximum number of processes in the simulation

int global_counter = -1;
int initial_conditions = 1;
int save_left_neighbor = -1, save_right_neighbor = -1;
int global_id = -1;        // This process's global id

// Global mapping array: for each original id (index) we store the current owner rank,
// or -1 if that process has failed.
int global_map[MAX_PROCS];
int total_global_procs = 0; // Total number of processes in the simulation

int run_ring(MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int counter = global_counter;
    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    pid_t pid = getpid();
    printf("run_ring: rank %d (pid %ld), global_id=%d, size=%d, initial counter=%d\n",
           rank, (long)pid, global_id, size, counter);
    fflush(stdout);

    MPI_Status st;
    int rc;
    if (initial_conditions == 1)
    {
        // For original processes, initialize counter and global id from rank.
        counter = rank;
        global_id = rank;
        initial_conditions = 0;
    }
    else
    {
        // Recovery: exchange saved neighbor counters.
        int recovery_from_left, recovery_from_right;
        rc = MPI_Sendrecv(&save_left_neighbor, 1, MPI_INT, right, RECOVERY_TAG,
                          &recovery_from_left, 1, MPI_INT, left, RECOVERY_TAG,
                          comm, &st);
        if (rc != MPI_SUCCESS) return rc;
        rc = MPI_Sendrecv(&save_right_neighbor, 1, MPI_INT, left, RECOVERY_TAG,
                          &recovery_from_right, 1, MPI_INT, right, RECOVERY_TAG,
                          comm, &st);
        if (rc != MPI_SUCCESS) return rc;
        if (global_counter < 0)
        {
            counter = (recovery_from_left + recovery_from_right) / 2;
            if (recovery_from_left > 0)
                counter = recovery_from_left;
            if (recovery_from_right > 0)
                counter = recovery_from_right;
            if (counter < 0)
            {
                printf("Neighbors also died, counter lost, simulation is cooked.\n");
                exit(0);
            }
            printf("Rank %d: Recovered counter from neighbors: left=%d, right=%d, recovered=%d\n",
                   rank, recovery_from_left, recovery_from_right, counter);
        }
        // NEW: The new process's global_id was set by the survivors' leader before merging.
    }

    while (counter < 100)
    {
        int send_value = counter;
        int recv_from_left, recv_from_right;
        rc = MPI_Sendrecv(&send_value, 1, MPI_INT, right, 0,
                          &recv_from_left, 1, MPI_INT, left, 0,
                          comm, &st);
        if (rc != MPI_SUCCESS) return rc;
        rc = MPI_Sendrecv(&send_value, 1, MPI_INT, left, 1,
                          &recv_from_right, 1, MPI_INT, right, 1,
                          comm, &st);
        if (rc != MPI_SUCCESS) return rc;
        save_left_neighbor  = recv_from_left;
        save_right_neighbor = recv_from_right;
        printf("Rank %d (global_id %d): counter=%d, left_neighbor=%d, right_neighbor=%d\n",
               rank, global_id, counter, save_left_neighbor, save_right_neighbor);
        fflush(stdout);
        global_counter = counter;
        counter += size;
        sleep(1);
    }
    return MPI_SUCCESS;
}

//
// Child process code: new processes spawned to replace failed ones.
// NEW: Do not assume that the sender is rank 0; receive from any source.
static MPI_Comm child_process_code(MPI_Comm parent_ic)
{
    if (parent_ic == MPI_COMM_NULL)
    {
        fprintf(stderr, "Child with no parent? Exiting.\n");
        MPI_Finalize();
        exit(1);
    }
    MPI_Status st;
    int assigned_gid;
    MPI_Recv(&assigned_gid, 1, MPI_INT, MPI_ANY_SOURCE, MAP_TAG, parent_ic, &st);
    global_id = assigned_gid;
    printf("Child process: received assigned global_id=%d\n", global_id);
    fflush(stdout);

    MPI_Comm child_comm;
    MPI_Intercomm_merge(parent_ic, 1, &child_comm);
    MPI_Comm_disconnect(&parent_ic);
    int rank;
    MPI_Comm_rank(child_comm, &rank);
    printf("Child process merged into new communicator as rank %d. Starting ring with state recovery.\n", rank);
    fflush(stdout);
    initial_conditions = 0; // Do not reinitialize state
    return child_comm;
}

//
// replace_failed_ranks: Called after a failure is detected.
// 1) All survivors perform an Allgather to share their global_ids.
// 2) The process with the smallest global_id becomes the elected leader.
// 3) The elected leader computes which global ids are missing (failed),
//    spawns new processes, and sends each spawned process its assigned global id.
// 4) After merging, the updated global_map is broadcast to everyone.
//
static MPI_Comm replace_failed_ranks(MPI_Comm survivors_comm, int nfailed)
{
    int s_rank, s_size;
    MPI_Comm_rank(survivors_comm, &s_rank);
    MPI_Comm_size(survivors_comm, &s_size);

    // All survivors share their global ids.
    int *survivor_gids = malloc(s_size * sizeof(int));
    MPI_Allgather(&global_id, 1, MPI_INT, survivor_gids, 1, MPI_INT, survivors_comm);

    // Elect leader: the process with the minimum global id among survivors.
    int min_global_id = survivor_gids[0];
    for (int i = 1; i < s_size; i++) {
        if (survivor_gids[i] < min_global_id)
            min_global_id = survivor_gids[i];
    }
    int is_leader = (global_id == min_global_id);

    // Leader computes the failed ids by comparing the survivors to the original global_map.
    int failed_ids[nfailed];
    if (is_leader)
    {
        int count = 0;
        for (int i = 0; i < total_global_procs; i++) {
            int found = 0;
            for (int j = 0; j < s_size; j++) {
                if (survivor_gids[j] == i) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                failed_ids[count++] = i;
                if (count == nfailed)
                    break;
            }
        }
        // Update the global_map: mark missing ones as -1.
        for (int i = 0; i < total_global_procs; i++) {
            int present = 0;
            for (int j = 0; j < s_size; j++) {
                if (survivor_gids[j] == i) {
                    present = 1;
                    break;
                }
            }
            if (!present)
                global_map[i] = -1;
        }
    }
    free(survivor_gids);

    // All survivors spawn nfailed new processes.
    char *spawn_argv[] = { "./partial-restart.out", NULL };
    MPI_Comm intercomm;
    int errcodes[nfailed];
    MPI_Comm_spawn("./partial-restart.out", spawn_argv, nfailed,
                   MPI_INFO_NULL, 0, survivors_comm, &intercomm, errcodes);

    // Only the elected leader sends mapping info to each newly spawned process.
    if (is_leader)
    {
        for (int i = 0; i < nfailed; i++) {
            MPI_Send(&failed_ids[i], 1, MPI_INT, i, MAP_TAG, intercomm);
            // In our design, the spawned process adopts the same global id as the failed process.
            global_map[failed_ids[i]] = failed_ids[i];
        }
    }

    // Merge the spawned children with survivors.
    MPI_Comm new_comm;
    MPI_Intercomm_merge(intercomm, 0, &new_comm);
    MPI_Comm_disconnect(&intercomm);

    // Determine the leader in the new merged communicator.
    int new_comm_size, new_comm_rank;
    MPI_Comm_size(new_comm, &new_comm_size);
    MPI_Comm_rank(new_comm, &new_comm_rank);
    int *new_comm_gids = malloc(new_comm_size * sizeof(int));
    MPI_Allgather(&global_id, 1, MPI_INT, new_comm_gids, 1, MPI_INT, new_comm);
    int elected_root = 0;
    for (int i = 1; i < new_comm_size; i++) {
        if (new_comm_gids[i] < new_comm_gids[elected_root])
            elected_root = i;
    }
    free(new_comm_gids);
    // Broadcast the updated global_map to all processes.
    MPI_Bcast(global_map, total_global_procs, MPI_INT, elected_root, new_comm);

    if (is_leader) {
        printf("Updated global_map: ");
        for (int i = 0; i < total_global_procs; i++) {
            printf("%d ", global_map[i]);
        }
        printf("\n");
        fflush(stdout);
    }
    return new_comm;
}

//
// failure_handler: When a failure is detected, revoke and shrink the communicator,
// then call replace_failed_ranks to spawn new processes and update the global mapping.
//
static MPI_Comm failure_handler(MPI_Comm comm)
{
    fprintf(stderr, "Detected rank failure\n");
    MPIX_Comm_revoke(comm);
    MPIX_Comm_failure_ack(comm);
    MPI_Group fgroup;
    MPIX_Comm_failure_get_acked(comm, &fgroup);
    int fsize;
    MPI_Group_size(fgroup, &fsize);
    MPI_Group_free(&fgroup);
    if (fsize <= 0)
    {
        fprintf(stderr, "No failed ranks found? Strange.\n");
        MPI_Abort(comm, 1);
    }
    MPI_Comm survivors_comm;
    MPIX_Comm_shrink(comm, &survivors_comm);
    MPI_Comm_free(&comm);
    comm = replace_failed_ranks(survivors_comm, fsize);
    return comm;
}

//
// main: For the original (parent) processes, we initialize the global_map so that
// each process’s global id is its rank. In case of a failure, failure_handler is invoked.
//
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_Comm parent, comm;
    MPI_Comm_get_parent(&parent);

    int rc;
    if (parent == MPI_COMM_NULL)
    {
        comm = MPI_COMM_WORLD;
        // Initialize global_map and record the total number of processes.
        MPI_Comm_size(comm, &total_global_procs);
        for (int i = 0; i < total_global_procs; i++) {
            global_map[i] = i;
        }
        while (1)
        {
            rc = run_ring(comm);
            if (rc == MPI_SUCCESS)
            {
                MPI_Comm_free(&comm);
                MPI_Finalize();
                return 0;
            }
            else
            {
                comm = failure_handler(comm);
            }
        }
    }
    else
    {
        // Child (spawned) processes run child_process_code.
        comm = child_process_code(parent);
        while (1)
        {
            rc = run_ring(comm);
            if (rc == MPI_SUCCESS)
            {
                MPI_Comm_free(&comm);
                MPI_Finalize();
                exit(0);
            }
            else
            {
                comm = failure_handler(comm);
            }
        }
    }
    return 0;
}
