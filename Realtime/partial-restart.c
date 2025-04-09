#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <unistd.h>
#include <string.h>

#define RECOVERY_TAG 99
#define MAP_TAG 100
#define MAX_PROCS 100

#define SAVE_STEP 10

int global_counter = -1;
int initial_conditions = 1;
int save_left_neighbor = -1, save_right_neighbor = -1;
int global_id = -1;

// Global mapping array: for each original id (index) we store the current owner rank,
// or -1 if that process has failed.
int global_map[MAX_PROCS];
int total_global_procs = 0; // total number of processes in the simulation

static void update_global_map(MPI_Comm new_comm)
{
    int my_gid = global_id;
    int *all_gids = malloc(total_global_procs * sizeof(int));
    MPI_Allgather(&my_gid, 1, MPI_INT,
                all_gids, 1, MPI_INT, new_comm);

    for (int r = 0; r < total_global_procs; r++) {
        int their_gid = all_gids[r];
        if (their_gid >= 0 && their_gid < MAX_PROCS)
        global_map[ their_gid ] = r;
    }
    free(all_gids);
    printf("Updated global_map: ");
    for (int i = 0; i < total_global_procs; i++) {
        printf("%d ", global_map[i]);
    }
    printf("\n");
    fflush(stdout);
}
int run_ring(MPI_Comm comm)
{
    
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int counter = global_counter;
    int left, right;

    update_global_map(comm);

    pid_t pid = getpid();
    printf("run_ring: rank %d (pid %ld), global_id=%d, size=%d, initial counter=%d\n",
           rank, (long)pid, global_id, size, counter);
    fflush(stdout);

    MPI_Status st;
    int rc;
    if (initial_conditions == 1)
    {
        counter = rank;
        global_id = rank;
        initial_conditions = 0;
    }
    else
    {
        left  = global_map[(global_id - 1 + total_global_procs) % total_global_procs];
        right = global_map[(global_id + 1) % total_global_procs];
        int recovery_from_left, recovery_from_right;
        rc = MPI_Sendrecv(&save_right_neighbor, 1, MPI_INT, right, RECOVERY_TAG,
                          &recovery_from_left, 1, MPI_INT, left, RECOVERY_TAG,
                          comm, &st);
        if (rc == MPI_ERR_PROC_FAILED || rc == MPIX_ERR_REVOKED) return rc;
        rc = MPI_Sendrecv(&save_left_neighbor, 1, MPI_INT, left, RECOVERY_TAG,
                          &recovery_from_right, 1, MPI_INT, right, RECOVERY_TAG,
                          comm, &st);
        if (rc == MPI_ERR_PROC_FAILED || rc == MPIX_ERR_REVOKED) return rc;
        if (global_counter < 0)
        {
            counter = -1;
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
    }
    
    left  = global_map[(global_id - 1 + total_global_procs) % total_global_procs];
    right = global_map[(global_id + 1) % total_global_procs];

    while (counter < 50 * total_global_procs)
    {
        int send_value = counter;
        int recv_from_left, recv_from_right;
        rc = MPI_Sendrecv(&send_value, 1, MPI_INT, right, 0,
                          &recv_from_left, 1, MPI_INT, left, 0,
                          comm, &st);
        if (rc == MPI_ERR_PROC_FAILED || rc == MPIX_ERR_REVOKED) return rc;
        rc = MPI_Sendrecv(&send_value, 1, MPI_INT, left, 1,
                          &recv_from_right, 1, MPI_INT, right, 1,
                          comm, &st);
        if (rc == MPI_ERR_PROC_FAILED || rc == MPIX_ERR_REVOKED) return rc;

        int iteration = (counter - global_id) / total_global_procs;
        if (iteration % SAVE_STEP == 0)
        {
            printf("Rank %d (global_id %d): Saving state: counter=%d, , iteration=%d, left_neighbor=%d, right_neighbor=%d\n",
                   rank, global_id, counter, iteration, recv_from_left, recv_from_right);
            save_left_neighbor  = recv_from_left;
            save_right_neighbor = recv_from_right;
            global_counter = counter;
        }
        printf("Rank %d (global_id %d): counter=%d, left_neighbor=%d, right_neighbor=%d\n",
               rank, global_id, counter, save_left_neighbor, save_right_neighbor);
        fflush(stdout);
        counter += total_global_procs;
        sleep(1);
    }
    return MPI_SUCCESS;
}

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
    MPI_Recv(&assigned_gid, 1, MPI_INT, 0, MAP_TAG, parent_ic, &st);
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
    initial_conditions = 0;
    MPI_Comm_size(child_comm, &total_global_procs);
    return child_comm;
}

static MPI_Comm replace_failed_ranks(MPI_Comm survivors_comm, int nfailed)
{
    int s_rank, s_size;
    MPI_Comm_rank(survivors_comm, &s_rank);
    MPI_Comm_size(survivors_comm, &s_size);

    int min_global_id;
    
    // Everyone collectively finds the minimal global ID.
    MPI_Allreduce(&global_id, &min_global_id, 1, MPI_INT, MPI_MIN, survivors_comm);
    
    int is_leader = (global_id == min_global_id);

    int *survivor_gids = NULL;
    survivor_gids = malloc(s_size * sizeof(int));
    
    MPI_Allgather(&global_id, 1, MPI_INT, survivor_gids, 1, MPI_INT, survivors_comm);

    int failed_ids[nfailed];
    if (is_leader) {
        int count = 0;
        // Identify which global ids are missing
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
                if (count == nfailed) break;
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
        free(survivor_gids);
    }

    char *spawn_argv[] = { "./partial-restart.out", NULL };
    MPI_Comm intercomm;
    int errcodes[nfailed];
    MPI_Comm_spawn("./partial-restart.out", spawn_argv, nfailed,
                   MPI_INFO_NULL, 0, survivors_comm, &intercomm, errcodes);

    // Leader sends each new process its assigned global id.
    if (is_leader) {
        for (int i = 0; i < nfailed; i++) {
            MPI_Send(&failed_ids[i], 1, MPI_INT, i, MAP_TAG, intercomm);
        }
    }

    MPI_Comm new_comm;
    MPI_Intercomm_merge(intercomm, 0, &new_comm);
    MPI_Comm_disconnect(&intercomm);

    return new_comm;
}

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
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
    return comm;
}

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
