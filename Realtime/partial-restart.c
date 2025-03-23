#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <unistd.h>

#define RECOVERY_TAG 99

int global_counter = -1;
int initial_conditions = 1;
int save_left_neighbor = -1, save_right_neighbor = -1;

int run_ring(MPI_Comm comm)
{

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int counter = global_counter;

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    pid_t pid = getpid();
    printf("run_ring: rank %d (pid %ld), size=%d, initial counter=%d\n",
           rank, (long)pid, size, counter);
    fflush(stdout);

        
    MPI_Status st;
    int rc;
    if ( initial_conditions == 1)
    {
        counter = rank;
        initial_conditions = 0;
    }
    else
    {
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
            {
                counter = recovery_from_left;
            }
            if (recovery_from_right > 0)
            {
                counter = recovery_from_right;
            }
            if (counter < 0)
            {
                printf("Neighbors also died, counter lost, simulation is cooked.");
                exit(0);
            }
            printf("Rank %d: Recovered counter from neighbors: left=%d, right=%d, recovered=%d\n",
                   rank, recovery_from_left, recovery_from_right, counter);
        }
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


        printf("Rank %d: counter=%d, left_neighbor=%d, right_neighbor=%d\n",
               rank, counter, save_left_neighbor, save_right_neighbor);
        fflush(stdout);

        global_counter = counter;
        counter+=size;
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

    MPI_Comm child_comm;
    MPI_Intercomm_merge(parent_ic, 1, &child_comm);
    MPI_Comm_disconnect(&parent_ic);
    printf("Child process merged into new communicator. Starting ring with state recovery from neighbors.\n");
    fflush(stdout);
    initial_conditions = 0;
    return child_comm;
}

static MPI_Comm replace_failed_ranks(MPI_Comm survivors_comm, int nfailed)
{
    int s_rank;
    MPI_Comm_rank(survivors_comm, &s_rank);

    char *spawn_argv[] = { "./partial-restart.out", NULL };
    MPI_Comm intercomm;
    int errcodes[nfailed];

    MPI_Comm_spawn("./partial-restart.out", spawn_argv, nfailed, MPI_INFO_NULL, 0, survivors_comm, &intercomm, errcodes);
    MPI_Comm new_comm;
    MPI_Intercomm_merge(intercomm, 0, &new_comm);
    MPI_Comm_disconnect(&intercomm);

    printf("Survivors: done spawning and merging, new communicator formed.\n");
    fflush(stdout);
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
