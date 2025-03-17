#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <unistd.h>

int global_counter = -1;

int run_ring(MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int counter = global_counter;
    int left_neighbor = -1, right_neighbor = -1;

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    pid_t pid = getpid();
    printf("run_ring: rank %d (pid %ld), size=%d, initial counter=%d\n",
           rank, (long)pid, size, counter);
    fflush(stdout);

    while (counter < 1000)
    {
        int send_value = counter;
        int recv_from_left, recv_from_right;
        MPI_Status st;
        int rc;
        rc = MPI_Sendrecv(&send_value, 1, MPI_INT, right, 0,
                          &recv_from_left, 1, MPI_INT, left, 0,
                          comm, &st);
        if (rc != MPI_SUCCESS) return rc;

        rc = MPI_Sendrecv(&send_value, 1, MPI_INT, left, 1,
                          &recv_from_right, 1, MPI_INT, right, 1,
                          comm, &st);
        if (rc != MPI_SUCCESS) return rc;

        left_neighbor  = recv_from_left;
        right_neighbor = recv_from_right;

        if (counter < 0)
        {
            if (left_neighbor >= 0)
            {
                counter = left_neighbor;
            } else if (right_neighbor >= 0)
            {
                counter = right_neighbor;
            } else {
                counter = 0;
            }
        }

        printf("Rank %d: counter=%d, left_neighbor=%d, right_neighbor=%d\n",
               rank, counter, left_neighbor, right_neighbor);
        fflush(stdout);

        global_counter = counter;
        counter++;
        sleep(1);
    }
    return MPI_SUCCESS;
}

static void child_process_code(MPI_Comm parent_ic)
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
    int rc = run_ring(child_comm);
    if (rc != MPI_SUCCESS)
    {
        fprintf(stderr, "Child ring encountered error %d. Exiting.\n", rc);
    }
    MPI_Comm_free(&child_comm);
    MPI_Finalize();
    exit(0);
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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);

    if (parent == MPI_COMM_NULL)
    {
        MPI_Comm comm = MPI_COMM_WORLD;
        int rc;
        while (1) {
            rc = run_ring(comm);
            if (rc == MPI_SUCCESS)
            {
                break;
            } 
            else
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
            }
        }
        MPI_Finalize();
        return 0;
    } 
    else
    {
        child_process_code(parent);
    }
    return 0;
}
