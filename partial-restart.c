#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

void checkpoint(int rank, int counter) {
    char filename[256];
    sprintf(filename, "checkpoint_rank%d.dat", rank);
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen");
        return;
    }
    fprintf(fp, "%d\n", counter);
    fclose(fp);
    printf("Rank %d: checkpoint saved (counter=%d) to %s\n",
           rank, counter, filename);
    fflush(stdout);
}

int restore_checkpoint(int rank) {
    char filename[256];
    sprintf(filename, "checkpoint_rank%d.dat", rank);
    FILE *fp = fopen(filename, "r");
    int counter = 0;
    if (fp) {
        fscanf(fp, "%d", &counter);
        fclose(fp);
        printf("Rank %d: restored checkpoint (counter=%d) from %s\n",
               rank, counter, filename);
        fflush(stdout);
    }
    return counter;
}

int run_ring(MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    int counter = restore_checkpoint(rank);

    pid_t pid = getpid();
    printf("run_ring: rank %d (pid %ld), size=%d, start counter=%d\n",
           rank, (long)pid, size, counter);
    fflush(stdout);

    while (1) {
        int left_counter=0, right_counter=0;
        MPI_Status st;
        int rc;

        rc = MPI_Sendrecv(&counter, 1, MPI_INT, right, 0,
                          &left_counter, 1, MPI_INT, left, 0,
                          comm, &st);
        if (rc != MPI_SUCCESS) return rc;

        rc = MPI_Sendrecv(&counter, 1, MPI_INT, left, 1,
                          &right_counter, 1, MPI_INT, right, 1,
                          comm, &st);
        if (rc != MPI_SUCCESS) return rc;

        printf("Rank %d: counter=%d, left(%d)=%d, right(%d)=%d\n",
               rank, counter, left, left_counter, right, right_counter);
        fflush(stdout);

        counter++;

        if (counter % 5 == 0) {
            checkpoint(rank, counter);
        }

        sleep(1);
    }
    return MPI_SUCCESS;
}

static void child_process_code(MPI_Comm parent_ic)
{
    if (parent_ic == MPI_COMM_NULL) {
        fprintf(stderr, "Child with no parent? Exiting.\n");
        MPI_Finalize();
        exit(0);
    }

    MPI_Comm child_local;
    MPI_Intercomm_merge(parent_ic, 1, &child_local);

    MPI_Comm_disconnect(&parent_ic);
    printf("Child ranks formed child_local intracomm. Now running ring...\n");
    fflush(stdout);

    int rc = run_ring(child_local);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "Child ring got error %d -> exit.\n", rc);
    }
    MPI_Comm_free(&child_local);
    MPI_Finalize();
    exit(0);
}

static void replace_failed_ranks(MPI_Comm survivors_comm, int nfailed)
{
    int s_rank;
    MPI_Comm_rank(survivors_comm, &s_rank);

    MPI_Comm intercomm = MPI_COMM_NULL;

    if (s_rank == 0) {
        char *spawn_argv[] = { "./partial-restart.out", NULL };
        int errcodes[nfailed];
        printf("[Survivor rank 0] Spawning %d new ranks...\n", nfailed);
        fflush(stdout);

        MPI_Comm_spawn("./partial-restart.out", spawn_argv,
                       nfailed, MPI_INFO_NULL, 0,
                       survivors_comm, &intercomm,
                       errcodes);
    }
    MPI_Bcast(&intercomm, 1, MPI_UINT64_T, 0, survivors_comm);
    if (intercomm != MPI_COMM_NULL) {
        MPI_Comm_disconnect(&intercomm);
    }

    printf("[Survivors] done spawning, returning to ring with fewer ranks.\n");
    fflush(stdout);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);

    if (parent == MPI_COMM_NULL) {

        MPI_Comm comm = MPI_COMM_WORLD;
        int rc;
        while (1) {
            rc = run_ring(comm);
            if (rc == MPI_SUCCESS) {
                break;
            } else {
                fprintf(stderr, "Detected rank failure => partial replacement.\n");
                MPIX_Comm_revoke(comm);
                MPIX_Comm_failure_ack(comm);
                MPI_Group fgroup;
                MPIX_Comm_failure_get_acked(comm, &fgroup);
                int fsize;
                MPI_Group_size(fgroup, &fsize);
                MPI_Group_free(&fgroup);
                if (fsize <= 0) {
                    fprintf(stderr, "No failed ranks found? Strange.\n");
                    MPI_Abort(comm, 1);
                }
                MPI_Comm survivors_comm;
                MPIX_Comm_shrink(comm, &survivors_comm);
                MPI_Comm_free(&comm);
                comm = survivors_comm;
                replace_failed_ranks(comm, fsize);
            }
        }

        MPI_Finalize();
        return 0;
    }
    else {
        child_process_code(parent);
    }
    return 0;
}
