#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

void checkpoint(int rank, int counter) {
    char filename[256];
    sprintf(filename, "checkpoint_rank%d.dat", rank);
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("fopen");
        return;
    }
    fprintf(fp, "%d\n", counter);
    fclose(fp);
    printf("Rank %d: Checkpoint saved (counter = %d) to %s\n",
           rank, counter, filename);
    fflush(stdout);
}

int restore_checkpoint(int rank) {
    char filename[256];
    sprintf(filename, "checkpoint_rank%d.dat", rank);
    FILE *fp = fopen(filename, "r");
    int counter = 0;
    if (fp != NULL) {
        fscanf(fp, "%d", &counter);
        fclose(fp);
        printf("Rank %d: Restored checkpoint from %s (counter = %d)\n",
               rank, filename, counter);
        fflush(stdout);
    }
    return counter;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    int counter = restore_checkpoint(rank);

    pid_t pid = getpid();
    printf("Rank %d (PID %lu) started with counter = %d\n",
           rank, (unsigned long)pid, counter);
    fflush(stdout);

    while (1) {
        int left_counter  = 0;
        int right_counter = 0;
        MPI_Status status;
        int err;

        err = MPI_Sendrecv(&counter, 1, MPI_INT, right, 0,
                           &left_counter, 1, MPI_INT, left, 0,
                           comm, &status);
        if (err != MPI_SUCCESS) {
            fprintf(stderr, "Rank %d detected a failure. Aborting job...\n", rank);
            fflush(stderr);
            MPI_Abort(comm, 99);
            exit(1);
        }

        err = MPI_Sendrecv(&counter, 1, MPI_INT, left, 1,
                           &right_counter, 1, MPI_INT, right, 1,
                           comm, &status);
        if (err != MPI_SUCCESS) {
            fprintf(stderr, "Rank %d detected a failure. Aborting job...\n", rank);
            fflush(stderr);
            MPI_Abort(comm, 99);
            exit(1);
        }

        printf("Rank %d (PID %lu): my counter=%d, left(rank %d)=%d, right(rank %d)=%d\n",
               rank, (unsigned long)pid, counter, left, left_counter, right, right_counter);
        fflush(stdout);

        counter++;

        if (counter % 10 == 0) {
            checkpoint(rank, counter);
        }

        sleep(1);
    }

    MPI_Finalize();
    return 0;
}
