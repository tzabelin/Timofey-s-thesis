#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>

#define COMPUTATION_TIME 5
#define SAVE_STEP        10
#define DISK_LATENCY     50
#define FAILURE_RATE     0.2
#define FAILURE_INTERVAL 5

void get_time_str(char *buffer, int size, int time_val) {
    int hours   = time_val / 3600;
    int minutes = (time_val % 3600) / 60;
    int seconds = time_val % 60;
    snprintf(buffer, size, "%02d-%02d-%02d", hours, minutes, seconds);
}

void checkpoint(int rank, int counter, int *time_counter, int *last_checkpoint, FILE *out) {
    *time_counter += DISK_LATENCY;
    *last_checkpoint = counter;
    char time_str_buf[16];
    get_time_str(time_str_buf, sizeof(time_str_buf), *time_counter);
    fprintf(out, "[%s] Rank %d: *** In-memory checkpoint *** (counter = %d)\n",
            time_str_buf, rank, counter);
    fflush(out);
}

void restore_checkpoint(int rank, int *time_counter, int *counter, int *last_checkpoint, FILE *out) {
    *time_counter += DISK_LATENCY;
    *counter = *last_checkpoint;
    char time_str_buf[16];
    get_time_str(time_str_buf, sizeof(time_str_buf), *time_counter);
    fprintf(out, "[%s] Rank %d: *** Rolling back *** (restored counter = %d)\n",
            time_str_buf, rank, *counter);
    fflush(out);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    char filename[64];
    snprintf(filename, sizeof(filename), "rank_%d.log", rank);
    FILE *out = fopen(filename, "w");
    if (!out) {
        fprintf(stderr, "Rank %d: Error opening file %s\n", rank, filename);
        MPI_Abort(comm, 1);
        exit(1);
    }

    fprintf(out, "COMPUTATION_TIME=%d, SAVE_STEP=%d, DISK_LATENCY=%d, FAILURE_RATE=%f, FAILURE_INTERVAL=%d\n",
            COMPUTATION_TIME, SAVE_STEP, DISK_LATENCY, FAILURE_RATE, FAILURE_INTERVAL);
    fflush(out);

    srand(time(NULL) + rank);
    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    int dummy_time = 0;
    int counter = rank;
    int last_checkpoint = counter;

    char time_str_buf[16];
    get_time_str(time_str_buf, sizeof(time_str_buf), dummy_time);
    fprintf(out, "[%s] Rank %d (PID %lu) started with counter = %d\n",
            time_str_buf, rank, (unsigned long)getpid(), counter);
    fflush(out);

    while (counter < 1000) {
        dummy_time += COMPUTATION_TIME;

        int failed_local  = 0;
        for(int i = 0; i < (COMPUTATION_TIME/FAILURE_INTERVAL); i++) {
            double r = rand() / (double)RAND_MAX;
            if (r < FAILURE_RATE) {
                failed_local = 1;
            }
        }

        int failed_global = 0;
        MPI_Allreduce(&failed_local, &failed_global, 1, MPI_INT, MPI_SUM, comm);

        if (failed_global > 0) {
            restore_checkpoint(rank, &dummy_time, &counter, &last_checkpoint, out);
        }
        else
        {
            {
                MPI_Status status;
                int err;
                int left_counter  = 0;
                int right_counter = 0;

                err = MPI_Sendrecv(&counter, 1, MPI_INT, right, 0,
                                &left_counter, 1, MPI_INT, left, 0,
                                comm, &status);
                if (err != MPI_SUCCESS) {
                    fprintf(out, "Rank %d detected a failure in Sendrecv (left). Aborting job...\n", rank);
                    fflush(out);
                    MPI_Abort(comm, 99);
                    exit(1);
                }

                err = MPI_Sendrecv(&counter, 1, MPI_INT, left, 1,
                                &right_counter, 1, MPI_INT, right, 1,
                                comm, &status);
                if (err != MPI_SUCCESS) {
                    fprintf(out, "Rank %d detected a failure in Sendrecv (right). Aborting job...\n", rank);
                    fflush(out);
                    MPI_Abort(comm, 99);
                    exit(1);
                }

                get_time_str(time_str_buf, sizeof(time_str_buf), dummy_time);
                fprintf(out, "[%s] Rank %d (PID %lu): my counter=%d, left(rank %d)=%d, right(rank %d)=%d\n",
                        time_str_buf, rank, (unsigned long)getpid(),
                        counter, left, left_counter, right, right_counter);
                fflush(out);
            }

            counter += size;

            if ((counter - rank) % SAVE_STEP == 0) {
                checkpoint(rank, counter, &dummy_time, &last_checkpoint, out);
            }
        }
    }

    fclose(out);
    MPI_Finalize();
    return 0;
}
