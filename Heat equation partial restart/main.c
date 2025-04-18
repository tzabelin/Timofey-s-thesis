/* Heat equation solver in 2D. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "heat.h"
#include "fault.h"

int nsteps;

int main(int argc, char **argv)
{
    double a = 0.5;             //!< Diffusion constant
    field current, previous;    //!< Current and previous temperature fields

    double dt;                  //!< Time step
    //int nsteps;                 //!< Number of time steps

    int image_interval = 500;    //!< Image output interval

    int restart_interval = 200;  //!< Checkpoint output interval

    parallel_data parallelization; //!< Parallelization info

    int iter, iter0;               //!< Iteration counter

    double dx2, dy2;            //!< delta x and y squared

    double start_clock;        //!< Time stamps

    /* CHANGE Structure to hold neighbor checkpoint data */
    neighbor_data_buffers neighbor_checkpoint_buffers;

    MPI_Init(&argc, &argv);
    initialize(argc, argv, &current, &previous, &nsteps, &neighbor_checkpoint_buffers,
               &parallelization, &iter0);

    /* Output the initial field */
    //write_field(&current, iter0, &parallelization);
    MPI_Comm parent_comm;
    MPI_Comm_get_parent(&parent_comm);

    if (parent_comm == MPI_COMM_NULL) {
    iter0++; //PAY ATTENTION<--->> FOR NEW PROCESSES IT WILL MEAN START +1 after checkpoint
    }

    /* Largest stable time step */
    dx2 = current.dx * current.dx;
    dy2 = current.dy * current.dy;
    dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    /* Get the start time stamp */
    start_clock = MPI_Wtime();

    /* Time evolve */
    for (iter = iter0; iter < iter0 + nsteps; iter++) {
        exchange_init(&previous, &parallelization);
        evolve_interior(&current, &previous, a, dt);
        int rc = exchange_finalize(&parallelization);
        if (rc != MPI_SUCCESS) {
            printf("Error in exchange_init, rank %d\n", parallelization.rank);
            parallel_setup(&parallelization, rows, cols, parallelization.comm);
            process_restart(&current, &parallelization, &iter, &neighbor_checkpoint_buffers, 0);
        }
        else{
            evolve_edges(&current, &previous, a, dt);
            if (iter % image_interval == 0) {
                write_field(&current, iter, &parallelization);
            }
        /* write a checkpoint now and then for easy restarting */
            if (iter % restart_interval == 0) {
                //CHANGED
                save_neighbor(&current, &parallelization, &neighbor_checkpoint_buffers, iter);
            }
            /* Swap current field so that it will be used
                as previous for next iteration step */
            swap_fields(&current, &previous);
        }
        printf("Rank %d: Iteration %d\n", parallelization.rank, iter);
    }

    printf("Rank %d: Iteration %d Finished\n", parallelization.rank, iter);
    /* Determine the CPU time used for the iteration */
    if (parallelization.rank == 0) {
        printf("Iteration took %.3f seconds.\n", (MPI_Wtime() - start_clock));
        printf("Reference value at 5,5: %f\n", 
                        previous.data[idx(5, 5, current.ny + 2)]);
    }

    write_field(&current, iter, &parallelization);

    finalize(&current, &previous, &parallelization);
    //CHANGED
    free_neighbor_data(&neighbor_checkpoint_buffers);
    MPI_Barrier(parallelization.comm);
    MPI_Finalize();

    return 0;
}
