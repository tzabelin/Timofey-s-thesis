/* Heat equation solver in 2D. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "fault.h"
#include "heat.h"

int main(int argc, char **argv)
{
    double a = 0.5;             //!< Diffusion constant
    field current, previous;    //!< Current and previous temperature fields

    double dt;                  //!< Time step
    int nsteps;                 //!< Number of time steps

    int image_interval = 500;    //!< Image output interval

    int restart_interval = 200;  //!< Checkpoint output interval

    parallel_data parallelization; //!< Parallelization info

    int iter, iter0;               //!< Iteration counter

    double dx2, dy2;            //!< delta x and y squared

    double start_clock;        //!< Time stamps
    
    /* Pointers to hold neighbor checkpoint data */
    double *neighbor_up    = NULL;
    double *neighbor_down  = NULL;
    double *neighbor_left  = NULL;
    double *neighbor_right = NULL;
    
    MPI_Init(&argc, &argv);

    initialize(argc, argv, &current, &previous, &nsteps, 
               &parallelization, &iter0);

    /* Output the initial field */
    write_field(&current, iter0, &parallelization);
    iter0++;

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
        exchange_finalize(&parallelization);
        evolve_edges(&current, &previous, a, dt);
        if (iter % image_interval == 0) {
            write_field(&current, iter, &parallelization);
        }
       /* write a checkpoint now and then for easy restarting */
        if (iter % restart_interval == 0) {
            /* Free the allocated neighbor checkpoint data buffers */
            if (neighbor_up) {
                free(neighbor_up);
                neighbor_up = NULL;
            }
            if (neighbor_down) {
                free(neighbor_down);
                neighbor_down = NULL;
            }
            if (neighbor_left) {
                free(neighbor_left);
                neighbor_left = NULL;
            }
            if (neighbor_right) {
                free(neighbor_right);
                neighbor_right = NULL;
            }
            save_neighbor(&current, &parallelization,
                &neighbor_up, &neighbor_down,
                &neighbor_left, &neighbor_right);
        }
        /* Swap current field so that it will be used
            as previous for next iteration step */
        swap_fields(&current, &previous);
    }
    /* Determine the CPU time used for the iteration */
    if (parallelization.rank == 0) {
        printf("Iteration took %.3f seconds.\n", (MPI_Wtime() - start_clock));
        printf("Reference value at 5,5: %f\n", 
                        previous.data[idx(5, 5, current.ny + 2)]);
    }

    write_field(&current, iter, &parallelization);

    finalize(&current, &previous, &parallelization);
    MPI_Barrier(parallelization.comm);
    MPI_Finalize();

    return 0;
}
