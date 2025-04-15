/* Setup routines for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "heat.h"
#include "pngwriter.h"
#include "fault.h"

#define NSTEPS 1000  // Default number of iteration steps


/* Initialize the heat equation solver */
void initialize(int argc, char *argv[], field *current,
                field *previous, int *nsteps, neighbor_data_buffers *neighbor_checkpoint_buffers, parallel_data *parallel, 
                int *iter0)
{
    /*
     * Following combinations of command line arguments are possible:
     * No arguments:    use default field dimensions and number of time steps
     * One argument:    read initial field from a given file
     * Two arguments:   initial field from file and number of time steps
     * Three arguments: field dimensions (rows,cols) and number of time steps
     */


    rows = 2000;             //!< Field dimensions with default values
    cols = 2000;

    //TP: multiply by two for a safety factor
    char input_file[64*2];        //!< Name of the optional input file

    int read_file = 0;

    *nsteps = NSTEPS;
    *iter0 = 0;

    switch (argc) {
    case 1:
        /* Use default values */
        break;
    case 2:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_file = 1;
        break;
    case 3:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_file = 1;

        /* Number of time steps */
        *nsteps = atoi(argv[2]);
        break;
    case 4:
        /* Field dimensions */
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        /* Number of time steps */
        *nsteps = atoi(argv[3]);
        break;
    default:
        printf("Error: Unsupported number of command line arguments\n");
        exit(-1);
    }

   //HERE WAS CODE FOR READING THE CHECKPOINT. I CHANGED IT.
    MPI_Comm parent_comm;
    MPI_Comm_get_parent(&parent_comm);

    if (parent_comm != MPI_COMM_NULL) {
        int rank; // Rank within the group of newly spawned processes
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Just to put a number in rank
        printf("[Spawned Rank %d] Detected parent communicator. Participating in recovery.\n", rank);

        MPI_Comm new_comm, reordered_comm;
        int parent_size, new_rank, new_size, num_failed;
        int *failed_ranks = NULL;
        int my_original_rank = -1;
        int rc;

        // 1. Get size of parent group (survivors) for later
        rc = MPI_Comm_remote_size(parent_comm, &parent_size);
        if (rc != MPI_SUCCESS) { fprintf(stderr, "[Spawned Rank %d] Error getting parent size: %d\n", rank, rc); MPI_Abort(MPI_COMM_WORLD, rc); }

        rc = MPI_Intercomm_merge(parent_comm, 1 /* high */, &new_comm);
        if (rc != MPI_SUCCESS) { fprintf(stderr, "[Spawned Rank %d] Error merging communicator: %d\n", rank, rc); MPI_Abort(MPI_COMM_WORLD, rc); }
        MPI_Comm_free(&parent_comm); // Don't need the intercomm anymore

        MPI_Comm_rank(new_comm, &new_rank);
        MPI_Comm_size(new_comm, &new_size);
        printf("[Spawned Rank %d -> %d/%d in merged] Merge successful.\n", rank, new_rank, new_size);

        // Receive list of original failed ranks from leader (rank 0 in new_comm)
        num_failed = new_size - parent_size;
        if (num_failed <= 0) {
             fprintf(stderr, "[Spawned Rank %d -> %d/%d] Error: num_failed (%d) is not positive after merge.\n", rank, new_rank, new_size, num_failed);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
        failed_ranks = (int*)malloc(num_failed * sizeof(int));
        if (!failed_ranks) { MPI_Abort(MPI_COMM_WORLD, 1); }

        rc = MPI_Bcast(failed_ranks, num_failed, MPI_INT, 0, new_comm); // Rank 0 in new_comm is the survivor leader
        if (rc != MPI_SUCCESS) { fprintf(stderr, "[Spawned Rank %d -> %d/%d] Error receiving failed ranks: %d\n", rank, new_rank, new_size, rc); MPI_Abort(MPI_COMM_WORLD, rc); }
        printf("[Spawned Rank %d -> %d/%d] Received %d failed ranks.\n", rank, new_rank, new_size, num_failed);

        // Determine original rank
        int my_spawn_index = new_rank - parent_size; // Index among the spawned processes (0-based)
        if (my_spawn_index < 0 || my_spawn_index >= num_failed) {
            fprintf(stderr, "[Spawned Rank %d -> %d/%d] Error: Invalid spawn index %d (num_failed=%d)\n", rank, new_rank, new_size, my_spawn_index, num_failed);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        my_original_rank = failed_ranks[my_spawn_index];
        printf("[Spawned Rank %d -> %d/%d] My target original rank is %d.\n", rank, new_rank, new_size, my_original_rank);
        free(failed_ranks); // Don't need this anymore

        rc = MPI_Comm_split(new_comm, 0 /* color */, my_original_rank /* key */, &reordered_comm);
        if (rc != MPI_SUCCESS) { fprintf(stderr, "[Spawned Rank %d -> %d/%d] Error splitting communicator: %d\n", rank, new_rank, new_size, rc); MPI_Abort(MPI_COMM_WORLD, rc); }
        MPI_Comm_free(&new_comm); // Don't need the merged comm anymore

        int final_rank, final_size;
        MPI_Comm_rank(reordered_comm, &final_rank);
        MPI_Comm_size(reordered_comm, &final_size);
        if(final_rank != my_original_rank) {
             fprintf(stderr, "[Spawned Rank %d -> %d/%d] *** ERROR ***: Final rank %d does not match target original rank %d after split!\n", rank, new_rank, new_size, final_rank, my_original_rank);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("[Spawned Rank %d -> %d/%d] Split successful. Final Rank %d/%d.\n", rank, new_rank, new_size, final_rank, final_size);

        // Now setup the parallel structure with the correct communicator and dimensions
        printf("[Spawned Rank %d] Setting up parallel structure with new dimensions: rows=%d, cols=%d\n", final_rank, rows, cols);
        parallel_setup(parallel, rows, cols, reordered_comm);

        // Idk what happened, abort
         if(parallel->rank != final_rank || parallel->size != final_size) {
             fprintf(stderr, "[Rank %d] *** ERROR ***: Parallel structure rank/size (%d/%d) mismatch final comm (%d/%d)!\n", final_rank, parallel->rank, parallel->size, final_rank, final_size);
             MPI_Abort(MPI_COMM_WORLD, 1);
         }

        set_field_dimensions(current, rows, cols, parallel);
        set_field_dimensions(previous, rows, cols, parallel);
        generate_field(current, parallel);
        allocate_field(previous);
        copy_field(current, previous);


        printf("[Rank %d] Initialization complete for spawned process.\n", final_rank);
        
        initialize_neighbor_buffers(parallel, neighbor_checkpoint_buffers);
        process_restart(current, parallel, iter0, neighbor_checkpoint_buffers, 1);
    } else {
        parallel_setup(parallel, rows, cols, MPI_COMM_WORLD);
        if (parallel->rank == 0)
            printf("Generating default data from: rows, columns, nsteps: %d, %d\n", rows, cols);
        set_field_dimensions(current, rows, cols, parallel);
        set_field_dimensions(previous, rows, cols, parallel);
        generate_field(current, parallel);
        allocate_field(previous);
        copy_field(current, previous);
        initialize_neighbor_buffers(parallel, neighbor_checkpoint_buffers);
    }
}

/* Generate initial temperature field.  Pattern is disc with a radius
 * of nx_full / 6 in the center of the grid.
 * Boundary conditions are (different) constant temperatures outside the grid */
void generate_field(field *temperature, parallel_data *parallel)
{
    int i, j, ind, width;
    double radius;
    int dx, dy;
    int dims[2], coords[2], periods[2];

    /* Allocate the temperature array, note that
     * we have to allocate also the ghost layers */
    temperature->data =
        malloc_2d(temperature->nx + 2, temperature->ny + 2);

    MPI_Cart_get(parallel->comm, 2, dims, periods, coords);

    /* Radius of the source disc */
    radius = temperature->nx_full / 6.0;

    width = temperature->ny + 2;
    for (i = 0; i < temperature->nx + 2; i++) {
        for (j = 0; j < temperature->ny + 2; j++) {
            /* Distance of point i, j from the origin */
            dx = i + coords[0] * temperature->nx -
                 temperature->nx_full / 2 + 1;
            dy = j + coords[1] * temperature->ny -
                 temperature->ny_full / 2 + 1;
            ind = idx(i, j, width);
            if (dx * dx + dy * dy < radius * radius) {
                temperature->data[ind] = 5.0;
            } else {
                temperature->data[ind] = 65.0;
            }
        }
    }

    /* Boundary conditions */
    // Left boundary
    if (coords[1] == 0) {
        for (i = 0; i < temperature->nx + 2; i++) {
            ind = idx(i, 0, width);
            temperature->data[ind] = 20.0;
        }
    }
    // Right boundary
    if (coords[1] == dims[1] - 1) {
        for (i = 0; i < temperature->nx + 2; i++) {
            ind = idx(i, temperature->ny + 1, width);
            temperature->data[ind] = 70.0;
        }
    }
    // Upper boundary
    if (coords[0] == 0) {
        for (j = 0; j < temperature->ny + 2; j++) {
            ind = idx(0, j, width);
            temperature->data[ind] = 85.0;
        }
    }
    // Lower boundary
    if (coords[0] == dims[0] - 1) {
        for (j = 0; j < temperature->ny + 2; j++) {
            ind = idx(temperature->nx + 1, j, width);
            temperature->data[ind] = 5.0;
        }
    }

}

/* Set dimensions of the field. Note that the nx is the size of the first
 * dimension and ny the second. */
void set_field_dimensions(field *temperature, int nx, int ny,
                          parallel_data *parallel)
{
    int nx_local, ny_local;
    int dims[2], coords[2], periods[2];

    MPI_Cart_get(parallel->comm, 2, dims, periods, coords);
    nx_local = nx / dims[0];
    ny_local = ny / dims[1];

    temperature->dx = DX;
    temperature->dy = DY;
    temperature->nx = nx_local;
    temperature->ny = ny_local;
    temperature->nx_full = nx;
    temperature->ny_full = ny;
}

void parallel_setup(parallel_data *parallel, int nx, int ny, MPI_Comm comm)
{
    int nx_local;
    int ny_local;
    int world_size;
    int dims[2] = {0, 0};
    int periods[2] = { 0, 0 };

    /* Set grid dimensions */
    MPI_Comm_size(comm, &world_size);
    MPI_Dims_create(world_size, 2, dims);
    nx_local = nx / dims[0];
    ny_local = ny / dims[1];

    if (nx_local * dims[0] != nx) {
        printf("Cannot divide grid evenly to processors in x-direction "
               "%d x %d != %d\n", nx_local, dims[0], nx);
        fflush(stdout);
        MPI_Abort(comm, -2);
    }
    if (ny_local * dims[1] != ny) {
        printf("Cannot divide grid evenly to processors in y-direction "
               "%d x %d != %d\n", ny_local, dims[1], ny);
        fflush(stdout);
        MPI_Abort(comm, -2);
    }

    /* Create cartesian communicator */
    MPI_Barrier(comm);
    MPI_Cart_create(comm, 2, dims, periods, 1, &parallel->comm);
    MPI_Cart_shift(parallel->comm, 0, 1, &parallel->nup, &parallel->ndown);
    MPI_Cart_shift(parallel->comm, 1, 1, &parallel->nleft,
                   &parallel->nright);

    MPI_Comm_size(parallel->comm, &parallel->size);
    MPI_Comm_rank(parallel->comm, &parallel->rank);
    printf("Rank %d: Setting up parallel environment\n", parallel->rank);

    //if (parallel->rank == 0) {
        printf("Rank %d Using domain decomposition %d x %d\n", parallel->rank, dims[0], dims[1]);
        printf("Rank %d Local domain size %d x %d\n", parallel->rank, nx_local, ny_local);
    //}

    /* Create datatypes for halo exchange */
    MPI_Type_vector(nx_local + 2, 1, ny_local + 2, MPI_DOUBLE,
                    &parallel->columntype);
    MPI_Type_contiguous(ny_local + 2, MPI_DOUBLE, &parallel->rowtype);
    MPI_Type_commit(&parallel->columntype);
    MPI_Type_commit(&parallel->rowtype);

    /* Create datatype for subblock needed in text I/O
     *   Rank 0 uses datatype for receiving data into full array while
     *   other ranks use datatype for sending the inner part of array */
    int sizes[2] = {nx_local + 2, ny_local + 2};
    int subsizes[2] = { nx_local, ny_local };
    int offsets[2] = {1, 1};
    if (parallel->rank == 0) {
        sizes[0] = nx;
        sizes[1] = ny;
        offsets[0] = 0;
        offsets[1] = 0;
    }

    MPI_Type_create_subarray(2, sizes, subsizes, offsets, MPI_ORDER_C,
                             MPI_DOUBLE, &parallel->subarraytype);
    MPI_Type_commit(&parallel->subarraytype);

    /* Create datatypes for restart I/O
     * For boundary ranks also the ghost layer (boundary condition) 
     * is written */

    int coords[2];
    MPI_Cart_coords(parallel->comm, parallel->rank, 2, coords);
    sizes[0] = nx + 2;
    sizes[1] = ny + 2;
    offsets[0] = 1 + coords[0] * nx_local;
    offsets[1] = 1 + coords[1] * ny_local;
    if (coords[0] == 0) {
       offsets[0] -= 1;
       subsizes[0] += 1;
    }
    if (coords[0] == dims[0] - 1) {
       subsizes[0] += 1;
    }
    if (coords[1] == 0) {
       offsets[1] -= 1;
       subsizes[1] += 1;
    }
    if (coords[1] == dims[1] - 1) {
       subsizes[1] += 1;
    }

    MPI_Type_create_subarray(2, sizes, subsizes, offsets, MPI_ORDER_C,
                             MPI_DOUBLE, &parallel->filetype);
    MPI_Type_commit(&parallel->filetype);

    sizes[0] = nx_local + 2;
    sizes[1] = ny_local + 2;
    offsets[0] = 1;
    offsets[1] = 1;
    if (coords[0] == 0) {
       offsets[0] = 0;
    }
    if (coords[1] == 0) {
       offsets[1] = 0;
    }

    MPI_Type_create_subarray(2, sizes, subsizes, offsets, MPI_ORDER_C,
                             MPI_DOUBLE, &parallel->restarttype);
    MPI_Type_commit(&parallel->restarttype);
    set_error_handler(parallel->comm);
}

/* Deallocate the 2D arrays of temperature fields */
void finalize(field *temperature1, field *temperature2,
              parallel_data *parallel)
{
    free_2d(temperature1->data);
    free_2d(temperature2->data);

    MPI_Type_free(&parallel->rowtype);
    MPI_Type_free(&parallel->columntype);
    MPI_Type_free(&parallel->subarraytype);
    MPI_Type_free(&parallel->restarttype);
    MPI_Type_free(&parallel->filetype);

}

