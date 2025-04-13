#include <mpi.h>
#include <mpi-ext.h>
#include <stdlib.h>
#include "heat.h"
#include "fault.h"

int initialize_neighbor_buffers(parallel_data *parallel, neighbor_data_buffers *buffers)
{
    MPI_Aint lb;

    buffers->up = NULL;
    buffers->down = NULL;
    buffers->left = NULL;
    buffers->right = NULL;
    buffers->self = NULL;

    buffers->raw_ptr_up = NULL;
    buffers->raw_ptr_down = NULL;
    buffers->raw_ptr_left = NULL;
    buffers->raw_ptr_right = NULL;
    buffers->raw_ptr_self_buffer = NULL;

    int mpi_ret = MPI_Type_get_extent(parallel->restarttype, &lb, &buffers->extent);
    if (mpi_ret != MPI_SUCCESS) {
        printf("[Rank %d] Error: MPI_Type_get_extent failed with code %d.\n",
                parallel->rank, mpi_ret);
        return -1;
    }
    if (buffers->extent <= 0) {
        printf("[Rank %d] Error: Invalid extent (%ld) for restarttype.\n",
                parallel->rank, (long)buffers->extent);
        return -1;
    }

    if (parallel->nup != MPI_PROC_NULL) {
        buffers->raw_ptr_up = malloc(buffers->extent);
        if (!buffers->raw_ptr_up) {
            printf("[Rank %d] Failed to allocate memory for neighbor up buffer");
            return -1;
        }
        buffers->up = (double*)((char*)buffers->raw_ptr_up - lb);
    }

    if (parallel->ndown != MPI_PROC_NULL) {
        buffers->raw_ptr_down = malloc(buffers->extent);
        if (!buffers->raw_ptr_down) {
            printf("[Rank %d] Failed to allocate memory for neighbor down buffer");
            free_neighbor_data(buffers);
            return -1;
        }
        buffers->down = (double*)((char*)buffers->raw_ptr_down - lb);
    }

    if (parallel->nleft != MPI_PROC_NULL) {
        buffers->raw_ptr_left = malloc(buffers->extent);
        if (!buffers->raw_ptr_left) {
            printf("[Rank %d] Failed to allocate memory for neighbor left buffer");
            free_neighbor_data(buffers);
            return -1;
        }
        buffers->left = (double*)((char*)buffers->raw_ptr_left - lb);
    }

    if (parallel->nright != MPI_PROC_NULL) {
        buffers->raw_ptr_right = malloc(buffers->extent);
        if (!buffers->raw_ptr_right) {
            printf("[Rank %d] Failed to allocate memory for neighbor right buffer");
            free_neighbor_data(buffers);
            return -1;
        }
        buffers->right = (double*)((char*)buffers->raw_ptr_right - lb);
    }

    buffers->raw_ptr_self_buffer = malloc(buffers->extent);
    if (!buffers->raw_ptr_self_buffer) {
        printf("[Rank %d] Failed to allocate memory for self buffer");
        free_neighbor_data(buffers);
        return -1;
    }
    buffers->self = (double*)((char*)buffers->raw_ptr_self_buffer - lb);


    return 0;
}




void save_neighbor(field *temperature, parallel_data *parallel,
                        neighbor_data_buffers *buffers, int iter)
{
    MPI_Request requests[8];
    int req_count = 0;

    if (parallel->nup != MPI_PROC_NULL) {
        MPI_Irecv(buffers->up, 1, parallel->restarttype,
                  parallel->nup, 21, parallel->comm, &requests[req_count++]);
    }
    if (parallel->ndown != MPI_PROC_NULL) {
        MPI_Irecv(buffers->down, 1, parallel->restarttype,
                  parallel->ndown, 21, parallel->comm, &requests[req_count++]);
    }
    if (parallel->nleft != MPI_PROC_NULL) {
        MPI_Irecv(buffers->left, 1, parallel->restarttype,
                  parallel->nleft, 21, parallel->comm, &requests[req_count++]);
    }
    if (parallel->nright != MPI_PROC_NULL) {
        MPI_Irecv(buffers->right, 1, parallel->restarttype,
                  parallel->nright, 21, parallel->comm, &requests[req_count++]);
    }

    if (parallel->nup != MPI_PROC_NULL) {
        MPI_Isend(temperature->data, 1, parallel->restarttype,
                  parallel->nup, 21, parallel->comm, &requests[req_count++]);
    }
    if (parallel->ndown != MPI_PROC_NULL) {
        MPI_Isend(temperature->data, 1, parallel->restarttype,
                  parallel->ndown, 21, parallel->comm, &requests[req_count++]);
    }
    if (parallel->nleft != MPI_PROC_NULL) {
        MPI_Isend(temperature->data, 1, parallel->restarttype,
                  parallel->nleft, 21, parallel->comm, &requests[req_count++]);
    }
    if (parallel->nright != MPI_PROC_NULL) {
        MPI_Isend(temperature->data, 1, parallel->restarttype,
                  parallel->nright, 21, parallel->comm, &requests[req_count++]);
    }

    memcpy(buffers->self, temperature->data, buffers->extent);

    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    buffers->nx_full = temperature->nx_full;
    buffers->ny_full = temperature->ny_full;

    buffers->iteration = iter;
}

void error_handler(MPI_Comm *comm, int *error_code, ...)
{
    int initial_size, initial_dims, initial_rank;

    int eclass;
    MPI_Error_class(*error_code, &eclass);

    // Only handle process failure errors with ULFM features
    if (MPI_ERR_PROC_FAILED != eclass && MPIX_ERR_REVOKED != eclass) {
        // Not a failure we can handle here, print error and abort
        char err_string[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(*error_code, err_string, &len);
        printf("[Rank %d] Unrecoverable MPI Error: %s\n", initial_rank, err_string);
        MPI_Abort(MPI_COMM_WORLD, *error_code); // Use initial_rank if available
        return; // Should not be reached
    }

    // --- Recovery process starts ---
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get rank in WORLD for messages before comm is fixed
    printf("[WorldRank %d / InitialRank %d] Info: Entering error handler for comm %p. Error code: %d\n",
            world_rank, initial_rank, (void*)*comm, *error_code);


    // 1. Revoke the communicator
    MPIX_Comm_revoke(*comm);
    MPI_Barrier(comm); // Ensure all processes are synchronized before proceeding

    // 2. Acknowledge the failure (needed for subsequent ULFM calls)
    MPIX_Comm_failure_ack(*comm);

    // 3. Get the group of failed processes (processes that acknowledged)
    MPI_Group failed_group;
    MPIX_Comm_failure_get_acked(*comm, &failed_group);
    int nfailed;
    MPI_Group_size(failed_group, &nfailed);
    MPI_Group_free(&failed_group); // Free the group handle

    if (nfailed <= 0) {
        printf("[WorldRank %d / InitialRank %d] Error: Handler invoked, but no failed ranks detected by MPIX_Comm_failure_get_acked. Aborting.\n",
                world_rank, initial_rank);
        // It's possible the error was MPIX_ERR_REVOKED on a process that didn't fail itself
        // If nfailed is 0, we might still need to shrink and proceed if the comm was revoked.
        // However, spawning 0 processes will fail later. Let's check shrink result.
       MPI_Abort(*comm, 1); // Abort seems safest if state is unexpected
    } else {
         printf("[WorldRank %d / InitialRank %d] Info: Detected %d failed process(es).\n",
                 world_rank, initial_rank, nfailed);
    }


    // 4. Create a new communicator containing only surviving processes
    MPI_Comm survivors_comm;
    MPIX_Comm_shrink(*comm, &survivors_comm);
    int s_rank, s_size;
    MPI_Comm_rank(survivors_comm, &s_rank);
    MPI_Comm_size(survivors_comm, &s_size);

    printf("[WorldRank %d / InitialRank %d] Info: Shrunk communicator created (size %d). My rank in survivors_comm: %d.\n",
            world_rank, initial_rank, s_size, s_rank);


    // We need the initial ranks of the survivors to figure out which initial ranks failed.
    int *survivor_initial_ranks = (int*)malloc(s_size * sizeof(int));
    if (!survivor_initial_ranks) {
         printf("[Rank %d] Error: Failed to allocate memory for survivor initial ranks. Aborting.\n", s_rank);
         MPI_Abort(survivors_comm, 1);
    }

    // Gather all initial_rank values from survivors onto rank 0 of survivors_comm
    MPI_Gather(&initial_rank, 1, MPI_INT,
               survivor_initial_ranks, 1, MPI_INT, 0, // Root is rank 0 in survivors_comm
               survivors_comm);

    int *failed_initial_ranks = NULL;
    if (s_rank == 0) {
        failed_initial_ranks = (int*)malloc(nfailed * sizeof(int));
         if (!failed_initial_ranks) {
             printf("[Rank %d] Error: Failed to allocate memory for failed initial ranks. Aborting.\n", s_rank);
             MPI_Abort(survivors_comm, 1);
        }

        // Determine which initial ranks are missing
        int failed_count = 0;
        for (int r = 0; r < initial_size; ++r) {
            int found = 0;
            for (int i = 0; i < s_size; ++i) {
                if (survivor_initial_ranks[i] == r) {
                    found = 1;
                    break;
                }
            }
            if (found == 0) {
                if (failed_count < nfailed) {
                    failed_initial_ranks[failed_count++] = r;
                     printf("[Rank %d] Info: Identified initial_rank %d as failed.\n", s_rank, r);
                } else {
                     printf("[Rank %d] Warning: Found more missing ranks than reported failures (%d > %d)! Something is inconsistent.\n", s_rank, failed_count+1, nfailed);
                     // Decide how to handle this - maybe abort? For now, proceed with nfailed.
                }
            }
        }
         if (failed_count != nfailed) {
            printf("[Rank %d] Error: Number of identified missing ranks (%d) does not match reported failures (%d). Aborting.\n", s_rank, failed_count, nfailed);
            free(survivor_initial_ranks);
            free(failed_initial_ranks);
            MPI_Abort(survivors_comm, 1);
        }
    }

    // 5. Spawn replacement processes
    //    All survivors must call spawn collectively.
    //    We assume the executable is the same as the current one.
    //    argv[0] might not be reliable, provide executable name explicitly if needed.
    char *spawn_argv[] = { NULL }; // No specific args needed for replacements in this example
    MPI_Comm intercomm;
    int *spawn_errcodes = (int*)malloc(nfailed * sizeof(int)); // Error codes for each spawned process
     if (!spawn_errcodes && nfailed > 0) { // Check nfailed > 0 for allocation
         printf("[Rank %d] Error: Failed to allocate memory for spawn error codes. Aborting.\n", s_rank);
         free(survivor_initial_ranks); // Clean up previous allocation
         if(s_rank == 0) free(failed_initial_ranks);
         MPI_Abort(survivors_comm, 1);
    }

    printf("[Rank %d / InitialRank %d] Info: Spawning %d replacement processes...\n", s_rank, initial_rank, nfailed);
    int spawn_ret = MPI_Comm_spawn("./heat_ulm", // Adjust executable name if necessary
                       spawn_argv, nfailed, MPI_INFO_NULL,
                       0, // Root rank within survivors_comm initiating spawn
                       survivors_comm, // The intracommunicator of survivors
                       &intercomm,     // The resulting intercommunicator
                       (nfailed > 0) ? spawn_errcodes : MPI_ERRCODES_IGNORE); // Ignore if nfailed is 0

    if (spawn_ret != MPI_SUCCESS) {
        char err_string[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(spawn_ret, err_string, &len);
        printf("[Rank %d] Error: MPI_Comm_spawn failed: %s. Aborting.\n", s_rank, err_string);
        free(survivor_initial_ranks);
        if(s_rank == 0) free(failed_initial_ranks);
        if(nfailed > 0) free(spawn_errcodes);
        MPI_Abort(survivors_comm, spawn_ret);
    }
     printf("[Rank %d / InitialRank %d] Info: Spawn call successful.\n", s_rank, initial_rank);
     if (nfailed > 0) free(spawn_errcodes); // Don't need these after checking return code


    // 6. Assign original ranks (initial_rank) to replacements
    //    The leader (rank 0 in survivors_comm) sends each new process its assigned initial_rank.
    //    New processes will receive this after MPI_Init via MPI_Comm_get_parent.
    if (s_rank == 0) {
        for (int i = 0; i < nfailed; ++i) {
            printf("[Rank %d] Info: Sending initial_rank %d to new process %d.\n", s_rank, failed_initial_ranks[i], i);
            MPI_Send(&failed_initial_ranks[i], 1, MPI_INT,
                     i, // Send to rank 'i' in the remote group (the spawned processes)
                     99, // Tag for initial rank assignment
                     intercomm); // Use the intercommunicator
        }
        free(failed_initial_ranks); // No longer needed
    }
    free(survivor_initial_ranks); // No longer needed by anyone


    // 7. Merge the intercommunicator into a new intracommunicator
    MPI_Comm merged_comm;
     printf("[Rank %d / InitialRank %d] Info: Merging intercommunicator...\n", s_rank, initial_rank);
    MPI_Intercomm_merge(intercomm, 0, &merged_comm); // 0: survivors are "low" group
    MPI_Comm_disconnect(&intercomm); // Disconnect and free intercomm resources

    int merged_rank, merged_size;
    MPI_Comm_rank(merged_comm, &merged_rank);
    MPI_Comm_size(merged_comm, &merged_size);
     printf("[Rank %d / InitialRank %d] Info: Merge successful. New comm size %d, my rank %d.\n", s_rank, initial_rank, merged_size, merged_rank);


    // Sanity check: the new size should match the original size
    if (merged_size != initial_size) {
        printf("[InitialRank %d / MergedRank %d] Error: Merged communicator size (%d) does not match initial size (%d). Aborting.\n",
                initial_rank, merged_rank, merged_size, initial_size);
        MPI_Abort(merged_comm, 1);
    }

    // 8. Reorder the new communicator based on initial_rank to match original topology.
    //    Use MPI_Comm_split with initial_rank as the key. The rank in the split_comm
    //    will correspond to the original rank order.
    MPI_Comm ordered_comm;
    MPI_Comm_split(merged_comm, 0, // color: all in the same group
                   initial_rank,   // key: sort based on original rank
                   &ordered_comm);
    MPI_Comm_free(&merged_comm); // Free the intermediate merged communicator

    int ordered_rank, ordered_size;
    MPI_Comm_rank(ordered_comm, &ordered_rank);
    MPI_Comm_size(ordered_comm, &ordered_size);

    // Sanity check: rank in ordered_comm should match initial_rank
    if (ordered_rank != initial_rank) {
         printf("[InitialRank %d / OrderedRank %d] Error: Rank mismatch after split/reorder! Aborting.\n",
                 initial_rank, ordered_rank);
         MPI_Abort(ordered_comm, 1);
    }
     printf("[InitialRank %d] Info: Reordering successful. My rank %d confirmed.\n", initial_rank, ordered_rank);


    // 9. Recreate the Cartesian topology on the ordered communicator
    MPI_Comm new_cart_comm;
    int periods[2] = {0, 0}; // Assuming non-periodic grid as in original setup
    MPI_Cart_create(ordered_comm, 2, initial_dims, periods,
                    0, // No reordering allowed here, already ordered by split
                    &new_cart_comm);
    MPI_Comm_free(&ordered_comm); // Free the intermediate ordered communicator

    if (new_cart_comm == MPI_COMM_NULL) {
         printf("[InitialRank %d] Error: Failed to recreate Cartesian communicator. Aborting.\n", initial_rank);
         // Cannot easily abort on new_cart_comm if null, use WORLD or self
         MPI_Abort(MPI_COMM_WORLD, 1);
    }
     printf("[InitialRank %d] Info: Cartesian communicator recreated successfully.\n", initial_rank);


    // 10. Update the communicator pointer for the caller
    MPI_Comm_free(comm); // Free the old (revoked) communicator handle pointed to by comm
    *comm = new_cart_comm;

    // 11. Set an error handler on the new communicator
    //     Could recursively set self, or MPI_ERRORS_RETURN/MPI_ERRORS_ARE_FATAL
    MPI_Comm_set_errhandler(*comm, MPI_ERRORS_RETURN); // Or use the custom handler again if nested failures should be handled
    // set_error_handler(*comm); // If you want recursive recovery attempts

    printf("[InitialRank %d] Info: Error handler finished. New communicator created and assigned.\n", initial_rank);
}

void set_error_handler(MPI_Comm comm)
{
    MPI_Errhandler handler;
    MPI_Comm_create_errhandler(error_handler, &handler);
    MPI_Comm_set_errhandler(comm, handler);
}

void free_neighbor_data(neighbor_data_buffers *buffers)
{
    if (buffers)
    {
        if (buffers->raw_ptr_up) free(buffers->raw_ptr_up);
        if (buffers->raw_ptr_down) free(buffers->raw_ptr_down);
        if (buffers->raw_ptr_left) free(buffers->raw_ptr_left);
        if (buffers->raw_ptr_right) free(buffers->raw_ptr_right);
        if (buffers->raw_ptr_self_buffer) free(buffers->raw_ptr_self_buffer);

        buffers->up = NULL;
        buffers->down = NULL;
        buffers->left = NULL;
        buffers->right = NULL;
        buffers->self = NULL;

        buffers->raw_ptr_up = NULL;
        buffers->raw_ptr_down = NULL;
        buffers->raw_ptr_left = NULL;
        buffers->raw_ptr_right = NULL;
        buffers->raw_ptr_self_buffer = NULL;
    }
}


void process_restart(field *temperature, parallel_data *parallel, int *iter)
{
    MPI_File fp;
    MPI_Offset disp;

    int nx, ny;

    // open the file and write the dimensions
    MPI_File_open(MPI_COMM_WORLD, CHECKPOINT, MPI_MODE_RDONLY,
                  MPI_INFO_NULL, &fp);

    // read grid size and current iteration
    MPI_File_read_all(fp, &nx, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_read_all(fp, &ny, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_read_all(fp, iter, 1, MPI_INT, MPI_STATUS_IGNORE);
    // set correct dimensions to MPI metadata
    //parallel_setup(parallel, nx, ny, MPI_COMM_WORLD);
    // set local dimensions and allocate memory for the data
    set_field_dimensions(temperature, nx, ny, parallel);
    allocate_field(temperature);


    disp = 3 * sizeof(int);
    MPI_File_set_view(fp, 0, MPI_DOUBLE, parallel->filetype, "native", 
                      MPI_INFO_NULL);
    MPI_File_read_at_all(fp, disp, temperature->data,
                          1, parallel->restarttype, MPI_STATUS_IGNORE);
    MPI_File_close(&fp);
}