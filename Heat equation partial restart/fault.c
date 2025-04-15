#include <mpi.h>
#include <mpi-ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "heat.h"
#include "fault.h"

#define RESTART_EXCHANGE_TAG 77

extern int rows, cols, nsteps;
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
    if (*error_code != MPI_ERR_PROC_FAILED  && *error_code != MPIX_ERR_REVOKED) 
        MPI_Abort(MPI_COMM_WORLD, *error_code );
    int size, rank, rc;
    int num_failed = 0;
    MPI_Group failed_group;
    int* failed_ranks = NULL;
    int* failed_ranks_global = NULL;
    int is_leader = 0;
    int leader_rank_survivors = 0; // Rank 0 in survivors comm acts as leader
    MPI_Comm survivors_comm, new_comm, reordered_comm, spawn_intercomm;

    // Get rank and size of the communicator *before* the failure was handled
    MPI_Comm_rank(*comm, &rank);
    MPI_Comm_size(*comm, &size);

    printf("[Rank %d/%d] Entering error handler with error code %d.\n", rank, size, *error_code);

    rc = MPIX_Comm_failure_ack(*comm);
    if (rc != MPI_SUCCESS) {
        printf("[Rank %d] ERROR: MPIX_Comm_failure_ack failed: %d\n", rank, rc);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    rc = MPIX_Comm_failure_get_acked(*comm, &failed_group);
     if (rc != MPI_SUCCESS) {
        printf("[Rank %d] ERROR: MPIX_Comm_failure_get_acked failed: %d\n", rank, rc);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Group_size(failed_group, &num_failed);
    printf("[Rank %d] Acknowledged failure. Detected %d failed processes.\n", rank, num_failed);

    
    // MPIX_Comm_agree ensures all survivors agree on the failure state.
    int agreement_flag = (num_failed > 0); // Agree on whether there *was* a failure acked by someone.
    rc = MPIX_Comm_agree(*comm, &agreement_flag);
    if (rc != MPI_SUCCESS) {
         printf("[Rank %d] ERROR: MPIX_Comm_agree failed: %d\n", rank, rc);
         MPI_Abort(MPI_COMM_WORLD, rc);
    }
    printf("[Rank %d] Agreement reached. Failure confirmed: %d\n", rank, agreement_flag);

    if (!agreement_flag) {
         printf("[Rank %d] No failure agreed upon, returning. Strange state?\n", rank);
         MPI_Group_free(&failed_group);
         return;
    }


    rc = MPIX_Comm_shrink(*comm, &survivors_comm);
    if (rc != MPI_SUCCESS) {
        printf("[Rank %d] ERROR: MPIX_Comm_shrink failed: %d\n", rank, rc);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    int survivors_rank, survivors_size;
    MPI_Comm_rank(survivors_comm, &survivors_rank);
    MPI_Comm_size(survivors_comm, &survivors_size);
    printf("[Rank %d -> %d/%d in survivors] Shrink successful.\n", rank, survivors_rank, survivors_size);


    //Spawn
    if (survivors_rank == leader_rank_survivors)
    {
        is_leader = 1;
        printf("[Rank %d is Leader] Spawning %d new processes.\n", rank, num_failed);
        int *errcodes = (int*)malloc(sizeof(int) * num_failed);

        char rows_str[32];
        char cols_str[32];
        char nsteps_str[32];
        snprintf(rows_str, sizeof(rows_str), "%d", rows);
        snprintf(cols_str, sizeof(cols_str), "%d", cols);
        snprintf(nsteps_str, sizeof(nsteps_str), "%d", nsteps);

         char *spawn_argv[] = {
                            rows_str, cols_str, nsteps_str, NULL
                            };
        printf("[Rank %d Leader] Spawn command: %s\n", rank, spawn_argv[0]);
        rc = MPI_Comm_spawn("./heat_mpi", spawn_argv, num_failed, MPI_INFO_NULL,
                            leader_rank_survivors, survivors_comm, &spawn_intercomm, errcodes);
                            printf("[Rank %d Leader] Spawned processes successfully.\n", rank);
        free(errcodes);
        if (rc != MPI_SUCCESS)
        {
             printf("[Rank %d Leader] ERROR: MPI_Comm_spawn failed: %d\n", rank, rc);
             MPI_Abort(MPI_COMM_WORLD, rc);
        }
        printf("[Rank %d is Leader] Spawn successful.\n", rank);
    } 
    else
    {
         rc = MPI_Comm_spawn(NULL, NULL, num_failed, MPI_INFO_NULL,
                            leader_rank_survivors, survivors_comm, &spawn_intercomm, MPI_ERRCODES_IGNORE);
         if (rc != MPI_SUCCESS)
         {
             printf("[Rank %d Survivor] ERROR: MPI_Comm_spawn collective call failed: %d\n", rank, rc);
             MPI_Abort(MPI_COMM_WORLD, rc); // Abort everyone
        }
          printf("[Rank %d Survivor] Participated in spawn.\n", rank);
    }

    // Merge survivors and newly spawned processes
    rc = MPI_Intercomm_merge(spawn_intercomm, 0, &new_comm);
    if (rc != MPI_SUCCESS) {
         printf("[Rank %d Survivor] ERROR: MPI_Intercomm_merge failed: %d\n", rank, rc);
         MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_free(&spawn_intercomm);
    MPI_Comm_free(&survivors_comm);

    int new_rank, new_size;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);
    printf("[Rank %d -> %d/%d in merged comm] Merge successful. Expected size: %d\n", rank, new_rank, new_size, size);


    // Determine original ranks of failed processes and distribute
    MPI_Group comm_group;
    failed_ranks = (int*)malloc(num_failed * sizeof(int));
    if (!failed_ranks) { MPI_Abort(MPI_COMM_WORLD, 1); }

    if (is_leader)
    {
        MPI_Comm_group(*comm, &comm_group);
        int* ranks_in_failed_group = (int*)malloc(num_failed * sizeof(int));
        if (!ranks_in_failed_group) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < num_failed; i++) {
            ranks_in_failed_group[i] = i; // Ranks 0..num_failed-1 within failed_group
        }
        // Translate ranks from failed_group to original comm_group
        MPI_Group_translate_ranks(failed_group, num_failed, ranks_in_failed_group,
                                  comm_group, failed_ranks);
        free(ranks_in_failed_group);
        MPI_Group_free(&comm_group);

        // Sort failed ranks for consistent assignment
        qsort(failed_ranks, num_failed, sizeof(int), (int (*)(const void *, const void *))strcmp);

        printf("[Rank %d Leader] Original ranks of failed processes: ", rank);
        for(int i=0; i<num_failed; ++i) printf("%d ", failed_ranks[i]);
        printf("\n");
    }

    // Broadcast the list of failed original ranks to everyone in the new communicator
    // The leader (rank 0 in the *merged* comm, which was rank 0 in survivors) broadcasts
    printf("[Rank %d Survivor] Broadcasting failed ranks.\n", rank);
    MPI_Bcast(failed_ranks, num_failed, MPI_INT, 0, new_comm);
    printf("[Rank %d Survivor] Broadcast complete.\n", rank);

    // Recreate communicator with original rank ordering
    int my_original_rank = rank;
    int current_new_rank;

    // Split the communicator
    printf("[Rank %d Survivor] Splitting communicator with original rank %d.\n", rank, my_original_rank);
    rc = MPI_Comm_split(new_comm, 0, my_original_rank, &reordered_comm);
    
    MPI_Comm_rank(reordered_comm, &current_new_rank);
    printf("[Rank %d Survivor] Split complete. New rank %d/%d.\n", rank, current_new_rank, new_size);
     if (rc != MPI_SUCCESS) {
         printf("[Rank %d (orig %d)] ERROR: MPI_Comm_split failed: %d\n", current_new_rank, my_original_rank, rc);
         MPI_Abort(MPI_COMM_WORLD, rc);
     }
    MPI_Comm_free(&new_comm);


    MPI_Comm_free(comm);
    *comm = reordered_comm;

    free(failed_ranks);
    MPI_Group_free(&failed_group);


    int final_rank, final_size;
    MPI_Comm_rank(*comm, &final_rank);
    MPI_Comm_size(*comm, &final_size);
    printf("[Rank %d (orig %d)] Exiting error handler. New rank %d/%d. Comm handle updated.\n", current_new_rank, my_original_rank, final_rank, final_size);

    //TF, we are cooked
    if (final_rank != my_original_rank)
    {
         printf("[Rank %d (orig %d)] *** ERROR ***: Final rank %d does not match target original rank %d!\n", current_new_rank, my_original_rank, final_rank, my_original_rank);
         MPI_Abort(MPI_COMM_WORLD, 1);
    }
    printf("That's all, folks!\n");
}

void set_error_handler(MPI_Comm comm)
{
    //MPI_Errhandler handler;
    //MPI_Comm_create_errhandler(error_handler, &handler);
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
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


void process_restart(field *temperature, parallel_data *parallel, int *iter, neighbor_data_buffers *buffers, int newbie_flag)
{

    int rank;
    MPI_Comm_rank(parallel->comm, &rank);

    printf("[Rank %d] Entering process_restart (newbie=%d). Last checkpoint iter: %d\n",
           rank, newbie_flag, buffers->iteration);


    int agreed_iter = -1;

    MPI_Allreduce(&buffers->iteration, &agreed_iter, 1, MPI_INT, MPI_MAX, parallel->comm);
    *iter = agreed_iter;

    printf("[Rank %d] Restarting from iteration %d.\n", rank, *iter);


    MPI_Aint lb = 0; // Lower bound, needed for pointer arithmetic
    MPI_Type_get_true_extent(parallel->restarttype, &lb, &buffers->extent); // Use true extent

    void *raw_recv_up = NULL, *raw_recv_down = NULL, *raw_recv_left = NULL, *raw_recv_right = NULL;
    double *recv_up = NULL, *recv_down = NULL, *recv_left = NULL, *recv_right = NULL;

    // Allocate receive buffers only if the neighbor exists
    if (parallel->nup != MPI_PROC_NULL) {
        raw_recv_up = malloc(buffers->extent);
        if (!raw_recv_up) MPI_Abort(parallel->comm, 1);
        recv_up = (double*)((char*)raw_recv_up - lb);
    }
     if (parallel->ndown != MPI_PROC_NULL) {
        raw_recv_down = malloc(buffers->extent);
         if (!raw_recv_down) MPI_Abort(parallel->comm, 1);
        recv_down = (double*)((char*)raw_recv_down - lb);
    }
    if (parallel->nleft != MPI_PROC_NULL) {
        raw_recv_left = malloc(buffers->extent);
         if (!raw_recv_left) MPI_Abort(parallel->comm, 1);
        recv_left = (double*)((char*)raw_recv_left - lb);
    }
     if (parallel->nright != MPI_PROC_NULL) {
        raw_recv_right = malloc(buffers->extent);
         if (!raw_recv_right) MPI_Abort(parallel->comm, 1);
        recv_right = (double*)((char*)raw_recv_right - lb);
    }


    MPI_Request requests[8]; // Max 4 sends + 4 receives
    int req_count = 0;
    int rc;


    if (parallel->nup != MPI_PROC_NULL) {
        rc = MPI_Irecv(recv_up, 1, parallel->restarttype, parallel->nup,
                       RESTART_EXCHANGE_TAG, parallel->comm, &requests[req_count++]);
        if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
    }
    if (parallel->ndown != MPI_PROC_NULL) {
        rc = MPI_Irecv(recv_down, 1, parallel->restarttype, parallel->ndown,
                       RESTART_EXCHANGE_TAG, parallel->comm, &requests[req_count++]);
        if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
    }
    if (parallel->nleft != MPI_PROC_NULL) {
        rc = MPI_Irecv(recv_left, 1, parallel->restarttype, parallel->nleft,
                       RESTART_EXCHANGE_TAG, parallel->comm, &requests[req_count++]);
        if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
    }
    if (parallel->nright != MPI_PROC_NULL) {
        rc = MPI_Irecv(recv_right, 1, parallel->restarttype, parallel->nright,
                       RESTART_EXCHANGE_TAG, parallel->comm, &requests[req_count++]);
        if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
    }


    void* send_buffer = newbie_flag ? temperature->data : buffers->self;
    if (!send_buffer && newbie_flag && temperature->data == NULL) {
         fprintf(stderr, "[Rank %d Newbie] Error: temperature->data is NULL before sending in restart.\n", rank);
         MPI_Abort(parallel->comm, 1);
    }
     if (!send_buffer && !newbie_flag && buffers->self == NULL) {
         fprintf(stderr, "[Rank %d Survivor] Error: buffers->self is NULL before sending in restart.\n", rank);
         MPI_Abort(parallel->comm, 1);
     }


    if (parallel->nup != MPI_PROC_NULL) {
        rc = MPI_Isend(send_buffer, 1, parallel->restarttype, parallel->nup,
                       RESTART_EXCHANGE_TAG, parallel->comm, &requests[req_count++]);
        if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
    }
    if (parallel->ndown != MPI_PROC_NULL) {
        rc = MPI_Isend(send_buffer, 1, parallel->restarttype, parallel->ndown,
                       RESTART_EXCHANGE_TAG, parallel->comm, &requests[req_count++]);
         if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
   }
    if (parallel->nleft != MPI_PROC_NULL) {
        rc = MPI_Isend(send_buffer, 1, parallel->restarttype, parallel->nleft,
                       RESTART_EXCHANGE_TAG, parallel->comm, &requests[req_count++]);
         if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
   }
    if (parallel->nright != MPI_PROC_NULL) {
        rc = MPI_Isend(send_buffer, 1, parallel->restarttype, parallel->nright,
                       RESTART_EXCHANGE_TAG, parallel->comm, &requests[req_count++]);
         if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
   }


    rc = MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
     if (rc != MPI_SUCCESS) {
        printf("Rank %d] ERROR during recovery\n", rank);
        MPI_Abort(MPI_COMM_WORLD, rc);
        }

    if (newbie_flag) {
        printf("[Rank %d Newbie] Restoring state from received neighbor data.\n", rank);
        int restored = 0;
        if (parallel->nup != MPI_PROC_NULL && recv_up != NULL) {
            memcpy(temperature->data, recv_up, buffers->extent);
            restored = 1;
            printf("[Rank %d Newbie] Restored state using data received from UP neighbor.\n", rank);
        } else if (parallel->ndown != MPI_PROC_NULL && recv_down != NULL) {
            memcpy(temperature->data, recv_down, buffers->extent);
             restored = 1;
             printf("[Rank %d Newbie] Restored state using data received from DOWN neighbor.\n", rank);
        } else if (parallel->nleft != MPI_PROC_NULL && recv_left != NULL) {
             memcpy(temperature->data, recv_left, buffers->extent);
             restored = 1;
             printf("[Rank %d Newbie] Restored state using data received from LEFT neighbor.\n", rank);
        } else if (parallel->nright != MPI_PROC_NULL && recv_right != NULL) {
             memcpy(temperature->data, recv_right, buffers->extent);
             restored = 1;
             printf("[Rank %d Newbie] Restored state using data received from RIGHT neighbor.\n", rank);
        }

        if (!restored && parallel->size > 1) {
             fprintf(stderr, "[Rank %d Newbie] Error: Could not receive state data from any neighbor!\n", rank);
             MPI_Abort(parallel->comm, -1);
        } else if (!restored && parallel->size == 1) {
             printf("[Rank %d Newbie] Warning: Running in size 1, no neighbors to restore from. State might be incorrect.\n", rank);
        }

    } else {
        printf("[Rank %d Survivor] Restoring state from local checkpoint buffer (buffers->self).\n", rank);
        if (buffers->self == NULL) {
             fprintf(stderr, "[Rank %d Survivor] Error: buffers->self is NULL. Cannot restore state.\n", rank);
             MPI_Abort(parallel->comm, -1);
        }
         // Ensure temperature->data is allocated (should be by initialize)
         if (temperature->data == NULL) {
             fprintf(stderr, "[Rank %d Survivor] Error: temperature->data is NULL before memcpy.\n", rank);
             MPI_Abort(parallel->comm, -1);
         }

         if (buffers->extent <= 0) {
             fprintf(stderr, "[Rank %d Survivor] Error: Invalid buffers->extent (%lld) provided for memcpy. Cannot restore.\n", rank, (long long)buffers->extent);
             MPI_Abort(parallel->comm, -1);
         }

         printf("[Rank %d Survivor] Using memcpy to restore %lld bytes from buffers->self to temperature->data.\n",
                rank, (long long)buffers->extent);

         memcpy(temperature->data, buffers->self, buffers->extent);
    }

    if (raw_recv_up) free(raw_recv_up);
    if (raw_recv_down) free(raw_recv_down);
    if (raw_recv_left) free(raw_recv_left);
    if (raw_recv_right) free(raw_recv_right);


    MPI_Barrier(parallel->comm); // Ensure all processes are synchronized before halo exchange
    printf("[Rank %d] process_restart finished successfully.\n", rank);
    return;
}