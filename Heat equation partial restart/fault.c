#include <mpi.h>
#include <stdlib.h>
#include "heat.h"
#include "fault.h"


void save_neighbor(field *temperature, parallel_data *parallel,
                   double **neighbor_up, double **neighbor_down,
                   double **neighbor_left, double **neighbor_right)
{
    int type_size_bytes, num_doubles;
    MPI_Request requests[8];
    int req_count = 0;
    int flag;

    /* Determine the size in bytes of one checkpoint block*/
       MPI_Aint lb, extent;
       MPI_Type_get_extent(parallel->restarttype, &lb, &extent);
       if (parallel->nup != MPI_PROC_NULL) {
        void *tmp = malloc(extent);
        if (!tmp) {
            return;
        }
        *neighbor_up = (double*)((char*)tmp - lb);
        } else {
            *neighbor_up = NULL;
        }
        
        if (parallel->ndown != MPI_PROC_NULL) {
        void *tmp = malloc(extent);
        if (!tmp) {
            return;
        }
            *neighbor_down = (double*)((char*)tmp - lb);
        } else {
        *neighbor_down = NULL;
        }
        
        if (parallel->nright != MPI_PROC_NULL) {
            void *tmp = malloc(extent);
            if (!tmp) {
                return;
            }
            *neighbor_right = (double*)((char*)tmp - lb);
        } else {
            *neighbor_right = NULL;
        }
        
        if (parallel->nleft  != MPI_PROC_NULL) {
            void *tmp = malloc(extent);
            if (!tmp) {
                return;
            }
            *neighbor_left = (double*)((char*)tmp - lb);
        } else {
            *neighbor_left = NULL;
        }

    if (parallel->nup != MPI_PROC_NULL) {
        MPI_Irecv(*neighbor_up, 1, parallel->restarttype,
                  parallel->nup, 21, parallel->comm, &requests[req_count++]);
    }
    if (parallel->ndown != MPI_PROC_NULL) {
        MPI_Irecv(*neighbor_down, 1, parallel->restarttype,
                  parallel->ndown, 21, parallel->comm, &requests[req_count++]);
    }
    if (parallel->nleft != MPI_PROC_NULL) {
        MPI_Irecv(*neighbor_left, 1, parallel->restarttype,
                  parallel->nleft, 21, parallel->comm, &requests[req_count++]);
    }
    if (parallel->nright != MPI_PROC_NULL) {
        MPI_Irecv(*neighbor_right, 1, parallel->restarttype,
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

    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}
