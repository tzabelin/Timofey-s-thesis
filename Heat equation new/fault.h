
#include "heat.h"

typedef struct {
    double *up;
    double *down;
    double *left;
    double *right;
    double *self;
    void *raw_ptr_up;
    void *raw_ptr_down;
    void *raw_ptr_left;
    void *raw_ptr_right;
    void *raw_ptr_self_buffer;
    MPI_Aint extent;
    int iteration;
    int nx_full, ny_full;
} neighbor_data_buffers;

void save_neighbor(field *temperature, parallel_data *parallel,
    neighbor_data_buffers *buffers, int iter);

void error_handler(MPI_Comm *comm, int *error_code, ...);
void free_neighbor_data(neighbor_data_buffers *buffers) ;
int initialize_neighbor_buffers(parallel_data *parallel, neighbor_data_buffers *buffers);
