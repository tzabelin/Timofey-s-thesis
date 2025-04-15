
#include "heat.h"



void save_neighbor(field *temperature, parallel_data *parallel,
    neighbor_data_buffers *buffers, int iter);

void error_handler(MPI_Comm *comm, int *error_code, ...);
void free_neighbor_data(neighbor_data_buffers *buffers) ;
int initialize_neighbor_buffers(parallel_data *parallel, neighbor_data_buffers *buffers);
void process_restart(field *temperature, parallel_data *parallel, int *iter, neighbor_data_buffers *neighbor_checkpoint_buffers, int newbie_flag);