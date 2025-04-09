#!/bin/bash
make
mpirun -n 2 ./heat_mpi

