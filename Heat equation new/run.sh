#!/bin/bash
rm heat_mpi
rm "HEAT_RESTART.dat"
rm core.o setup.o utilities.o io.o main.o fault.o pngwriter.o  
make
mpirun --with-ft ulfm --host localhost:8 --oversubscribe -n 2 ./heat_mpi

