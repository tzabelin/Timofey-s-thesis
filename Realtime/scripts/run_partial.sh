mpicc -o partial-restart.out partial-restart.c
mpirun --with-ft ulfm --host localhost:8 --oversubscribe -n 3 --continuous ./partial-restart.out
