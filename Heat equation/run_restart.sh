#!/bin/bash

attempt=0

while :
do
    echo "=== Starting attempt #$((attempt+1)) ==="
    mpirun -n 2 ./heat_mpi
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== Job completed successfully! ==="
        exit 0
    else
        echo "=== Job failed with code $rc; restarting... ==="
        attempt=$((attempt+1))
    fi
done

