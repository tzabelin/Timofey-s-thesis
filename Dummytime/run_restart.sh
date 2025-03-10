#!/bin/bash

mpicc -o checkpoint-restart.out checkpoint-restart.c
attempt=0

while :
do
    echo "=== Starting attempt #$((attempt+1)) ==="
    mpirun -n 2 ./checkpoint-restart.out
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== Job completed successfully! ==="
        exit 0
    else
        echo "=== Job failed with code $rc; restarting... ==="
        attempt=$((attempt+1))
    fi
done

