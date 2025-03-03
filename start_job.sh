#!/bin/bash

max_restarts=5
attempt=0

while [ $attempt -lt $max_restarts ]; do
    echo "=== Starting attempt #$((attempt+1)) ==="
    mpirun -n 2 ./checkpoint-restart.out
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== Job completed successfully! ==="
        exit 0
    else
        echo "=== Job failed with code $rc; restarting... ==="
        attempt=$((attempt+1))
        sleep 2
    fi
done

echo "=== Reached max restarts ($max_restarts). Exiting. ==="
exit 1

