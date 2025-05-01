#!/bin/bash

COMPUTATION_TIME=1
SAVE_STEP=100
DISK_LATENCY=10
KILL_RATE=0.1

echo "COMPUTATION_TIME=$COMPUTATION_TIME" > outputs
echo "SAVE_STEP=$SAVE_STEP" >> outputs
echo "DISK_LATENCY=$DISK_LATENCY" >> outputs
echo "KILL_RATE=$KILL_RATE" >> outputs

mpicc -o checkpoint-restart.out checkpoint-restart.c \
  -DCOMPUTATION_TIME=$COMPUTATION_TIME \
  -DSAVE_STEP=$SAVE_STEP\
  -DDISK_LATENCY=$DISK_LATENCY$

./run_restart.sh > outputs &
./killer.sh $KILL_RATE &

