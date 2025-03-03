#!/bin/bash

COMPUTATION_TIME=1
SAVE_STEP=10
KILL_RATE=0.1

mpicc -o checkpoint-restart.out checkpoint-restart.c \
  -DCOMPUTATION_TIME=$COMPUTATION_TIME \
  -DSAVE_STEP=$SAVE_STEP

./run_restart.sh > outputs &
./killer.sh $KILL_RATE &

