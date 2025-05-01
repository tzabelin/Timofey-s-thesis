#!/bin/bash

echo "STARTING"


./run_partial.sh > "outputs_partial.txt" &
./killer_partial.sh $KILL_RATE > "killer_log_partial.txt" &

