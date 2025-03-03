#!/bin/bash

KILL_RATE=${1:-0.1}

echo "Killer started with kill probability = $KILL_RATE"

while true
do
    SLEEP_TIME=$((1 + RANDOM % 5))
    sleep $SLEEP_TIME

    RANDOM_FLOAT=$(awk -v seed="$RANDOM" 'BEGIN { srand(seed); print rand(); }')

    if (( $(echo "$RANDOM_FLOAT < $KILL_RATE" | bc -l) )); then
        pids=( $(pgrep -f "checkpoint-restart.out") )
        
        if [ ${#pids[@]} -gt 0 ]; then
            kill_pid=${pids[$RANDOM % ${#pids[@]}]}
            echo "[KILLER] Killing process PID=$kill_pid"
            kill -9 $kill_pid
        fi
    fi
done

