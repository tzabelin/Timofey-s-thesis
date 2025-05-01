#!/bin/bash

KILL_RATE=${1:-0.1}
KILL_INTERVAL=5
DURATION=60  # total run time in seconds
END_TIME=$((SECONDS + DURATION))

echo "Killer started with kill probability = $KILL_RATE"
echo "Killing process every $KILL_INTERVAL seconds with probability $KILL_RATE"

while [ $SECONDS -lt $END_TIME ]
do
    sleep $KILL_INTERVAL

    RANDOM_FLOAT=$(awk -v seed="$RANDOM" 'BEGIN { srand(seed); print rand(); }')

    should_kill=$(awk -v val="$RANDOM_FLOAT" -v threshold="$KILL_RATE" \
        'BEGIN { if (val < threshold) { print 1 } else { print 0 } }')

    if [ "$should_kill" -eq 1 ]; then
        pids=( $(pgrep -f "checkpoint-restart.out") )
        if [ ${#pids[@]} -gt 0 ]; then
            kill_pid=${pids[0]}
            echo "[KILLER] Killing process PID=$kill_pid"
            kill -9 "$kill_pid"
        fi
    fi
done

echo "Killer finished after $DURATION seconds."
