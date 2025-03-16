#!/bin/bash

COMPUTATION_TIMES="5 10 15"
SAVE_STEPS="10 50 100"
DISK_LATENCIES="10 20 30"
KILL_RATES="0.05 0.1 0.2"

for cT in $COMPUTATION_TIMES; do
  for sS in $SAVE_STEPS; do
    for dL in $DISK_LATENCIES; do
      for kR in $KILL_RATES; do

        OUTFILE="outputs_CT${cT}_SS${sS}_DL${dL}_KR${kR}.log"

        echo "====================================================="
        echo "Starting benchmark with:"
        echo "  COMPUTATION_TIME=$cT"
        echo "  SAVE_STEP=$sS"
        echo "  DISK_LATENCY=$dL"
        echo "  KILL_RATE=$kR"
        echo "Logging to: $OUTFILE"
        echo "====================================================="
	rm -f checkpoint_rank0.dat checkpoint_rank1.dat
        mpicc -o checkpoint-restart.out checkpoint-restart.c \
          -DCOMPUTATION_TIME=$cT \
          -DSAVE_STEP=$sS \
          -DDISK_LATENCY=$dL

        echo "COMPUTATION_TIME=$cT"       >  "$OUTFILE"
        echo "SAVE_STEP=$sS"             >> "$OUTFILE"
        echo "DISK_LATENCY=$dL"         >> "$OUTFILE"
        echo "KILL_RATE=$kR"            >> "$OUTFILE"
        echo "-----------------------------------------" >> "$OUTFILE"
        ./run_restart.sh >> "$OUTFILE" 2>&1 &
        RUN_PID=$!
        ./killer.sh $kR &
        KILLER_PID=$!
        wait $RUN_PID
        kill -9 $KILLER_PID 2>/dev/null

        echo "Finished run with cT=$cT, sS=$sS, dL=$dL, kR=$kR"
        echo "Results saved to $OUTFILE"
        echo "====================================================="
        echo

      done
    done
  done
done

