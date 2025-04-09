#!/bin/bash

KILL_RATE=0.5

rm "HEAT_RESTART.dat"
rm *.png
rm "killer_log.txt"
make

./run_restart.sh &
./killer.sh $KILL_RATE >"killer_log.txt" &

