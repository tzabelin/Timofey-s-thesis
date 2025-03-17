set terminal pngcairo enhanced font "Arial,12" size 800,600
set output 'expected_progress.png'

set title "Expected Computation Steps Over Time"
set xlabel "Time (HH:MM:SS)"
set ylabel "Computation Steps"

# Tell gnuplot that the x‑axis holds time data (seconds since 0)
set xdata time
set timefmt "%s"
set format x "%H:%M:%S"

# Set tics every 3600 seconds (1 hour)
set xtics 7200

set xrange ["0":"59457"]
set yrange [0:1000]

set grid
set key off

plot '-' using 1:2 with lines linecolor rgb "purple" linewidth 2 title "Expected Progress"
0 0
59457 1000
e

