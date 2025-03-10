set terminal pngcairo enhanced font "Arial,12" size 800,600
set output 'expected_progress.png'

set title "Expected Computation Steps Over Time"
set xlabel "Time (HH:MM:SS)"
set ylabel "Computation Steps"

set xrange [0:117170]
set yrange [0:1000]

set grid
set key off

# Format x-axis to display time in HH:MM:SS
set xdata time
set timefmt "%s"
set format x "%H:%M:%S"

# Plot a straight line from (0,0) to (117170,1000)
plot '-' using 1:2 with lines linecolor rgb "purple" linewidth 2 title "Expected Progress"
0 0
117170 1000
e

