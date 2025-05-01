set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'plot.png'

set title "Computation Steps Over Time"
set xlabel "Time Elapsed (seconds)"
set ylabel "Computation Steps"
set key bottom right
set grid

plot "outputs" using 1:2 with lines title "Computation Steps (Rank 0)"
