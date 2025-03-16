set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'plot.png'

set title "Computation steps over time"
set xlabel "Time (HH-MM-SS)"
set ylabel "Computation Steps"
set xdata time
set timefmt "%H-%M-%S"
set format x "%H:%M:%S"
set key bottom right
set xtics rotate by -45
set grid

plot "outputs" using 1:2 with linespoints title "Computation Steps" lw 2 lc rgb "blue"

