set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'plot.png'

set title "Computation Steps Over Time"
set xlabel "Time (HH-MM-SS)"
set ylabel "Computation Steps"

set xdata time
set timefmt "%H-%M-%S"
set format x "%H:%M:%S"

set key bottom right
set xtics
set grid

plot "< grep 'my counter=' rank_1.log | grep 'Rank 1' | sed 's/\\[\\([0-9][0-9]-[0-9][0-9]-[0-9][0-9]\\)\\].*my counter=\\([0-9]*\\).*/\\1 \\2/'" \
     using 1:2 with lines title "Computation Steps (Rank 1)"
