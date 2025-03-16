set terminal pngcairo enhanced font "Arial,12" size 800,600
set output 'expected_progress.png'

set title "Expected Computation Steps Over Time"
set xlabel "Time (HH:MM:SS)"
set ylabel "Computation Steps"

set xrange [0:4696875]
set yrange [0:1000]

set grid
set key off

# Define a custom modulus function for positive numbers
mymod(x,y) = x - y*int(x/y)

# Function to convert seconds to HH:MM:SS format
sec2time(t) = sprintf("%02d:%02d:%02d", int(t/3600), int(mymod(t,3600)/60), int(mymod(t,60)))

# Set xtics at start, middle, and end points
set xtics (sec2time(0) 0, sec2time(2348437.5) 2348437.5, sec2time(4696875) 4696875)

plot '-' using 1:2 with lines linecolor rgb "purple" linewidth 2 title "Expected Progress"
0 0
4696875 1000
e

