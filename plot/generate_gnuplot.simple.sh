#!/bin/bash
#
#set -x
#
# generate a gnuplot command file for this test
#
# assume data files of the form: $dir/$dst_host.dat
#
# Note that you'll need to edit this files by hand to adjust the title and x axis 
#  for best looking plots
#
dst_host=$1
dir="."

echo "Generating gnuplot file for test to host: "$dst_host

gpfile=$dir/$dst_host.gp
rm -f $gpfile

echo "# file to plot iperf3 results for file $NAME" >> $gpfile
echo "set term png" >> $gpfile
echo "set key width -12" >> $gpfile
echo "set output \"$dir/$dst_host.png\" " >> $gpfile
echo "set grid xtics" >> $gpfile
echo "set grid ytics"  >> $gpfile
echo "set grid linewidth 1"  >> $gpfile
echo "# change title as necessary" >> $gpfile
echo "set title \"TCP performance, 100G host to $dst_host \\n " >> $gpfile
echo "#set title \"TCP performance: YYY to XXX \; $dst_host: 10G Host to XG Host, rtt = XX\" " >> $gpfile
echo "set xlabel \"time \(seconds\)\" " >> $gpfile
echo "set ylabel \"Bandwidth \(Gbits/second\)\" " >> $gpfile
echo "set y2label \"TCP Retransmits\" " >> $gpfile
echo "set xrange [0:62]"  >> $gpfile
echo "# change yrange as necessary" >> $gpfile
echo "set yrange [0:40]"  >> $gpfile
echo "set ytics nomirror"  >> $gpfile
echo "set y2tics"  >> $gpfile
echo "#set y2range [0:100]"  >> $gpfile
echo "# dont plot when retransmits = 0"  >> $gpfile
echo "set datafile missing '0'"  >> $gpfile
echo "set pointsize 1"  >> $gpfile
echo ""  >> $gpfile
#
echo "plot \"$dir/$dst_host.dat\" using 1:3 title '1 stream' with linespoints lw 2 pt 2, \\" >> $gpfile
echo "  \"$dir/$dst_host.dat\" using 1:4 title '$OS retransmits' with points pt 7 axes x1y2" >> $gpfile

echo "Generating .png file..."
gnuplot $gpfile
echo "Done."

