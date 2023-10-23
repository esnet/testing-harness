#!/bin/bash
#
#set -x
#
# generate a gnuplot command file for this test
#
# Note that you'll need to edit this files by hand to adjust the title and x axis 
#  for best looking plots
#
#dir=$1
base=$1
OS=$2
opts=$3
dir="."

gpfile=$dir/$base.gp
rm -f $gpfile

echo "# file to plot iperf3 results for file $NAME" >> $gpfile
echo "set term png" >> $gpfile
echo "set key width -12" >> $gpfile
echo "set grid xtics" >> $gpfile
echo "set grid ytics"  >> $gpfile
echo "set grid linewidth 1"  >> $gpfile
echo "# change title as necessary" >> $gpfile
echo "#set title \"TCP performance: YYY to XXX \; CentOS 6.5 vs CentOS 7.2 \\n $base: 10G Host to XG Host, rtt = XX\" " >> $gpfile
echo "set xlabel \"time \(seconds\)\" " >> $gpfile
echo "set ylabel \"Bandwidth \(Gbits/second\)\" " >> $gpfile
#echo "set y2label \"TCP Retransmits\" " >> $gpfile
echo "set xrange [0:62]"  >> $gpfile
echo "# change yrange as necessary" >> $gpfile
echo "set yrange [0:15]"  >> $gpfile
echo "set ytics nomirror"  >> $gpfile
echo "set y2tics"  >> $gpfile
echo "#set y2range [0:100]"  >> $gpfile
echo "# dont plot when retransmits = 0"  >> $gpfile
echo "set datafile missing '0'"  >> $gpfile
echo "set pointsize 1"  >> $gpfile
echo ""  >> $gpfile
#
# plot 1:
echo "set output \"$dir/$base.1.png\" " >> $gpfile
echo "set title \"TCP performance: CentOS6 vs CentOS7 \\n $base " >> $gpfile
echo "plot \"$dir/$base:CentOS6.dat\" using 1:3 title 'CentOS6' with linespoints lw 2 pt 2, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7.dat\" using 1:3 title 'CentOS7' with linespoints lw 2 pt 5, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:FQ.dat\" using 1:3 title 'CentOS7 + FQ' with linespoints lw 2 pt 7, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:FQ:8g.dat\" using 1:3 title 'CentOS7 + FQ + 8G pacing' with linespoints lw 2 pt 9" >> $gpfile
echo ""  >> $gpfile

# plot 2:
#echo "set output \"$dir/$base.2.png\" " >> $gpfile
#echo "set title \"TCP performance: FQ vs FQ + 8G pacing \\n $base " >> $gpfile
#echo "plot \"$dir/$base:CentOS7:FQ.dat\" using 1:3 title 'CentOS7 + FQ' with linespoints lw 2 pt 2, \\" >> $gpfile
#echo "  \"$dir/$base:CentOS7:FQ:8g.dat\" using 1:3 title 'CentOS7 + FQ + pacing' with linespoints lw 2 pt 5" >> $gpfile
#echo ""  >> $gpfile

# plot 3:
#echo "set output \"$dir/$base.3.png\" " >> $gpfile
#echo "set title \"TCP performance: sum of 4 streams \\n $base " >> $gpfile
#echo "plot \"$dir/$base:CentOS6:P4.sum.dat\" using 1:3 title 'CentOS6' with linespoints lw 2 pt 2, \\" >> $gpfile
#echo "  \"$dir/$base:CentOS7:P4.sum.dat\" using 1:3 title 'CentOS7' with linespoints lw 2 pt 5, \\" >> $gpfile
#echo "  \"$dir/$base:CentOS7:FQ:P4.sum.dat\" using 1:3 title 'CentOS7 + FQ' with linespoints lw 2 pt 7" >> $gpfile
#echo ""  >> $gpfile

# plot 4:
#echo "set output \"$dir/$base.4.png\" " >> $gpfile
#echo "set title \"TCP performance: sum of 4 streams with 8G pacing \\n $base " >> $gpfile
#echo "plot \"$dir/$base:CentOS7:P4.sum.dat\" using 1:3 title 'CentOS7' with linespoints lw 2 pt 2, \\" >> $gpfile
#echo "  \"$dir/$base:CentOS7:FQ:P4.sum.dat\" using 1:3 title 'CentOS7 + FQ' with linespoints lw 2 pt 5, \\" >> $gpfile
#echo "  \"$dir/$base:CentOS7:FQ:P4:8g.sum.dat\" using 1:3 title 'CentOS7 + FQ + pacing ' with linespoints lw 2 pt 7" >> $gpfile
#echo ""  >> $gpfile

# plot 5:
#echo "set output \"$dir/$base.5.png\" " >> $gpfile
#echo "set title \"TCP performance: 4 streams \\n $base " >> $gpfile
#echo "plot \"$dir/$base:CentOS6:P4.dat\" using 1:3 title 'CentOS6' with points lw 2 pt 2, \\" >> $gpfile
#echo "  \"$dir/$base:CentOS7:P4.dat\" using 1:3 title 'CentOS7' with points lw 2 pt 5, \\" >> $gpfile
#echo "  \"$dir/$base:CentOS7:FQ:P4.dat\" using 1:3 title 'CentOS7 + FQ' with points lw 2 pt 7" >> $gpfile
#echo ""  >> $gpfile

# plot 6:
#echo "set output \"$dir/$base.6.png\" " >> $gpfile
#echo "set title \"TCP performance: 4 streams with 2.4G pacing \\n $base " >> $gpfile
#echo "plot \"$dir/$base:CentOS7:P4.dat\" using 1:3 title 'CentOS7' with points lw 2 pt 2, \\" >> $gpfile
#echo "  \"$dir/$base:CentOS7:FQ:P4.dat\" using 1:3 title 'CentOS7 + FQ' with points lw 2 pt 5, \\" >> $gpfile
#echo "  \"$dir/$base:CentOS7:FQ:P4:2g.dat\" using 1:3 title 'CentOS7 + FQ + pacing ' with points lw 2 pt 7" >> $gpfile
#echo ""  >> $gpfile

# combine plots 3,4,5,6 into a single multiplot
echo "set term png size 1000,800 " >> $gpfile
echo "set output \"$dir/$base.multiplot.png\" " >> $gpfile
echo "set size 1,1" >> $gpfile
echo "set origin 0,0" >> $gpfile
echo "set multiplot layout 2,2 columnsfirst title \"Parallel Stream Testing\" " >> $gpfile

echo "set title \"TCP performance: sum of 4 streams \\n $base " >> $gpfile
echo "plot \"$dir/$base:CentOS6:P4.sum.dat\" using 1:3 title 'CentOS6' with linespoints lw 2 pt 2, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:P4.sum.dat\" using 1:3 title 'CentOS7' with linespoints lw 2 pt 5, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:FQ:P4.sum.dat\" using 1:3 title 'CentOS7 + FQ' with linespoints lw 2 pt 7" >> $gpfile
echo ""  >> $gpfile

echo "set yrange [0:16]"  >> $gpfile
echo "set title \"TCP performance: sum of 4 streams with pacing \\n $base " >> $gpfile
echo "plot \"$dir/$base:CentOS7:P4.sum.dat\" using 1:3 title 'CentOS7' with linespoints lw 2 pt 2, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:FQ:P4.sum.dat\" using 1:3 title 'CentOS7 + FQ' with linespoints lw 2 pt 5, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:FQ:P4:8g.sum.dat\" using 1:3 title 'CentOS7 + FQ + 8G pacing ' with linespoints lw 2 pt 7, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:FQ:P4:2g.sum.dat\" using 1:3 title 'CentOS7 + FQ + 2.4G pacing ' with linespoints lw 2 pt 9" >> $gpfile
echo ""  >> $gpfile

echo "set yrange [0:10]"  >> $gpfile
echo "set title \"TCP performance: 4 streams \\n $base " >> $gpfile
echo "plot \"$dir/$base:CentOS6:P4.dat\" using 1:3 title 'CentOS6' with points lw 1 pt 2, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:P4.dat\" using 1:3 title 'CentOS7' with points lw 1 pt 5, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:FQ:P4.dat\" using 1:3 title 'CentOS7 + FQ' with points lw 1 pt 7" >> $gpfile
echo ""  >> $gpfile

echo "set title \"TCP performance: 4 streams with pacing \\n $base " >> $gpfile
echo "plot \"$dir/$base:CentOS7:FQ:P4.dat\" using 1:3 title 'CentOS7 + FQ' with points lw 1 pt 2, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:FQ:P4:8g.dat\" using 1:3 title 'CentOS7 + FQ + 8G pacing ' with points lw 1 pt 7, \\" >> $gpfile
echo "  \"$dir/$base:CentOS7:FQ:P4:2g.dat\" using 1:3 title 'CentOS7 + FQ + 2.4G pacing ' with points lw 1 pt 9" >> $gpfile
echo ""  >> $gpfile

echo "unset multiplot" >> $gpfile


gnuplot $gpfile

