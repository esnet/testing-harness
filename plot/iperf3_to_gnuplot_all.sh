#!/bin/bash
set -x
#
# redo iperf3_to_gnuplot.py on P4 files


MYPATH="/Users/tierney/scripts"

#FILES=src-cmd:*:*
# if run format_pscheduler.py first, use this
FILES=src-cmd:*.json

for f in $FILES
do
  echo "Processing $f file... "
  base=`basename $f .json`
  \rm -f $base.dat
  #$MYPATH/iperf3_to_gnuplot.py -f $f -o $base.dat 
  # do option to sum parrel streams (-s)
  $MYPATH/iperf3_to_gnuplot.py -s -f $f -o $base.dat 
  exit
done



exit

# run gnuplot on all files ending in .gp in this dir
#

FILES=*.gp
for f in $FILES
do
  echo "Processing $f file... "
  gnuplot $f.gp
done

