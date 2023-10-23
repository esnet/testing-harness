#!/bin/bash
#set -x
#
# generate a gnuplot command file for all files in a directory ending in .dat
#
# Note that you'll need to edit this files by hand to adjust the title and x axis 
#  for best looking plots
#
# need to set these for now..
OS="CentOS7"

MYPATH="/local/scripts/tcp-test-scripts"
MYPATH="/Users/tierney/Desktop/TCP-stack-testing/tcp-test-results/scripts"

rename 's/lbl-diskpt1.es.net/LBL/' *
rename 's/lbl-pt1.es.net/LBL/' *

FILES=*.dat
for f in $FILES
do
  base=`echo $f | awk -F ":" '{print $1}'`
  echo "Processing $f file... "
 
# older runs
#  $MYPATH/generate_gnuplot.v1.sh $base $OS $OPTS
# newer
# 1G or 10G
#  $MYPATH/generate_gnuplot.sh $base $OS 
#  $MYPATH/generate_gnuplot.500m.sh $base $OS 
  $MYPATH/generate_gnuplot.uk.sh $base $OS 
  gnuplot $base.gp

done

