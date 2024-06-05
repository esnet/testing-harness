#!/bin/bash
# sample script to run a long running collect.py job in the background to make sure tty output does not cause it to hang
# run this script with 'nohup' 
# eg.:  nohup run_collect.sh &
#set -x

# cd to dir with .ini files, etc.
cd ~/tuning-tests

echo "Starting collect.py"
$HOME/scripts/collect.py -j iperf3-tuning-test.ini -H hostlist.csv -i eth100 -l test.log -o iperf3-results-6.4  & > collect-stderr.log 2>&1

