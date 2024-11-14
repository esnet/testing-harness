#!/bin/bash

# JSON output from pscheduler is a mess, and not parsable. This script will clean it up.
#
# note: summarize_all.py script now does this in python 
#

# Check if input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <inputfile>"
  exit 1
fi

inputfile="$1"
outputfile="${inputfile}.fixed.json"

# Step 1: Replace escape sequences and save to test.out
sed -e 's/\\n/\n/g' -e 's/\\"/"/g' "$inputfile" > tmp.out

# Step 2: Remove the first 3 lines and lines from 'Participant' to the end, and save to fixed.out
sed -e '1,3d' -e '/Participant/,$d' tmp.out > "$outputfile"

# for debugging: Extract the .start and .end.sum_sent fields from the JSON in fixed.out
#start=$(jq .start "$outputfile")
end_sum_sent=$(jq .end.sum_sent "$outputfile")

# Display some of the results
#echo "Start: $start"
echo "End Sum Sent: $end_sum_sent"

rm tmp.out


