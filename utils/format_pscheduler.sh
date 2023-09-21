#!/bin/bash
#set -x

# the JSON output from pscheduler is pretty unreadable, so run this script to clean it up
# it also contains 2 sections: Participant 0 and Participant 1.
# for this purpose, we only need the Sender (Participant 0)

# Define the awk command to run
awk_command='awk '\''{gsub(/\\n/, "\n"); gsub(/\\"/, "\"");}1'\'''

awk_command2='awk '\''BEGIN { count = 0 } /{/ { count++; if (count == 2) { printing = 1 } } printing { print }'\'''

tmp_file="tmp.out"
tmp2_file="tmp2.out"

find . -type f -regex '.*:[0-9]*' -size +5000c -print | while IFS= read -r file; do

  # To Remove 'src-cmd:"' from the input filename
  # But this break summarize_iperf script
  #stripped_filename=$(echo $file | sed 's/^[^:]*://')
  # output_file="${stripped_filename}.json"

  # Define the output file name as "name.json"
  output_file="${file}.json"
  echo "Output file is: $output_file"

  echo "Cleaning up JSON: $file to $tmp_file"
  # Run the awk command and save the output to the specified file
  #echo "Running: $awk_command $file > $output_file"
  eval "$awk_command" $file > $tmp_file

  echo "strip first part of $tmp_file to $tmp2_file"
  eval "$awk_command2" $tmp_file > $tmp2_file

  lnum=$(grep -n "Participant" $tmp2_file | cut -d: -f1)
  echo "Participant occurs at line $lnum"

  # get everything up to that point in the file
  echo "extract sender side info only from $tmp2_file to $output_file"
  head -n $(($lnum - 1)) $tmp2_file > $output_file

done


