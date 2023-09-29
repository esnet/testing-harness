#!/usr/bin/env python3

import os
import re
import subprocess

# Define the awk commands
# First get rid of extra newline (\n) junk
awk_command = r'''awk '{gsub(/\\n/, "\n"); gsub(/\\"/, "\"");}1' '''
# grab content starting at the 2nd "{"
awk_command2 = r'''awk 'BEGIN { count = 0 } /{/ { count += 1; if (count == 2) { printing = 1 } } printing { print }' '''

tmp_file = "tmp.out"
tmp2_file = "tmp2.out"

for root, dirs, files in os.walk('.'):
    for file in files:
        file_path = os.path.join(root, file)
        if re.match('.*:[0-9]+$', file_path) and os.path.getsize(file_path) > 5000:
            print(" ")
            print ("Processing file: ", file_path)

            # Check if the file contains the string "Participant"
            if not "Participant" in open(file_path).read():
                # Construct the output file name
                output_file = file_path + ".json"

                print(f"File in correct format. Copying file with 'Participant' to {output_file}")
                with open(output_file, "wb") as dest_file:
                    with open(file_path, "rb") as src_file:
                        dest_file.write(src_file.read())
            else:
                # Define the output file name as "name.json"
                output_file = file_path + ".json"
                print(f"   Cleaning up JSON: {file_path} to {tmp_file}")

                # Run the awk command and save the output to the specified file
                subprocess.run(f"{awk_command} {file_path} > {tmp_file}", shell=True)

                print(f"   Strip first part of {tmp_file} to {tmp2_file}")
                subprocess.run(f"{awk_command2} {tmp_file} > {tmp2_file}", shell=True)

                with open(tmp2_file, "r") as tmp2:
                    lines = tmp2.readlines()
                    lnum = next((i + 1 for i, line in enumerate(lines) if "Participant" in line), None)
                    if lnum is not None:
                        lnum -= 1

                        # Get everything up to that point in the file
                        print(f"   Extract sender side info only from {tmp2_file} to output file: {output_file}")
                        with open(output_file, "w") as output:
                            output.writelines(lines[:lnum])
                    else:
                        print(f"*** ERROR: Sending side Participant line not found, so skipping: {file_path}")

