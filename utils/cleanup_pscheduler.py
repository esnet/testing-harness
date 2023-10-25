#!/usr/bin/env python3

# convert pscheduler output files for successful runs to more readable JSON files
# and rename as .json

import os
import subprocess
import re

# Define the pattern to match filenames
pattern = "src-cmd:[^:]*:([0-9]+)$"

# Define the minimum file size
min_file_size = 2000

# Specify the root directory to start the search
root_directory = "."

# Recursively search for matching files in subdirectories
for folder_path, subfolders, files in os.walk(root_directory):
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        #print ("checking: ",filename)
        match = re.search(pattern, filename)
        file_size = os.path.getsize(file_path)
        
        if match and file_size > min_file_size:
            # Extract the integer N from the matched pattern
            n = int(match.group(1))
            #print(f"   Matched filename: {file_path}, N: {n}, File Size: {file_size} bytes")

            jq_cmd = f"cat {file_path} | jq '.' > {file_path}.json"
            print ("Running command: ", jq_cmd)

            subprocess.run(f"{jq_cmd}", shell=True)
        else:
            #print(f"   Skipping file: {file_path}. Name does not match or file too small {file_size}")
            continue


