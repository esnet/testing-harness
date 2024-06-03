#!/usr/bin/env python3
import os
import json
import argparse
from glob import glob

def calculate_averages(data):
    # Extract all CPU load metrics
    cpu_loads = []
    for host in data['sysstat']['hosts']:
        for stat in host['statistics']:
            cpu_loads.extend(stat['cpu-load'])
    
    print (f"Found {len(cpu_loads)} in this file")
    # Skip the first 5 and last 5 values
    cpu_loads = cpu_loads[4:-4]
    
    if not cpu_loads:
        return None

    # Initialize sums and count
    sums = {
        "usr": 0, "sys": 0, "nice": 0, "iowait": 0, "irq": 0,
        "soft": 0, "steal": 0, "guest": 0, "gnice": 0, "idle": 0
    }
    count = 0

    # Sum each metric
    for load in cpu_loads:
        for key in sums:
            sums[key] += load[key]
        count += 1

    # Calculate averages
    averages = {key: value / count for key, value in sums.items()}
    return averages

def main(input_dir, output_file):
    # Initialize the output file
    with open(output_file, 'w') as f:
        f.write("CPU Averages:\n")

    # Process each JSON file in the directory
    for json_file in glob(os.path.join(input_dir, '*.json')):
        print(f"Processing {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
            averages = calculate_averages(data)
            if averages:
                with open(output_file, 'a') as f:
                    f.write(f"{json_file}: {averages}\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files to calculate average CPU metrics.')
    parser.add_argument('-i', '--input_dir', required=True, help='Input directory containing JSON files')
    parser.add_argument('-o', '--output_file', required=True, help='Output file to save the results')
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_file)

