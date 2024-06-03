#!/usr/bin/env python3
#
# computes averages of mpstat JSON output for all .json files in a directory
#
import os
import json
import argparse
from glob import glob
from collections import defaultdict

def calculate_averages(cpu_loads):
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
    averages = {key: round(value / count, 2) for key, value in sums.items() if value != 0}
    return averages

def process_data(data):
    cpu_loads = defaultdict(list)
    for host in data['sysstat']['hosts']:
        for stat in host['statistics']:
            for load in stat['cpu-load']:
                cpu_loads[load['cpu']].append(load)
    
    print(f"Found {sum(len(loads) for loads in cpu_loads.values())} entries in this file")
    
    # Skip the first 4 and last 1 values, to ensure 'steady-state'
    for cpu, loads in cpu_loads.items():
        cpu_loads[cpu] = loads[4:-1]

    averages = {cpu: calculate_averages(loads) for cpu, loads in cpu_loads.items() if loads}
    return averages, cpu_loads

def write_output_human(averages, output_file, json_file):
    with open(output_file, 'a') as f:
        f.write(f"{json_file}:\n")
        for cpu, avg in averages.items():
            f.write(f"CPU {cpu}:\n")
            for key, value in avg.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

def write_output_json(averages, output_file, json_file):
    with open(output_file, 'a') as f:
        output = {json_file: averages}
        json.dump(output, f, indent=4)
        f.write(",\n")

def main(input_dir, output_file, output_format):
    # Initialize the output file
    with open(output_file, 'w') as f:
        if output_format == 'json':
            f.write("[\n")
        else:
            f.write("CPU Averages:\n")

    all_cpu_loads = defaultdict(list)

    # Process each JSON file in the directory
    for json_file in glob(os.path.join(input_dir, '*.json')):
        print(f"Processing {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
            averages, cpu_loads = process_data(data)
            if averages:
                if output_format == 'json':
                    write_output_json(averages, output_file, json_file)
                else:
                    write_output_human(averages, output_file, json_file)

                # Collect all CPU loads for overall averages
                for cpu, loads in cpu_loads.items():
                    all_cpu_loads[cpu].extend(loads)

    # Calculate overall averages
    overall_averages = {cpu: calculate_averages(loads) for cpu, loads in all_cpu_loads.items() if loads}
    
    if overall_averages:
        if output_format == 'json':
            with open(output_file, 'a') as f:
                output = {"overall": overall_averages}
                json.dump(output, f, indent=4)
                f.write("\n]")
        else:
            with open(output_file, 'a') as f:
                f.write("Overall CPU Averages:\n")
                for cpu, avg in overall_averages.items():
                    f.write(f"CPU {cpu}:\n")
                    for key, value in avg.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files to calculate average CPU metrics.')
    parser.add_argument('-i', '--input_dir', default='.', help='Input directory containing JSON files (default: current directory)')
    parser.add_argument('-o', '--output_file', help='Output file to save the results')
    parser.add_argument('-f', '--format', choices=['human', 'json'], default='human', help='Output format (default: human-readable)')
    
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = 'summary.json' if args.format == 'json' else 'summary.txt'
    
    main(args.input_dir, args.output_file, args.format)


