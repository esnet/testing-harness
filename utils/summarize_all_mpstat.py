#!/usr/bin/env python3

# summarize all mpstat in a directory tree

import os
import re
import json
import argparse
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
    
    # Skip the first 4 and last 1 values, to ensure 'steady-state'
    for cpu, loads in cpu_loads.items():
        cpu_loads[cpu] = loads[4:-1]

    averages = {cpu: calculate_averages(loads) for cpu, loads in cpu_loads.items() if loads}
    return averages, cpu_loads

def find_files_and_extract_info():
    cwd = os.getcwd()
    pattern = re.compile(r"mpstat:(\d+\.\d+\.\d+\.\d+):.*\.json")

    results = []

    for root, dirs, files in os.walk(cwd):
        for file in files:
            match = pattern.match(file)
            if match:
                ip_address = match.group(1)
                test_name = os.path.basename(root)
                results.append({"file": os.path.join(root, file), "ip_address": ip_address, "test_name": test_name})
    
    return results

def main(input_dir, output_format):
    all_cpu_loads = defaultdict(lambda: defaultdict(list))

    results = find_files_and_extract_info()

    for result in results:
        json_file = result['file']
        test_name = result['test_name']
        ip_address = result['ip_address']
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            averages, cpu_loads = process_data(data)

            for cpu, loads in cpu_loads.items():
                all_cpu_loads[(test_name, ip_address)][cpu].extend(loads)

    overall_averages = {
        (test_name, ip_address): {cpu: calculate_averages(loads) for cpu, loads in cpu_data.items() if loads}
        for (test_name, ip_address), cpu_data in all_cpu_loads.items()
    }
    
    if overall_averages:
        if output_format == 'json':
            output = {
                f"{test_name} - {ip_address}": averages
                for (test_name, ip_address), averages in overall_averages.items()
            }
            with open('mpstat-summary.json', 'w') as f:
                json.dump(output, f, indent=4)
            print(f"Results saved to mpstat-summary.json")
        else:
            for (test_name, ip_address), averages in overall_averages.items():
                print(f"Overall CPU Averages for {test_name} - Host: {ip_address}:")
                for cpu, avg in averages.items():
                    avg_str = '   '.join(f"{key}: {value}" for key, value in avg.items())
                    print(f"   CPU {cpu}:   {avg_str}")
                print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files to calculate average CPU metrics.')
    parser.add_argument('-i', '--input_dir', default='.', help='Input directory containing JSON files (default: current directory)')
    parser.add_argument('-f', '--format', choices=['human', 'json'], default='human', help='Output format (default: human-readable)')
    
    args = parser.parse_args()
    
    main(args.input_dir, args.format)


