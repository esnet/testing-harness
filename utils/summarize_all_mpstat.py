#!/usr/bin/env python3

# summarize all mpstat and iperf3 output in a directory tree
# run with -h to see options

import os
import re
import json
import argparse
import csv
import sys
from collections import defaultdict

def calculate_averages(cpu_loads):
# note: this routine is use to calculate average for within a test, AND for set of tests

    if not cpu_loads:
        print("calculate_averages Error: no cpu data provided")
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
    #print ("   calculate_averages returns: ", averages)
    return averages

def process_cpu_data(data):
    cpu_loads = defaultdict(list)
    for host in data['sysstat']['hosts']:
        for stat in host['statistics']:
            for load in stat['cpu-load']:
                cpu_loads[load['cpu']].append(load)
    
    # Skip the first 4 and last 1 values, to ensure 'steady-state'
    for cpu, loads in cpu_loads.items():
        cpu_loads[cpu] = loads[4:-1]

    averages = {cpu: calculate_averages(loads) for cpu, loads in cpu_loads.items() if loads}
    #print ("process_cpu_data returns: ", cpu_loads)
    return averages, cpu_loads

def find_files():
    cwd = os.getcwd()
    mpstat_pattern = re.compile(r"mpstat:(\d+\.\d+\.\d+\.\d+):.*\.json")
    src_cmd_pattern = re.compile(r"src-cmd:(\d+\.\d+\.\d+\.\d+):.*")

    results = []

    for root, dirs, files in os.walk(cwd):
        mpstat_count = 0
        src_cmd_count = 0
        for file in files:
            mpstat_match = mpstat_pattern.match(file)
            src_cmd_match = src_cmd_pattern.match(file)
            if mpstat_match:
                ip_address = mpstat_match.group(1)
                test_name = os.path.basename(root)
                results.append({"file": os.path.join(root, file), "ip_address": ip_address, "test_name": test_name})
                mpstat_count += 1
            elif src_cmd_match:
                ip_address = src_cmd_match.group(1)
                test_name = os.path.basename(root)
                results.append({"file": os.path.join(root, file), "ip_address": ip_address, "test_name": test_name})
                src_cmd_count += 1

        file_count = mpstat_count + src_cmd_count
        if file_count == 0:
            print(f"No relevant files found in directory: {root}. Skipping...")
            continue

        #print(f"In directory {root}: Found {mpstat_count} mpstat files and {src_cmd_count} src-cmd files")

    #print ("will look at these files: ", results)
    return results

def extract_throughput(src_cmd_file):
    with open(src_cmd_file, 'r') as f:
        for line in f:
            if 'sender' in line and 'CPU' not in line:
                # XXX: what if Mbits/sec, not Gbits??
                throughput = re.search(r'(\d+\.\d+) Gbits/sec', line)
                if throughput:
                    tput = float(throughput.group(1)) # group(1) ensures a single number
                    #print ("   extract_throughput returns: ", tput)
                    return tput
    return None

def write_to_csv(output_file, data, throughput_values):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['IP', 'test_name', 'thruput', 'CPU number', 'cpu_user', 'cpu_sys', 'cpu_soft', 'cpu_irq', 'cpu_idle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (test_name, ip_address), averages in data.items():
            for cpu, avg in averages.items():
                writer.writerow({
                    'IP': ip_address,
                    'test_name': test_name,
                    'thruput': throughput_values.get((test_name, ip_address), ''),
                    'CPU number': cpu,
                    'cpu_user': avg.get('usr', ''),
                    'cpu_sys': avg.get('sys', ''),
                    'cpu_soft': avg.get('soft', ''),
                    'cpu_irq': avg.get('irq', ''),
                    'cpu_idle': avg.get('idle', '')
                })
    print(f"Results saved to {output_file}")

def main(input_dir, output_format, output_file):

    all_cpu_loads = defaultdict(lambda: defaultdict(list))
    throughput_values = {}

    if not output_file:
        if output_format == 'csv':
            output_file = 'test-summary.csv'
        elif output_format == 'json':
            output_file = 'test-summary.json'
        else:
            output_file = 'test-summary.txt'

    results = find_files()  # build a dict of filename, testname, IP

    print ("\nCollecting Results from all files...")
    for result in results:
        input_file = result['file']  # includes full path
        test_name = result['test_name']
        ip_address = result['ip_address']
        
        #print ("loading file: ", input_file)
        fname = os.path.basename(input_file)
        if fname.startswith('src-cmd'):
            #print ("Extracting throughput from file: ", input_file)
            throughput = extract_throughput(input_file)
            if throughput is not None:
                throughput_values[(test_name, ip_address)] = throughput
            else:
                print (f"   Throughput not found in file {input_file}")
        elif fname.startswith('mpstat'):
            with open(input_file, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    parent_dir = os.path.basename(os.path.dirname(input_file))
                    filename_with_parent = os.path.join(parent_dir, fname)
                    print("   Error loading file: ", filename_with_parent)
                    sys.exit()
                    continue
                averages, cpu_loads = process_cpu_data(data)

                for cpu, loads in cpu_loads.items():
                    all_cpu_loads[(test_name, ip_address)][cpu].extend(loads)

    overall_averages = {}
    print ("\nComputing CPU Averages...")
    #overall_averages = {cpu: calculate_averages(loads) for cpu, loads in all_cpu_loads.items() if loads}  # compact version
    for (test_name, ip_address), cpu_data in all_cpu_loads.items():
        averages = {}
        #print("cpu_data.items: ",cpu_data.items())

        for cpu, loads in cpu_data.items():
            #print ("  in loop: loads = ", loads)
            if loads:
                averages[cpu] = calculate_averages(loads)
                #print (f"   CPU {cpu} averages for test {test_name}  IP {ip_address} ", averages[cpu])
        overall_averages[(test_name, ip_address)] = averages
    
    if overall_averages:
        if output_format == 'csv':
            write_to_csv(output_file, overall_averages, throughput_values)
        elif output_format == 'json':
            output = {
                f"{test_name} - {ip_address}": averages
                for (test_name, ip_address), averages in overall_averages.items()
            }
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=4)
            print(f"Results saved to file: ", output_file)
        else:
            print ("\nSummary of all testing: \n")
            for (test_name, ip_address), averages in overall_averages.items():
               if (test_name, ip_address) in throughput_values and throughput_values[(test_name, ip_address)] is not None:
                   print(f"Average Throughput for test {test_name} to Host: {ip_address}: {throughput_values[(test_name, ip_address)]:.2f} Gbps")
                   for cpu, avg in averages.items():
                        avg_str = '   '.join(f"{key:4s}: {value:4.2f}" for key, value in avg.items())
                        print(f"   CPU {cpu}:   {avg_str}")
                   print()
               else:
                   print(f"\nNo throughput data available for test {test_name} to Host: {ip_address}")
    else:
        print("ERROR: calculate_averages returned no results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files to calculate average CPU metrics.')
    parser.add_argument('-i', '--input_dir', default='.', help='Input directory containing JSON files (default: current directory)')
    parser.add_argument('-f', '--format', choices=['human', 'json', 'csv'], default='human', help='Output format (default: human-readable)')
    parser.add_argument('-o', '--output_file', help='Output filename (default = mpstat-summary.{csv,json,txt)')

    args = parser.parse_args()

    main(args.input_dir, args.format, args.output_file)

