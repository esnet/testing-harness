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
import statistics

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
    mpstat_snd_pattern = re.compile(r"mpstat-sender:(\d+\.\d+\.\d+\.\d+):.*\.json")
    mpstat_rcv_pattern = re.compile(r"mpstat-receiver:(\d+\.\d+\.\d+\.\d+):.*\.json")
    src_cmd_pattern = re.compile(r"src-cmd:(\d+\.\d+\.\d+\.\d+):.*")

    results = []

    for root, dirs, files in os.walk(cwd):
        file_cnt = 0
        for file in files:
            mpstat_snd_match = mpstat_snd_pattern.match(file)
            mpstat_rcv_match = mpstat_rcv_pattern.match(file)
            src_cmd_match = src_cmd_pattern.match(file)
            test_name = os.path.basename(root)
            if mpstat_snd_match:
                file_cnt += 1
                ip_address = mpstat_snd_match.group(1) 
                results.append({"file": os.path.join(root, file), "ip_address": ip_address, "test_name": test_name, "type": "mpstat_snd"})
            if mpstat_rcv_match:
                file_cnt += 1
                ip_address = mpstat_rcv_match.group(1) 
                results.append({"file": os.path.join(root, file), "ip_address": ip_address, "test_name": test_name, "type": "mpstat_rcv"})
            if src_cmd_match:
                file_cnt += 1
                ip_address = src_cmd_match.group(1) 
                results.append({"file": os.path.join(root, file), "ip_address": ip_address, "test_name": test_name, "type": "src_cmd"})

        if file_cnt == 0:
            print(f"No relevant files found in directory: {root}. Skipping...")
            continue

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


def write_to_csv(output_file, cpu_data, throughput_values):

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['IP', 'test_name', 'ave_thruput', 'max_thruput', 'stdev_tput', 'snd_or_rcv', 'CPU_number', 'cpu_user', 'cpu_sys', 'cpu_soft', 'cpu_irq', 'cpu_idle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in cpu_data:
            (test_name, ip_address, type), averages = entry
            ave_throughput = throughput_values.get((test_name, ip_address), [])
            if len(ave_throughput) == 0:
                continue
            if len(ave_throughput) > 5:
                print("Warning: have > 5 throughput values for this test. Is this intentional?")
            ave_tput = round(statistics.mean(ave_throughput), 2) if ave_throughput else None
            max_throughput = max(ave_throughput, default=None)
            stdev_throughput = round(statistics.stdev(ave_throughput), 2) if len(ave_throughput) > 1 else None

            # Initialize a row with common values
            row = {
                'IP': ip_address,
                'test_name': test_name,
                'ave_thruput': ave_tput,
                'max_thruput': max_throughput,
                'stdev_tput': stdev_throughput,
                'snd_or_rcv': type,
            }

            # Write row for each CPU, filling in CPU metrics
            for cpu, avg in averages.items():
                row['CPU_number'] = cpu
                row['cpu_user'] = avg.get('usr', '')
                row['cpu_sys'] = avg.get('sys', '')
                row['cpu_soft'] = avg.get('soft', '')
                row['cpu_irq'] = avg.get('irq', '')
                row['cpu_idle'] = avg.get('idle', '')
                writer.writerow(row)

            
    print(f"Results saved to {output_file}")

def write_to_json(output_file, cpu_averages, throughput_values):

    output = {}
    for entry in cpu_averages:
        (test_name, ip_address, type), averages = entry
        if len(throughput_values.get((test_name, ip_address), [])) == 0:
            continue
        avg_throughput = round(statistics.mean(throughput_values.get((test_name, ip_address), [])), 2)
        max_throughput = round(max(throughput_values.get((test_name, ip_address), []), default=None), 2)
        stdev_throughput = round(statistics.stdev(throughput_values.get((test_name, ip_address), [])), 2) if len(throughput_values.get((test_name, ip_address), [])) > 1 else None

        cpu_averages_data = {
            "sender": {},
            "receiver": {}
        }

        for cpu, avg in averages.items():
            cpu_type = "sender" if type == "mpstat_snd" else "receiver"
            cpu_averages_data[cpu_type][f"CPU {cpu}"] = {key: round(value, 2) for key, value in avg.items()}

        output[f"{test_name} - {ip_address}"] = {
            "test_name": test_name,
            "IP": ip_address,
            "average_throughput": avg_throughput,
            "max_throughput": max_throughput,
            "stdev_throughput": stdev_throughput,
            "cpu_averages": cpu_averages_data
        }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to file: {output_file}")



def main(input_dir, output_format, output_file):

    snd_cpu_loads = defaultdict(lambda: defaultdict(list))
    rcv_cpu_loads = defaultdict(lambda: defaultdict(list))
    throughput_values = defaultdict(list)

    if not output_file:
        if output_format == 'csv':
            output_file = 'test-summary.csv'
        elif output_format == 'json':
            output_file = 'test-summary.json'
        else:
            output_file = 'test-summary.txt'

    results = find_files()  # build a dict of filename, testname, IP, type

    print ("\nCollecting Results from all files...")
    for result in results:
        input_file = result['file']  # includes full path
        test_name = result['test_name']
        ip_address = result['ip_address']
        type = result['type']
        
        #print ("loading file: ", input_file)
        fname = os.path.basename(input_file)
        if type == 'src_cmd':
            #print ("Extracting throughput from file: ", input_file)
            throughput = extract_throughput(input_file)
            if throughput is not None:
                throughput_values[(test_name, ip_address)].append(throughput)
            else:
                print (f"   Throughput not found in file {input_file}")
        if type == 'mpstat_snd' or type == 'mpstat_rcv':
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
                    #print (f"cpu: {cpu}, type: {type}, loads: {loads}")
                    if type == 'mpstat_snd':
                        snd_cpu_loads[(test_name, ip_address)][cpu].extend(loads)
                    else:
                        rcv_cpu_loads[(test_name, ip_address)][cpu].extend(loads)

    print(f"\n Got mpstat data from {len(snd_cpu_loads)} sender files and {len(rcv_cpu_loads)} receiver files")
    overall_averages = {}
    print ("\nComputing CPU Averages...")
    for (test_name, ip_address), cpu_data in snd_cpu_loads.items():
        averages = {}
        #print("snd_cpu_data.items: ",snd_cpu_loads.items())

        for cpu, loads in cpu_data.items():
            #print ("  in loop: loads = ", loads)
            if loads:
                averages[cpu] = calculate_averages(loads)
                #print (f"   CPU {cpu} averages for test {test_name}  IP {ip_address} ", averages[cpu])
        overall_averages[(test_name, ip_address, 'sender')] = averages

    for (test_name, ip_address), cpu_data in rcv_cpu_loads.items():
        averages = {}

        for cpu, loads in cpu_data.items():
            if loads:
                averages[cpu] = calculate_averages(loads)
        overall_averages[(test_name, ip_address, 'receiver')] = averages

    if overall_averages:
        # Sort the dictionary by test_name and ip_address
        #sorted_averages = sorted( overall_averages.items(), key=lambda x: (x[0][0], x[0][1]))
        # Sort the dictionary by ip_address, then test_name
        sorted_averages = sorted( overall_averages.items(), key=lambda x: (x[0][1], x[0][0]))
        overall_averages = dict(sorted_averages)  # Convert back to dictionary if needed

        if output_format == 'csv':
            write_to_csv(output_file, sorted_averages, throughput_values)
        elif output_format == 'json':
            write_to_json(output_file, sorted_averages, throughput_values)
        else:
            prev_ip_address = prev_test_name = ""
            print ("\nSummary of all testing: \n")
            for (test_name, ip_address, type), averages in sorted_averages:
               if (test_name, ip_address) in throughput_values and throughput_values[(test_name, ip_address)]:
                   if test_name != prev_test_name or ip_address != prev_ip_address: # only print for new test
                       avg_throughput = statistics.mean(throughput_values[(test_name, ip_address)])
                       max_throughput = max(throughput_values[(test_name, ip_address)])
                       stdev_throughput = statistics.stdev(throughput_values[(test_name, ip_address)])
                       print(f"\nAve Tput for test {test_name} to Host: {ip_address}: {avg_throughput:.2f} Gbps, Max: {max_throughput:.2f} Gbps,  stdev: {stdev_throughput:.2f}")
                   if type == 'sender':
                       for cpu, avg in averages.items():
                            avg_str = '   '.join(f"{key:4s}: {value:4.2f}" for key, value in avg.items())
                            print(f"    Sender CPU {cpu}:   {avg_str}")
                   else:
                       for cpu, avg in averages.items():
                            avg_str = '   '.join(f"{key:4s}: {value:4.2f}" for key, value in avg.items())
                            print(f"  Receiver CPU {cpu}:   {avg_str}")
               else:
                   print(f"\nNo throughput data available for test {test_name} to Host: {ip_address}")
               prev_test_name = test_name
               prev_ip_address = ip_address
    else:
        print("ERROR: calculate_averages returned no results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files to calculate average CPU metrics.')
    parser.add_argument('-i', '--input_dir', default='.', help='Input directory containing JSON files (default: current directory)')
    parser.add_argument('-f', '--format', choices=['human', 'json', 'csv'], default='human', help='Output format (default: human-readable)')
    parser.add_argument('-o', '--output_file', help='Output filename (default = mpstat-summary.{csv,json,txt)')

    args = parser.parse_args()

    main(args.input_dir, args.format, args.output_file)


