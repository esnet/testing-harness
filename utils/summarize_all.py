#!/usr/bin/env python3

# new version that uses output of pscheduler, and gets CPU from iperf3
# If want detailed CPU from mpstat, use summarize_all_mpstat.py 
#
# note: there is some untested code in here to process mpstat files
# XXX: To Do: merge this program with summarize_all_mpstat.py
# XXX: .csv and .json output options not yet tested!

import os
import re
import json
import argparse
import csv
import sys
from collections import defaultdict
import statistics
import subprocess

# Define global variables for default values
verbose = 0
max_tput_summary = 0

def load_iperf3_json(file_path):
    filename = os.path.basename(file_path)
    new_filepath = file_path + ".fixed.json"
    try:
        if not os.path.exists(new_filepath):
            result = subprocess.run(['fix-pscheduler-json.sh', file_path], capture_output=True, text=True, check=True)

        result = subprocess.run(['jq', '.end', new_filepath], capture_output=True, text=True, check=True)
        try:
            data = json.loads(result.stdout)
        except:
            print(f"Error parsing data in file: {file_path}")
            return None, None, None, None

        result = subprocess.run(['jq', '.start.test_start.num_streams', new_filepath], capture_output=True, text=True, check=True)
        num_streams = int(result.stdout.split('\n')[0])

        send_cpu = float(data["cpu_utilization_percent"]["host_total"])
        recv_cpu = float(data["cpu_utilization_percent"]["remote_total"])
        return data, num_streams, send_cpu, recv_cpu
    except subprocess.CalledProcessError as e:
        return None, None, None, None

def calculate_cpu_averages(cpu_loads):
    if not cpu_loads:
        return None

    sums = {
        "usr": 0, "sys": 0, "nice": 0, "iowait": 0, "irq": 0,
        "soft": 0, "steal": 0, "guest": 0, "gnice": 0, "idle": 0
    }
    count = 0

    for load in cpu_loads:
        for key in sums:
            sums[key] += load[key]
        count += 1

    averages = {key: round(value / count, 1) for key, value in sums.items() if value != 0}
    return averages

def process_cpu_data(data):
    cpu_loads = defaultdict(list)
    for host in data['sysstat']['hosts']:
        for stat in host['statistics']:
            for load in stat['cpu-load']:
                cpu_loads[load['cpu']].append(load)
    
    for cpu, loads in cpu_loads.items():
        cpu_loads[cpu] = loads[4:-1]

    averages = {cpu: calculate_cpu_averages(loads) for cpu, loads in cpu_loads.items() if loads}
    return averages, cpu_loads

def find_files():
    cwd = os.getcwd()
    mpstat_snd_pattern = re.compile(r"mpstat-sender:([\w\.\-]+):.*\.json")
    mpstat_rcv_pattern = re.compile(r"mpstat-receiver:([\w\.\-]+):.*\.json")
    # also, dont match on *fixed* either
    src_cmd_pattern = re.compile(r"src-cmd:([\w\.\-]+):(?!.*fixed*)")


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
            if verbose:
                print(f"No iperf3 or mpstat files found in directory: {root}. Skipping...")
            continue

    return results

def extract_throughput(src_cmd_file):
    data, num_streams, send_cpu, recv_cpu = load_iperf3_json(src_cmd_file)
    if data:
        tput = float(data["sum_sent"]["bits_per_second"]) / 1000000000
        retrans = data["sum_sent"]["retransmits"]
        if verbose:
            print(f"loaded JSON results: tput={tput} Gbps, retrans={retrans}, send CPU={send_cpu}, recv CPU={recv_cpu}")
        return tput, retrans, num_streams, send_cpu, recv_cpu
    else:
        with open(src_cmd_file, 'r') as f:
            for line in f:
                if 'sender' in line and 'CPU' not in line:
                    throughput = re.search(r'(\d+\.\d+)\s*Gbits/sec\s*(\d+)', line)
                    if throughput:
                        tput = float(throughput.group(1))
                        retrans = int(throughput.group(2))
                        return tput, retrans, 1, None, None
    return None, None, None, None, None

def write_human_readable(throughput_values, retrans_values, snd_cpu_values, rcv_cpu_values):
    print("\nResult Summary, sorted by Average Throughput:")

    #print ("debug: retrans_values dict: ", retrans_values)
    #print ("debug: cpu_values dict: ", snd_cpu_values)

    # Sort results by average throughput within each IP address
    sorted_results = {}
    for (test_name, ip_address), tputs in throughput_values.items():
        avg_tput = round(statistics.mean(tputs), 1)
        min_tput = round(min(tputs), 1)
        max_tput = round(max(tputs), 1)
        stdev_tput = round(statistics.stdev(tputs), 1) if len(tputs) > 1 else 0.0
        avg_retrans = int(statistics.mean(retrans_values[(test_name, ip_address)])) if (test_name, ip_address) in retrans_values else 0
        avg_snd_cpu = int(statistics.mean(snd_cpu_values[(test_name, ip_address)])) if (test_name, ip_address) in snd_cpu_values else 0
        avg_rcv_cpu = int(statistics.mean(rcv_cpu_values[(test_name, ip_address)])) if (test_name, ip_address) in snd_cpu_values else 0
        # only include count if value > 0
        ntests = sum(1 for x in tputs if x > 0)
        
        # Store the data in a nested dictionary sorted by IP
        if ip_address not in sorted_results:
            sorted_results[ip_address] = []
        
        sorted_results[ip_address].append({
            "test_name": test_name,
            "avg_tput": avg_tput,
            "stdev_tput": stdev_tput,
            "min_tput": min_tput,
            "max_tput": max_tput,
            "avg_retrans": avg_retrans,
            "avg_snd_cpu": avg_snd_cpu,
            "avg_rcv_cpu": avg_rcv_cpu
        })

    # Sort each IP's test entries by average throughput in descending order
    for ip_address in sorted_results:
        sorted_results[ip_address].sort(key=lambda x: x["avg_tput"], reverse=True)

    # Print the sorted results
    for ip_address, tests in sorted_results.items():
        print(f"Host: {ip_address}  (average result for {ntests} tests)")
        for test in tests:
            print(f"    Test Name: {test['test_name']}, Ave Throughput: {test['avg_tput']} Gbps (std: {test['stdev_tput']}, min: {test['min_tput']}, max: {test['max_tput']}),  Retr: {test['avg_retrans']}, snd cpu: {test['avg_snd_cpu']}%, rcv cpu: {test['avg_rcv_cpu']}%")


def write_to_csv(output_file, throughput_values, cpu_averages):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Test Name', 'IP', 'Avg Throughput (Gbps)', 'Min Throughput (Gbps)', 'Max Throughput (Gbps)', 'CPU Type', 'CPU Data']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for (test_name, ip_address), tputs in throughput_values.items():
            avg_tput = statistics.mean(tputs)
            min_tput = min(tputs)
            max_tput = max(tputs)
            for cpu_type in ['sender', 'receiver']:
                if (test_name, ip_address, cpu_type) in cpu_averages:
                    row = {
                        'Test Name': test_name,
                        'IP': ip_address,
                        'Avg Throughput (Gbps)': avg_tput,
                        'Min Throughput (Gbps)': min_tput,
                        'Max Throughput (Gbps)': max_tput,
                        'CPU Type': cpu_type,
                        'CPU Data': cpu_averages[(test_name, ip_address, cpu_type)]
                    }
                    writer.writerow(row)
    print(f"CSV output saved to {output_file}")

def write_to_json(output_file, throughput_values, cpu_averages):
    output = {}
    for (test_name, ip_address), tputs in throughput_values.items():
        avg_tput = statistics.mean(tputs)
        min_tput = min(tputs)
        max_tput = max(tputs)
        output[f"{test_name}_{ip_address}"] = {
            "average_throughput": avg_tput,
            "min_throughput": min_tput,
            "max_throughput": max_tput,
            "cpu_averages": {
                "sender": cpu_averages.get((test_name, ip_address, 'sender'), {}),
                "receiver": cpu_averages.get((test_name, ip_address, 'receiver'), {})
            }
        }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"JSON output saved to {output_file}")

def main(args):
    use_iperf3_cpu = args.use_iperf3_cpu
    throughput_values = defaultdict(list)
    retrans_values = defaultdict(list)
    snd_cpu_loads = defaultdict(list)
    rcv_cpu_loads = defaultdict(list)
    cpu_averages = {}

    results = find_files()
    for result in results:
        input_file = result['file']
        test_name = result['test_name']
        ip_address = result['ip_address']
        type = result['type']
        
        if type == 'src_cmd':
            throughput, retrans, num_streams, send_cpu, recv_cpu = extract_throughput(input_file)
            if throughput is not None:
                throughput_values[(test_name, ip_address)].append(throughput)
                retrans_values[(test_name, ip_address)].append(retrans)
                snd_cpu_loads[(test_name, ip_address)].append(send_cpu)
                rcv_cpu_loads[(test_name, ip_address)].append(recv_cpu)
            else:
                if verbose:
                    print(f"Throughput not found in file {input_file}")
        elif not use_iperf3_cpu and (type == 'mpstat_snd' or type == 'mpstat_rcv'):
            with open(input_file, 'r') as f:
                try:
                    data = json.load(f)
                    averages, cpu_loads = process_cpu_data(data)
                    for cpu, loads in cpu_loads.items():
                        if type == 'mpstat_snd':
                            snd_cpu_loads[(test_name, ip_address)][cpu].extend(loads)
                        else:
                            rcv_cpu_loads[(test_name, ip_address)][cpu].extend(loads)
                except:
                    print(f"Error loading mpstat file: {input_file}")
                    continue

    if args.format == 'human':
        write_human_readable(throughput_values, retrans_values, snd_cpu_loads, rcv_cpu_loads)
    elif args.format == 'csv':
        output_file = args.output_file or "summary.csv"
        write_to_csv(output_file, throughput_values, cpu_averages)
    elif args.format == 'json':
        output_file = args.output_file or "summary.json"
        write_to_json(output_file, throughput_values, cpu_averages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files to calculate average CPU metrics.')
    parser.add_argument('-i', '--input_dir', default='.', help='Input directory containing JSON files (default: current directory)')
    parser.add_argument('-f', '--format', choices=['human', 'json', 'csv'], default='human', help='Output format (default: human-readable)')
    parser.add_argument('-o', '--output_file', help='Output filename (default = mpstat-summary.{csv,json,txt)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    parser.add_argument('--use_iperf3_cpu', action='store_true', help='Use CPU data from iperf3 JSON instead of mpstat files')
    
    args = parser.parse_args()
    verbose = args.verbose
    main(args)


