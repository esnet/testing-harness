#!/usr/bin/env python3

# summarize all mpstat and iperf3 output in a directory tree
# run with -h to see options
#
# typical use:
#   copy send and receiver output files from collect.py to a directory tree such as:
#      test_name
#          1stream
#              send
#              rcv
#          8streams
#              send
#              rcv
# then cd to test_name/1stream, and run this command from there.
# Note that is should work fine to copy everything to just '1stream', but
#    its easier to detect missing data if keep send and rcv separate.

# TO DO:
#   - add num_streams to csv and JSON output
#   - test csv and JSON with num_streams > 1


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
verbose = 0  # set to get warning about files with missing data
max_tput_summary = 0  # set to get summary of max tput as well as average tput
DEFAULT_IRQ_CORE = 4
DEFAULT_IPERF3_CORE = 5
NUM_STREAMS = 1   # XXX: get this from the JSON results!!

def load_iperf3_json(file_path):

# NOTE: iperf3 JSON seems to be messed up, and the standard python JSON decoder
#   can not parse it. As a workaround, use 'jq', and then throw out extra stuff
#   XXX: need to report this bug

    filename = os.path.basename(file_path)  # Extract filename from file_path
    #print ("Getting tput data from file: ", filename)
    try:
        # Call jq to extract the desired data
        result = subprocess.run(['jq', '.end.sum_sent', file_path], capture_output=True, text=True, check=True)
        # Remove newlines and everything after "}"
        result = result.stdout.replace('\n', '').split('}')[0] + '}'
        #print ("jq result: ", result)
        # Load the JSON output
        try:
            data = json.loads(result)
        except:
            print(f"Error parsing data in file: ", file_path)
            return None, None
        #print ("got sum_sent data: ", data)
        # also get num_streams
        result = subprocess.run(['jq', '.start.test_start.num_streams', file_path], capture_output=True, text=True, check=True)
        num_streams = int(result.stdout.split('\n')[0])  # just grab first line due to iperf3 JSON bug
        #print (f"{file_path}: Got num_streams: ", num_streams)
        return data, num_streams
    except subprocess.CalledProcessError as e:
        #print("Error calling jq:", e)
        # probably not JSON
        return None, None


def calculate_cpu_averages(cpu_loads):
    # note: this routine is use to calculate average for within a test, AND for set of tests

    if not cpu_loads:
        print("calculate_cpu_averages Error: no cpu data provided")
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

    # Calculate CPU averages
    averages = {key: round(value / count, 1) for key, value in sums.items() if value != 0}
    #print ("   calculate_cpu_averages returns: ", averages)
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

    averages = {cpu: calculate_cpu_averages(loads) for cpu, loads in cpu_loads.items() if loads}
    #print ("process_cpu_data returns: ", cpu_loads)
    return averages, cpu_loads

def find_files():
    cwd = os.getcwd()
    mpstat_snd_pattern = re.compile(r"mpstat-sender:([\w\.\-]+):.*\.json")
    mpstat_rcv_pattern = re.compile(r"mpstat-receiver:([\w\.\-]+):.*\.json")
    src_cmd_pattern = re.compile(r"src-cmd:([\w\.\-]+):.*")

    results = []

    for root, dirs, files in os.walk(cwd):
        file_cnt = 0
        for file in files:
            #print("Checking file: ", file)
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

    #print ("will look at these files: ", results)
    return results

def extract_throughput(src_cmd_file):
    data, num_streams = load_iperf3_json(src_cmd_file)
    if data:
         tput = float(data["bits_per_second"]) / 1000000000  # in Gbps
         retrans = data["retransmits"] 
         if verbose:
             print(f"loaded JSON results: tput={tput} Gbps, retrans={retrans}")
         return tput, retrans, num_streams
    else: # not JSON, so assume normal iperf3 output format
        with open(src_cmd_file, 'r') as f:
            for line in f:
                if 'sender' in line and 'CPU' not in line:
                    # XXX: what if Mbits/sec, not Gbits??
                    throughput = re.search(r'(\d+\.\d+)\s*Gbits/sec\s*(\d+)', line)

                    if throughput:
                        tput = float(throughput.group(1)) # group(1) ensures a single number
                        retrans = int(throughput.group(2))
                        #print ("   extract_throughput returns: ", tput)
                        return tput, retrans, 1   # XXX: need to grab num_streams too!
    # If no throughput data is found, return None for both throughput and retrans
    return None, None, None


def write_to_csv(output_file, cpu_data, throughput_values, retrans_values):
   # XXX: need to add retrans!

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
            ave_tput = round(statistics.mean(ave_throughput), 1) if ave_throughput else None
            max_throughput = max(ave_throughput, default=None)
            stdev_throughput = round(statistics.stdev(ave_throughput), 1) if len(ave_throughput) > 1 else None

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

def write_to_json(output_file, cpu_averages, throughput_values, retrans_values):
   # XXX: need to add retrans!

    output = {}
    for entry in cpu_averages:
        (test_name, ip_address, type), averages = entry
        if len(throughput_values.get((test_name, ip_address), [])) == 0:
            continue
        avg_throughput = round(statistics.mean(throughput_values.get((test_name, ip_address), [])), 1)
        max_throughput = round(max(throughput_values.get((test_name, ip_address), []), default=None), 1)
        stdev_throughput = round(statistics.stdev(throughput_values.get((test_name, ip_address), [])), 1) if len(throughput_values.get((test_name, ip_address), [])) > 1 else None

        cpu_averages_data = {
            "sender": {},
            "receiver": {}
        }

        for cpu, avg in averages.items():
            cpu_type = "sender" if type == "mpstat_snd" else "receiver"
            cpu_averages_data[cpu_type][f"CPU {cpu}"] = {key: round(value, 1) for key, value in avg.items()}

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



def main(args):
    input_dir = args.input_dir
    output_format = args.format
    output_file = args.output_file
    irq_core = args.irq_core
    iperf3_core = args.iperf3_core

    snd_cpu_loads = defaultdict(lambda: defaultdict(list))
    rcv_cpu_loads = defaultdict(lambda: defaultdict(list))
    throughput_values = defaultdict(list)
    retrans_values = defaultdict(list)
    min_throughput_per_test = defaultdict(dict)
    max_throughput_per_test = defaultdict(dict)
    ave_throughput_per_test = defaultdict(dict)
    stdev_throughput_per_test = defaultdict(dict)
    ave_retrans_per_test = defaultdict(dict)

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
            throughput, retrans, num_streams = extract_throughput(input_file)
            #print (f"test name: {test_name}, IP: {ip_address}, tput: {throughput}")
            if throughput is not None:
                throughput_values[(test_name, ip_address)].append(throughput)
                retrans_values[(test_name, ip_address)].append(retrans)
            else:
                if verbose:
                   print (f"   Throughput not found in file {input_file}")
        if type == 'mpstat_snd' or type == 'mpstat_rcv':
            with open(input_file, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    parent_dir = os.path.basename(os.path.dirname(input_file))
                    filename_with_parent = os.path.join(parent_dir, fname)
                    print("   Error loading mpstat file: ", input_file)
                    print("   JSON truncated? Continuing")
                    continue

                averages, cpu_loads = process_cpu_data(data)

                for cpu, loads in cpu_loads.items():
                    #print (f"cpu: {cpu}, type: {type}, loads: {loads}")
                    if type == 'mpstat_snd':
                        snd_cpu_loads[(test_name, ip_address)][cpu].extend(loads)
                    else:
                        rcv_cpu_loads[(test_name, ip_address)][cpu].extend(loads)

    print(f"\n Got iperf3 data from {len(throughput_values)} files")
    print(f"\n Got mpstat data from {len(snd_cpu_loads)} sender files and {len(rcv_cpu_loads)} receiver files")
    overall_cpu_averages = {}
    print ("\nComputing CPU Averages...")
    for (test_name, ip_address), cpu_data in snd_cpu_loads.items():
        cpu_averages = {}
        #print("snd_cpu_data.items: ",snd_cpu_loads.items())

        for cpu, loads in cpu_data.items():
            #print ("  in loop: loads = ", loads)
            if loads:
                cpu_averages[cpu] = calculate_cpu_averages(loads)
                #print (f"   CPU {cpu} averages for test {test_name}  IP {ip_address} ", cpu_averages[cpu])
        overall_cpu_averages[(test_name, ip_address, 'sender')] = cpu_averages

    for (test_name, ip_address), cpu_data in rcv_cpu_loads.items():
        cpu_averages = {}

        for cpu, loads in cpu_data.items():
            if loads:
                cpu_averages[cpu] = calculate_cpu_averages(loads)
        overall_cpu_averages[(test_name, ip_address, 'receiver')] = cpu_averages

    if overall_cpu_averages:
        # Sort the dictionary by test_name and ip_address
        sorted_cpu_averages = sorted( overall_cpu_averages.items(), key=lambda x: (x[0][1], x[0][0]))
        overall_cpu_averages = dict(sorted_cpu_averages)  # Convert back to dictionary if needed

        if output_format == 'csv':
            write_to_csv(output_file, sorted_cpu_averages, throughput_values, retrans_values)
        elif output_format == 'json':
            write_to_json(output_file, sorted_cpu_averages, throughput_values, retrans_values)
        else:
            prev_ip_address = prev_test_name = ""
             
            print ("\nSummary of all testing: \n")
            for (test_name, ip_address, type), cpu_averages in sorted_cpu_averages:
               #if (test_name, ip_address) in throughput_values and throughput_values[(test_name, ip_address)]:
               if (test_name, ip_address) in throughput_values:
                   if test_name != prev_test_name or ip_address != prev_ip_address: # only print for new test
                       numtests = len(throughput_values[(test_name, ip_address)])
                       avg_throughput = statistics.mean(throughput_values[(test_name, ip_address)])
                       min_throughput = min(throughput_values[(test_name, ip_address)])
                       max_throughput = max(throughput_values[(test_name, ip_address)])
                       min_throughput_per_test[ip_address][test_name] = min_throughput # save for summary at the end
                       max_throughput_per_test[ip_address][test_name] = max_throughput # save for summary at the end
                       ave_throughput_per_test[ip_address][test_name] = avg_throughput # save for summary at the end
                       if numtests > 1:
                           tput_array = throughput_values[(test_name, ip_address)]
                           stdev_throughput = statistics.stdev(tput_array)
                       else:
                           print ("Warning: only 1 test found.")
                           stdev_throughput = 0
                       stdev_throughput_per_test[ip_address][test_name] = stdev_throughput # save for summary at the end
                       avg_retrans = int(statistics.mean(retrans_values[(test_name, ip_address)]))
                       ave_retrans_per_test[ip_address][test_name] = avg_retrans # save for summary at the end
                       print(f"\nTest {test_name} to Host: {ip_address}   (num tests: {numtests})")
                       print(f"       Throughput:   Mean: {avg_throughput:.1f} Gbps   Min: {min_throughput:.1f} Gbps  Max: {max_throughput:.1f} Gbps   STDEV: {stdev_throughput:.1f}   Retr: {avg_retrans}")
                   if type == 'sender':
                       total_snd = {}
                       for cpu, avg in cpu_averages.items():
                            cpu = int(cpu)
                            total_snd[cpu] = sum(value for key, value in avg.items() if key != 'idle')
                            avg_str = '   '.join(f"{key:4s}: {value:4.1f}" for key, value in avg.items())
                            print(f"     Sender CPU {cpu}:   {avg_str}   Total:{total_snd[cpu]:3.1f}")
                   else:
                       total_rcv = {}
                       for cpu, avg in cpu_averages.items():
                            cpu = int(cpu)
                            total_rcv[cpu] = sum(value for key, value in avg.items() if key != 'idle')
                            avg_str = '   '.join(f"{key:4s}: {value:4.1f}" for key, value in avg.items())
                            print(f"   Receiver CPU {cpu}:   {avg_str}   Total:{total_rcv[cpu]:3.1f}")
                       #print (f"num_streams: {num_streams}; total_snd: ", total_snd)
                       #print (f"len of total_rcv: {len(total_rcv)}, total_rcv: ", total_rcv)
                       if num_streams == 1:
                          if len(total_rcv) > 0:
                             try:
                                 if total_snd[irq_core] > 90:
                                      print ("    ** Throughput appears to be limited by IRQ on the Send Host **")
                                 elif total_snd[iperf3_core] > 90:
                                      print ("    ** Throughput appears to be limited by application CPU on the Send Host **")
                                 elif total_rcv[irq_core] > 90:
                                      print ("     ** Throughput appears to be limited by application CPU on the Receive Host **")
                                 elif total_rcv[iperf3_core] > 90:
                                      print ("     ** Throughput appears to be limited by application CPU on the Receive Host **")
                                 else:
                                      print ("     ** Throughput appears to be memory limited, CWND limited, or unknown **")
                             except:
                                  print(f"Error getting core usage for cores {irq_core} or {iperf3_core}." )
               else:
                   if verbose:
                       print(f"\nNo throughput data available for test {test_name} to Host: {ip_address}")

               prev_test_name = test_name
               prev_ip_address = ip_address

    else:
        print("   ERROR: no CPU data (mpstat files) found. Current version of this program requires them. Exiting..")
        print("   Re-run collect.py with mpstat specified in the ini file")
        sys.exit()

    # Sort max_throughput_per_test by maximum throughput for each IP
    sorted_max_throughput = {}
    for ip_address, tests in max_throughput_per_test.items():
        sorted_tests = sorted(tests.items(), key=lambda x: x[1], reverse=True)
        sorted_max_throughput[ip_address] = sorted_tests

    # Print sorted max_throughput_per_test
    if max_tput_summary:
        print("\n\nResult Summary, sorted by Max Throughput:")
        for ip_address, tests in sorted_max_throughput.items():
            print(f"IP Address: {ip_address}")
            for test_name, max_throughput in tests:
                print(f"     Test Name: {test_name}, Max Throughput: {max_throughput:.1f} Gbps")

    sorted_ave_throughput = {}
    for ip_address, tests in ave_throughput_per_test.items():
        sorted_tests = sorted(tests.items(), key=lambda x: x[1], reverse=True)
        sorted_ave_throughput[ip_address] = sorted_tests

    # Print sorted ave_throughput_per_test
    print("\n\nResult Summary, sorted by Average Throughput:")
    for ip_address, tests in sorted_ave_throughput.items():
        print(f"IP Address: {ip_address}")
        for test_name, ave_throughput in tests:
            stdev = stdev_throughput_per_test[ip_address][test_name]
            min_throughput = min_throughput_per_test[ip_address][test_name]
            max_throughput = max_throughput_per_test[ip_address][test_name]
            rtrans = int(ave_retrans_per_test[ip_address][test_name])
            print(f"    Test Name: {test_name}, Ave Throughput: {ave_throughput:.1f} Gbps (std: {stdev:.1f}, min: {min_throughput:.1f}, max: {max_throughput:.1f}),  Retr: {rtrans}")
            if verbose and stdev > 10:
                print("  ** high stdev! ** ", throughput_values[(test_name, ip_address)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files to calculate average CPU metrics.')
    parser.add_argument('-i', '--input_dir', default='.', help='Input directory containing JSON files (default: current directory)')
    parser.add_argument('-f', '--format', choices=['human', 'json', 'csv'], default='human', help='Output format (default: human-readable)')
    parser.add_argument('-o', '--output_file', help='Output filename (default = mpstat-summary.{csv,json,txt)')

    parser.add_argument('--irq_core', type=int, default=DEFAULT_IRQ_CORE, help=f'Core number for IRQ (default: {DEFAULT_IRQ_CORE})')
    parser.add_argument('--iperf3_core', type=int, default=DEFAULT_IPERF3_CORE, help=f'Core number for iperf3 (default: {DEFAULT_IPERF3_CORE})')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')

    args = parser.parse_args()
    verbose = args.verbose # set global

    main(args)


