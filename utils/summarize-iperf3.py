#!/usr/bin/env python3
#
# Summarize results of iperf3 testing .json files collected by testing_harness

import os
import json
import sys
from tabulate import tabulate  # Make sure to install tabulate: pip install tabulate
import statistics

# Specify the directory to start the search from
directory_path = "."

# Create a list to store the extracted data
data = []

# Create a nested dictionary to store and calculate the average throughput and retransmits
average_throughput = {}

# Traverse all directories and files recursively
for root, dirs, files in os.walk(directory_path):
    for filename in files:
        if filename.startswith("src-cmd"):
            file_path = os.path.join(root, filename)

            # Open and load the JSON file
            #print("Reading file:", file_path)
            with open(file_path, "r") as json_file:
                try:
                    json_data = json.load(json_file)
                except:
                    print("JSON load error. Not a JSON file?")
                    sys.exit(-1)

            # Extract bits_per_second and retransmits from sum_sent
            dest_host = json_data["start"]["connecting_to"]["host"]
            nstreams = json_data["start"]["test_start"]["num_streams"]
            fq_rate = float(json_data["start"]["test_start"]["fqrate"]) / 1000000000
            cong = json_data["end"]["sender_tcp_congestion"]

            gbits_per_second = float(json_data["end"]["sum_sent"]["bits_per_second"]) / 1000000000
            retransmits = json_data["end"]["sum_sent"]["retransmits"]

            # Create a key for the nested dictionary based on dest_host, nstreams, cong, and fq_rate
            key = (dest_host, nstreams, cong, fq_rate)

            # Add the throughput and retransmits to the nested dictionary
            if key in average_throughput:
                average_throughput[key][0].append(gbits_per_second)
                average_throughput[key][1].append(retransmits)
            else:
                average_throughput[key] = [[gbits_per_second], [retransmits]]

            # Append the extracted data to the list
            data.append([dest_host, nstreams, cong, fq_rate, gbits_per_second, retransmits])

# Create headers for the average throughput table
avg_headers = ["Dest Host", "Streams", "CC Alg", "Pacing (Gbps)", "Throughput (Gbps)", "Stddev", "Retransmits"]

# Calculate and format the averages and standard deviation for each combination of dest_host, nstreams, cong, fq_rate
average_table_data = []

for key, values in average_throughput.items():
    avg_throughput = sum(values[0]) / len(values[0])
    avg_retransmits = sum(values[1]) / len(values[1])
    throughput_values = values[0]  # List of throughput values
    std_dev_throughput = statistics.stdev(throughput_values)  # Calculate stddev
    dest_host, nstreams, cong, fq_rate = key
    avg_throughput_formatted = "{:.2f}".format(avg_throughput)
    std_dev_throughput_formatted = "{:.2f}".format(std_dev_throughput)  # Format stddev
    average_table_data.append([dest_host, nstreams, cong, fq_rate, avg_throughput_formatted, std_dev_throughput_formatted, int(avg_retransmits)])

# Sort the data by columns: dest_host, nstreams, and cong
average_table_data = sorted(average_table_data, key=lambda x: (x[0], x[1], x[3]))

# Create a list to store the final formatted data
formatted_average_table_data = []

# Iterate through the average_table_data and add separators
prev_dest_host = None
for row in average_table_data:
    dest_host, nstreams, cong, fq_rate, avg_throughput, stddev_throughput, avg_retransmits = row

    # Check if dest_host is different from the previous row
    if dest_host != prev_dest_host:
        # Add a separator row with double lines
        separator_row = ["=" * len(header) for header in avg_headers]
        formatted_average_table_data.extend([separator_row, row])
        prev_dest_host = dest_host  # Update the previous dest_host
    else:
        formatted_average_table_data.append(row)

# Print the average throughput table with double lines for separators
print("\nAverage Throughput for 5 tests:")
print(tabulate(formatted_average_table_data, headers=avg_headers, tablefmt="grid", disable_numparse=True))

# Create headers for the main data table
data_headers = ["Dest Host", "Num Streams", "CC Alg", "Pacing (Gbps)", "Throughput", "Retransmits"]

# Sort the data by columns: dest_host, nstreams, cong, and fq_rate
data = sorted(data, key=lambda x: (x[0], x[1], x[3], x[2]))

# Print the main data table
print("\nResults from each individual test:")
print(tabulate(data, headers=data_headers, tablefmt="grid"))


