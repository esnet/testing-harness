#!/usr/bin/env python3

import os
import subprocess
import argparse

# Function to ping a host using the system's native ping command
def ping_host(host):
    try:
        # Run the ping command and capture the output
        output = subprocess.check_output(['ping', '-c', '4', host], universal_newlines=True)

        # Check if the output contains the word "received" to determine reachability
        if "received" in output:
            return True
        else:
            return False
    except subprocess.CalledProcessError:
        return False

# Function to check if a port is open using 'nc'
def check_port_open(host, port):
    try:
        subprocess.check_output(['nc', '-z', '-w', '2', host, port], universal_newlines=True)
        return True
    except subprocess.CalledProcessError:
        return False

# Function to read hosts from a file and ping them and check port 443
def ping_and_check_port_from_file(filename, check_port):
    reachable_hosts = []
    unreachable_hosts = []

    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                # Ignore lines that start with "#"
                if not line.startswith("#") and line:
                    host = line
                    print ("Checking host: ", host)
                    is_reachable = ping_host(host)
                    if check_port:
                        is_port_open = check_port_open(host, '443')
                        if is_reachable and is_port_open:
                            reachable_hosts.append(host)
                        else:
                            unreachable_hosts.append(host)
                    else:
                        if is_reachable:
                            reachable_hosts.append(host)
                        else:
                            unreachable_hosts.append(host)

        return reachable_hosts, unreachable_hosts
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return [], []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check host reachability and port 443 status.')
    parser.add_argument('filename', metavar='filename', type=str,
                        help='Path to the hosts file')
    parser.add_argument('--check-port', action='store_true',
                        help='Check if port 443 is open using nc')

    args = parser.parse_args()

    reachable_hosts, unreachable_hosts = ping_and_check_port_from_file(args.filename, args.check_port)

    print("Reachable Hosts:")
    for host in reachable_hosts:
        print(f"{host} is reachable")

    if args.check_port:
        print("\nHosts with Port 443 Open:")
        for host in reachable_hosts:
            print(f"{host} has port 443 open")

    print("\nUnreachable Hosts:")
    for host in unreachable_hosts:
        print(f"{host} is unreachable")


