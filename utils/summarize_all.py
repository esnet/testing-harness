#!/usr/bin/env python3

import os
import re
import json
import argparse
import csv
import statistics
from collections import defaultdict

# Globals
verbose = 0
from_pscheduler = 0  # kept for compatibility with older workflows


def get_nested(data, *keys, default=None):
    """Safe nested getter."""
    cur = data
    try:
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return default


def load_iperf_fields(file_path):
    """
    Load iperf3 JSON and extract:
      - num_streams
      - send_cpu (host_total)
      - recv_cpu (remote_total)
      - mss (tcp_mss_default)
      - fq_rate
    Returns (data, num_streams, send_cpu, recv_cpu, mss, fq_rate)
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: JSON parse error in file {file_path}: {e}")
        return None, None, None, None, None, None

    # num_streams
    num_streams = get_nested(data, "start", "test_start", "num_streams")

    # MSS and fq_rate (may be None)
    mss = get_nested(data, "start", "tcp_mss_default")
    fq_rate = get_nested(data, "start", "fq_rate")

    # CPU
    send_cpu = get_nested(data, "end", "cpu_utilization_percent", "host_total")
    recv_cpu = get_nested(data, "end", "cpu_utilization_percent", "remote_total")

    if verbose:
        print(
            f"load_iperf_fields({file_path}): num_streams={num_streams}, "
            f"mss={mss}, fq_rate={fq_rate}, send_cpu={send_cpu}, recv_cpu={recv_cpu}"
        )

    return data, num_streams, send_cpu, recv_cpu, mss, fq_rate


def extract_throughput(src_cmd_file):
    """
    Extract throughput, retrans, num_streams, send/recv CPU, MSS, FQ rate
    from an iperf3 JSON file, with robust handling of different layouts.
    """
    data, num_streams, send_cpu, recv_cpu, mss, fq_rate = load_iperf_fields(src_cmd_file)

    if data is not None:
        # Try several common iperf3 layouts for the summary section
        sum_section = (
            get_nested(data, "end", "sum_sent")
            or get_nested(data, "end", "sum")
            or data.get("sum_sent")
            or data.get("sum")
        )

        if sum_section is None:
            if verbose:
                print(f"Warning: no sum/sum_sent section in {src_cmd_file}")
            # pass back everything except throughput/retrans
            return None, None, num_streams, send_cpu, recv_cpu, mss, fq_rate

        bits_per_second = sum_section.get("bits_per_second")
        if bits_per_second is None:
            if verbose:
                print(f"Warning: no bits_per_second in summary for {src_cmd_file}")
            return None, None, num_streams, send_cpu, recv_cpu, mss, fq_rate

        tput = float(bits_per_second) / 1e9  # Gbit/s
        retrans = sum_section.get("retransmits", 0)

        if verbose:
            print(
                f"extract_throughput({src_cmd_file}): tput={tput} Gbps, "
                f"retrans={retrans}, num_streams={num_streams}"
            )

        return tput, retrans, num_streams, send_cpu, recv_cpu, mss, fq_rate

    # Fallback: parse old-style text output if JSON wasn't usable
    try:
        with open(src_cmd_file, "r") as f:
            for line in f:
                if "sender" in line and "CPU" not in line:
                    m = re.search(r"(\d+\.\d+)\s*Gbits/sec\s*(\d+)", line)
                    if m:
                        tput = float(m.group(1))
                        retrans = int(m.group(2))
                        return tput, retrans, num_streams, send_cpu, recv_cpu, mss, fq_rate
    except Exception as e:
        if verbose:
            print(f"Warning: fallback parse failed for {src_cmd_file}: {e}")

    return None, None, num_streams, send_cpu, recv_cpu, mss, fq_rate


def find_files(input_dir="."):
    """
    Walk directory tree and find iperf3 src-cmd:* files.
    """
    results = []
    # match files ending in .N
    iperf_pattern = re.compile(r"src-cmd:(?P<host>[^:]+):\d+$")
    # match files ending in .json
    #iperf_pattern = re.compile(r"src-cmd:(?P<host>[^:]+):\d+(?:\.json)?$")

    for root, dirs, files in os.walk(input_dir):
        file_cnt = 0
        for file in files:
            full_path = os.path.join(root, file)

            m = iper_pattern = iperf_pattern.search(file)
            if m:
                ip = m.group("host")
                test_name = os.path.basename(root)
                results.append(
                    {
                        "file": full_path,
                        "test_name": test_name,
                        "ip_address": ip,
                        "type": "src_cmd",
                    }
                )
                file_cnt += 1

        if file_cnt == 0 and verbose:
            print(f"No iperf3 files found in directory: {root}. Skipping...")

    return results


def write_human_readable(
    throughput_values,
    retrans_values,
    snd_cpu_values,
    rcv_cpu_values,
    mss_values,
    fq_rate_values,
):
    """
    Human-readable summary with MSS and FQ_Rate on the Host: line.
    """
    print("\nResult Summary, sorted by Average Throughput:")

    # Aggregate per host and per test_name
    per_host_tests = defaultdict(list)  # ip -> list of test dicts
    host_summary = {}  # ip -> aggregate info for Host: line

    for (test_name, ip_address), tputs in throughput_values.items():
        if not tputs:
            continue

        clean_tputs = [t for t in tputs if isinstance(t, (int, float)) and t > 0]
        if not clean_tputs:
            continue

        nruns = len(clean_tputs)  # number of files/runs for this test_name+host
        avg_tput = round(statistics.mean(clean_tputs), 1)
        min_tput = round(min(clean_tputs), 1)
        max_tput = round(max(clean_tputs), 1)
        stdev_tput = round(statistics.stdev(clean_tputs), 1) if nruns > 1 else 0.0

        rlist = retrans_values.get((test_name, ip_address), [])
        rlist = [r for r in rlist if isinstance(r, (int, float))]
        avg_retrans = int(statistics.mean(rlist)) if rlist else 0

        snd_list = snd_cpu_values.get((test_name, ip_address), [])
        snd_list = [x for x in snd_list if isinstance(x, (int, float))]
        avg_snd_cpu = int(statistics.mean(snd_list)) if snd_list else 0

        rcv_list = rcv_cpu_values.get((test_name, ip_address), [])
        rcv_list = [x for x in rcv_list if isinstance(x, (int, float))]
        avg_rcv_cpu = int(statistics.mean(rcv_list)) if rcv_list else 0

        mss_list = mss_values.get((test_name, ip_address), [])
        mss_list = [m for m in mss_list if isinstance(m, (int, float))]
        avg_mss = int(statistics.mean(mss_list)) if mss_list else None

        fq_list = fq_rate_values.get((test_name, ip_address), [])
        fq_list = [f for f in fq_list if isinstance(f, (int, float))]
        avg_fq = int(statistics.mean(fq_list)) if fq_list else None

        per_host_tests[ip_address].append(
            {
                "test_name": test_name,
                "avg_tput": avg_tput,
                "min_tput": min_tput,
                "max_tput": max_tput,
                "stdev_tput": stdev_tput,
                "avg_retrans": avg_retrans,
                "avg_snd_cpu": avg_snd_cpu,
                "avg_rcv_cpu": avg_rcv_cpu,
            }
        )

        # host-level summary (we'll fill ntests later as number of test types)
        hs = host_summary.setdefault(
            ip_address,
            {
                "ntests": 0,        # will be overwritten
                "mss": None,
                "fq_rate": None,
                "max_avg_tput": 0.0,
            },
        )
        if avg_tput > hs["max_avg_tput"]:
            hs["max_avg_tput"] = avg_tput
        if hs["mss"] is None and avg_mss is not None:
            hs["mss"] = avg_mss
        if hs["fq_rate"] is None and avg_fq is not None:
            hs["fq_rate"] = avg_fq

    # For each host, sort tests by avg_tput desc
    for ip in per_host_tests:
        per_host_tests[ip].sort(key=lambda t: t["avg_tput"], reverse=True)

    # Now set ntests = number of test types (test_name groups) per host
    for ip, tests in per_host_tests.items():
        host_summary[ip]["ntests"] = len(tests)

    # Sort hosts by their best test
    def host_key(ip):
        info = host_summary.get(ip, {})
        return info.get("max_avg_tput", 0.0)

    for ip in sorted(per_host_tests.keys(), key=host_key, reverse=True):
        hs = host_summary.get(ip, {})
        mss_display = hs.get("mss")
        fq_display = hs.get("fq_rate")
        mss_display = mss_display if mss_display is not None else "?"
        fq_display = fq_display if fq_display is not None else "?"

        ntests = hs.get("ntests", 0)  # now "5" for 5 test types

        print(
            f"Host: {ip}  (MSS = {mss_display}, FQ_Rate = {fq_display}) "
            f"(average result for {ntests} tests)"
        )
        for test in per_host_tests[ip]:
            print(
                f"    Test Name: {test['test_name']}, "
                f"Ave Throughput: {test['avg_tput']} Gbps, "
                f"Min: {test['min_tput']} Gbps, "
                f"Max: {test['max_tput']} Gbps, "
                f"StdDev: {test['stdev_tput']} Gbps, "
                f"retrans: {test['avg_retrans']}, "
                f"snd cpu: {test['avg_snd_cpu']}%, "
                f"rcv cpu: {test['avg_rcv_cpu']}%"
            )


def write_to_csv(output_file, throughput_values, snd_cpu_values, rcv_cpu_values):
    fieldnames = [
        "test_name",
        "ip_address",
        "avg_tput",
        "min_tput",
        "max_tput",
        "stdev_tput",
        "avg_snd_cpu",
        "avg_rcv_cpu",
    ]
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (test_name, ip_address), tputs in throughput_values.items():
            if not tputs:
                continue
            clean_tputs = [t for t in tputs if isinstance(t, (int, float)) and t > 0]
            if not clean_tputs:
                continue

            ntests = len(clean_tputs)
            avg_tput = round(statistics.mean(clean_tputs), 1)
            min_tput = round(min(clean_tputs), 1)
            max_tput = round(max(clean_tputs), 1)
            stdev_tput = round(statistics.stdev(clean_tputs), 1) if ntests > 1 else 0.0

            snd_list = snd_cpu_values.get((test_name, ip_address), [])
            snd_list = [x for x in snd_list if isinstance(x, (int, float))]
            avg_snd_cpu = int(statistics.mean(snd_list)) if snd_list else 0

            rcv_list = rcv_cpu_values.get((test_name, ip_address), [])
            rcv_list = [x for x in rcv_list if isinstance(x, (int, float))]
            avg_rcv_cpu = int(statistics.mean(rcv_list)) if rcv_list else 0

            writer.writerow(
                {
                    "test_name": test_name,
                    "ip_address": ip_address,
                    "avg_tput": avg_tput,
                    "min_tput": min_tput,
                    "max_tput": max_tput,
                    "stdev_tput": stdev_tput,
                    "avg_snd_cpu": avg_snd_cpu,
                    "avg_rcv_cpu": avg_rcv_cpu,
                }
            )


def write_to_json(output_file, throughput_values, snd_cpu_values, rcv_cpu_values):
    results = []
    for (test_name, ip_address), tputs in throughput_values.items():
        if not tputs:
            continue
        clean_tputs = [t for t in tputs if isinstance(t, (int, float)) and t > 0]
        if not clean_tputs:
            continue

        ntests = len(clean_tputs)
        avg_tput = round(statistics.mean(clean_tputs), 1)
        min_tput = round(min(clean_tputs), 1)
        max_tput = round(max(clean_tputs), 1)
        stdev_tput = round(statistics.stdev(clean_tputs), 1) if ntests > 1 else 0.0

        snd_list = snd_cpu_values.get((test_name, ip_address), [])
        snd_list = [x for x in snd_list if isinstance(x, (int, float))]
        avg_snd_cpu = int(statistics.mean(snd_list)) if snd_list else 0

        rcv_list = rcv_cpu_values.get((test_name, ip_address), [])
        rcv_list = [x for x in rcv_list if isinstance(x, (int, float))]
        avg_rcv_cpu = int(statistics.mean(rcv_list)) if rcv_list else 0

        results.append(
            {
                "test_name": test_name,
                "ip_address": ip_address,
                "avg_tput": avg_tput,
                "min_tput": min_tput,
                "max_tput": max_tput,
                "stdev_tput": stdev_tput,
                "avg_snd_cpu": avg_snd_cpu,
                "avg_rcv_cpu": avg_rcv_cpu,
            }
        )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def main(args):
    global verbose
    input_dir = args.input_dir

    throughput_values = defaultdict(list)
    retrans_values = defaultdict(list)
    snd_cpu_loads = defaultdict(list)
    rcv_cpu_loads = defaultdict(list)
    mss_values = defaultdict(list)
    fq_rate_values = defaultdict(list)

    results = find_files(input_dir)
    for result in results:
        input_file = result["file"]
        test_name = result["test_name"]
        ip_address = result["ip_address"]

        # New line per your request:
        print(f"Processing file: {input_file}")

        tput, retrans, num_streams, send_cpu, recv_cpu, mss, fq_rate = extract_throughput(
            input_file
        )

        if tput is not None:
            throughput_values[(test_name, ip_address)].append(tput)
            retrans_values[(test_name, ip_address)].append(retrans)
            if send_cpu is not None:
                snd_cpu_loads[(test_name, ip_address)].append(send_cpu)
            if recv_cpu is not None:
                rcv_cpu_loads[(test_name, ip_address)].append(recv_cpu)
            if mss is not None:
                mss_values[(test_name, ip_address)].append(mss)
            if fq_rate is not None:
                fq_rate_values[(test_name, ip_address)].append(fq_rate)
        else:
            if verbose:
                print(f"Throughput not found in file {input_file}")

    if args.format == "human":
        write_human_readable(
            throughput_values,
            retrans_values,
            snd_cpu_loads,
            rcv_cpu_loads,
            mss_values,
            fq_rate_values,
        )
    elif args.format == "csv":
        output_file = args.output_file or "summary.csv"
        write_to_csv(output_file, throughput_values, snd_cpu_loads, rcv_cpu_loads)
    elif args.format == "json":
        output_file = args.output_file or "summary.json"
        write_to_json(output_file, throughput_values, snd_cpu_loads, rcv_cpu_loads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize iperf3 JSON results."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        default=".",
        help="Input directory containing iperf3 JSON files (default: current directory)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["human", "json", "csv"],
        default="human",
        help="Output format (default: human-readable)",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="Output filename when using csv/json format",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )

    args = parser.parse_args()
    verbose = args.verbose
    main(args)

