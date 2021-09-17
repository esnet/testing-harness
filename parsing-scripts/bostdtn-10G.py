"""
BOST-DTN ANALYSIS

1. ALL HOSTS
2. ESNET HOSTS
3. NON-ESNET HOSTS
"""

import os
import json
import csv
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

rootdir = "/Users/eashan22/Desktop/Internship 2021/bbrv2/Brian's Project/"

def _filereader(path, verbose=False):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        tempList = []
        # This skips the first row of the CSV file.
        next(csv_reader)

        for enum, row in enumerate(csv_reader):
            if verbose:
                print(f'{enum}.\t{row[0]}, {row[1]}, {row[2]}')
            tempList.append( (row[0],row[1],row[2]) )
            line_count += 1
        if verbose:
            print(f'Processed {line_count} lines.')
    return tempList

rtt_data = _filereader(os.path.join(rootdir,'test-hosts.bost-dtn.csv'),False)

def traverse_esnet(path):
    fileList_less_30, fileList_greater_30 = [], []
    c1, c2 = 0, 0
    for root, directories, files in os.walk(path):
        for file in files:
            # Filtering es.net names
            if file.startswith("ss") and file.endswith("json") and file.find("es.net")!=-1:
            # if file.startswith("ss") and file.endswith("json") and file.find("cern-773")!=-1:
                
                for enum, rtt_file in enumerate(rtt_data):
                    # Filtering rtt in ms
                    if file.find(rtt_file[0])!=-1 and float(rtt_file[1].split("ms")[0])<30:
                        print(file)
                        fileList_less_30.append(os.path.join(root,file))
                        c1+=1
                    elif file.find(rtt_file[0])!=-1 and float(rtt_file[1].split("ms")[0])>=30:
                        print(file)
                        fileList_greater_30.append(os.path.join(root,file))
                        c2+=1
    
    print(f"Total files: {c1+c2}")
    return fileList_less_30, fileList_greater_30

def traverse_non_esnet(path):
    fileList_less_30, fileList_greater_30 = [], []
    c1, c2 = 0, 0
    for root, directories, files in os.walk(path):
        for file in files:
            # Filtering non-es.net file names, and hosts with 1G
            if file.startswith("ss") and file.endswith("json") and file.find("es.net")==-1 and file.find("psl01.pic.es")==-1 and file.find("btw-bw.t1.grid.kiae.ru")==-1 and file.find("psb.hpc.utfsm.cl")==-1:
                    for enum, rtt_file in enumerate(rtt_data):
                        # Filtering rtt in ms
                        if file.find(rtt_file[0])!=-1 and float(rtt_file[1].split("ms")[0])<30:
                            fileList_less_30.append(os.path.join(root,file))
                            c1+=1
                        elif file.find(rtt_file[0])!=-1 and float(rtt_file[1].split("ms")[0])>=30:
                            fileList_greater_30.append(os.path.join(root,file))
                            c2+=1

    print(f"Total files: {c1+c2}")
    return fileList_less_30, fileList_greater_30


paths = [
    (
        "bost-dtn-10G/2021-08-02:23:01/pscheduler_bbr2_p1",
        "bost-dtn-10G/2021-08-02:23:01/pscheduler_bbr2_p16",
    ),
    (
        "bost-dtn-10G/2021-08-02:23:01/pscheduler_cubic_p1",
        "bost-dtn-10G/2021-08-02:23:01/pscheduler_cubic_p16",
    ),
    (
        "bost-dtn-10G/2021-07-30:19:21/pscheduler_both_p16",
        "bost-dtn-10G/2021-07-31:04:13/pscheduler_both_p16",
        "bost-dtn-10G/2021-07-31:14:26/pscheduler_both_p16",
        "bost-dtn-10G/2021-08-02:23:01/pscheduler_both_p16",
        "bost-dtn-10G/10G-to-ESnet/pscheduler_both_p16",
    )
]

'''Path traversal Demo'''
# pscheduler_bbr2  = os.path.join(rootdir, paths[2][4])
# filenames_less_30, filenames_greater_30 = traverse_esnet(pscheduler_bbr2)
# filenames_less_30, filenames_greater_30 = traverse_non_esnet(pscheduler_bbr2)


# ============================================================================
print("\n")
print(50*"=")
print("ESNET HOSTS")
print("RTT less than 30ms")
print(50*"=")
print("\n")
# ============================================================================

if not len(paths[0])==0:
    for q1 in paths[0]:
        print(f"=== {q1} ===")
        pscheduler_bbr2  = os.path.join(rootdir, q1)
        filenames_less_30, filenames_greater_30 = traverse_esnet(pscheduler_bbr2)

        data_seg = []
        tput_bbr2_p1 = []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_less_30):
            try:
                path = Path(f)
                data = [json.loads(line) for line in open(f, 'r')]

                bbr2_data_seg, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in enumerate(data):
                    if "interval" in d.keys() and j==0:
                        start_time = data[j]['interval']['time'] # Start time of the test
                    elif key2 in data[j].keys():
                        end_time = data[j-1]['interval']['time'] # End time of the test
                        mss = data[j-1]['interval']['mss'] # maximum segment size

                        bbr2_data_seg = data[j][key2]
                        data_seg.append( bbr2_data_seg )

                        throughput = (bbr2_data_seg*mss*8)/(end_time-start_time)/1e9
                        tput_bbr2_p1.append( throughput )

            except Exception as e:
                print(e)

        print("Throughput (P1)")
        print(f"BBRv2 - M, C.V : {np.mean(tput_bbr2_p1):.4f}, {(np.std(tput_bbr2_p1)/np.mean(tput_bbr2_p1)):.4f}")
        print("")

if not len(paths[1])==0:
    for q2 in paths[1]:
        print(f"=== {q2} ===")
        pscheduler_cubic = os.path.join(rootdir, q2)
        filenames_less_30, filenames_greater_30 = traverse_esnet(pscheduler_cubic)

        data_seg = []
        tput_p1_cubic = []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_less_30):
            try:
                path = Path(f)
                # The MongoDB JSON dump has one object per line, so this works for me.
                data = [json.loads(line) for line in open(f, 'r')]

                cubic_data_seg, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in enumerate(data):
                    if "interval" in d.keys() and j==0:
                        start_time = data[j]['interval']['time'] # Start time of the test
                    elif key1 in data[j].keys():
                        end_time = data[j-1]['interval']['time'] # End time of the test
                        mss = data[j-1]['interval']['mss'] # maximum segment size

                        cubic_data_seg = data[j][key1]
                        data_seg.append( cubic_data_seg )

                        throughput = (cubic_data_seg*mss*8)/(end_time-start_time)/1e9
                        tput_p1_cubic.append( throughput )

            except Exception as e:
                print(e)

        print("Throughput (P1)")
        print(f"CUBIC - M, C.V : {np.mean(tput_p1_cubic):.4f}, {(np.std(tput_p1_cubic)/np.mean(tput_p1_cubic)):.4f}")
        print("")

if not len(paths[2])==0:
    for q3 in paths[2]:
        print(f"=== {q3} ===")
        pscheduler_both = os.path.join(rootdir, q3)
        filenames_less_30, filenames_greater_30 = traverse_esnet(pscheduler_both)

        data_seg_sum_cubic, data_seg_sum_bbr2 = [], []
        tput_cubic_p16, tput_bbr2_p16 = [], []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_less_30):
            try:
                path = Path(f)
                data = [json.loads(line) for line in open(f, 'r')]

                throughput_cubic, throughput_bbr2, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in zip(range(len(data)),data):
                    try:
                        if "interval" in d.keys() and j==0:
                            start_time = d['interval']['time'] # Start time of the test
                        if "streams" in d.keys():
                            end_time = data[j-1]['interval']['time'] # End time of the test
                            mss = data[j-1]['interval']['mss'] # maximum segment size

                            cubic_data_seg_list, bbr2_data_seg_list = [], []
                            for j in range(len(d['streams'])):
                                if "cubic" in d['streams'][j]['cc']:
                                    cubic_data_seg_list.append(d['streams'][j]['data_segs'])
                                elif "bbr2" in d['streams'][j]['cc']:
                                    bbr2_data_seg_list.append(d['streams'][j]['data_segs'])

                            data_seg_sum_cubic.append( sum(cubic_data_seg_list) )
                            throughput_cubic = (sum(cubic_data_seg_list)*mss*8)/(end_time-start_time)/1e9
                            tput_cubic_p16.append( throughput_cubic )

                            data_seg_sum_bbr2.append( sum(bbr2_data_seg_list) )
                            throughput_bbr2 = (sum(bbr2_data_seg_list)*mss*8)/(end_time-start_time)/1e9
                            tput_bbr2_p16.append( throughput_bbr2 )

                    except Exception as e:
                        print(e)

            except Exception as e:
                print(e)

        print("Throughput (P16)")
        print(f"BBRv2 - M, C.V : {np.mean(tput_bbr2_p16):.4f}, {(np.std(tput_bbr2_p16)/np.mean(tput_bbr2_p16)):.4f}")
        print(f"CUBIC - M, C.V : {np.mean(tput_cubic_p16):.4f}, {(np.std(tput_cubic_p16)/np.mean(tput_cubic_p16)):.4f}")
        print("")


# ============================================================================
print("\n")
print(50*"=")
print("ESNET HOSTS")
print("RTT greater than 30ms")
print(50*"=")
print("\n")
# ============================================================================

if not len(paths[0])==0:
    for q1 in paths[0]:
        print(f"=== {q1} ===")
        pscheduler_bbr2  = os.path.join(rootdir, q1)
        filenames_less_30, filenames_greater_30 = traverse_esnet(pscheduler_bbr2)

        data_seg = []
        tput_bbr2_p1 = []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_greater_30):
            try:
                path = Path(f)
                data = [json.loads(line) for line in open(f, 'r')]

                bbr2_data_seg, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in enumerate(data):
                    if "interval" in d.keys() and j==0:
                        start_time = data[j]['interval']['time'] # Start time of the test
                    elif key2 in data[j].keys():
                        end_time = data[j-1]['interval']['time'] # End time of the test
                        mss = data[j-1]['interval']['mss'] # maximum segment size

                        bbr2_data_seg = data[j][key2]
                        data_seg.append( bbr2_data_seg )

                        throughput = (bbr2_data_seg*mss*8)/(end_time-start_time)/1e9
                        tput_bbr2_p1.append( throughput )

            except Exception as e:
                print(e)

        print("Throughput (P1)")
        print(f"BBRv2 - M, C.V : {np.mean(tput_bbr2_p1):.4f}, {(np.std(tput_bbr2_p1)/np.mean(tput_bbr2_p1)):.4f}")
        print("")

if not len(paths[1])==0:
    for q2 in paths[1]:
        print(f"=== {q2} ===")
        pscheduler_cubic = os.path.join(rootdir, q2)
        filenames_less_30, filenames_greater_30 = traverse_esnet(pscheduler_cubic)

        data_seg = []
        tput_p1_cubic = []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_greater_30):
            try:
                path = Path(f)
                # The MongoDB JSON dump has one object per line, so this works for me.
                data = [json.loads(line) for line in open(f, 'r')]

                cubic_data_seg, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in enumerate(data):
                    if "interval" in d.keys() and j==0:
                        start_time = data[j]['interval']['time'] # Start time of the test
                    elif key1 in data[j].keys():
                        end_time = data[j-1]['interval']['time'] # End time of the test
                        mss = data[j-1]['interval']['mss'] # maximum segment size

                        cubic_data_seg = data[j][key1]
                        data_seg.append( cubic_data_seg )

                        throughput = (cubic_data_seg*mss*8)/(end_time-start_time)/1e9
                        tput_p1_cubic.append( throughput )

            except Exception as e:
                print(e)

        print("Throughput (P1)")
        print(f"CUBIC - M, C.V : {np.mean(tput_p1_cubic):.4f}, {(np.std(tput_p1_cubic)/np.mean(tput_p1_cubic)):.4f}")
        print("")

if not len(paths[2])==0:
    for q3 in paths[2]:
        print(f"=== {q3} ===")
        pscheduler_both = os.path.join(rootdir, q3)
        filenames_less_30, filenames_greater_30 = traverse_esnet(pscheduler_both)

        data_seg_sum_cubic, data_seg_sum_bbr2 = [], []
        tput_cubic_p16, tput_bbr2_p16 = [], []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_greater_30):
            try:
                path = Path(f)
                data = [json.loads(line) for line in open(f, 'r')]

                throughput_cubic, throughput_bbr2, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in zip(range(len(data)),data):
                    try:
                        if "interval" in d.keys() and j==0:
                            start_time = d['interval']['time'] # Start time of the test
                        if "streams" in d.keys():
                            end_time = data[j-1]['interval']['time'] # End time of the test
                            mss = data[j-1]['interval']['mss'] # maximum segment size

                            cubic_data_seg_list, bbr2_data_seg_list = [], []
                            for j in range(len(d['streams'])):
                                if "cubic" in d['streams'][j]['cc']:
                                    cubic_data_seg_list.append(d['streams'][j]['data_segs'])
                                elif "bbr2" in d['streams'][j]['cc']:
                                    bbr2_data_seg_list.append(d['streams'][j]['data_segs'])

                            data_seg_sum_cubic.append( sum(cubic_data_seg_list) )
                            throughput_cubic = (sum(cubic_data_seg_list)*mss*8)/(end_time-start_time)/1e9
                            tput_cubic_p16.append( throughput_cubic )

                            data_seg_sum_bbr2.append( sum(bbr2_data_seg_list) )
                            throughput_bbr2 = (sum(bbr2_data_seg_list)*mss*8)/(end_time-start_time)/1e9
                            tput_bbr2_p16.append( throughput_bbr2 )

                    except Exception as e:
                        print(e)

            except Exception as e:
                print(e)

        print("Throughput (P16)")
        print(f"BBRv2 - M, C.V : {np.mean(tput_bbr2_p16):.4f}, {(np.std(tput_bbr2_p16)/np.mean(tput_bbr2_p16)):.4f}")
        print(f"CUBIC - M, C.V : {np.mean(tput_cubic_p16):.4f}, {(np.std(tput_cubic_p16)/np.mean(tput_cubic_p16)):.4f}")
        print("")

# ============================================================================
print("\n")
print(50*"=")
print("NON-ESNET HOSTS")
print("RTT less than 30ms")
print(50*"=")
# ============================================================================

print("\n")
if not len(paths[0])==0:
    for q1 in paths[0]:
        print(f"=== {q1} ===")
        pscheduler_bbr2  = os.path.join(rootdir, q1)
        filenames_less_30, filenames_greater_30 = traverse_non_esnet(pscheduler_bbr2)

        data_seg = []
        tput_bbr2_p1 = []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_less_30):
            try:
                path = Path(f)
                data = [json.loads(line) for line in open(f, 'r')]

                bbr2_data_seg, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in enumerate(data):
                    if "interval" in d.keys() and j==0:
                        start_time = data[j]['interval']['time'] # Start time of the test
                    elif key2 in data[j].keys():
                        end_time = data[j-1]['interval']['time'] # End time of the test
                        mss = data[j-1]['interval']['mss'] # maximum segment size

                        bbr2_data_seg = data[j][key2]
                        data_seg.append( bbr2_data_seg )

                        throughput = (bbr2_data_seg*mss*8)/(end_time-start_time)/1e9
                        tput_bbr2_p1.append( throughput )

            except Exception as e:
                print(e)

        print("Throughput (P1)")
        print(f"BBRv2 - M, C.V : {np.mean(tput_bbr2_p1):.4f}, {(np.std(tput_bbr2_p1)/np.mean(tput_bbr2_p1)):.4f}")
        print("")

if not len(paths[1])==0:
    for q2 in paths[1]:
        print(f"=== {q2} ===")
        pscheduler_cubic = os.path.join(rootdir, q2)
        filenames_less_30, filenames_greater_30 = traverse_non_esnet(pscheduler_cubic)

        data_seg = []
        tput_p1_cubic = []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_less_30):
            try:
                path = Path(f)
                # The MongoDB JSON dump has one object per line, so this works for me.
                data = [json.loads(line) for line in open(f, 'r')]

                cubic_data_seg, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in enumerate(data):
                    if "interval" in d.keys() and j==0:
                        start_time = data[j]['interval']['time'] # Start time of the test
                    elif key1 in data[j].keys():
                        end_time = data[j-1]['interval']['time'] # End time of the test
                        mss = data[j-1]['interval']['mss'] # maximum segment size

                        cubic_data_seg = data[j][key1]
                        data_seg.append( cubic_data_seg )

                        throughput = (cubic_data_seg*mss*8)/(end_time-start_time)/1e9
                        tput_p1_cubic.append( throughput )

            except Exception as e:
                print(e)

        print("Throughput (P1)")
        print(f"CUBIC - M, C.V : {np.mean(tput_p1_cubic):.4f}, {(np.std(tput_p1_cubic)/np.mean(tput_p1_cubic)):.4f}")
        print("")

if not len(paths[2])==0:
    for q3 in paths[2]:
        print(f"=== {q3} ===")
        pscheduler_both = os.path.join(rootdir, q3)
        filenames_less_30, filenames_greater_30 = traverse_non_esnet(pscheduler_both)

        data_seg_sum_cubic, data_seg_sum_bbr2 = [], []
        tput_cubic_p16, tput_bbr2_p16 = [], []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_less_30):
            try:
                path = Path(f)
                data = [json.loads(line) for line in open(f, 'r')]

                throughput_cubic, throughput_bbr2, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in zip(range(len(data)),data):
                    try:
                        if "interval" in d.keys() and j==0:
                            start_time = d['interval']['time'] # Start time of the test
                        if "streams" in d.keys():
                            end_time = data[j-1]['interval']['time'] # End time of the test
                            mss = data[j-1]['interval']['mss'] # maximum segment size

                            cubic_data_seg_list, bbr2_data_seg_list = [], []
                            for j in range(len(d['streams'])):
                                if "cubic" in d['streams'][j]['cc']:
                                    cubic_data_seg_list.append(d['streams'][j]['data_segs'])
                                elif "bbr2" in d['streams'][j]['cc']:
                                    bbr2_data_seg_list.append(d['streams'][j]['data_segs'])

                            data_seg_sum_cubic.append( sum(cubic_data_seg_list) )
                            throughput_cubic = (sum(cubic_data_seg_list)*mss*8)/(end_time-start_time)/1e9
                            tput_cubic_p16.append( throughput_cubic )

                            data_seg_sum_bbr2.append( sum(bbr2_data_seg_list) )
                            throughput_bbr2 = (sum(bbr2_data_seg_list)*mss*8)/(end_time-start_time)/1e9
                            tput_bbr2_p16.append( throughput_bbr2 )

                    except Exception as e:
                        print(e)

            except Exception as e:
                print(e)

        print("Throughput (P16)")
        print(f"BBRv2 - M, C.V : {np.mean(tput_bbr2_p16):.4f}, {(np.std(tput_bbr2_p16)/np.mean(tput_bbr2_p16)):.4f}")
        print(f"CUBIC - M, C.V : {np.mean(tput_cubic_p16):.4f}, {(np.std(tput_cubic_p16)/np.mean(tput_cubic_p16)):.4f}")
        print("")

# ============================================================================
print("\n")
print(50*"=")
print("NON-ESNET HOSTS")
print("RTT greater than 30ms")
print(50*"=")
# ============================================================================

print("\n")
if not len(paths[0])==0:
    for q1 in paths[0]:
        print(f"=== {q1} ===")
        pscheduler_bbr2  = os.path.join(rootdir, q1)
        filenames_less_30, filenames_greater_30 = traverse_non_esnet(pscheduler_bbr2)

        data_seg = []
        tput_bbr2_p1 = []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_greater_30):
            try:
                path = Path(f)
                data = [json.loads(line) for line in open(f, 'r')]

                bbr2_data_seg, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in enumerate(data):
                    if "interval" in d.keys() and j==0:
                        start_time = data[j]['interval']['time'] # Start time of the test
                    elif key2 in data[j].keys():
                        end_time = data[j-1]['interval']['time'] # End time of the test
                        mss = data[j-1]['interval']['mss'] # maximum segment size

                        bbr2_data_seg = data[j][key2]
                        data_seg.append( bbr2_data_seg )

                        throughput = (bbr2_data_seg*mss*8)/(end_time-start_time)/1e9
                        tput_bbr2_p1.append( throughput )

            except Exception as e:
                print(e)

        print("Throughput (P1)")
        print(f"BBRv2 - M, C.V : {np.mean(tput_bbr2_p1):.4f}, {(np.std(tput_bbr2_p1)/np.mean(tput_bbr2_p1)):.4f}")
        print("")

if not len(paths[1])==0:
    for q2 in paths[1]:
        print(f"=== {q2} ===")
        pscheduler_cubic = os.path.join(rootdir, q2)
        filenames_less_30, filenames_greater_30 = traverse_non_esnet(pscheduler_cubic)

        data_seg = []
        tput_p1_cubic = []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_greater_30):
            try:
                path = Path(f)
                # The MongoDB JSON dump has one object per line, so this works for me.
                data = [json.loads(line) for line in open(f, 'r')]

                cubic_data_seg, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in enumerate(data):
                    if "interval" in d.keys() and j==0:
                        start_time = data[j]['interval']['time'] # Start time of the test
                    elif key1 in data[j].keys():
                        end_time = data[j-1]['interval']['time'] # End time of the test
                        mss = data[j-1]['interval']['mss'] # maximum segment size

                        cubic_data_seg = data[j][key1]
                        data_seg.append( cubic_data_seg )

                        throughput = (cubic_data_seg*mss*8)/(end_time-start_time)/1e9
                        tput_p1_cubic.append( throughput )

            except Exception as e:
                print(e)

        print("Throughput (P1)")
        print(f"CUBIC - M, C.V : {np.mean(tput_p1_cubic):.4f}, {(np.std(tput_p1_cubic)/np.mean(tput_p1_cubic)):.4f}")
        print("")

if not len(paths[2])==0:
    for q3 in paths[2]:
        print(f"=== {q3} ===")
        pscheduler_both = os.path.join(rootdir, q3)
        filenames_less_30, filenames_greater_30 = traverse_non_esnet(pscheduler_both)

        data_seg_sum_cubic, data_seg_sum_bbr2 = [], []
        tput_cubic_p16, tput_bbr2_p16 = [], []
        key1, key2, key3 = 'cubic_data_segs', 'bbr2_data_segs', 'bbr_data_segs'

        for i,f in enumerate(filenames_greater_30):
            try:
                path = Path(f)
                data = [json.loads(line) for line in open(f, 'r')]

                throughput_cubic, throughput_bbr2, throughput, mss, start_time, end_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for j,d in zip(range(len(data)),data):
                    try:
                        if "interval" in d.keys() and j==0:
                            start_time = d['interval']['time'] # Start time of the test
                        if "streams" in d.keys():
                            end_time = data[j-1]['interval']['time'] # End time of the test
                            mss = data[j-1]['interval']['mss'] # maximum segment size

                            cubic_data_seg_list, bbr2_data_seg_list = [], []
                            for j in range(len(d['streams'])):
                                if "cubic" in d['streams'][j]['cc']:
                                    cubic_data_seg_list.append(d['streams'][j]['data_segs'])
                                elif "bbr2" in d['streams'][j]['cc']:
                                    bbr2_data_seg_list.append(d['streams'][j]['data_segs'])

                            data_seg_sum_cubic.append( sum(cubic_data_seg_list) )
                            throughput_cubic = (sum(cubic_data_seg_list)*mss*8)/(end_time-start_time)/1e9
                            tput_cubic_p16.append( throughput_cubic )

                            data_seg_sum_bbr2.append( sum(bbr2_data_seg_list) )
                            throughput_bbr2 = (sum(bbr2_data_seg_list)*mss*8)/(end_time-start_time)/1e9
                            tput_bbr2_p16.append( throughput_bbr2 )

                    except Exception as e:
                        print(e)

            except Exception as e:
                print(e)

        print("Throughput (P16)")
        print(f"BBRv2 - M, C.V : {np.mean(tput_bbr2_p16):.4f}, {(np.std(tput_bbr2_p16)/np.mean(tput_bbr2_p16)):.4f}")
        print(f"CUBIC - M, C.V : {np.mean(tput_cubic_p16):.4f}, {(np.std(tput_cubic_p16)/np.mean(tput_cubic_p16)):.4f}")
        print("")