"""
BUFFER TESTING
"""

import os
import json
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

rootdir = "/Users/eashan22/Desktop/Internship 2021/bbrv2/Brian's Project/buffer-testing"

def traverse(path):
    fileList = []
    c = 0
    for root, directories, files in os.walk(path):
        for file in files:
            if file.endswith("json"):
                fileList.append(os.path.join(root,file))
                c+=1
    print(f"Total files: {c}")
    return fileList

path = [
    "corsa.8MB.loop/pscheduler_both_p16", 
    "corsa.9MB.loop/pscheduler_both_p16",
    "corsa.12MB.loop/pscheduler_both_p16",
    "corsa.16MB.loop/pscheduler_both_p16",
    "corsa.24MB.loop/pscheduler_both_p16",
    "corsa.32MB.loop.1/pscheduler_both_p16",
    "corsa.32MB.loop.2/pscheduler_both_p16",
    "corsa.64MB.loop/pscheduler_both_p16",
    "corsa.100MB.loop/pscheduler_both_p16",
    "corsa.100MB.loop.2/pscheduler_both_p16"
]

for p in path:
    print(f"\n===================\n{p}\n===================")
    pscheduler_both_p16 = os.path.join(rootdir, p)
    filenames = traverse(pscheduler_both_p16)
    print()

    data_seg_sum_cubic, data_seg_sum_bbr2 = [], []
    tput_cubic_p16, tput_bbr2_p16 = [], []
    key1, key2 = 'cubic_data_segs', 'bbr2_data_segs'

    for i,f in enumerate(filenames):
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
                        throughput_cubic = ((sum(cubic_data_seg_list)*mss*8)/(end_time-start_time))/1e9
                        tput_cubic_p16.append( throughput_cubic )

                        data_seg_sum_bbr2.append( sum(bbr2_data_seg_list) )
                        throughput_bbr2 = ((sum(bbr2_data_seg_list)*mss*8)/(end_time-start_time))/1e9
                        tput_bbr2_p16.append( throughput_bbr2 )

                except Exception as e:
                    print(e)

        except Exception as e:
            print(e)

    print("Throughput")
    print(f"BBRv2 - Mean: {np.mean(tput_bbr2_p16):.5f} | Std. dev.: {np.std(tput_bbr2_p16):.5f} |  Coef. of Variance: {(np.std(tput_bbr2_p16)/np.mean(tput_bbr2_p16)):.5f}  |  Variance: {np.var(tput_bbr2_p16):.5f}")
    print(f"CUBIC - Mean: {np.mean(tput_cubic_p16):.5f} | Std. dev.: {np.std(tput_cubic_p16):.5f} |  Coef. of Variance: {(np.std(tput_cubic_p16)/np.mean(tput_cubic_p16)):.5f}  |  Variance: {np.var(tput_cubic_p16):.5f}")

    print("Data Segment")
    print(f"BBRv2 - Mean: {np.mean(data_seg_sum_bbr2):.5f}   |  Std. Dev.: {np.std(data_seg_sum_bbr2):.5f}  |  Coef. of Variance: {(np.std(data_seg_sum_bbr2)/np.mean(data_seg_sum_bbr2)):.5f}  |  Variance: {np.var(data_seg_sum_bbr2):.5f}")
    print(f"CUBIC - Mean: {np.mean(data_seg_sum_cubic):.5f}  |  Std. Dev.: {np.std(data_seg_sum_cubic):.5f}  |  Coef. of Variance: {(np.std(data_seg_sum_cubic)/np.mean(data_seg_sum_cubic)):.5f}  |  Variance: {np.var(data_seg_sum_cubic):.5f}")
