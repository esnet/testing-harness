from __future__ import absolute_import, print_function

# --- System ---
import os
import sys

# --- Utility ---
import pandas as pd
import numpy as np
import math
import random
import warnings
import datetime, time
import argparse
import pathlib
warnings.filterwarnings('ignore')

# --- Plot --
import seaborn as sns
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Testpoint Statistics')
parser.add_argument('-P', '--path', default="data/statistics-5.csv", type=str,
                    help='Enter the complete path of the CSV file')
parser.add_argument('-O', '--output', default="output", type=str,
                    help='Enter the output folder path for plotting')                    
args = parser.parse_args()
print("")

dataPath = args.path
print(f"File {args.path} loaded for plotting!\n")

path = pathlib.Path(args.output)
path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(dataPath)
# columnList = df.columns

# Dropping columns that are not required at the moment
df = df.drop(columns=[ 'Unnamed: 0', 'UUID', 'HOSTNAME', #'ALIAS',
                       'THROUGHPUT (Receiver)', 'LATENCY (min.)', 'LATENCY (max.)', 
                       'CONGESTION (Receiver)', 'BYTES (Receiver)'
                     ])

# Removing `gbit` string from the pacing number for analysis
pacing = df['PACING'].values
for i, p in enumerate(pacing):
    v, _ = p.split("gbit")
    pacing[i] = float(v)
df['PACING'] = pacing

# Spliting the congestion type into binary format
df['TYPE'] = (df['CONGESTION (Sender)'] == 'cubic').astype(int) # Cubic = 1 & BBRV2 = 0

# Joining congestion type and stream to create unique type for plotting x-axis
df['TYPE-STREAM'] = df['CONGESTION (Sender)'].str[:] + "_" + df['STREAMS'].map(str)

# Converting human-readable timestamps into UNIX time format
unixtime = []
for i in range( len(df['TIMESTAMP']) ):
    
    unixtime.append(datetime.datetime.strptime(df['TIMESTAMP'][i], '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())

df['UNIX'] = unixtime
# df.head(5)

joblist = df['TYPE-STREAM'].unique()
hostlist = df['ALIAS'].unique()
pacingList = df['PACING'].unique()

print("Job list: ", joblist)
print("Host list: ", hostlist)
print("Pacing list: ", pacingList)

print("Plotting all type-stream vs throughput values")
for host in hostlist:

    print("Plotting for ", str(host))
    cubic, bbrv2 = [], []

    for p in pacingList: # range(1, 11):
        cubic.append( df.loc[ ( df['TYPE-STREAM'].str.startswith('cubic') ) & (df['ALIAS']==host) & (df['PACING']==p) ] )
        bbrv2.append( df.loc[ ( df['TYPE-STREAM'].str.startswith('bbr2') )  & (df['ALIAS']==host) & (df['PACING']==p) ] )

    fig, axs = plt.subplots(2, 5, figsize = (28,8), constrained_layout=True)

    for ax, idx in zip(axs.flat, range(0, 10)):
        ax.plot (cubic[idx]['TYPE-STREAM'], cubic[idx]['THROUGHPUT (Sender)'], 'o-', color='blue', label='cubic')
        ax.plot (bbrv2[idx]['TYPE-STREAM'], bbrv2[idx]['THROUGHPUT (Sender)'], 'o-', color='green', label='bbrv2')
        ax.set_title(str(host)+", Pacing "+str(idx+1))
        ax.set(xlabel='congestion type', ylabel='throughput')
        ax.grid(True)
        ax.legend()
    plt.savefig("output/"+str(host)+".png", dpi=200)
    plt.show()


print("Plotting MAX for the type-stream vs throughput values")
for host in hostlist:

    print("Plotting for ", str(host))

    cubic, bbrv2 = [], []
    cubic_types = ['cubic_1', 'cubic_4', 'cubic_8', 'cubic_16']
    bbr2_types = ['bbr2_1', 'bbr2_4', 'bbr2_8', 'bbr2_16']

    for p in pacingList:
        cubic_1  = df.loc[ ( df['TYPE-STREAM']=='cubic_1')  & (df['ALIAS']==host) & (df['PACING']==p) ]
        cubic_4  = df.loc[ ( df['TYPE-STREAM']=='cubic_4')  & (df['ALIAS']==host) & (df['PACING']==p) ]
        cubic_8  = df.loc[ ( df['TYPE-STREAM']=='cubic_8')  & (df['ALIAS']==host) & (df['PACING']==p) ]
        cubic_16 = df.loc[ ( df['TYPE-STREAM']=='cubic_16') & (df['ALIAS']==host) & (df['PACING']==p) ]
        
        try:
            tput_cubic = [ max(cubic_1['THROUGHPUT (Sender)']), max(cubic_4['THROUGHPUT (Sender)']), max(cubic_8['THROUGHPUT (Sender)']), max(cubic_16['THROUGHPUT (Sender)'])]    
        except Exception as e:
            tput_cubic = [0, 0, 0, 0]
            # print(e)
            # pass
        cubic.append(tput_cubic)

        bbr_1  = df.loc[ ( df['TYPE-STREAM']=='bbr2_1')  & (df['ALIAS']==host) & (df['PACING']==p) ]
        bbr_4  = df.loc[ ( df['TYPE-STREAM']=='bbr2_4')  & (df['ALIAS']==host) & (df['PACING']==p) ]
        bbr_8  = df.loc[ ( df['TYPE-STREAM']=='bbr2_8')  & (df['ALIAS']==host) & (df['PACING']==p) ]
        bbr_16 = df.loc[ ( df['TYPE-STREAM']=='bbr2_16') & (df['ALIAS']==host) & (df['PACING']==p) ]

        try:
            tput_bbr = [ max(bbr_1['THROUGHPUT (Sender)']), max(bbr_4['THROUGHPUT (Sender)']), max(bbr_8['THROUGHPUT (Sender)']), max(bbr_16['THROUGHPUT (Sender)'])]
        except Exception as e:
            tput_bbr = [0, 0, 0, 0]
            # print(e)
        bbrv2.append(tput_bbr)

    fig = plt.figure(figsize=(17, 11))

    plt.title (str(host), fontsize=20)
    for idx, p in zip(range(len(cubic)), pacingList):
        plt.plot (cubic_types, cubic[idx], 'o-', label='cubic, pacing:'+str(p))
        plt.plot (bbr2_types, bbrv2[idx],  '*-', label='bbrv2, pacing:'+str(p))
    
    axes = plt.gca()
    
    plt.xlabel('Congestion Type', fontsize=18)
    plt.xticks(fontsize=14)

    axes.set_ylim([-0.2e10,1.2e10])
    plt.yticks(fontsize=14)
    plt.ylabel('Throughput (Gbps/sec)', fontsize=18)
    
    plt.minorticks_on()
    plt.grid (b=True, which='major', linestyle='-', linewidth=1.5)
    plt.grid (b=True, which='minor', linestyle='--', linewidth=1)
    plt.legend (loc=1, fontsize=6)
    plt.savefig("output/"+str(host)+"(max).png", dpi=200)
    plt.show()
