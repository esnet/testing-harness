'''
**
** Project Lead: Eashan Adhikarla
** Mentor: Ezra Kissel
**
** Date Created: June 17' 2021
** Last Modified: June 29' 2021
**
'''

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

import os
import pandas as pd
import requests
import json
import argparse
import logging
import errno
import re, datetime


try:
    os.makedirs('data')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# logging.basicConfig(filename='data/iperf3.log', level=logging.DEBUG)

# Create the elasticsearch client
HOST = 'nersc-tbn-6.testbed100.es.net'
PORT = 9200
TIMEOUT = 30
MAX_TRIES = 10
VERIFY_CERTS = True
RETRY = True

es = Elasticsearch( host=HOST, port=PORT,
                    timeout=TIMEOUT,
                    max_retries=MAX_TRIES,
                    verify_certs=VERIFY_CERTS,
                    retry_on_timeout=RETRY
                  )

if not es.ping():
    raise ValueError("Connection failed")
    # logging.error("Connection failed\n")
else:
    print("Connection successful!")
    # logging.info("Connection successful!\n")


class clr:
    """
    Defining colors for the print syntax coloring
    """
    H   = '\033[35m' # Header
    B   = '\033[94m' # Blue
    G   = '\033[36m' # Green
    W   = '\033[93m' # Warning
    F   = '\033[91m' # Fail
    E   = '\033[0m'  # End
    BD  = '\033[1m'  # Bold
    UL  = '\033[4m'  # Underline


class GETTER:
    """
    Get class to get current information about different Indexs
    """
    def __init__(self, term, sum=0):
        self.term = term
        self.sum = sum

    def getIndexList(self, term):
        idx=[]
        indices_dict = es.indices.get_alias(self.term)
        if isinstance(indices_dict, dict) and indices_dict is not None:
            print(f"'{clr.G}{len(indices_dict)}{clr.E}' indexes found!")
            # logging.info(f"'{clr.G}{len(indices_dict)}{clr.E}' indexes found!")
            for k,v in indices_dict.items():
                idx.append(k)
            return idx
        else:
            print (f"{clr.F}Empty dict!{clr.E}")
            # logging.info(f"{clr.F}Empty dict!{clr.E}")

    def getIndexDetails(self, iperf3, jobmeta, column_list, total_docs=1000):
        df = pd.DataFrame(columns=column_list)

        for i in range(len(jobmeta)):
            try:
                # Given a index name, finds all the iperf3_documents in the index
                # index: Index name as a string
                # body: Empty equivalent to '*' to get all the values
                # size: To some random max value for getting all the iperf3_documents
                jobmeta_result = es.search(index=jobmeta[i],
                                body={"query":{"match_all":{}}},
                                size=total_docs,
                                )
                # print(f"\n{jobmeta[i]} ---> {clr.G}{jobmeta_result['hits']['total']['value']}{clr.E} documents\n")
                # logging.info(f"\n{jobmeta[i]} ---> {clr.G}{jobmeta_result['hits']['total']['value']}{clr.E} documents\n")
                jobmeta_documents = [doc for doc in jobmeta_result['hits']['hits']]

                for j in range(len(iperf3)):
                    iperf3_result = es.search(index=iperf3[j],
                                        body={"query":{"match_all":{}}},
                                        size=total_docs,
                                        )
                    # print(f"\n{iperf3[i]} ---> {clr.G}{iperf3_result['hits']['total']['value']}{clr.E} documents\n")
                    # logging.info(f"\n{iperf3[i]} ---> {clr.G}{iperf3_result['hits']['total']['value']}{clr.E} documents\n")
                    iperf3_documents = [doc for doc in iperf3_result['hits']['hits']]

                    for iperfdoc in range(len(iperf3_documents)):
                        # ---------------------
                        # For each job/document
                        # ---------------------
                        # Format: dict_keys(['end', '@version', 'start', 'uuid', 'intervals', '@timestamp'])
                        # uuid
                        uuid = iperf3_documents[iperfdoc]['_source']['uuid']
                        for jobmetadoc in range(len(jobmeta_documents)):
                            try:
                                iter_uuid = jobmeta_documents[jobmetadoc]['_source']['iter_uuids']
                            except:
                                pass
                            if uuid in iter_uuid:
                                # Hostname
                                hostname = jobmeta_documents[jobmetadoc]['_source']['hostname']

                                # @timestamp
                                timestamp = iperf3_documents[iperfdoc]['_source']['@timestamp']
                    
                                # start (Format: dict_keys(['cookie', 'test_start', 'tcp_mss_default', 'version', 'connected', 'sndbuf_actual',
                                #                           'rcvbuf_actual', 'sock_bufsize', 'system_info', 'timestamp', 'connecting_to']))
                                start_dict = iperf3_documents[iperfdoc]['_source']['start']
                                num_streams = start_dict['test_start']['num_streams']

                                # Congestion Type
                                sender_congestion = iperf3_documents[iperfdoc]['_source']['end']['sender_tcp_congestion']
                                receiver_congestion = iperf3_documents[iperfdoc]['_source']['end']['receiver_tcp_congestion']

                                # ---------------------
                                # For each stream/flow
                                # ---------------------
                                # Intervals
                                intervals_list = iperf3_documents[iperfdoc]['_source']['intervals']
                                for itrv in range(len(intervals_list)):
                                    for streams in range(num_streams):
                                        # Format: dict_keys(['receiver', 'sender'])
                                        # 'sender': dict_keys(['retransmits', 'max_rtt', 'sender', 'start', 'bytes', 'mean_rtt', 'end'
                                        #                      'max_snd_cwnd', 'bits_per_second', 'socket', 'seconds', 'min_rtt'])
                                        # 'receiver': dict_keys(['end', 'bits_per_second', 'sender', 'start', 'socket', 'seconds', 'bytes'])

                                        sender_throughput = iperf3_documents[iperfdoc]['_source']['end']['streams'][streams]['sender']['bits_per_second']
                                        sender_min_rtt = iperf3_documents[iperfdoc]['_source']['end']['streams'][streams]['sender']['min_rtt']/2
                                        sender_max_rtt = iperf3_documents[iperfdoc]['_source']['end']['streams'][streams]['sender']['max_rtt']/2
                                        sender_mean_rtt = iperf3_documents[iperfdoc]['_source']['end']['streams'][streams]['sender']['mean_rtt']/2
                                        sender_retransmits = iperf3_documents[iperfdoc]['_source']['end']['streams'][streams]['sender']['retransmits']
                                        receiver_throughput = iperf3_documents[iperfdoc]['_source']['end']['streams'][streams]['receiver']['bits_per_second']

                                        # print(f"uuid: {uuid}\nhostname: {hostname}\ntimestamp: {timestamp}\nnum_streams: {num_streams}\nsender_throughput: {sender_throughput}\nreceiver_throughput: {receiver_throughput}\nsender_min_rtt: {sender_min_rtt}\nsender_max_rtt: {sender_max_rtt}\nsender_mean_rtt: {sender_mean_rtt}\nsender_congestion: {sender_congestion}\nreceiver_congestion: {receiver_congestion}\nsender_retransmits: {sender_retransmits}\n\n")
                                        # logging.info(f"uuid: {uuid}\nhostname: {hostname}\ntimestamp: {timestamp}\nnum_streams: {num_streams}\nsender_throughput: {sender_throughput}\nreceiver_throughput: {receiver_throughput}\nsender_min_rtt: {sender_min_rtt}\nsender_max_rtt: {sender_max_rtt}\nsender_mean_rtt: {sender_mean_rtt}\nsender_congestion: {sender_congestion}\nreceiver_congestion: {receiver_congestion}\nsender_retransmits: {sender_retransmits}\n\n")

                                        df = df.append({'UUID':uuid,
                                                        'HOSTNAME':hostname,
                                                        'TIMESTAMP':timestamp,
                                                        'STREAMS':num_streams,
                                                        'THROUGHPUT (Sender)':sender_throughput,
                                                        'THROUGHPUT (Receiver)':receiver_throughput,
                                                        'LATENCY (min.)':sender_min_rtt,
                                                        'LATENCY (max.)':sender_max_rtt,
                                                        'LATENCY (mean)':sender_mean_rtt,
                                                        'CONGESTION (Sender)':sender_congestion,
                                                        'CONGESTION (Receiver)':receiver_congestion,
                                                        'RETRANSMITS':sender_retransmits,
                                                        }, ignore_index=True)                                    
            except:
                pass
        return df


def main(verbose=False):
    print("\nStarting ELK testpoint stats retrieval...")
    # logging.info("\nStarting ELK testpoint stats retrieval...")

    parser = argparse.ArgumentParser(description='Testpoint Statistics')
    parser.add_argument('-t', '--term', default="*", type=str,
                        help='The search term to find the indexes {"*", "iperf3*", "jobmeta*", "bbrmon*"}')
    args = parser.parse_args()

    print(f"Chosen index type: {args.term}")
    # logging.info(f"Chosen index type: {args.term}")
    get = GETTER(args.term)

    # STEP 1. Get all the indices in the ELK given a term.
    indexes = get.getIndexList(args.term)

    if verbose:
        for e,i in enumerate(indexes):
            print(f"{e}: {i}")
            # logging.info(f"{e}: {i}")

    # Filtering indexes by type
    sorted_index = sorted(indexes)
    iperf3 = [i for i in sorted_index if i.startswith("iperf3")]
    jobmeta = [i for i in sorted_index if i.startswith("jobmeta")]
    ss = [i for i in sorted_index if i.startswith("ss")]
    bbrmon_pscheduler = [i for i in sorted_index if i.startswith("bbrmon-pscheduler")]
    bbrmon_jobmeta = [i for i in sorted_index if i.startswith("bbrmon-jobmeta")]
    bbrmon_ss = [i for i in sorted_index if i.startswith("bbrmon-ss")]
    bbrmon_tcptrace = [i for i in sorted_index if i.startswith("bbrmon-tcptrace")]
    miscellaneous = [i for i in sorted_index if i not in (iperf3+jobmeta+ss+bbrmon_jobmeta+bbrmon_pscheduler+bbrmon_ss+bbrmon_tcptrace)]
    
    flag = len(iperf3+jobmeta+ss+bbrmon_jobmeta+bbrmon_pscheduler+bbrmon_ss+bbrmon_tcptrace+miscellaneous) == len(sorted_index)
    if not flag:
        print("New indexes got added, missing in the working list!")
        # logging.info("New indexes got added, missing in the working list!")
        missing = sorted_index - (iperf3+jobmeta+ss+bbrmon_jobmeta+bbrmon_pscheduler+bbrmon_ss+bbrmon_tcptrace+miscellaneous)
        for e,i in enumerate(indexes):
            print(f"{e}: {i}")
            # logging.info(f"{e}: {i}")

    pandas_column_list = [
                        'UUID',
                        'HOSTNAME',
                        'TIMESTAMP',
                        'STREAMS',
                        'THROUGHPUT (Sender)', 'THROUGHPUT (Receiver)',
                        'LATENCY (min.)', 'LATENCY (max.)', 'LATENCY (mean)',
                        'CONGESTION (Sender)', 'CONGESTION (Receiver)',
                        'RETRANSMITS',
                        ]

    # STEP 2. getIndexDetails to retrieve the statistics of every testpoint
    # and every stream/flow wrt index
    index_response = get.getIndexDetails(iperf3, jobmeta, pandas_column_list)

    # STEP 3. Create a Pandas Dataframe to make it easier for the model to read.
    if args.term=="*":
        filename = "all"
    else:
        filename = args.term[:-1]

    try:
        index_response.to_csv('data/'+str(filename)+'.csv')
        print(f"{str(filename)}.csv file written!")
    except:
        raise ValueError("Cannot write the file")


if __name__ == "__main__":
    main()

