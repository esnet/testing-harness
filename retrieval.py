'''
**
** Project Lead: Eashan Adhikarla
** Mentor: Ezra Kissel
**
** Date Created: June 17' 2021
** Last Modified: June 24' 2021
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


try:
    os.makedirs('data')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

logging.basicConfig(filename='data/iperf3.log', level=logging.DEBUG)

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
    logging.error("Connection failed")
else:
    print("Connection successful!")
    logging.info("Connection successful!")


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
            logging.info(f"'{clr.G}{len(indices_dict)}{clr.E}' indexes found!")
            for k,v in indices_dict.items():
                idx.append(k)
            return idx
        else:
            print (f"{clr.F}Empty dict!{clr.E}")
            logging.info(f"{clr.F}Empty dict!{clr.E}")

    def getIndexDetails(self, indexes, column_list, total_docs=10):
        df = pd.DataFrame(columns=column_list)
        for i in range(1): # len(indexes)):
            try:
                # Given a index name, finds all the documents in the index
                # index: Index name as a string
                # body: Empty equivalent to '*' to get all the values
                # size: To some random max value for getting all the documents
                result = es.search(index=indexes[i],
                                body={"query":{"match_all":{}}},
                                size=total_docs,
                                )
                print(f"\n{indexes[i]} ---> {clr.G}{result['hits']['total']['value']}{clr.E} documents\n")
                logging.info(f"\n{indexes[i]} ---> {clr.G}{result['hits']['total']['value']}{clr.E} documents\n")

                documents = [doc for doc in result['hits']['hits']]

                for j in range(len(documents)):
                    # ---------------------
                    # For each job/document
                    # ---------------------
                    # Format: dict_keys(['end', '@version', 'start', 'uuid', 'intervals', '@timestamp'])
                    # uuid
                    uuid = documents[j]['_source']['uuid']
                    # @timestamp
                    timestamp = documents[j]['_source']['@timestamp']
                    # start (Format: dict_keys(['cookie', 'test_start', 'tcp_mss_default', 'version', 'connected', 'sndbuf_actual',
                    #                           'rcvbuf_actual', 'sock_bufsize', 'system_info', 'timestamp', 'connecting_to']))
                    start_dict = documents[j]['_source']['start']
                    num_streams = start_dict['test_start']['num_streams']

                    # ---------------------
                    # For each stream/flow
                    # ---------------------
                    # Intervals
                    intervals_list = documents[j]['_source']['intervals']
                    for k in range(len(intervals_list)):
                        for m in range(num_streams):
                            # Format: dict_keys(['receiver', 'sender'])
                            # 'sender': dict_keys(['retransmits', 'max_rtt', 'sender', 'start', 'bytes', 'mean_rtt', 'end'
                            #                      'max_snd_cwnd', 'bits_per_second', 'socket', 'seconds', 'min_rtt'])
                            # 'receiver': dict_keys(['end', 'bits_per_second', 'sender', 'start', 'socket', 'seconds', 'bytes'])

                            sender_start = documents[j]['_source']['end']['streams'][m]['sender']['start']
                            sender_end = documents[j]['_source']['end']['streams'][m]['sender']['end']
                            sender_retransmits = documents[j]['_source']['end']['streams'][m]['sender']['retransmits']
                            sender_bytes = documents[j]['_source']['end']['streams'][m]['sender']['bytes']
                            sender_min_rtt = documents[j]['_source']['end']['streams'][m]['sender']['min_rtt']
                            sender_max_rtt = documents[j]['_source']['end']['streams'][m]['sender']['max_rtt']
                            sender_mean_rtt = documents[j]['_source']['end']['streams'][m]['sender']['mean_rtt']
                            sender_bps = documents[j]['_source']['end']['streams'][m]['sender']['bits_per_second']

                            receiver_start = documents[j]['_source']['end']['streams'][m]['receiver']['start']
                            receiver_end = documents[j]['_source']['end']['streams'][m]['receiver']['end']
                            receiver_seconds = documents[j]['_source']['end']['streams'][m]['receiver']['seconds']
                            receiver_bytes = documents[j]['_source']['end']['streams'][m]['receiver']['bytes']
                            receiver_bps = documents[j]['_source']['end']['streams'][m]['receiver']['bits_per_second']

                            # print(f"uuid: {uuid}\ntimestamp: {timestamp}\nnum_streams: {num_streams}\nsender_start: {sender_start}\n sender_end: {sender_end}\n sender_retransmits: {sender_retransmits}\nsender_bytes: {sender_bytes}\n sender_min_rtt: {sender_min_rtt}\n sender_max_rtt: {sender_max_rtt}\nsender_mean_rtt: {sender_mean_rtt}\n sender_bps: {sender_bps}\n receiver_start: {receiver_start}\nreceiver_end: {receiver_end}\n receiver_seconds: {receiver_seconds}\n receiver_bytes: {receiver_bytes}\nreceiver_bps: {receiver_bps}\n\n")
                            logging.info(f"uuid: {uuid}\ntimestamp: {timestamp}\nnum_streams: {num_streams}\nsender_start: {sender_start}\n sender_end: {sender_end}\n sender_retransmits: {sender_retransmits}\nsender_bytes: {sender_bytes}\n sender_min_rtt: {sender_min_rtt}\n sender_max_rtt: {sender_max_rtt}\nsender_mean_rtt: {sender_mean_rtt}\n sender_bps: {sender_bps}\n receiver_start: {receiver_start}\nreceiver_end: {receiver_end}\n receiver_seconds: {receiver_seconds}\n receiver_bytes: {receiver_bytes}\nreceiver_bps: {receiver_bps}\n\n")

                            df = df.append({'uuid':uuid,
                                            'timestamp':timestamp,
                                            'num_streams':num_streams,
                                            'sender_start':sender_start,
                                            'sender_end':sender_end,
                                            'sender_retransmits':sender_retransmits,
                                            'sender_bytes':sender_bytes,
                                            'sender_min_rtt':sender_min_rtt,
                                            'sender_max_rtt':sender_max_rtt,
                                            'sender_mean_rtt':sender_mean_rtt,
                                            'sender_bps':sender_bps,
                                            'receiver_start':receiver_start,
                                            'receiver_end':receiver_end,
                                            'receiver_seconds':receiver_seconds,
                                            'receiver_bytes':receiver_bytes,
                                            'receiver_bps':receiver_bps
                                            }, ignore_index=True)

            except:
                pass
            # print("\nTotal docs found: ", self.sum)
            print("Done parsing!")
        return df


# class PLOT:
#     def __init__(self):
#         self.

#     def rtt(self): # By algorithm type {cubic, bbrv2} - plot mean, min, max
#         return NotImplemented

#     def bps(self): # By algorithm type {cubic, bbrv2}
#         return NotImplemented


def main():
    print("\nStarting ELK testpoint stats retrieval...")
    logging.info("\nStarting ELK testpoint stats retrieval...")

    parser = argparse.ArgumentParser(description='Testpoint Statistics')
    parser.add_argument('-t', '--term', default="iperf3*", type=str,
                        help='The search term to find the indexes {"*", "iperf3*", "jobmeta*", "bbrmon*"}')
    args = parser.parse_args()

    # indexTypes = ["*", "iperf3*", "jobmeta*", "bbrmon*"]
    # term_ = indexTypes[1]

    print(f"Chosen index type: {args.term}")
    logging.info(f"Chosen index type: {args.term}")
    get = GETTER(args.term)

    # STEP 1. Get all the indices in the ELK given a term.
    indexes = get.getIndexList(args.term)
    for e,i in enumerate(indexes):
        print(f"{e}: {i}")
        logging.info(f"{e}: {i}")

    pandas_column_list = ['uuid',
                          'timestamp',
                          'num_streams',
                          'sender_start','sender_end','sender_retransmits','sender_bytes','sender_min_rtt','sender_max_rtt','sender_mean_rtt','sender_bps',
                          'receiver_start','receiver_end','receiver_seconds','receiver_bytes','receiver_bps',
                         ]

    # STEP 2. getIndexDetails to retrieve the statistics of every testpoint
    # and every stream/flow wrt index
    index_response = get.getIndexDetails(indexes, pandas_column_list)

    # STEP 3. Create a Pandas Dataframe to make it easier for the model to read.
    try:
        index_response.to_csv('data/'+str(args.term[:-1])+'.csv')
        print(f"{str(args.term[:-1])}.csv file written!")
    except:
        raise ValueError("Cannot write the file")

    # STEP 4. Plot some of the data for sanity checks and analysis 


if __name__ == "__main__":
    main()

