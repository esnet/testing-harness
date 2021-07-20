'''
**
** Project Lead: Eashan Adhikarla
** Mentor: Ezra Kissel
**
** Date Created: June 17' 2021
** Last Modified: July 13' 2021
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
import glob

try:
    os.makedirs('data')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

logging.basicConfig(filename='data/statistics.log', level=logging.INFO) #logging.DEBUG)

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
    raise ValueError("\nConnection failed")
    logging.error("\nConnection failed")
else:
    print("\nConnection successful!")
    logging.info("\nConnection successful!")


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
            print (f"\n'{clr.G}{len(indices_dict)}{clr.E}' indexes found!")
            logging.info (f"'{len(indices_dict)}' indexes found!")
            for k,v in indices_dict.items():
                idx.append(k)
            return idx
        else:
            print (f"{clr.F}Empty dict!{clr.E}")
            logging.info (f"Empty dict!")

    def getIndexDetails_bbrmon(self, bbrmon_pscheduler, column_list, total_docs=10000):
        df = pd.DataFrame(columns=column_list)

        for i in range(len(bbrmon_pscheduler)):
            bbrmon_pscheduler_result = es.search(index=bbrmon_pscheduler[i],
                                                 body={"query":{"match_all":{}}},
                                                 size=total_docs,
                                                )
            bbrmon_pscheduler_documents = [docs for docs in bbrmon_pscheduler_result['hits']['hits']]
            
            for bbrmondoc in range(len(bbrmon_pscheduler_documents)):
            # ---------------------
            # For each job/document
            # ---------------------
            # Format:
                try:
                    uuid = bbrmon_pscheduler_documents[bbrmondoc]['_source']['id']
                    src_hostname = bbrmon_pscheduler_documents[bbrmondoc]['_source']['meta']['source']['hostname']
                    dst_hostname = bbrmon_pscheduler_documents[bbrmondoc]['_source']['meta']['destination']['hostname']
                    throughput = bbrmon_pscheduler_documents[bbrmondoc]['_source']['result']['throughput']
                    retransmits = bbrmon_pscheduler_documents[bbrmondoc]['_source']['result']['retransmits']

                    mean_rtt, sum_rtt = 0.0, 0.0
                    mean_btys, sum_byts = 0.0, 0.0
                    intvl_len = len(bbrmon_pscheduler_documents[bbrmondoc]['_source']['result']['intervals']['json'])
                    for intvl in range(intvl_len):
                        sum_rtt += bbrmon_pscheduler_documents[bbrmondoc]['_source']['result']['intervals']['json'][intvl]['streams'][0]['rtt']/2 # Divide by 2 to get the latency
                        sum_byts += bbrmon_pscheduler_documents[bbrmondoc]['_source']['result']['intervals']['json'][intvl]['streams'][0]['throughput-bytes']

                    mean_rtt = (sum_rtt/intvl_len)
                    mean_btys = (sum_byts/intvl_len)

                    # print (f"uuid: {uuid}\nsrc hostname: {src_hostname}\ndst hostname: {dst_hostname}\nthroughput: {throughput}\nLatency: {mean_rtt}\nretransmits: {retransmits}\nbytes: {mean_btys}\n\n")
                    # logging.info (f"uuid: {uuid}\nsrc hostname: {src_hostname}\ndst hostname: {dst_hostname}\nthroughput: {throughput}\nLatency: {mean_rtt}\nretransmits: {retransmits}\nbytes: {mean_btys}\n\n")

                    df = df.append({'UUID':uuid,
                                    'SRC HOSTNAME':src_hostname,
                                    'DST HOSTNAME':dst_hostname,
                                    'THROUGHPUT (mean)':throughput,
                                    'LATENCY (mean)':mean_rtt,
                                    'RETRANSMITS (mean)':retransmits,
                                    'BYTES (mean)':mean_btys,
                                    }, ignore_index=True)

                except Exception as e:
                    print("Exception: ", e)
        return df

    def getIndexDetails(self, iperf3, jobmeta, column_list, interval=False, total_docs=10000):
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
                # print (f"\n{jobmeta[i]} ---> {clr.G}{jobmeta_result['hits']['total']['value']}{clr.E} documents\n")
                jobmeta_documents = [doc for doc in jobmeta_result['hits']['hits']]

                for j in range(len(iperf3)):
                    iperf3_result = es.search(index=iperf3[j],
                                        body={"query":{"match_all":{}}},
                                        size=total_docs,
                                        )
                    # print (f"\n{iperf3[i]} ---> {clr.G}{iperf3_result['hits']['total']['value']}{clr.E} documents\n")
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
                                alias = jobmeta_documents[jobmetadoc]['_source']['alias']
                                pacing = jobmeta_documents[jobmetadoc]['_source']['pacing'] 

                                if uuid in iter_uuid:
                                    # Hostname
                                    hostname = jobmeta_documents[jobmetadoc]['_source']['hostname']

                                    # @timestamp
                                    timestamp = iperf3_documents[iperfdoc]['_source']['@timestamp']

                                    # start (Format: dict_keys(['cookie', 'test_start', 'tcp_mss_default', 'version', 'connected', 'sndbuf_actual',
                                    #                           'rcvbuf_actual', 'sock_bufsize', 'system_info', 'timestamp', 'connecting_to']))
                                    start_dict = iperf3_documents[iperfdoc]['_source']['start']
                                    num_streams = start_dict['test_start']['num_streams']

                                    if not interval:
                                        # Bits per second (bps) is the throughput
                                        sender_throughput = iperf3_documents[iperfdoc]['_source']['end']['streams'][0]['sender']['bits_per_second']
                                        receiver_throughput = iperf3_documents[iperfdoc]['_source']['end']['streams'][0]['receiver']['bits_per_second']

                                        # Round trip time (rtt) for the LATENCY calculation
                                        sender_min_rtt = iperf3_documents[iperfdoc]['_source']['end']['streams'][0]['sender']['min_rtt']/2
                                        sender_max_rtt = iperf3_documents[iperfdoc]['_source']['end']['streams'][0]['sender']['max_rtt']/2
                                        sender_mean_rtt = iperf3_documents[iperfdoc]['_source']['end']['streams'][0]['sender']['mean_rtt']/2

                                        # Retransmits for LOSS calculation
                                        sender_retransmits = iperf3_documents[iperfdoc]['_source']['end']['streams'][0]['sender']['retransmits']

                                        # Congestion Type
                                        sender_congestion = iperf3_documents[iperfdoc]['_source']['end']['sender_tcp_congestion']
                                        receiver_congestion = iperf3_documents[iperfdoc]['_source']['end']['receiver_tcp_congestion']

                                        # Transmitted bytes for the testpoint
                                        receiver_bytes = iperf3_documents[iperfdoc]['_source']['end']['streams'][0]['receiver']['bytes']

                                        # print (f"uuid: {uuid}\nhostname: {hostname}\nalias: {alias}\ntimestamp: {timestamp}\nnum_streams: {num_streams}\nsender_throughput: {sender_throughput}\nreceiver_throughput: {receiver_throughput}\nlatency (min): {sender_min_rtt}\nlatency (max): {sender_max_rtt}\nlatency (mean): {sender_mean_rtt}\nsender_retransmits: {sender_retransmits}\nsender_congestion: {sender_congestion}\nreceiver_congestion: {receiver_congestion}\nreceiver_bytes: {receiver_bytes}\n\n")
                                        # logging.info (f"uuid: {uuid}\nhostname: {hostname}\nalias: {alias}\ntimestamp: {timestamp}\nnum_streams: {num_streams}\nsender_throughput: {sender_throughput}\nreceiver_throughput: {receiver_throughput}\nlatency (min): {sender_min_rtt}\nlatency (max): {sender_max_rtt}\nlatency (mean): {sender_mean_rtt}\nsender_retransmits: {sender_retransmits}\nsender_congestion: {sender_congestion}\nreceiver_congestion: {receiver_congestion}\nreceiver_bytes: {receiver_bytes}\n\n")

                                        df = df.append({'UUID':uuid,
                                                        'HOSTNAME':hostname,
                                                        'ALIAS':alias,
                                                        'TIMESTAMP':timestamp,
                                                        'STREAMS':num_streams,
                                                        'PACING':pacing,
                                                        'THROUGHPUT (Sender)':sender_throughput,
                                                        'THROUGHPUT (Receiver)':receiver_throughput,
                                                        'LATENCY (min.)':sender_min_rtt,
                                                        'LATENCY (max.)':sender_max_rtt,
                                                        'LATENCY (mean)':sender_mean_rtt,
                                                        'RETRANSMITS':sender_retransmits,
                                                        'CONGESTION (Sender)':sender_congestion,
                                                        'CONGESTION (Receiver)':receiver_congestion,
                                                        'BYTES (Receiver)':receiver_bytes,
                                                        }, ignore_index=True)

                                    if interval:
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

                                                # print (f"uuid: {uuid}\nhostname: {hostname}\ntimestamp: {timestamp}\nnum_streams: {num_streams}\nsender_throughput: {sender_throughput}\nreceiver_throughput: {receiver_throughput}\nsender_min_rtt: {sender_min_rtt}\nsender_max_rtt: {sender_max_rtt}\nsender_mean_rtt: {sender_mean_rtt}\nsender_congestion: {sender_congestion}\nreceiver_congestion: {receiver_congestion}\nsender_retransmits: {sender_retransmits}\n\n")
                                                logging.info (f"uuid: {uuid}\nhostname: {hostname}\nalias: {alias}\ntimestamp: {timestamp}\nnum_streams: {num_streams}\nsender_throughput: {sender_throughput}\nreceiver_throughput: {receiver_throughput}\nsender_min_rtt: {sender_min_rtt}\nsender_max_rtt: {sender_max_rtt}\nsender_mean_rtt: {sender_mean_rtt}\nsender_congestion: {sender_congestion}\nreceiver_congestion: {receiver_congestion}\nsender_retransmits: {sender_retransmits}\n\n")

                                                df = df.append({'UUID':uuid,
                                                                'HOSTNAME':hostname,
                                                                'ALIAS':alias,
                                                                'TIMESTAMP':timestamp,
                                                                'STREAMS':num_streams,
                                                                'PACING':pacing,
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
            except:
                pass
        return df


class TIMEWINDOW:
    """
    Filter the index according to the provided time window
    """
    def __init__(self, from_date, to_date):
        self.from_date = from_date
        self.to_date = to_date

    def timeFormatter(self):
        if self.to_date=="empty":  #is None:
            curr = datetime.datetime.now()
            self.to_date = f"{curr.year}-{curr.month}-{curr.day}"

        self.from_date = datetime.datetime.strptime(self.from_date, '%Y-%m-%d').date()
        self.to_date   = datetime.datetime.strptime(self.to_date, '%Y-%m-%d').date()
        time_window = (self.from_date, self.to_date)
        return time_window

    def filterWindow(self, indexes, timewindow):
        newIndexList=[]
        from_date, to_date = timewindow
        for ind in indexes:
            try:
                match = re.search('\d{4}.\d{2}.\d{2}', ind)
                date = datetime.datetime.strptime(match.group(), '%Y.%m.%d').date()
                flag2 = from_date < date < to_date
                if flag2:
                    newIndexList.append(ind)
            except:
                pass
        print (f"'{clr.G}{len(newIndexList)}{clr.E}' filtered indexes in time window ({clr.H}FROM:{clr.E}{from_date}, {clr.H}TO:{clr.E}{to_date})\n")
        logging.info (f"'{len(newIndexList)}' filtered indexes in time window (FROM:{from_date}, TO:{to_date})\n")
        return newIndexList


def main(verbose=False):
    print("Starting ELK testpoint stats retrieval...")
    logging.info ("Starting ELK testpoint stats retrieval...")

    parser = argparse.ArgumentParser(description='Testpoint Statistics')
    parser.add_argument('-t', '--term', default="*", type=str,
                        help='The search term to find the indexes {"*", "iperf3*", "jobmeta*", "bbrmon*"}')
    parser.add_argument('-F', '--from_date', default="2021-05-01", type=str,
                        help='Get all the testpoint stats starting from this date | Format: yyyy-mm-dd')
    parser.add_argument('-T', '--to_date', default="empty", type=str,
                        help='Get all the testpoint stats until this date | Format: yyyy-mm-dd')
    parser.add_argument('-i', '--interval', action='store_true',
                        help='Interval=True will retreive per test per interval statistics')
    parser.add_argument('-o','--type', default="iperf3", type=str,
                        help='Chose option for pulling type of index details {iperf3, bbrmon}')
    args = parser.parse_args()
    for arg in vars(args):
        print (f"{arg} {getattr(args, arg) : ^25}")
        logging.info (f"{arg} {getattr(args, arg)}")

    get = GETTER(args.term)

    # ----------------------------------------------------
    # STEP 1. Get all the indices in the ELK given a term.
    # ----------------------------------------------------
    indexes = get.getIndexList(args.term)
    if verbose:
        for e,i in enumerate(indexes):
            print (f"{e}: {i}")
            logging.info (f"{e}: {i}")

    # Filtering indexes based on time window
    tw = TIMEWINDOW(args.from_date, args.to_date)
    timewindow = tw.timeFormatter()
    filteredIndex = tw.filterWindow(indexes, timewindow)

    # Filtering indexes by type
    sorted_index = sorted(filteredIndex)
    iperf3 = [i for i in sorted_index if i.startswith("iperf3")]
    jobmeta = [i for i in sorted_index if i.startswith("jobmeta")]
    ss = [i for i in sorted_index if i.startswith("ss")]
    bbrmon_pscheduler = [i for i in sorted_index if i.startswith("bbrmon-pscheduler")]
    bbrmon_jobmeta = [i for i in sorted_index if i.startswith("bbrmon-jobmeta")]
    bbrmon_ss = [i for i in sorted_index if i.startswith("bbrmon-ss")]
    bbrmon_tcptrace = [i for i in sorted_index if i.startswith("bbrmon-tcptrace")]
    miscellaneous = [i for i in sorted_index if i not in (iperf3+jobmeta+ss+bbrmon_jobmeta+bbrmon_pscheduler+bbrmon_ss+bbrmon_tcptrace)]

    flag1 = len(iperf3+jobmeta+ss+bbrmon_jobmeta+bbrmon_pscheduler+bbrmon_ss+bbrmon_tcptrace+miscellaneous) == len(sorted_index)
    if not flag1:
        print("New indexes got added, missing in the working list!")
        logging.info ("New indexes got added, missing in the working list!")
        missing = sorted_index - (iperf3+jobmeta+ss+bbrmon_jobmeta+bbrmon_pscheduler+bbrmon_ss+bbrmon_tcptrace+miscellaneous)
        for e,i in enumerate(missing):
            print (f"{e}: {i}")
            logging.info (f"{e}: {i}")

    iperf3_column_list = ['UUID',
                          'HOSTNAME',
                          'ALIAS',
                          'TIMESTAMP',
                          'STREAMS',
                          'PACING',
                          'THROUGHPUT (Sender)', 'THROUGHPUT (Receiver)',
                          'LATENCY (min.)', 'LATENCY (max.)', 'LATENCY (mean)',
                          'RETRANSMITS',
                          'CONGESTION (Sender)', 'CONGESTION (Receiver)',
                          'BYTES (Receiver)',
                         ]
    bbrmon_column_list = ['UUID',
                          'SRC HOSTNAME',
                          'DST HOSTNAME',
                          'THROUGHPUT (mean)',
                          'LATENCY (mean)',
                          'RETRANSMITS (mean)',
                          'BYTES (mean)',
                          ]

    # ---------------------------------------------------------------------
    # STEP 2.a. getIndexDetails to retrieve the statistics of every testpoint
    # and every stream/flow wrt index
    # STEP 2.b. Create a Pandas Dataframe to make it easier for the model to read
    # ---------------------------------------------------------------------
    if args.type == "bbrmon":
        if len(bbrmon_pscheduler)==0:
            print("No indexes in the given time window!")
        else:
            index_response = get.getIndexDetails_bbrmon(bbrmon_pscheduler, bbrmon_column_list, total_docs=10000)
    elif args.type == "iperf3": # considering else as iperf3 only*
        index_response = get.getIndexDetails(iperf3, jobmeta, iperf3_column_list, interval=False, total_docs=10000)

    print(f"Records: {clr.G}{len(index_response)}{clr.E}")
    # --------------------------------------------------------------------------
    # STEP 3. Writer to write the dataframe into a csv file
    # takes care of the naming, if ran the script multiple times
    # --------------------------------------------------------------------------
    # files = os.listdir("data")
    
    files = glob.glob("data/*.csv") # Reading all the previously written files
    last_filename = files[-1]  # Get the last file name

    num = int(last_filename.split("-")[1].split(".")[0]) # Extract the number from the last filename

    if args.term=="*":
        filename = "statistics-"+str(num+1)
    else:
        filename = args.term[:-1]

    try:
        index_response.to_csv('data/'+str(filename)+'.csv')
        print (f"{str(filename)}.csv file written!")
        logging.info (f"'{clr.G}{str(filename)}.csv{clr.E}' file written!")
    except:
        raise ValueError("{clr.F}File not written!{clr.E}")


if __name__ == "__main__":
    main()
