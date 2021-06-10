#!/usr/bin/python3
#
# Parse ss.logs and create JSON objects to send to ELK
# Usage:
#    infile=ss.log [dir=data_directory] [uuid=uuid] [summary_only=true] ss_log_parser.py
# 
# based on version from Google in https://github.com/google/bbr
#
# todo: 
#   - strip out control port info
#   - lots of code leanup
#      

import os
import socket
import sys
import time
import json
import uuid

#DEBUG = True   # enable debugging output?
DEBUG = False   # enable debugging output?

try:
    outdir = os.environ['dir']
except:
    outdir = "."
outfile = os.environ['infile']+".json"
try:
    summary_only = os.environ['summary_only']
except:
    summary_only = False

try:
    uuid =  os.environ['uuid']
except:
    uuid = uuid.uuid1()
    uuid = f"{uuid}"

dataport = 5201  # only grab data for traffic on this port
all_data = {}   # data for all time:            <time>: time_data
all_list = []   # list of dicts to dump as JSON

def debug(s):
    if DEBUG:
        print('DEBUG: %s' % s)

def ldebug(s,d):
    #localized debug
    if d:
        print('DEBUG: %s' % s)

def median(nums):
    """Return median of all numbers."""

    if len(nums) == 0:
        return 0
    sorted_nums = sorted(nums)
    n = len(sorted_nums)
    m = n - 1
    return (sorted_nums[int(n/2)] + sorted_nums[int(m/2)]) / 2.0

def read_file():
    """Read the ss.log file and parse into a dictionary."""
    time_data = {}  # data for the current timestamp: <port>: { field: value }, used to compute retrans rate
    time_secs = -1
    infile =  os.environ['infile']
    ss_log_path = f"%s/%s" % (outdir,infile)
    #print('reading path: %s' % (ss_log_path))
    try:
        f = open(ss_log_path)
    except:
        print ("file %s not found " % ss_log_path)
        sys.exit(-1)

    # Read a timestamp line, or per-flow tuple line, or EOF.
    line = f.readline()
    debug('readline 1 => %s' % (line))
    while True:
        #debug('line => %s' % (line))

        # If the file is done or data for current time is done, save time data.
        if not line or line.startswith('# ') and len(time_data):
            debug('all_data time %d => time_data %s' %
                  (time_secs,  time_data))
            all_data[time_secs] = time_data
            time_data = {}

        if not line:
            return all_data

        # Check to see if we have data for a new point in time
        if line.startswith('# '):
            time_secs = float(line[2:])
            assert time_secs > 0, time_secs
            #debug('time_secs = %s' % (time_secs))
            # Read ss column headers ("State...")
            line = f.readline()
            #debug('readline column headers => %s' % (line))
            # Read next line
            line = f.readline()
            continue

        # Parse line with 4-tuple
        debug('readline for 4-tuple => %s' % (line))
        if not line or line.startswith('# '):
            continue   # No live sockets with ports maching the ss query...
        if len(line.split()) != 5:
            sys.stderr.write('unable to find 4-tuple in this line. skipping: %s' % (line))
            line = f.readline()
            continue
        flow_data = {}
        
        h = line.split()[3]
        ip1 = h.split(':')[0]
        try:  # note: only works for IPv4
            port1 = int(h.split(':')[1])
        except:
            print ("port not found, skipping, %s" % {h})
            # skip 2 lines
            line = f.readline()
            line = f.readline()
            continue
        h = line.split()[4]
        ip2 = h.split(':')[0]
        port2 = int(h.split(':')[1])
        debug ('found connection for %s:%d - %s:%d' % (ip1, port1, ip2, port2))

        # Read line with flow stats
        line = f.readline()
        debug('readline flow stats => %s' % (line))
        assert line, 'expected flow stats for port %d' % (port1)
        if port1 != dataport and port2 != dataport:
	    #print ("not an iperf3 flow, skipping")
            pass
        else:
	    #print ("found an iperf3 connection")
            stats = line.strip().split()
            debug('stats: %s' % (stats))
            flow_data['ip1'] = ip1
            flow_data['port1'] = port1
            flow_data['ip2'] = ip2
            flow_data['port2'] = port2
            flow_data['cc'] = 'default'  # initialize to something in case not bbr2 or cubic
            flow_data['time'] = time_secs # time of first entry for this stream
            flow_data['data_segs_out'] = 0  # initialize

            for item in stats:
                #print ("checking item: ", item)
                if item.startswith('cwnd:'):
                    flow_data['cwnd'] = int(item[item.rfind(':') + 1:])
                elif item.startswith('bytes_acked:'):
                    flow_data['bytes_acked'] = int(item[item.rfind(':') + 1:])
                elif item.startswith('retrans:'):
                    flow_data['retrans'] = int(item[item.rfind('/') + 1:])
                elif item.startswith('data_segs_out:'):
                    flow_data['data_segs_out'] = int(item[item.rfind(':') + 1:])
                elif item.startswith('rtt:'):
	            # rtt from ss is rtt/rttvar, in ms
                    rtt = float(item[item.find(':') + 1:item.rfind('/')]) / 1000
                    flow_data['rtt'] =  '%.3f' % rtt
                elif item.startswith('unacked:'):
                    flow_data['unacked'] = int(item[item.find(':') + 1:])
                elif item.startswith('mss:'):
                    flow_data['mss'] = int(item[item.find(':') + 1:])
                elif item.startswith('cubic'):
                    flow_data['cc'] = 'cubic'
                elif item.startswith('bbr2'):
                    flow_data['cc'] = 'bbr2'
                elif item.startswith('pmtu'):
                    flow_data['pmtu'] = int(item[item.find(':') + 1:])
                # tried capturing bytes_sent and delivered, but these appear to not be useful to compute throughput
                #elif item.startswith('bytes_sent'):  # NOTE, not useful, seems to be current outstanding number of bytes
                #    flow_data['bytes_sent'] = int(item[item.find(':') + 1:])
                #elif item.startswith('delivered'):
                #    flow_data['delivered'] = int(item[item.find(':') + 1:])
                elif item.startswith('minrtt'):
                    flow_data['minrtt'] = '%.3f' % float(item[item.find(':') + 1:])
                elif item.startswith('mrtt:'):    # bbr2 only
                    flow_data['mrtt'] = '%.3f' % float(item[item.find(':') + 1:])
                # need to figure out how to grab next string, as these dont use key:value syntax
                #elif item.startswith('pacing_rate'):  
                #    flow_data['pacing_rate'] = 
                #elif item.startswith('delivery_rate'):  
                #    flow_data['delivery_rate'] = 

            #debug('time_data for time %s port %d: %s' % (time_secs, port1, flow_data))
            if not 'cwnd' in flow_data:
                print('unable to find cwnd in line: %s' % (line))
                continue
            if flow_data['data_segs_out'] > 0:   #ignore if no data
                time_data[port1] = flow_data
                all_list.append(flow_data)
        # Move on to the next line:
        line = f.readline()

def compute_summary_info (f, all_data):
    """Log average retransmit rate and more for each flow and globally.
 
    Computing retransmit rate as follows:

    Notes: ss output includes the following
       segs_out: The total number of segments sent.
       data_segs_out: The number of segments sent containing a positive length data segment.
       retrans: how many times the retransmission occured (from man page). 
           Q: tcpEStatsPerfSegsRetrans from rfc4898.txt?
	   "The number of segments transmitted containing at least some retransmitted data"
       bytes_acked
       bytes_sent
       bytes_retrans   (bbr2 only)

       Best way to compute retrans rate is as follows (method from Google team)
          - keep track of the final data_segs_out value for the stream
              NOTE: for parallel transfers be sure to use source port, not dest port
          - keep track of the final retrans value for the stream
          - retrans rate = retrans / tot_data_segs

    """
    debug  = False
    #debug  = True

    cc = {}  # congestion control per port
    last_data_segs_out = {}  # last data_segs_out per port
    last_retrans =       {}  # last retransmitted packet count per port
    stream_info = {}      # summary info to convert to JSON object
    total_bytes = [0] * 65000  # init to 0 for any possible port #; probably a better way to do this....
    i = total_retrans = bbr2_retrans = cubic_retrans = cubic_data_segs = bbr2_data_segs = tot_bytes = 0
    start_time = 0.0

    for t in sorted(all_data.keys()):
        time_data = all_data[t]
        ldebug(time_data,debug)
        for port1, flow_data in time_data.items():
            if start_time == 0.0:
                start_time = flow_data.get('time')
            ldebug('port %d flow_data %s' % (port1, flow_data),debug)
            cc[port1] = flow_data.get('cc', 0)
            ldebug('port %d is using congestion control %s' % (port1, cc[port1]),debug)
            last_data_segs_out[port1] = flow_data.get('data_segs_out', 0)
            ldebug('port %d last_data_segs_out=%s' %
                  (port1, last_data_segs_out[port1]), debug)
            last_retrans[port1] = flow_data.get('retrans', 0)

            # this not useful  -blt
            #bytes =  flow_data.get('delivered', 0)
            #try:
            #    total_bytes[port1] += bytes
            #except:
            #    total_bytes[port1] = 0  # initialize 1st time
            ##    total_bytes[port1] += bytes
            #tot_bytes += bytes # total for all streams

            ldebug('port %d last_retrans=%d' % (port1, last_retrans[port1]), debug)
            #ldebug('loop cnt: %d, bytes = %d, tot_bytes = %d ' %(i,bytes,tot_bytes), debug)
            i += 1

    end_time = flow_data.get('time')
    total_time = end_time - start_time
    total_data_segs_out = 0
    num_streams = len(last_data_segs_out)  # note that for iperf3 there will also be a control stream
    streams = []
    i = 0

    for port1 in sorted(last_data_segs_out):
        if last_data_segs_out[port1] > 100:   # ignore control channel
            # need to round so that very small numbers in scientific notation dont break plotting tools later
            retrans = round(float(last_retrans[port1]) / float(last_data_segs_out[port1]),8)
            s = {}
            s['port'] = port1
            s['cc'] = cc[port1]
            s['data_segs'] = last_data_segs_out[port1]
            s['rtrans_rate'] =  f"{retrans:.8f}"  # to make sure not in scientific notation
            streams.append(s)
            #print('stream %d: data seg = %d ;  retrans rate = %.8f (%s)' % (i,last_data_segs_out[port1],retrans,cc[port1]) )
        else:
            num_streams -= 1   # dont count control stream
        
        total_retrans += last_retrans[port1]
        total_data_segs_out += last_data_segs_out[port1]
        if cc[port1] == "bbr2":
             bbr2_retrans += last_retrans[port1]
             bbr2_data_segs += last_data_segs_out[port1]
        else:  # XXX: Assumes cubic if not bbr
             cubic_retrans += last_retrans[port1]
             cubic_data_segs += last_data_segs_out[port1]
        #print('last_retrans: %d, last_data_segs_out: %d' % (last_retrans[port1], last_data_segs_out[port1]) )
        #print('bbr2_data_segs: %d, cubic_data_segs: %d' % (bbr2_data_segs, cubic_data_segs) )
        #print('bbr2_retrans: %d, cubic_retrans: %d' % (bbr2_retrans, cubic_retrans) )

        i += 1

    if total_data_segs_out == 0:
        total_retrans_rate = 0
    else:
        total_retrans_rate = round(float(total_retrans) / float(total_data_segs_out),8)
        if cubic_data_segs > 0:
           cubic_retrans_rate = round(float(cubic_retrans) / float(cubic_data_segs),8)
        else:
           cubic_retrans_rate = 0
        if bbr2_data_segs > 0:
           bbr2_retrans_rate = round(float(bbr2_retrans) / float(bbr2_data_segs),8)
        else:
           bbr2_retrans_rate = 0

    # not needed??
    total_retrans_rate = f"{total_retrans_rate:.8f}"  # to make sure not in scientific notation
    cubic_retrans_rate = f"{cubic_retrans_rate:.8f}"  # to make sure not in scientific notation
    bbr2_retrans_rate = f"{bbr2_retrans_rate:.8f}"  # to make sure not in scientific notation
    #print ('Total data segs: %d ; Total Retransmit rate: %s ; cubic retrans: %s ; bbr2 retrans: %s ; Time: %.1f  ' % (total_data_segs_out, total_retrans_rate, cubic_retrans_rate, bbr2_retrans_rate, total_time ))

    # add to dict
    stream_info['streams'] = streams
    stream_info['num_streams'] = num_streams
    stream_info['total_retrans'] = total_retrans
    stream_info['total_data_segs_out'] = total_data_segs_out
    stream_info['cubic_data_segs'] = cubic_data_segs
    stream_info['cubic_retrans_rate'] = cubic_retrans_rate
    stream_info['bbr2_data_segs'] = bbr2_data_segs
    stream_info['bbr2_retrans_rate'] = bbr2_retrans_rate

    # also compute median srtt for all srtt samples we took from periodic ss dumps.
    rtts = []
    for t in sorted(all_data.keys()):
        time_data = all_data[t]
        for port1, flow_data in time_data.items():
            #debug('port %d flow_data %s' % (port1, flow_data))
            if 'rtt' in flow_data:
                rtt = float(flow_data['rtt'])
                rtts.append(rtt)

    p50_rtt = median(rtts)
    p50_rtt = p50_rtt * 1000.0   # convert to ms
    # Write p50 srtt sample (in secs) we took across all flows.
    stream_info['p50_rtt'] = p50_rtt

    return stream_info

def log_retrans_rate(f, summary):
    # dump summary object
    json.dump(summary, f)
    f.write("\n")
    #print (json.dumps(summary))
    ss = summary.get('streams')
    for j in range(len(ss)):
        # also dump per stream objects
        s = ss[j]
        # put UUID in every object
        s['uuid'] = summary.get('uuid')
        json.dump(s, f)
        f.write("\n")
        #print (json.dumps(s))


def log_intervals(f, data, uuid):
    # rework this dictionary into a form elastic will be happy with, and add uuid
    for j in range(len(data)):
        interval = data[j]
        interval['uuid'] = f"{uuid}"
        interval = {'interval': interval}
        debug (interval)
        json.dump(interval, f)
        f.write("\n")


def main():
    """Main function to run everything."""
    read_file()
    if len(all_list) == 0:
       print ("Error: no ss data found")
       sys.exit(-1)

    f = open(os.path.join(outdir, outfile), 'w')
    print ("writing results to file: %s/%s" % (outdir,outfile))

    if not summary_only:
        log_intervals(f,all_list, uuid)

    summary = compute_summary_info(f, all_data)
    summary['uuid'] = uuid
    # logs retrans and other summary info
    log_retrans_rate(f, summary)
    f.flush()
    os.sync()
    print ('--- ss to JSON complete ---')
    return 0

if __name__ == '__main__':
    sys.exit(main())
