#!/usr/bin/env python

"""
Extract iperf data from json blob and format for gnuplot.

Output file contains:

time, bytes_sent, throughput_Gbps, retransmits, stream ID
"""

import json
import os
import sys

from optparse import OptionParser

import pprint
# for debugging, so output to stderr to keep verbose
# output out of any redirected stdout.
pp = pprint.PrettyPrinter(indent=4, stream=sys.stderr)


def generate_output(iperf, options):
    """Do the actual formatting."""
    for i in iperf.get('intervals'):
        stream_id = 0
        for ii in i.get('streams'):
            if options.verbose:
                pp.pprint(ii)
            row = '{0} {1} {2} {3} {4} {5}\n'.format(
                round(float(ii.get('start')), 2),
                ii.get('bytes'),
                # to Gbits/sec
                round(float(ii.get('bits_per_second')) / (1000*1000*1000), 3),
                ii.get('retransmits'),
                round(float(ii.get('snd_cwnd')) / (1000*1000), 2),
                stream_id
            )
	    #print ("row: ", row)
            yield row
            stream_id += 1


def summed_output(iperf, options):
    """Format summed output."""

    for i in iperf.get('intervals'):

        row_header = None

        byte = list()
        bits_per_second = list()
        retransmits = list()
        snd_cwnd = list()

        for ii in i.get('streams'):
            if options.verbose:
                pp.pprint(i)
            # grab the first start value
            if row_header is None:
                row_header = round(float(ii.get('start')), 2)
            # aggregate the rest of the values
            byte.append(ii.get('throughput-bytes'))
            bits_per_second.append(float(ii.get('throughput-bits')) / (1000*1000*1000))
            retransmits.append(ii.get('retransmits'))
            snd_cwnd.append(float(ii.get('tcp-window-size')) / (1000*1000))

        row = '{h} {b} {bps} {r} {s}\n'.format(
            h=row_header,
            b=sum(byte),
            bps=round(sum(bits_per_second), 3),
            r=sum(retransmits),
            s=round(sum(snd_cwnd) / len(snd_cwnd), 2)
        )

        yield row


def main():
    """Execute the read and formatting."""
    usage = '%prog [ -f FILE | -o OUT | -v | --sum | --split ]'
    parser = OptionParser(usage=usage)
    parser.add_option('-f', '--file', metavar='FILE',
                      type='string', dest='filename',
                      help='Input filename.')
    parser.add_option('-o', '--output', metavar='OUT',
                      type='string', dest='output',
                      help='Optional file to append output to.')
    parser.add_option('-s', '--sum',
                      dest='summed', action='store_true', default=False,
                      help='Sum all parallel streams into a single value.')
    parser.add_option('-S', '--split',
                      dest='split', action='store_true', default=False,
                      help='Generate a separate file per stream.')
    parser.add_option('-v', '--verbose',
                      dest='verbose', action='store_true', default=False,
                      help='Verbose debug output to stderr.')
    options, _ = parser.parse_args()

    if not options.filename:
        parser.error('Filename is required.')

    file_path = os.path.normpath(options.filename)

    if not os.path.exists(file_path):
        parser.error('{f} does not exist'.format(f=file_path))

    with open(file_path, 'r') as fh:
        data = fh.read()

    try:
        iperf = json.loads(data)
    except Exception as ex:  # pylint: disable=broad-except
        parser.error('Could not parse JSON from file (ex): {0}'.format(str(ex)))

    if options.output:
        absp = os.path.abspath(options.output)
        output_dir, _ = os.path.split(absp)
        if not os.path.exists(output_dir):
            parser.error('Output file directory path {0} does not exist'.format(output_dir))
        fh = open(absp, 'a')
    else:
        fh = sys.stdout

    if options.summed:
        fmt = summed_output
    else:
        fmt = generate_output

    for i in fmt(iperf, options):
        #stream_id = i.split()[5]
	# NOT DONE: how to write to different files based on stream ID?? for --split option?
        fh.write(i)


if __name__ == '__main__':
    main()
