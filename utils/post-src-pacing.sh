#!/bin/bash
set -x

#eth=ens3f1
#eth=p2p1
eth=dtn1

# to reset to default:
/sbin/tc qdisc del dev $eth root

#rm -f /usr/bin/iperf3; ln -s /usr/bin/iperf3-dist /usr/bin/iperf3

