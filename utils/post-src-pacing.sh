#!/bin/bash
set -x

#eth=ens3f1
#eth=p2p1
eth=eth200

# to reset to default:
/sbin/tc qdisc del dev $eth root

/usr/sbin/sysctl -w net.ipv4.tcp_congestion_control=bbr2 

rm -f /usr/bin/iperf3; ln -s /usr/bin/iperf3-dist /usr/bin/iperf3

