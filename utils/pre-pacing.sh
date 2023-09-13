#!/bin/bash
set -x

#eth=ens3f1
#eth=p2p1
eth=eth200

# $1: pacing rate

echo "setting pacing to $1"

# first delete any existing setting
tc qdisc del dev $eth root
# set maxrate
tc qdisc add dev $eth root fq maxrate $1


