#!/bin/bash
set -x
# sets pacing rate

# requires 1st arg. If 2nd arg does not exist, then just clear any pacing setting
# $1: interface
# $2: rate

echo "setting interface $1 to paced rate of $2"

# clear previous settings
tc qdisc del dev $1 root

if [ -n "$2" ]; then
   # set new setting
   tc qdisc add dev $1 root fq maxrate $2
fi

#show results
tc qdisc show dev $1

