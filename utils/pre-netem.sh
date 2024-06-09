#!/bin/bash
#set -x

# assumes at least 1 of the following args
# $1: delay
# $2: loss
# $3: limit  (number of packets of buffer)

# sample use (Note that Units for lat are required)
# pre-netem.sh 5ms   # lat = 5ms, no loss
# pre-netem.sh 10ms 0.001  # lat = 10ms, 0.001% loss

# this script is meant to be called from collect.py.
# Add something like this to the .ini file:
#   netem-lat = 5
# or
#   lat-sweep = 1,5,20,50

NETEM_IP="10.200.1.1"  # Set IP address of netem host here
#NETEM_IP="192.168.120.48"  # Set IP address of netem host here
TEST_HOST="10.201.1.2"  # Set IP address of test endpoint (star-dtn1, VLAN 4014)
TC_CMD="/home/tierney/scripts/tc.sh"

echo "setting latency to $1"
echo "setting loss to $2"

# commented out for testing
ssh "$NETEM_IP" $TC_CMD clear
sleep 1
if [ "$2" != "None" ] ; then
    if [ "$3" != "None" ] ; then
        echo "calling:  $TC_CMD loss $1 $2 $3"
        ssh "$NETEM_IP" $TC_CMD loss $1 $2 $3 
    else
        echo "calling:  $TC_CMD /usr/sbin/tc.sh loss $1 $2"
        ssh "$NETEM_IP" $TC_CMD loss $1 $2 
    fi
else
    echo "calling:  $TC_CMD loss $1"
    ssh "$NETEM_IP" $TC_CMD loss $1 
fi
sleep 1
# script currently does not support 'show'
#ssh "$NETEM_IP" $TC_CMD show

# to test that latency change worked
ping -W 5 -c 2 "$TEST_HOST"

