#!/bin/bash
set -x

# assumes at least 1 of the following args
# $1: delay
# $2: loss
# $3: limit  (number of packets of buffer)

# this script is meant to be called from collect.py.
# Add something like this to the .ini file:
#   netem-lat = 5
# or
#   lat-sweep = 1,5,20,50

NETEM_IP="192.168.120.48"  # Set IP address of netem host here
TEST_HOST="10.201.1.2"  # Set IP address of test endpoint (star-dtn1, VLAN 4014)

echo "setting latency to $1"
echo "setting loss to $2"

# commented out for testing
ssh "$NETEM_IP" /usr/local/bin/tc.sh clear
sleep 1
if [ $2 != "None" ] ; then
    if [ $3 != "None" ] ; then
        echo "calling:  /usr/local/bin/tc.sh loss $1 $2 $3"
        ssh "$NETEM_IP" /usr/local/bin/tc.sh loss $1 $2 $3 
    else
        echo "calling:  /usr/local/bin/tc.sh loss $1 $2"
        ssh "$NETEM_IP" /usr/local/bin/tc.sh loss $1 $2 
    fi
else
    echo "calling:  /usr/local/bin/tc.sh loss $1"
    ssh "$NETEM_IP" /usr/local/bin/tc.sh loss $1 
fi
sleep 1
ssh "$NETEM_IP" /usr/local/bin/tc.sh show
# to test that latency change worked
ping -W 5 -c 2 "TEST_HOST"

