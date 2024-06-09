#!/bin/bash

#make sure SMT is off
echo off > /sys/devices/system/cpu/smt/control

# restore system to initial state
# for single stream tests
set_irq_affinity_cpulist.sh 4 eth100

# for parallel stream tests
#set_irq_affinity_bynode.sh 0 eth100


# set BIG TCP back to defaults
/usr/sbin/ip link set dev eth100 gro_max_size 65535 gso_max_size 65535
/usr/sbin/ip link set dev eth100 gso_ipv4_max_size 65535 gro_ipv4_max_size 65535

/usr/sbin/ip link set dev vlan4012 gso_ipv4_max_size 65535 gro_ipv4_max_size 65535
/usr/sbin/ip link set dev vlan4012 gso_max_size 65535 gro_max_size 65535

/usr/sbin/ip link set dev vlan3001 gso_ipv4_max_size 65535 gro_ipv4_max_size 65535
/usr/sbin/ip link set dev vlan3001 gso_max_size 65535 gro_max_size 65535
