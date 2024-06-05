#!/bin/bash
set -x
# enable BIG TCP (need to do this for eth100, and each VLAN, I think)
/usr/sbin/ip link set dev eth100 gro_max_size 185000 gso_max_size 185000
/usr/sbin/ip link set dev eth100 gso_ipv4_max_size 150000 gro_ipv4_max_size 150000
#
/usr/sbin/ip link set dev vlan4012 gro_max_size 185000 gso_max_size 185000
/usr/sbin/ip link set dev vlan4012 gso_ipv4_max_size 150000 gro_ipv4_max_size 150000
#
/usr/sbin/ip link set dev vlan3001 gro_max_size 185000 gso_max_size 185000
/usr/sbin/ip link set dev vlan3001 gso_ipv4_max_size 150000 gro_ipv4_max_size 150000

