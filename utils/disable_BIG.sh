#!/bin/bash

# set back to defaults
/usr/sbin/ip link set dev eth100 gro_max_size 65535 gso_max_size 65535
/usr/sbin/ip link set dev eth100 gso_ipv4_max_size 65535 gro_ipv4_max_size 65535

/usr/sbin/ip link set dev vlan4012 gso_ipv4_max_size 65535 gro_ipv4_max_size 65535
/usr/sbin/ip link set dev vlan4012 gso_max_size 65535 gro_max_size 65535

/usr/sbin/ip link set dev vlan3001 gso_ipv4_max_size 65535 gro_ipv4_max_size 65535
/usr/sbin/ip link set dev vlan3001 gso_max_size 65535 gro_max_size 65535
