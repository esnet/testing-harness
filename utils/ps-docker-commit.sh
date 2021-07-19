#!/usr/bin/bash -e
# sample script to commit a snapshot of your perfSONAR docker container
set -x

container_id=$(docker ps |grep "perfsonar" | awk '{print $1}')

if [ -z "$container_id" ]; then
   echo "no docker container found"
else
   echo "committing container "$container_id
   docker  commit -m "updated test harness" $container_id dtnaas/perfsonar-testpoint:latest
fi

