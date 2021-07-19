#!/usr/bin/bash -e
#set -x
# sample script to start and/or attach to a perfSONAR docker container

container_id=$(docker ps -a --filter name=perfsonar --format "{{.ID}} {{.Status}}" | awk '{print $1}')
status=$(docker ps -a --filter name=perfsonar --format "{{.ID}} {{.Status}}" | awk '{print $2}')

ssh_sock=$SSH_AUTH_SOCK

if [ -z "$container_id" ]; then
   echo "Setting-up new container..."
   docker run -d --name perfsonar --privileged --net=host -v /var/log/bbr2-testing:/var/log/bbr2 dtnaas/perfsonar-testpoint:latest
   # to bind to a specific set of CPUs, useful for 100G testing
   #docker run -d --name perfsonar --privileged --net=host --cpuset-cpus="12,13,14,15,16,17" -v /var/log/bbr2-testing:/var/log/bbr2 dtnaas/perfsonar-testpoint:latest
   container_id=$(docker ps --no-trunc | grep "perfsonar" | awk '{print $1}')
   sleep 3
else
   if [ $status = "Exited" ]; then
      echo "restarting existing container "$container_id
      docker start $container_id
      sleep 2
   fi
fi
echo "getting shell on container "$container_id
docker exec -it $container_id bash
