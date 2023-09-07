#!/usr/bin/bash -e
set -x
# sample script to start and/or attach to a perfSONAR docker container

# note: this will reuse old container. To force use of a newly build container, do a 'docker rm <ID>' first

container_id=$(docker ps -a --filter name=perfsonar --format "{{.ID}} {{.Status}}" | awk '{print $1}')
status=$(docker ps -a --filter name=perfsonar --format "{{.ID}} {{.Status}}" | awk '{print $2}')

ssh_sock=$SSH_AUTH_SOCK

if [ -z "$container_id" ]; then
   echo "Setting-up new container..."
   #docker run -d --name perfsonar --privileged --net=host -v /var/log/bbr3-testing:/var/log/bbr3 bltierney/perfsonar-testpoint-bbrv3-testing:latest
   # to bind to a specific set of CPUs, useful for 100G testing
   docker run -d --name perfsonar --privileged --net=host --cpuset-cpus="28,29,30,31" -v /data:/data -v /data/perfsonar:/var/log bltierney/perfsonar-testpoint-bbrv3-testing:latest
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
