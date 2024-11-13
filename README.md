# ESnet Network Test Harness

To install dependencies, run:

```
python3 setup.py install
```


This directory contains a customized version of ESnet's Network Test Harness for testing BBRv3.

Eventually this branch should probably be merged into the master branch.


This docker container has all tools needed (latest versions of ss and mpstat are required)
```
see: https://hub.docker.com/r/bltierney/perfsonar-testpoint-bbrv3-testing
```

# Docker container

The easiest way to use this test harness is via docker. Useful docker commands inslude:
```
 docker compose -f docker-compose.yml build
 docker compose -f docker-compose.yml up -d
 docker compose -f docker-compose.yml exec testpoint bash
 docker compose down
```


# Directories
```
ini: sample config files
lib: python code for each of the harness modules
plot: utilities needed to generate Gnuplot graphs
```

## Example execution

Sample command to run a single test:
~~~
$ ./collect.py -j ini/single.ini -o /data
~~~

Sample cron entry:
~~~
$ /harness/collect.py -v -i ens2 -j /harness/ini/perfsonar.ini -o /data -H /harness/ini/test-hosts.csv >> /tmp/harness.out 2>&1 
~~~

## iperf3 example output with default settings (localhost test)
```
$ python3 collect.py -j ini/iperf3.ini
[2021-03-31 16:37:53,266] [MainThread] INFO: Found 2 job definitions
[2021-03-31 16:37:53,266] [MainThread] INFO: Created job for item "iperf3_cubic"
[2021-03-31 16:37:53,267] [MainThread] INFO: Created job for item "iperf3_bbr2"
[2021-03-31 16:37:53,267] [MainThread] INFO: Starting jobs [2]
[2021-03-31 16:37:53,267] [MainThread] INFO: Executing runs for job iperf3_cubic and 2 iterations
[2021-03-31 16:37:53,267] [MainThread] INFO: Testing to 127.0.0.1 using "iperf3 -c 127.0.0.1 -J -A 1 -C cubic"
[2021-03-31 16:38:53,991] [MainThread] INFO: Executing runs for job iperf3_bbr2 and 1 iterations
[2021-03-31 16:38:53,991] [MainThread] INFO: Testing to 127.0.0.1 using "iperf3 -c 127.0.0.1 -J -A 1 -C bbr"
```

## Use with Netem

If you have a host running netem in your test path, you can specify netem sweep options in the .ini file.
See ini/perfsonar-throughput-example.ini for options. Note that the harness expects to find a script
/harness/utils/pre-netem.sh, and that you will likely need to customize that script for your setup. 

## Plotting the results

Instructions coming soon
- ### (plot/iperf3_to_gnuplot.py)
TBD
