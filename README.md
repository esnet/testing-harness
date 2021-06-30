## ESnet Network Test Harness

This directory contains files for ESnet's Network Test Harness.

Results can be sent to an Elastic server if desired.

This docker container has all tools needed (latest versions of ss and mpstat are required)
   https://hub.docker.com/r/bltierney/perfsonar-testpoint-bbrv2-testing

ini: sample config files
lib: python code for each of the harness modules
plot: utilities needed to generate Gnuplot graphs

### Example execution

Sample command to run a single test:
   ./collect.py -j ini/single.ini -o /var/log/bbr2 -H ini/test-hosts.csv 

Sample cron entry:
   /harness/collect.py -v -i ens2 -j /harness/ini/perfsonar.ini -o /var/log/bbr2 -H /harness/ini/test-hosts.csv -a bbr-mon.es.net >> /tmp/harness.out 2>&1 

### iperf3 example output with default settings (localhost test)
```
$ python3 collect.py -j ini/iperf3.ini
[2021-03-31 16:37:53,266] [MainThread] INFO: Found 2 job definitions
[2021-03-31 16:37:53,266] [MainThread] INFO: Created job for item "iperf3_cubic"
[2021-03-31 16:37:53,267] [MainThread] INFO: Created job for item "iperf3_bbr2"
[2021-03-31 16:37:53,267] [MainThread] INFO: Starting jobs [2]
[2021-03-31 16:37:53,267] [MainThread] INFO: Executing runs for job iperf3_cubic and 2 iterations
[2021-03-31 16:37:53,267] [MainThread] INFO: Testing to 127.0.0.1 using "iperf3 -c 127.0.0.1 -J -A 1 -C cubic"
[2021-03-31 16:38:53,991] [MainThread] INFO: Executing runs for job iperf3_bbr2 and 1 iterations
[2021-03-31 16:38:53,991] [MainThread] INFO: Testing to 127.0.0.1 using "iperf3 -c 127.0.0.1 -J -A 1 -C bbr2"
```

### use with netem

If you have a host running netem in your test path, you can specify netem sweep options in the .ini file.
See ini/perfsonar-throughput-example.ini for options. Note that the harness expects to find a script
/harness/utils/pre-netem.sh, and that you will likely need to customize that script for your setup. 


### Plotting

TBD

### ELK Statistics Retrieval

~~~
python3 --term "*" --from-date "2021-05-01" --to-date "2021-06-30"
~~~

Ignore the flag `--to-date` if retrieving until the current date.
The `log` and `csv` dataframe is files are going to be generated automatically in the `/data` folder.
