import os
import time
import signal
import socket
import subprocess
import logging
import re
import csv
import sys
import uuid
import json
import math
from pathlib import Path
from datetime import datetime
from subprocess import PIPE, STDOUT
from ping3 import ping, verbose_ping
from threading import Thread
from lib.ss import launch_ss
from lib.ss import ss_send_ampq
from lib.tcpdump import launch_tcpdump
from lib.tcptrace import launch_tcptrace
from lib.ampq import AMPQSender
from lib.profile import ProfileManager


loopbacks = ['localhost', '127.0.0.1', '::1']
csv_host_opts = ["hostname", "alias", "profile"]
log = logging.getLogger("harness")

comment_pattern = re.compile(r'\s*#.*$')


def skip_comments(lines):
    """
    A filter which skip/strip the comments and yield the
    rest of the lines

    :param lines: any object which we can iterate through such as a file
        object, list, tuple, or generator
    """
    global comment_pattern

    for line in lines:
        line = re.sub(comment_pattern, '', line).strip()
        if line:
            yield line


class Job:
    def __init__(self, name, cfg, outdir, hostlist, nic, archive):
        self.uuid = str(uuid.uuid4())
        self.name = name
        self.conf = cfg
        self.user = cfg.get('user', None)
        self.type = cfg.get('type', None)
        self.enabled = cfg.getboolean('enabled', True)
        self.instrument = cfg.getboolean('instrument', False)
        self.pre = cfg.get('pre', None)
        self.post = cfg.get('post', None)
        self.iters = cfg.getint('iterations', 1)
        self.src = cfg.get('src', None)
        self.dst = cfg.get('dst', None)
        self.alias = cfg.get('alias', None)
        input_string = cfg.get('src-cmd', None)
        # make sure there are no extra spaces in src-cmd, as this break pscheduler
        self.src_cmd = re.sub(r'\s{2,}', ' ', input_string)

        self.dst_cmd = cfg.get('dst-cmd', None)
        self.src_cmd_once = cfg.getboolean('src-cmd-once', False)
        self.dst_cmd_once = cfg.getboolean('dst-cmd-once', False)
        self.pre_src_cmd = cfg.get('pre-src-cmd', None)
        self.pre_dst_cmd = cfg.get('pre-dst-cmd', None)
        self.post_src_cmd = cfg.get('post-src-cmd', None)
        self.post_dst_cmd = cfg.get('post-dst-cmd', None)
        self.ss = cfg.getboolean('ss', False)
        self.tcptrace = cfg.getboolean('tcptrace', False)
        self.tcpdump = cfg.getboolean('tcpdump', False)
        self.tcpdump_filt = cfg.get('tcpdump-filt', None)
        self.iter_uuids = list()
        self.param_sweep = cfg.get('param-sweep', None)
        self.loss = cfg.get('netem-loss', None)
        self.lat = cfg.get('netem-lat', None)
        self.statexec = cfg.get('statexec', None)
        self.limit = cfg.get('netem-limit', None)
        self.pacing = cfg.get('pacing', None)
        self.lat_sweep = cfg.get('lat-sweep', None)
        self.limit_sweep = cfg.get('limit-sweep', None)
        self.profile_file = cfg.get('profile-file', None)
        self.profile = cfg.get('profile', None)
        self.profile_manager = ProfileManager(self.profile_file)
        self.nic = nic
        self.archive = archive
        self.vector_element = 0

        # determine where our util/script files are located
        abspath = os.path.dirname(os.path.abspath(__file__))
        self.util_path = os.path.join(abspath, "..", "utils")

        if self.lat_sweep and self.limit_sweep :
            log.error ("Error: Can not specify both lat-sweeep and limit-sweep in the same test ")
            sys.exit(-1)
        if self.lat_sweep :
            self.lat_val_list = []
            param_args = [x.strip() for x in self.lat_sweep.split(',')]
            for x in range( 0, len(param_args)) :  
                 self.lat_val_list.append(param_args[x])
            #  Hmm, this does not work...
            #log.info ("will test the following latencies: ", str(self.lat_val_list))
            logme = "will test the following latencies: " + str(self.lat_val_list) 
            log.info (logme)
        if self.limit_sweep :
            self.limit_val_list = []
            param_args = [x.strip() for x in self.limit_sweep.split(',')]
            for x in range( 0, len(param_args)) :  
                 self.limit_val_list.append(param_args[x])
            logme = "will test the following netem limits: " + str(self.limit_val_list) 
            log.info (logme)
        if self.param_sweep :
            param_args = [x.strip() for x in self.param_sweep.split(',')]
            self.param_name = param_args[0]
            self.param_val_list = []
            self.param_vector_list = []
            if param_args[1] == 'sys-module-increment' :
                # assumes values in config file are: start, end, increment
                for x in range( int(param_args[2]),int(param_args[3])+1,int(param_args[4]) ) :
                     self.param_val_list.append(x)
            elif param_args[1] == 'sys-module-list' :
                # assumes values in config file are a list of values
                for x in range( 2, len(param_args)) :
                     self.param_val_list.append(int(param_args[x]))
            elif param_args[1] == 'sys-module-vector-list' :
                # assumes values in config file are a list of values to replace vector element N
                self.vector_element = int(param_args[2])
                for x in range( 3, len(param_args)) :  # values start at location 3
                    self.param_val_list.append(int(param_args[x]))
                log.info (f"will replace vector element %d with the list of values:" % self.vector_element)
                try:  # get default values
                    fname = "/sys/module/" + self.param_name
                    param_vector_string = open(fname, 'r').read()
                except:
                    log.info ("Error: parameter file not found", fname)
                    sys.exit(-1)
                vl = [x.strip() for x in param_vector_string.split(',')]
                #log.info ("parm vector values from host: " , vl)
                for x in range( 0, len(vl)) :
                     self.param_vector_list.append(int(vl[x]))
                #log.info ("param_vector_list: " , self.param_vector_list)

            else :
                log.info ("param sweep type not supported")
                sys.exit(0)
            print ("will run a parameter sweep of variable %s over the following values: " % self.param_name, self.param_val_list)
            #log.info ("will run a parameter sweep of variable %s over the following values: " % self.param_name, self.param_val_list)

        self.hostname = socket.gethostname()
        self.start_time = datetime.now().strftime('%Y-%m-%d:%H:%M')
        self.outdir = os.path.join(outdir, self.start_time, self.name)

        self.hostlist = hostlist
        self.hosts = list()
        if self.hostlist:
            with open(self.hostlist, mode='r') as csv_file:
                csv_reader = csv.DictReader(skip_comments(csv_file))
                for row in csv_reader:
                    opts = dict()
                    for opt in csv_host_opts:
                        opts[opt] = row.get(opt, None)
                    self.hosts.append(opts)
                    log.debug(f"Adding host {opts['hostname']} with opts: {opts}")
        else:
            opts = {'hostname': self.dst,
                    'profile': self.profile,
                    'alias': self.alias}
            self.hosts.append(opts)
            log.debug(f"Adding host {self.dst} with opts: {opts}")

        if self.enabled:
            log.info(f"Created job for item \"{name}\"")
            log.debug(f"Creating output directory for job in {self.outdir}")
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
            log.debug(self.__dict__)
        else:
            log.info(f"item \"{name}\" not enabled")

    def _src_cmd_cb(self, host, cmd, ofname, uuid):
        pass

    def _handle_opts(self, host):
        # Configure any profiles for this host
        self.profile_manager.set_profile(host)

        # TODO: clean up this pacing stuff
        if self.pacing :
            if not self.nic :
                log.error("Error: must specify NIC if using pacing option")
                sys.exit(-1)
            of = os.path.join(self.outdir, "pacing.out")
            cmd = f"{self.util_path}/set-pacing.sh %s %s  > {of}  2>&1 &" % (self.nic, self.pacing)
            log.debug(f"calling {cmd}")
            try:
                p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
            except subprocess.CalledProcessError as err:
                print('ERROR setting pacing :', err)
            log.info(f"pacing set to %s" % self.pacing)
        else:
            if self.nic : # clear any pacing setting if nic is set but pacing is not set
                of = os.path.join(self.outdir, "pacing.out")
                cmd = f"{self.util_path}/set-pacing.sh %s > {of}  2>&1 &" % (self.nic)
                log.debug(f"calling {cmd}")
                try:
                    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                except subprocess.CalledProcessError as err:
                    print('ERROR clearing pacing:', err)
                log.debug(f"pacing cleared")


    def _export_md(self, host=None):
        def create_meta_host():
            md = {
                "iter_uuids": self.iter_uuids,
                "parent": self.uuid,
                "rtt": 0
            }
            # Ping the host and get the RTT (in seconds)

            try:
                ping_host = host['hostname']
            except:
                log.error(f"ERROR: host object does not contain a hostname")
                sys.exit()
            log.debug (f"Testing ping to host: {ping_host}")
            rtt_ms = ping(ping_host, unit='ms')

            if rtt_ms is not None:
                log.info(f"RTT to {host['hostname']}: {rtt_ms:.2f} ms")
            else:
                log.info(f"ping to {host['hostname']} FAILED. skipping this host")
                rtt_ms = 0
            # store RTT in the host dict
            #host['rtt'] = int(rtt_ms)
            host['rtt'] = math.ceil(rtt_ms) # always round up so .1 does not become 0

            md.update(host)
            md.update({"profile_settings": self.profile_manager.get_profile(host)})
            md.update({"rtt":int(rtt_ms)})
            return md

        def create_meta_job():
            # get dictionary representation of this job configuration object (copy)
            md = self.__dict__.copy()
            # remove things we don't want to export in job meta
            del md["conf"]
            del md["profile_manager"]
            del md["iter_uuids"]

            # jobmeta has no parent
            md["parent"] = None
            # get some NIC stats if needed
            if self.nic:
                md["NIC"] = self.nic
                cmd = f'ifconfig {self.nic} | grep -i MTU '
                cmd += ''' | awk '{print $NF}' ''' 
                log.debug(f"calling {cmd}")
                try:
                    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                except subprocess.CalledProcessError as err:
                    print('ERROR getting MTU:', err)
                else:
                    mtu = p.stdout.decode('utf-8').rstrip()
                log.debug(f"got interface MTU: {mtu}")
                cmd = f'ethtool {self.nic} | grep -i speed ' 
                cmd += ''' | awk '{print $NF}' '''  
                log.debug(f"calling {cmd}")
                try:
                    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                except subprocess.CalledProcessError as err:
                    print('ERROR getting MTU:', err)
                else:
                    nic_speed = p.stdout.decode('utf-8').rstrip()
                log.debug(f"got NIC speed: {nic_speed}")
                md["MTU"] = mtu
                md["NIC_speed"] = nic_speed
                if self.pacing :
                    md["pacing"] = self.pacing
            return md

        if host:
            dst = host['alias'] if host['alias'] else host['hostname']
            ofname=f"{self.outdir}/{dst}-meta.json"
            md = create_meta_host()
        else:
            ofname=f"{self.outdir}/jobmeta.json"
            md = create_meta_job()
        jstr = json.dumps(md)
        if self.archive:
            ampq = AMPQSender(self.archive, "jobmeta")
            ampq.send("jobmeta", jstr)
        try:
            f = open(ofname, 'w')
            f.write(jstr)
            f.close()
            log.debug(f"Saved meta info to file {ofname}")
        except Exception as e:
            log.error(f"Could not write meta info to {ofname}: {e}")
        return

    def _start_instr(self, dst, iter, ofname_suffix, interval=0.1):
        self.stop_instr = False


        # Collect ss stats
        if self.ss:
            ofname = os.path.join(self.outdir, f"ss:{ofname_suffix}")
            params = {'interval': interval,
                      'outfile': ofname,
                      'dst': dst
                      }
            self.ss_thread = launch_ss(params, lambda: self.stop_instr)

        # Collect tcpdump capture if enabled
        if self.tcpdump:
            ofname = os.path.join(self.outdir, f"tcpdump:{ofname_suffix}")
            params = {'interval': interval,
                      'outfile': ofname,
                      'dst': dst,
                      'filter': self.tcpdump_filt
                      }
            self.td_thread = launch_tcpdump(params, lambda: self.stop_instr)

    def _stop_instr(self, dst, iter, ofname_suffix):
        self.stop_instr = True
        if self.ss:
            self.ss_thread.join()
            # convert results to JSON, use UUID of job metadata to link
            #  FIXME: needs future uuid mapping per iteration
            #  FIXME: dont hardcode path to ss_log_parser.py
            cmd = f"uuid={self.iter_uuids[iter-1]} infile=ss:{ofname_suffix} dir={self.outdir} {self.util_path}/ss_log_parser.py"
            # for summary only, add summary_only=true
            log.debug(f"converting ss results to JSON: {cmd}")
            try:
                proc = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
            except Exception as e:
                log.info(f"Error running {cmd}: {e}")
                return
            ss_json = f"{self.outdir}/ss:{ofname_suffix}.json"
            log.debug(f"send ss results from file {ss_json} to archive {self.archive}")
            ss_send_ampq(ss_json, self.archive, self.iter_uuids[iter-1])
        if self.tcpdump:
            self.td_thread.join()
            if self.tcptrace:
                ofname = f"tcpdump:{ofname_suffix}"
                try:
                    launch_tcptrace(ofname, self.outdir, dst, iter, self.archive, self.iter_uuids[iter-1])
                except Exception as e:
                    log.error(f"Failed to collect tcptrace stats for {dst}: {e}")

    def _run_host_cmd(self, host, cmd, ofname, stop):
        if host == self.hostname or host in loopbacks or not host:
            initcmd = []
        elif self.user:
            initcmd = ["ssh", "-l", self.user, "-t", "-o", "StrictHostKeyChecking=no", hostname]
        else:
            initcmd = ["ssh", "-t", "-o", "StrictHostKeyChecking=no", hostname]

        log.debug(f"run_host_cmd: Running: \"{cmd}\" on \"{self.hostname}\", with output to file \"{ofname}\"")
        rcmd = initcmd + cmd.split(" ")
        log.debug(rcmd)
        try:
            proc = subprocess.Popen(rcmd, stdout=PIPE, stderr=STDOUT)
        except Exception as e:
            log.info(f"Error running {rcmd}: {e}")
            return
        outs = None
        errs = None
        if stop:
            while not stop():
                time.sleep(1)
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except TimeoutExpired:
                proc.kill()
        else:
            outs, errs = proc.communicate()
            proc.terminate()

        time.sleep(2)

        if not ofname:
            return
        try:
            f = open(ofname, 'wb')
            if not outs:
                outs = proc.stdout.read()
            f.write(outs)
            f.close()
        except Exception as e:
            log.error(f"Could not write output for \"{ofname}\": {e}")
            return

    def run(self):
        log.info(f"Executing runs for job {self.name} and {self.iters} iterations")
        self._export_md()

        for item in self.hosts:
            dst = item.get("hostname", None)
            # First handle any options for this host
            try:
                self._handle_opts(item)
            except Exception as e:
                log.error(f"Could not handle host options for {dst}: {e}")
                continue
            # generate UUIDs for each iteration
            uuids = list()
            for it in range(0, self.iters):
                uuids.append(str(uuid.uuid4()))
            self.iter_uuids = uuids
            # also export a child jobmeta for each host
            self._export_md(item)

            cmd = self.src_cmd.format(dst=dst)

            log.info(f"Testing {self.iters} times to {dst} using \"{cmd}\"")

            if item['rtt'] == 0:
                 # ping must have failed, so skip this host
                 log.info(f"Error: ping to {dst} failed")
                 continue

            for iter in range(1, int(self.iters)+1):
                log.debug(f"Starting iteration # {iter} for host {dst}")
                if self.lat_sweep :
                    for lat in self.lat_val_list :
                        rtt=f"%s" % (float(lat) * 2)
                        ofname_suffix = f"{dst}:{iter}:{rtt}ms"
                        ofname = os.path.join(self.outdir, f"pre-netem:{dst}:{iter}:{rtt}ms")
                        pcmd = f"{self.util_path}/pre-netem.sh %sms %s %s > {ofname}  2>&1 &" % (lat, self.loss, self.limit)
                        log.info (f"Running command to set netem latency: %s" % pcmd)
                        try:
                            status = os.system(pcmd)
                            #log.debug (f"netem script return code: %s " % status)
                        except:
                            log.info ("Error setting netem, Exitting ")
                            sys.exit(-1)
                        time.sleep(5)

                        if self.param_sweep :  # if both lat_sweep and param_sweep
                            self.param_sweep_loop (dst, cmd, iter, lat, self.limit)
                        else :
                            self.subrun(dst, cmd, iter, ofname_suffix)
                elif self.limit_sweep :
                    for limit in self.limit_val_list :
                        ofname_suffix = f"{dst}:{iter}:{limit}"
                        ofname = os.path.join(self.outdir, f"pre-netem:{dst}:{iter}:{limit}")
                        pcmd = f"{self.util_path}/pre-netem.sh %s %s %s > {ofname}  2>&1 &" % (self.lat, self.loss, limit)
                        log.info (f"Running command to set netem latency: %s" % pcmd)
                        try:
                            status = os.system(pcmd)
                        except:
                            log.info ("Error setting netem, Exitting ")
                            sys.exit(-1)
                        time.sleep(5)

                        if self.param_sweep :  # if both limit_sweep and param_sweep
                            self.param_sweep_loop (dst, cmd, iter, self.lat, limit)
                        else :
                            self.subrun(dst, cmd, iter, ofname_suffix)
                elif self.lat :  # single netem lat only 
                    rtt=f"%s" % (float(self.lat) * 2)
                    ofname_suffix = f"{dst}:{iter}:{rtt}ms"
                    ofname = os.path.join(self.outdir, f"pre-netem:{dst}:{iter}:{rtt}ms")
                    pcmd = f"{self.util_path}pre-netem.sh %sms %s %s > {ofname}  2>&1 &" % (self.lat, self.loss, self.limit)
                    log.info (f"Running command to set netem latency: %s" % pcmd)
                    try:
                        status = os.system(pcmd)
                    except:
                        log.info ("Error setting netem, Exitting ")
                        sys.exit(-1)
                    time.sleep(5)
                    self.subrun(dst, cmd, iter, ofname_suffix)
                elif self.param_sweep :  # if param_sweep only
                    self.param_sweep_loop (dst, cmd, iter, self.lat, self.limit)
                else:
                    ofname_suffix = f"{dst}:{iter}"
                    self.subrun(dst, cmd, iter, ofname_suffix)
            # reset any profiles that were set as part of option handling
            self.profile_manager.clear_profile(item)


    def param_sweep_loop (self, dst, cmd, iter, lat, limit):
         for val in self.param_val_list :
                if self.vector_element > 0 : # special handling if param is a vector
                      #log.info (f"replacing value for parameter vector element number: %d" % self.vector_element)
                      loc = self.vector_element-1
                      self.param_vector_list[loc] = val
                      converted_list = [str(element) for element in self.param_vector_list]
                      vector_string = ",".join(converted_list)
                      pcmd = f"echo %s > /sys/module/%s" % (vector_string, self.param_name)
                else :
                      pcmd = f"echo %d > /sys/module/%s" % (val, self.param_name)
    
                log.info (f"Running command to set bbr2 param: %s" % pcmd)
                try:
                      os.system(pcmd)
                except:
                      log.info ("Error setting parameter value, Exitting ")
                      sys.exit(-1)
                param = self.param_name.split('/')[-1]
                if self.lat_sweep:
                      # put RTT, not lat, in the file name
                      rtt=f"%s" % (float(lat) * 2)
                      ofname_suffix = f"{dst}:{iter}:{param}:{val}:{rtt}ms"
                elif self.limit_sweep:
                      ofname_suffix = f"{dst}:{iter}:{param}:{val}:{limit}"
                else :
                      ofname_suffix = f"{dst}:{iter}:{param}:{val}"
                log.info (f"output files will have suffix: %s" % ofname_suffix)
                self.subrun(dst, cmd, iter, ofname_suffix)


    def subrun(self, dst, cmd, iter, ofname_suffix):
        # Start instrumentation (per iter)
        if self.instrument:
             self._start_instr(dst, iter, ofname_suffix)

        stop_threads = False
        jthreads = list()

        log.info (f"subrun: {self.pre_src_cmd}" )
        # handle pre-cmd invocation
        if self.pre_src_cmd:
            log.info (f"running pre_src_cmd: {self.pre_src_cmd}" )
            ofname = os.path.join(self.outdir, f"pre-src-cmd:{ofname_suffix}")
            th = Thread(target=self._run_host_cmd,
                                args=(self.src, self.pre_src_cmd, ofname, (lambda: stop_threads)))
            jthreads.append(th)
            th.start()
        if self.pre_dst_cmd:
            ofname = os.path.join(self.outdir, f"pre-dst-cmd:{ofname_suffix}")
            th = Thread(target=self._run_host_cmd,
                                args=(dst, self.pre_dst_cmd, ofname, (lambda: stop_threads)))
            jthreads.append(th)
            th.start()

        # wait for a bit
        time.sleep(2)


        # start target (dst) cmd
        if self.dst_cmd:
            ofname = os.path.join(self.outdir, f"dst-cmd:{ofname_suffix}")
            if self.statexec: 
                 prom_fname = ofname+".prom"
                 dst_cmd = f"/usr/local/bin/statexec -f {prom_fname} {self.dst_cmd}"
                 log.info (f"statexec option set. Running command: {dst_cmd}")
            else:
                 dst_cmd = self.dst_cmd
            log.debug(f"Launching dst thread on host: {dst}")
            th = Thread(target=self._run_host_cmd,
                                args=(dst, dst_cmd, ofname, (lambda: stop_threads)))
            jthreads.append(th)
            th.start()

        # wait for a bit more
        time.sleep(5)

        # finally, start and wait on src cmd (will block)
        if self.src_cmd:
           done = False
           cnt = 0
           while not done: # try up to 4 times for sucessful run
              log.info (f"Running iteration {iter}/{self.iters}, command: {cmd}")

              ofname = os.path.join(self.outdir, f"src-cmd:{ofname_suffix}")
              if self.statexec: 
                   prom_fname = ofname+".prom"
                   src_cmd = f"/usr/local/bin/statexec -f {prom_fname} {self.src_cmd}"
                   log.info (f"statexec option set. Running command: {src_cmd} \n")
              else:
                   src_cmd = self.src_cmd
              th = Thread(target=self._run_host_cmd,
                                args=(self.src, src_cmd, ofname, None))
              th.start()
              th.join()
              time.sleep(2)
              log.debug("size of results file %s is %d" % (ofname, os.path.getsize(ofname)))
              if os.path.getsize(ofname) > 1000 :
                 done = True
                 log.info (f"Test {iter} attempt {cnt} to host {dst} completed sucessfully...")
              elif cnt > 4:
                 done = True
                 log.info (f"Test {iter} attempt {cnt} to host {dst} FAILED. Giving up ...")
              else:
                 log.info (f"Test {iter} attempt {cnt} to host {dst} FAILED, trying again...")
                 try:
                     with open(ofname, "r") as file:
                         file_contents = file.read()
                         log.info(f"job output file contains: {file_contents}")
                 except:
                     log.error(f"output file '{ofname}' not found.")

                 time.sleep(2)
                 cnt += 1

        # invoke a callback with some context
        if self.src_cmd:
            self._src_cmd_cb(self.src, cmd, ofname, self.iter_uuids[iter-1])

        # wait for a bit
        time.sleep(2)
        stop_threads = True
        for th in jthreads:
            th.join()

        # These must run to completion
        if self.post_dst_cmd:
            th = Thread(target=self._run_host_cmd,
                                args=(dst, self.post_dst_cmd, None, None))
            th.start()
            th.join()

        if self.post_src_cmd:
            log.info (f"running post_src_cmd: {self.post_src_cmd}" )
            ofname = os.path.join(self.outdir, f"post-src-cmd:{ofname_suffix}")
            log.debug ("running post cmd: %s " % self.post_src_cmd)
            th = Thread(target=self._run_host_cmd,
                                args=(self.src, self.post_src_cmd, ofname, None))
            th.start()
            th.join()

        # Stop instrumentation (per iter)
        if self.instrument:
            self._stop_instr(dst, iter, ofname_suffix)
