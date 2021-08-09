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
from pathlib import Path
from datetime import datetime
from subprocess import PIPE, STDOUT
from threading import Thread
from lib.ss import launch_ss
from lib.ss import ss_send_ampq
from lib.tcpdump import launch_tcpdump
from lib.tcptrace import launch_tcptrace
from lib.ampq import AMPQSender
from lib.profile import ProfileManager, TrafficController

from model import SEEDEVERYTHING
from model import DATA
from model import RECEIVEFEATURES
from model import PACINGDATASET
from model import PACINGCLASSIFIER

loopbacks = ["localhost", "127.0.0.1", "::1"]
csv_host_opts = ["hostname", "alias", "profile"]
default_opts = ["profile-control-url", "traffic-control-url"]
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
    def __init__(self, name, cfg, defaults, outdir, hostlist, nic, archive):
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
        self.src_cmd = cfg.get('src-cmd', None)
        self.dst_cmd = cfg.get('dst-cmd', None)
        self.src_cmd_once = cfg.getboolean('src-cmd-once', False)
        self.dst_cmd_once = cfg.getboolean('dst-cmd-once', False)
        self.pre_src_cmd = cfg.get('pre-src-cmd', None)
        self.pre_dst_cmd = cfg.get('pre-dst-cmd', None)
        self.post_src_cmd = cfg.get('post-src-cmd', None)
        self.post_dst_cmd = cfg.get('post-dst-cmd', None)
        self.ss = cfg.getboolean('ss', True)
        self.tcptrace = cfg.getboolean('tcptrace', False)
        self.tcpdump = cfg.getboolean('tcpdump', False)
        self.tcpdump_filt = cfg.get('tcpdump-filt', None)
        self.iter_uuids = list()
        self.param_sweep = cfg.get('param-sweep', None)
        self.loss = cfg.get('netem-loss', None)
        self.lat = cfg.get('netem-lat', None)
        self.limit = cfg.get('netem-limit', None)
        self.pacing = cfg.getlist('pacing', list())
        self.lat_sweep = cfg.getlist('lat-sweep', list())
        self.limit_sweep = cfg.getlist('limit-sweep', list())
        self.profile_file = cfg.get('profile-file', None)
        self.profile = cfg.get('profile', None)
        self.profile_control_url = cfg.get('profile-control-url', None)
        self.traffic_control_url = cfg.get('traffic-control-url', None)
        self.nic = nic
        self.archive = archive
        self.vector_element = 0

        # support a "default" config section
        if defaults:
            for d in default_opts:
                if not getattr(self, d.replace("-", "_")):
                    setattr(self, d.replace("-", "_"), defaults.get(d, None))

        # Instantiate manager classes
        self.profile_manager = ProfileManager(self.profile_file, self.profile_control_url)
        self.tc = TrafficController(self.traffic_control_url)

        if self.pacing and not self.nic:
            raise Exception("Invalid job options: selecting pacing requires interface specification, see harness options")

        # determine where our util/script files are located
        abspath = os.path.dirname(os.path.abspath(__file__))
        self.util_path = os.path.join(abspath, "..", "utils")

        if self.lat_sweep and self.limit_sweep :
            log.error ("Error: Can not specify both lat-sweeep and limit-sweep in the same test ")
            sys.exit(-1)
        if self.lat_sweep:
            log.info (f"will test the following latencies: {self.lat_sweep}")
        if self.limit_sweep:
            log.info (f"will test the following limits: {self.limit_sweep}")
        if self.pacing:
            log.info (f"will test the following pacing rates: {self.pacing}")

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
        try:
            # First reset any profiles that were set as part of option handling
            self.profile_manager.clear_profile(host)
        except:
            # ignore any exceptions that may occur from a clear
            pass
        # Configure any profiles for this host
        self.profile_manager.set_profile(host)

    def _export_md(self, host=None, extra=None):
        def create_meta_host():
            md = {
                "iter_uuids": self.iter_uuids,
                "parent": self.uuid
            }
            md.update(host)
            md.update({"profile_settings": self.profile_manager.get_profile(host)})
            if extra:
                md.update(extra)
            return md

        def create_meta_job():
            # get dictionary representation of this job configuration object (copy)
            md = self.__dict__.copy()
            # remove things we don't want to export in job meta
            del md["conf"]
            del md["profile_manager"]
            del md["tc"]
            del md["iter_uuids"]

            # jobmeta has no parent
            md["parent"] = None
            # get some NIC stats if needed
            if self.nic:
                md["NIC"] = self.nic
                cmd = f"cat /sys/class/net/{self.nic}/mtu"
                log.debug(f"calling {cmd}")
                try:
                    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                except subprocess.CalledProcessError as err:
                    print('ERROR getting MTU:', err)
                else:
                    mtu = p.stdout.decode('utf-8').rstrip()
                log.debug(f"got interface MTU: {mtu}")
                cmd = f"cat /sys/class/net/{self.nic}/speed"
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
            return md

        if host:
            dst = host['alias'] if host['alias'] else host['hostname']
            if extra and isinstance(extra, dict):
                for k,v in extra.items():
                    dst += f"-{k}:{v}"
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
                    launch_tcptrace(ofname, self.outdir, dst, self.archive, self.iter_uuids[iter-1])
                except Exception as e:
                    log.error(f"Failed to collect tcptrace stats for {dst}: {e}")

    def _run_host_cmd(self, host, cmd, ofname, stop):
        log.debug(f"Running \"{cmd}\" on \"{host}\", with output to file \"{ofname}\"")
        if host == self.hostname or host in loopbacks or not host:
            initcmd = []
        elif self.user:
            initcmd = ["ssh", "-l", self.user, "-t", "-o", "StrictHostKeyChecking=no", host]
        else:
            initcmd = ["ssh", "-t", "-o", "StrictHostKeyChecking=no", host]

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
            # When host:None, cmd: iperf3, ofname: None, stop: False
            if not outs:
                outs = proc.stdout.read()
            return outs

        try:
            f = open(ofname, 'wb')
            if not outs:
                outs = proc.stdout.read()
            f.write(outs)
            f.close()
        except Exception as e:
            log.error(f"Could not write output for \"{ofname}\": {e}")
            return

    def _run_iters(self, dst, cmd, teststr):
        # generate UUIDs for each iteration
        uuids = list()
        for it in range(0, self.iters):
            uuids.append(str(uuid.uuid4()))
        self.iter_uuids = uuids

        for iter in range(1, int(self.iters)+1):
            ofname_suffix = f"{dst}:{teststr}:{iter}"
            log.info(f"Testing to {dst} using \"{cmd}\", iter {iter}")
            self.subrun(dst, cmd, iter, ofname_suffix)

    def run(self):
        log.info(f"Executing runs for job {self.name} and {self.iters} iterations")
        self._export_md()

        key = 'dynamic'

        for item in self.hosts:
            dst = item.get("hostname", None)

            # First handle any options for this host
            try:
                self._handle_opts(item)
            except Exception as e:
                log.error(f"Could not handle host options for {dst}: {e}")
                continue

            # format src command
            cmd = self.src_cmd.format(dst=dst)

            # first ping the host to make sure its up
            png = f'ping -W 5 -c 2 {dst} > /dev/null'
            status = os.system(png)
            if status: # ping failed, skip
                log.info(f"Error: ping to {dst} failed, error code: \"{status}\"")
                continue

            if key in self.pacing:
                # Let the model predict the pacing time for us
                print("\nDetected dynamic pacing")
                res = self._run_host_cmd(None, cmd, None, False)
                try:
                    harnessInput_dict = json.loads(res)
                    harnessInput_frmt_dict = json.dumps(harnessInput_dict, indent=4)

                    host = harnessInput_dict['start']['connecting_to']['host']
                    streams = harnessInput_dict['start']['test_start']['num_streams']

                    throughput = harnessInput_dict['end']['sum_sent']['bits_per_second']
                    retransmits = harnessInput_dict['end']['sum_sent']['retransmits']
                    
                    cc_type = harnessInput_dict['end']['sender_tcp_congestion']
                    
                    min_rtt = harnessInput_dict['end']['streams']['sender']['min_rtt']
                    max_rtt = harnessInput_dict['end']['streams']['sender']['max_rtt']
                    mean_rtt = harnessInput_dict['end']['streams']['sender']['mean_rtt']
                    
                    bytes_ = harnessInput_dict['end']['sum_sent']['bytes']

                    # print(harnessInput_frmt_dict)
                    print(f"host:{host}\nstreams:{streams}\nthroughput:{throughput}\nmin_rtt:{min_rtt}\nmax_rtt:{max_rtt}\nmean_rtt:{mean_rtt}\nretransmits:{retransmits}\ncc_type:{cc_type}\nbytes:{bytes_}")
                except Exception as e:
                    print(e)

            bufferData = [host, streams, throughput, min_rtt, max_rtt, retransmits, cc_type]
            getPacingRate(bufferData, phase='test')


            # XXX: need a generalize method to expand sweep options and collect md for each
            for pace in self.pacing:
                try:
                    # allow clear pacing to fail
                    self.tc.clear_pacing(self.nic)
                except:
                    pass
                try:
                    # but not set pacing
                    self.tc.set_pacing(self.nic, dst, pace)
                except Exception as e:
                    log.error(f"Could not set pacing: {e}")
                    continue
                log.info(f"Set pacing to {pace}")
                self._run_iters(dst, cmd, f"pacing:{pace}")
                # export a child jobmeta for each new "job" defined by a sweep parameter
                self._export_md(item, {"pacing": pace})

            # reset any profiles that were set as part of option handling
            self.profile_manager.clear_profile(item)

    def subrun(self, dst, cmd, iter, ofname_suffix):
        # Start instrumentation (per iter)
        if self.instrument:
             self._start_instr(dst, iter, ofname_suffix)

        stop_threads = False
        jthreads = list()

        # handle pre-cmd invocation
        if self.pre_src_cmd:
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
            log.debug(f"Launching dst thread on host: {dst}")
            th = Thread(target=self._run_host_cmd,
                                args=(dst, self.dst_cmd, ofname, (lambda: stop_threads)))
            jthreads.append(th)
            th.start()

        # wait for a bit
        time.sleep(5)

        # finally, start and wait on src cmd (will block)
        if self.src_cmd:
           done = False
           cnt = 0
           while not done: # try up to 4 times for sucessful run
              log.debug ("Starting test...")
              ofname = os.path.join(self.outdir, f"src-cmd:{ofname_suffix}")
              th = Thread(target=self._run_host_cmd,
                                args=(self.src, cmd, ofname, None))
              th.start()
              th.join()
              log.debug("size of results file %s is %d" % (ofname, os.path.getsize(ofname)))
              if os.path.getsize(ofname) > 1000 or cnt > 4:
                 done = True
              else:
                 log.info ("Test failed, trying again...")
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
            ofname = os.path.join(self.outdir, f"post-src-cmd:{ofname_suffix}")
            log.debug ("running post cmd: %s " % self.post_src_cmd)
            th = Thread(target=self._run_host_cmd,
                                args=(self.src, self.post_src_cmd, ofname, None))
            th.start()
            th.join()

        # Stop instrumentation (per iter)
        if self.instrument:
            self._stop_instr(dst, iter, ofname_suffix)
