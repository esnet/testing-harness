import os
import sys
import socket
import time
import threading
import logging
import time
import json
from subprocess import PIPE, STDOUT
from lib.ampq import AMPQSender


log = logging.getLogger("harness")

SS_PATH = "/usr/sbin/ss"
IP_MODE = socket.AF_INET
SS_INTERVAL_SECONDS = 0.5  # gather 'ss' stats each X seconds


def run(cmd, verbose=True):
    if verbose:
        print(f"running: {cmd}")
    status = os.system(cmd)
    if status != 0:
        sys.stderr.write(f"error {status} executing: {cmd}")


def ss_send_ampq(infile, archive=None, parent_job=None):
    time.sleep(1)  # to ensure file has made it to disk. Getting "file not found" error before added this.
    try:
        f = open(infile, 'r')
    except:
        log.debug(f"ERROR: ss json file {infile} not found")
        return
    if not archive:
        log.debug("ss archive not enabled")
        return

    try:
        data = [json.loads(line) for line in f]
    except Exception as e:
        log.error(f"error loading json file: {e}")
        return

    jstr = json.dumps(data)
    ampq = AMPQSender(archive, "ss_raw")
    ampq.send("ss_raw", jstr)


def ss_log_thread(params, stop):
    """Repeatedly run ss command and append log to file."""
    outfile = params['outfile']
    dest = params['dst']
    port = params.get('port', None)
    num_conns = params.get('cc', 1)
    interval = params.get('interval', SS_INTERVAL_SECONDS)

    t0 = time.time()
    t = t0
    port_cnt = num_conns
    f = open(outfile, 'w')
    f.truncate()
    f.close()

    if IP_MODE == socket.AF_INET6:
        ss_ip = f"[{dest}]"
    else:
        ss_ip = f"{dest}"

    if port:
        filt = f"dport = {port} and dst {ss_ip}"
    else:
        filt = f"dst = {ss_ip}"

    ss_cmd = f"{SS_PATH} -tinm {filt} >> {outfile}"
    log.debug(f"launching ss command: \"{ss_cmd}\"")

    while not stop():
        f = open(outfile, 'a')
        f.write('# %f\n' % (time.time(),))
        f.close()
        run(ss_cmd, verbose=False)
        t += interval
        to_sleep = t - time.time()
        if to_sleep > 0:
            time.sleep(to_sleep)


def launch_ss(params, stop):
    t = threading.Thread(target=ss_log_thread, args=(params, stop))
    t.start()
    return t
