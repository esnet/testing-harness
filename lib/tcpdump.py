import os
import sys
import socket
import time
import threading
import logging
import signal
import tempfile
import subprocess
from subprocess import PIPE, STDOUT


log = logging.getLogger("harness")


def run(cmd):
    """Runs a command in the background."""
    # Expand {sudo} field in commands as appropriate for this user.
    cmd = cmd.format(sudo='sudo' if os.geteuid() != 0 else '')

    outf = tempfile.NamedTemporaryFile(delete=False)
    errf = tempfile.NamedTemporaryFile(delete=False)
    proc = subprocess.Popen(cmd, shell=True, stdin=open(os.devnull, 'rb'),
                            stdout=outf, stderr=errf)
    proc.outf = outf
    proc.errf = errf
    log.debug(f"Running {cmd} in pid {proc.pid}")
    return proc


def tcpdump_thread(params, stop):
    outfile = params['outfile']
    dest = params['dst']
    filt = params.get('filter', None)

    # substitute destination host in filter
    filt = filt.format(dst=dest)

    if not filt:
        log.error("No tcpdump filter given for this job, skipping packet capture")
        return

    tcpdump_cmd = "{sudo} /usr/sbin/tcpdump -w {outfile} {filt}"
    tcpdump_cmd = tcpdump_cmd.format(sudo="{sudo}", outfile=outfile, filt=filt)
    proc = run(tcpdump_cmd)
    if stop:
        while not stop():
            time.sleep(1)
        kill = run("{sudo} killall -q tcpdump")
        kill.wait()
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except TimeoutExpired:
            proc.kill()


def launch_tcpdump(params, stop):
    t = threading.Thread(target=tcpdump_thread, args=(params, stop))
    t.start()
    return t
