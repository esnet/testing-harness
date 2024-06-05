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

# run mpstat for specified cores while other tests are running.
# Note: this will run a before and after tests, so cant just average all the results

log = logging.getLogger("harness")


def run(cmd):
    """Runs a command in the background."""

    outf = tempfile.NamedTemporaryFile(delete=False)
    errf = tempfile.NamedTemporaryFile(delete=False)
    proc = subprocess.Popen(cmd, shell=True, stdin=open(os.devnull, 'rb'),
                            stdout=outf, stderr=errf)
    proc.outf = outf
    proc.errf = errf
    log.debug(f"Running {cmd} in pid {proc.pid}")
    return proc

def mpstat_thread(params, stop):
    outfile = params['outfile']
    cores = params['cores']

    # XXX: assumes 60 second tests, and dont want to use killall, as then the JSON file will be corrupted
    # capturign 28 2 second intervals to make sure mpstat finishes first
    mpstat_cmd = f"mpstat -P {cores} -o JSON 2 28 > {outfile} "
    proc = run(mpstat_cmd)
    if stop:
        while not stop():
            time.sleep(1)
#        kill = run("killall -q mpstat")
#        kill.wait()
#        proc.terminate()
#        try:
#            proc.wait(timeout=1)
#        except TimeoutExpired:
#            proc.kill()


def launch_mpstat(params, stop):
    t = threading.Thread(target=mpstat_thread, args=(params, stop))
    t.start()
    return t
