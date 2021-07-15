import os
import sys
import re
import logging
import subprocess
import json
import shutil
from subprocess import PIPE, STDOUT
from lib.ampq import AMPQSender

log = logging.getLogger("harness")

MIN_PKTS = 100


def run(cmd):
    """Runs a command in the background."""
    # Expand {sudo} field in commands as appropriate for this user.
    cmd = cmd.format(sudo='sudo' if os.geteuid() != 0 else '')
    proc = subprocess.Popen(cmd, shell=True, stdin=open(os.devnull, 'rb'),
                            stdout=PIPE, stderr=STDOUT)
    log.debug(f"Running {cmd} in pid {proc.pid}")
    return proc


def send_ampq(data, archive=None):
    if not archive:
        log.debug("tcptrace archive not enabled")
        return
    jstr = json.dumps(data)
    ampq = AMPQSender(archive, "tcptrace_raw")
    ampq.send("tcptrace_raw", jstr)


def launch_tcptrace(fname, outdir, host, iter, archive=None, parent_job=None):
    infile = f"{outdir}/{fname}"
    ofname = os.path.join(outdir, f"tcptrace:{host}:{iter}.out") # capture output of tcptrace command in a file too
    tcptrace_cmd = f"/usr/bin/tcptrace -Slr --output_dir={outdir} --output_prefix={host}. {infile}  > {ofname}  2>&1"
    log.debug(f"calling {tcptrace_cmd}")
    proc = run(tcptrace_cmd)
    outs, errs = proc.communicate()
    conns = list()
    conn = dict()
    conn["job"] = parent_job
    for line in outs.decode('utf-8').splitlines():
        l = line.strip()
        l = re.sub(" +", " ", l)
        if not len(l):
            continue
        if l.startswith("TCP connection"):
            conn["id"] = l.split(" ")[2].split(":")[0]
        if l.startswith("complete conn"):
            conn["complete_conn"] = l.split(":")[1].strip()
        if l.startswith("first packet"):
            conn["first_packet"] = l.split(":", 1)[1].strip()
        if l.startswith("last packet"):
            conn["last_packet"] = l.split(":", 1)[1].strip()
        if l.startswith("elapsed time"):
            conn["elapsed_time"] = l.split(":", 1)[1].strip()
        if l.startswith("filename"):
            conn["filename"] = l.split(":", 1)[1].strip()
        if l.startswith("========"):
            if int(conn["total_packets"]) > MIN_PKTS:
                conns.append(conn)
            conn = dict()
            conn["job"] = parent_job
        if l.startswith("host"):
            host, port = l.split(" ")[2].split(":")
            if "src_host" in conn:
                conn["dst_host"] = host
                conn["dst_port"] = port
            else:
                conn["src_host"] = host
                conn["src_port"] = port
        if l.startswith("total packets") and not "total_packets" in conn:
            conn["total_packets"] = l.split(":")[1].strip()
        # begin two-column parse
        if "->" in l:
            cA, cB, blah = l.split(":")
            cB = cB.split(":")[0].strip()
            conn["s->d"] = dict()
            conn["d->s"] = dict()

        elif "s->d" in conn and "d->s" in conn:
            try:
                parts = l.split(":")
                k0 = parts[0]
                k0 = re.sub(" ", "_", k0)
                v0, k1 = parts[1].strip().split(" ", 1)
                k1 = re.sub(" ", "_", k1)
                v1 = parts[2].strip()
                c = conn["s->d"]
                c[k0] = v0.split(" ")[0]
                c = conn["d->s"]
                c[k1] = v1.split(" ")[0]
            except:
                pass

    # save last conn parsed
    if "id" in conn and int(conn["total_packets"]) > MIN_PKTS:
        conns.append(conn)
    send_ampq(conns, archive)
    _finalize(fname, outdir)


def _finalize(fname, outdir):
    # after tcptrace, compress dump file
    gzip_exe = shutil.which("gzip")
    cmd = f"{gzip_exe} {outdir}/{fname}"
    log.debug(f"compressing dump files: {cmd}")
    try:
        proc = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    except Exception as e:
        log.error(f"Error running {cmd}: {e}")

    # also compress xpl files too
    cmd = f"{gzip_exe} {outdir}/*.xpl"
    log.debug(f"compressing xplot files: {cmd}")
    try:
        proc = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    except Exception as e:
        log.error(f"Error running {cmd}: {e}")
        return


def main():
    trace_path = os.environ['infile']
    launch_tcptrace(trace_path, 'ezra.duckdns.org')


if __name__ == '__main__':
    sys.exit(main())
