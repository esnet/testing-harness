#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from logging import handlers
from configparser import ConfigParser
from lib import *


log = logging.getLogger("harness")

JOB_TYPE_MAP = {
    'gridftp': GridFTP,
    'perfsonar': perfSONAR,
    'iperf3': iperf3,
    'default': Job
}


class TestingDaemon:
    def __init__(self, conf):
        self.conf = conf
        self.log_file = conf['log_file']
        self.hostlist = conf['hostlist']
        self.outdir = conf['outdir']
        self.verbose = conf['verbose']
        self.quiet = conf['quiet']
        self.nic = conf['nic']
        self.archive = conf['archive_url']
        self.jobs = list()

        # Setup logging
        form = '[%(asctime)s] [%(threadName)s] %(levelname)s: %(msg)s'
        level = logging.DEBUG if self.verbose else logging.INFO
        level = logging.CRITICAL if self.quiet else level

        if self.log_file == "stdout":
            fh = logging.StreamHandler(sys.stdout)
        else:
            fh = logging.handlers.RotatingFileHandler(
                self.log_file, maxBytes=(1024*1024*8), backupCount=7)
        fh.setFormatter(logging.Formatter(form))
        log.addHandler(fh)
        log.setLevel(level)

        # Setup output directory
        try:
            Path(self.outdir).mkdir(exist_ok=True)
        except Exception as e:
            log.error(f"Could not create result output directory \"{self.outdir}\": {e}")
            exit(1)

    def _setup(self):
        jfile = self.conf['job_file']
        parser = ConfigParser(allow_no_value=True)
        try:
            jobs = parser.read(jfile)
            sections = parser.sections()
        except Exception as e:
            log.error(f"Could not get job definition config file: {jfile}: {e}")
            return

        log.info(f"Found {len(sections)} job definitions")
        log.debug(sections)

        for s in sections:
            try:
                typ = parser[s]['type'].lower()
                jclass = JOB_TYPE_MAP.get(typ, Job)
                job = jclass(s, parser[s], self.outdir, self.hostlist, self.nic)
                self.jobs.append(job)
            except Exception as e:
                log.error(f"Could not create job from config \"{s}\": {e}")
                continue

    def start(self):
        self._setup()
        if not self.jobs:
            return

        log.info("Starting jobs [{}]".format(len(self.jobs)))
        for job in self.jobs:
            if job.enabled:
                job.run(self.archive)
            else:
                log.info(f"Skipping disabled job \"{job.name}\"")


def _read_config(fpath):
    if not fpath:
        return {}
    parser = ConfigParser(allow_no_value=True)
    try:
        parser.read(fpath)
    except Exception as e:
        raise AttributeErorr(f"Could not read harness config file: {fpath}: {e}")
    return {}


def main():
    parser = argparse.ArgumentParser(description='Network performance testing harness')
    parser.add_argument('-a', '--archive', default=None, type=str,
                        help='The complete URL of an RabbitMQ host (for ELK collection)')
    parser.add_argument('-m', '--mesh', default=None, type=str, help="URL of a pS MeshConfig (instead of MA URL)")
    parser.add_argument('-p', '--prometheus', action='store_true', help='Enable Prometheus collector')
    parser.add_argument('-l', '--log', default="stdout", help="Path to log file")
    parser.add_argument('-c', '--config', default=None, type=str, help="Path to harness configuration file")
    parser.add_argument('-i', '--interface', default=None, type=str, help="collect info on this NIC to store in jobmeta file")
    parser.add_argument('-j', '--jobs', default="jobs.ini", type=str, help="Path to job configuration file")
    parser.add_argument('-H', '--hostlist', default=None, type=str,
                        help="Path to a file containing a list of hosts to test against")
    parser.add_argument('-o', '--outdir', default=datetime.now().isoformat(),
                        type=str, help="Output directory for writing results")
    parser.add_argument('-v', '--verbose', action='store_true', help='Produce verbose output from the app')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode, no logging output')

    args = parser.parse_args()

    conf = {'job_file': args.jobs,
            'archive_url': args.archive,
            'mesh_url': args.mesh,
            'log_file': args.log,
            'outdir': args.outdir,
            'nic': args.interface,
            'hostlist': args.hostlist}
    conf.update(**_read_config(args.config))
    conf.update(**{k: v for k, v in args.__dict__.items() if v is not None})

    app = TestingDaemon(conf)
    app.start()


if __name__ == "__main__":
    main()
