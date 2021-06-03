import json
import logging
from lib.ampq import AMPQSender

from .job import Job


log = logging.getLogger("harness")

class iperf3(Job):
    def run(self):
        super().run()

    def _src_cmd_cb(self, host, cmd, ofname, uuid):
        if "iperf" in cmd and self.archive:
            try:
                f = open(ofname, "r")
                outs = f.read()
                jobj = json.loads(outs)
                jobj["uuid"] = uuid
                ampq = AMPQSender(self.archive, "iperf3")
                ampq.send("iperf3", json.dumps(jobj))
                f.close()
            except Exception as e:
                log.error(f"Could not generate iperf3 output for ampq send: {e}")
