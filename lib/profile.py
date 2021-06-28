import os
import json
import logging
import requests
import yaml
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


log = logging.getLogger("harness")

DEF_BASE_URL = "https://localhost:5000/api/dtnaas/agent/tc"

class TrafficController(object):
    def __init__(self, url=DEF_BASE_URL, user="admin", passwd="admin"):
        self._url = url
        self._user = user
        self._passwd = passwd

    def _call(self, fn, url, data=None):
        hdrs = {"Content-type": "application/json"}
        try:
            res = fn(url,
                     data=data,
                     auth=(self._user, self._passwd),
                     headers=hdrs,
                     verify=False)
            log.debug(f"Traffic Control agent response: {res.status_code} ({res.text.strip()})")
        except Exception as e:
            log.error(f"Could not set traffic control: {e}")
            return

    def set_pacing(self, iface, dst, rate):
        iface_parts = iface.split(".")
        tagged = False
        if len(iface_parts) > 1:
            tagged = True
        ep = f"{self._url}/pacing"
        d = {"interface": iface_parts[0],
             "ip": dst,
             "maxrate": rate,
             "tagged": tagged
             }
        data = json.dumps(d)
        log.debug(f"Setting : {data}")
        self._call(requests.post, ep, data)

    def update_pacing(self, iface, dst, maxrate):
        pass

    def clear_pacing(self, iface):
        iface_parts = iface.split(".")
        ep = f"{self._url}/pacing"
        d = {"interface": iface_parts[0]}
        data = json.dumps(d)
        log.debug(f"Setting : {data}")
        self._call(requests.delete, ep, data)

class NetemHandler(object):
    def __init__(self, url=DEF_BASE_URL, user="admin", passwd="admin"):
        self._url = url
        self._user = user
        self._passwd = passwd

    def _call(self, fn, url, data=None):
        hdrs = {"Content-type": "application/json"}
        try:
            res = fn(url,
                     data=data,
                     auth=(self._user, self._passwd),
                     headers=hdrs,
                     verify=False)
            log.debug(f"Netem agent response: {res.status_code} ({res.text.strip()})")
        except Exception as e:
            log.error(f"Could not set NetemHandler profile: {e}")
            return

    def run(self, profile):
        ep = f"{self._url}/netem"
        data = json.dumps(profile)
        log.debug(f"Setting netem agent profile: {data}")
        self._call(requests.post, ep, data)

    def reset(self, profile):
        ep = f"{self._url}/netem"
        data = json.dumps(profile)
        log.debug(f"Clearing netem agent profile: {data}")
        self._call(requests.delete, ep, data)

class ProfileManager():
    def __init__(self, pfile):
        self._profiles = dict()
        self._handlers = dict()
        self._handlers["netem"] = NetemHandler()
        
        if not pfile:
            return

        with open(pfile, "r") as yfile:
            try:
                data = yaml.safe_load(yfile)
                for k,v in data.items():
                    if isinstance(v, dict):
                        if (k == "profiles"):
                            self._profiles.update(v)
            except Exception as e:
                raise Exception(f"Could not load configuration file: {entry}: {e}")
            else:
                log.debug(f"Read {len(self._profiles)} profiles from {pfile}")
            yfile.close()

    def get_profile(self, host):
        profile = host.get("profile", None)
        if not profile:
            return
        pinfo = self._profiles.get(profile, None)
        if not pinfo:
            raise Exception(f"No such profile: {profile}")
        return pinfo

    def set_profile(self, host):
        pinfo = self.get_profile(host)
        if pinfo:
            # assuming all profiles are handled by our Netam Agent for now
            self._handlers["netem"].run(pinfo)

    def clear_profile(self, host):
        pinfo = self.get_profile(host)
        if pinfo:
            self._handlers["netem"].reset(pinfo)
