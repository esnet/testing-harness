#!/usr/bin/env python3

import os
import sys
import csv
import json
from collections import defaultdict
from pathlib import Path

def generate_output(iperf):
    for i in iperf.get('intervals'):
        stream_id = 0
        for ii in i.get('streams'):
            ret = {"time": round(float(ii.get('start')), 2),
                   "bytes": ii.get('throughput-bytes'),
                   "gbps": round(float(ii.get('throughput-bits')) / 1e9, 3),
                   "retransmits": ii.get('retransmits'),
                   "rtt": ii.get("rtt"),
                   "end": ii.get("end"),
                   "stream_id": stream_id}
            yield ret
            stream_id += 1

def get_stats_split(dat):
    vals = generate_output(dat)
    evn = {}
    odd = {}
    skey = "summ"
    evn[skey] = [0, 0, 0, 0]
    odd[skey] = [0, 0, 0, 0]
    for v in vals:
        time = round(v["time"],1)
        sid = int(v["stream_id"])
        bytes = v["bytes"]
        bps = v["gbps"]
        retrans = v["retransmits"]
        rtt = int(v["rtt"])
        end = v["end"]
        if sid == 0:
            evn[time] = [bytes, bps, retrans, rtt]
            evn[skey][0] += bytes
            evn[skey][1] += bps
            evn[skey][2] += retrans
            evn[skey][3] += rtt
        elif sid == 1:
            odd[time] = [bytes, bps, retrans, rtt]
            odd[skey][0] += bytes
            odd[skey][1] += bps
            odd[skey][2] += retrans
            odd[skey][3] += rtt
        elif sid % 2 == 0:  # even ID > 0
            evn[time][0] += bytes
            evn[time][1] += bps
            evn[time][2] += retrans
            evn[time][3] += rtt
            evn[skey][0] += bytes
            evn[skey][1] += bps
            evn[skey][2] += retrans
            evn[skey][3] += rtt
        else: #  odd ID > 1
            odd[time][0] += bytes
            odd[time][1] += bps
            odd[time][2] += retrans
            odd[time][3] += rtt
            odd[skey][0] += bytes
            odd[skey][1] += bps
            odd[skey][2] += retrans
            odd[skey][3] += rtt
    return (round(evn[skey][1]/end, 3), round(odd[skey][1]/end, 3))

def get_stats(dat):
    vals = generate_output(dat)
    stats = {}
    skey = "summ"
    stats[skey] = [0, 0, 0, 0]
    for v in vals:
        time = round(v["time"],1)
        sid = int(v["stream_id"])
        bytes = v["bytes"]
        bps = v["gbps"]
        retrans = v["retransmits"]
        rtt = int(v["rtt"])
        end = v["end"]

        # summary stats
        stats[skey][0] += bytes
        stats[skey][1] += bps
        stats[skey][2] += retrans
        stats[skey][3] += rtt

        # interval stats
        if sid == 0:
            stats[time] = [bytes, bps, retrans, rtt]
        else:
            stats[time][0] += bytes
            stats[time][1] += bps
            stats[time][2] += retrans
            stats[time][3] += rtt
    return (round(stats[skey][1]/end, 3))

def main():
    fnames = defaultdict(list)
    res = defaultdict(list)

    p = Path(sys.argv[1])
    files = list(p.glob('**/ss*.json'))
    for pf in files:
        key = os.path.join(pf.parent, pf.stem.split(":",1)[1])
        host = pf.stem.split(":")[1]
        with pf.open() as f:
            for jobj in f:
                try:
                    dat = json.loads(jobj)
                except:
                    f.close()
                    continue
                if "streams" in dat:
                    ccs = set()
                    for s in dat["streams"]:
                        ccs.add(s["cc"])
                    for cc in ccs:
                        retrastr = f'{cc}_retrans_rate'
                        reordstr = f'{cc}_reordering'
                        obj = {"streams": dat["num_streams"],
                               "cc": cc,
                               "type": "both" if len(ccs) > 1 else cc,
                               "ss_throughput": dat["throughput"],
                               "p50_rtt": int(dat["p50_rtt"]),
                               "retransmit_rate": dat[retrastr],
                               "reorder_segs": dat[reordstr]}
                        res[host].append(obj)
                        fnames[key].append(obj)
            f.close()

    # go through iperf output to gather and validate additional info
    files = list(p.glob('**/src-cmd*'))
    out = list()
    csvout = csv.writer(sys.stdout)
    for pf in files:
        key = os.path.join(pf.parent, pf.name.split(":",1)[1])
        host = pf.name.split(":")[1]
        with pf.open() as f:
            try:
                dat = json.loads(f.read())
                if "intervals" not in dat:
                    raise
            except:
                f.close()
                continue
            
            obj = fnames[key]
            fmt = "{host},{cc},{tp},{rtt}"
            for o in obj:
                if o["type"] == "both":
                    tp1,tp2 = get_stats_split(dat)
                    tp = tp1 if o["cc"] == "cubic" else tp2
                else:
                    tp = get_stats(dat)
                out.append((host,
                            o["type"],
                            o["streams"],
                            o["cc"],
                            tp,
                            o["retransmit_rate"],
                            o["reorder_segs"],
                            o["p50_rtt"]))
            f.close()
    #output
    for r in sorted(out, key=lambda x: x[-1]):
        csvout.writerow(r)

if __name__ == "__main__":
    main()
