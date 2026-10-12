"""Native path only, same 20 clips as side_by_side.py, with speculation diagnostics for misses."""
import json, os, random, statistics, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
ref = json.load(open(r"D:\wisper\lab\bench_out.json", encoding="utf-8"))
clips = sorted(ref.values(), key=lambda v: os.path.getsize(v["file"]))
random.seed(3)
clips = sorted(random.sample(clips[:-1], 20), key=lambda v: os.path.getsize(v["file"]))[::2]
ms = []
hits = 0
for v in clips:
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:47821/debug/simulate?tail=" + sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:47821/debug/simulate", open(v["file"], "rb").read(),
        {"Content-Type": "application/octet-stream"}), timeout=300))
    ms.append(r["release_to_text_ms"]); hits += r["speculation_hit"]
    print(r["release_to_text_ms"], "HIT" if r["speculation_hit"] else "miss", r["chunks"], json.dumps(r["diag"]), flush=True)
v = sorted(ms)
print(f"median {statistics.median(v):.0f} ms, p90 {v[int(.9 * len(v)) - 1]:.0f}, max {v[-1]}, ready at release {hits}/{len(v)}")
