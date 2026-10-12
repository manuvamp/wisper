"""Chunking settings: speed at release vs accuracy. Each clip is played in real time through the live push-to-talk
path and released right on the last word (tail 0.05 s, the hard case). Accuracy = words that differ from one
full-length pass over the whole clip (what you'd get with no chunking at all).

usage: python chunk_test.py [n_clips]
"""
import difflib, json, os, random, re, statistics as st, sys, urllib.parse, urllib.request, uuid
sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import decode_audio

CONFIGS = {
    "now (5 s chunks)": {},
    "3 s chunks": {"chunk_min": 3.0, "pause": 0.25},
    "3 s + context": {"chunk_min": 3.0, "pause": 0.25, "context": True},
    "reuse running pass": {"reuse_running": True},
    "all three": {"chunk_min": 3.0, "pause": 0.25, "context": True, "reuse_running": True},
}
URL = "http://127.0.0.1:47821"


def post(path, data, ctype):
    req = urllib.request.Request(URL + path, data, {"Content-Type": ctype})
    return json.load(urllib.request.urlopen(req, timeout=300))


def full_pass(path):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.webm\"\r\n\r\n").encode() + \
        open(path, "rb").read() + f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"source\"\r\n\r\ntest\r\n--{b}--\r\n".encode()
    return post("/v1/audio/transcriptions", body, f"multipart/form-data; boundary={b}")["text"]


def words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def wdiff(a, b):
    sm = difflib.SequenceMatcher(None, words(a), words(b), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


files = sorted(f for f in (os.path.join(r"D:\wisper\lab\clips", x) for x in os.listdir(r"D:\wisper\lab\clips"))
               if 6 <= len(decode_audio(f)) / 16000 <= 28)
random.seed(7)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
files = random.sample(files, min(n, len(files)))
res = {k: [] for k in CONFIGS}
total_words = 0
for i, f in enumerate(files, 1):
    ref = full_pass(f)
    total_words += len(words(ref))
    secs = len(decode_audio(f)) / 16000
    for name, opts in CONFIGS.items():
        q = "/debug/simulate?tail=0.05&opts=" + urllib.parse.quote(json.dumps(opts))
        r = post(q, open(f, "rb").read(), "application/octet-stream")
        res[name].append({"clip": os.path.basename(f), "secs": round(secs, 1), "ms": r["release_to_text_ms"],
                          "chunks": r["chunks"], "hit": r["speculation_hit"], "wdiff": wdiff(ref, r["text"]),
                          "exact": ref == r["text"], "ref": ref, "text": r["text"]})
    print(f"{i}/{len(files)} {secs:4.1f}s  " + "  ".join(f"{k.split(' (')[0]}: {res[k][-1]['ms']}ms/{res[k][-1]['wdiff']}w"
                                                    for k in CONFIGS), flush=True)
json.dump(res, open(r"D:\wisper\lab\chunk_test.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\n{len(files)} clips of 6-28 s, {total_words} words, released right on the last word")
for name, rows in res.items():
    ms = sorted(r["ms"] for r in rows)
    print(f"{name:20s} median {st.median(ms):4.0f} ms  p90 {ms[int(.9 * (len(ms) - 1))]:4.0f}  max {ms[-1]:4d}  | "
          f"words differing {sum(r['wdiff'] for r in rows):3d}/{total_words}  identical texts {sum(r['exact'] for r in rows)}/{len(rows)}")
