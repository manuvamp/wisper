"""Does the GPU's idle clock (P8, ~200 MHz) slow the first pass after a pause? Times one 5 s clip cold vs warm."""
import json, os, subprocess, sys, time
sys.path.insert(0, r"D:\wisper\service")
os.add_dll_directory(r"D:\wisper\bin\cuda")
os.environ["PATH"] = r"D:\wisper\bin\cuda" + os.pathsep + os.environ["PATH"]
os.environ["HF_HUB_OFFLINE"] = "1"
from faster_whisper import WhisperModel, decode_audio
import numpy as np

m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")
ref = json.load(open(r"D:\wisper\lab\bench_out.json", encoding="utf-8"))
clip = sorted(ref.values(), key=lambda v: os.path.getsize(v["file"]))[len(ref) // 3]["file"]
audio = decode_audio(clip)[: 16000 * 5]


def clocks():
    return subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.gr", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def run(tag):
    c = clocks()
    t0 = time.perf_counter()
    segs, _ = m.transcribe(audio, language="en", beam_size=1, without_timestamps=True, condition_on_previous_text=False)
    text = " ".join(s.text for s in segs)
    print(f"{tag:22s} {1000 * (time.perf_counter() - t0):5.0f} ms   clocks before: {c}", flush=True)
    return text


run("load warm-up")
for idle in (15, 15):
    time.sleep(idle)
    run(f"after {idle}s idle")
    for i in range(3):
        run(f"  back-to-back {i + 1}")
# keep-warm trick: a tiny GPU op right when the key goes down, then the real pass ~1 s later
for _ in range(2):
    time.sleep(15)
    t0 = time.perf_counter()
    list(m.transcribe(np.zeros(8000, "float32"), language="en", beam_size=1, without_timestamps=True)[0])
    print(f"  nudge took {1000 * (time.perf_counter() - t0):.0f} ms", flush=True)
    time.sleep(1.0)
    run("after nudge + 1s")
