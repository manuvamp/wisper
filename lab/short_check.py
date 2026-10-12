"""Live-service check for short dictations: soft versions of your short clips must come back with text, and pure
silence / background noise / a key click must still come back empty (no 'Thank you.' from nothing)."""
import glob, io, json, os, sqlite3, sys, time, urllib.request, uuid, wave
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from faster_whisper import decode_audio


def send(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n").encode() + \
        buf.getvalue() + f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"source\"\r\n\r\ntest\r\n--{b}--\r\n".encode()
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:47821/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={b}"}),
        timeout=60))
    return r["text"], (time.perf_counter() - t0) * 1000


db = sqlite3.connect(r"D:\wisper\data\wisper.db")
rows = db.execute("select final_text, audio_file from dictations where audio_s <= 4 and audio_file is not null "
                  "and final_text != '' and coalesce(source,'live')='live'").fetchall()
clips = [(t, decode_audio(f)) for t, f in rows if os.path.exists(f)]
for vol in (1.0, 0.25, 0.1):
    empty, ms = [], []
    for ref, a in clips:
        text, t = send(a * vol)
        ms.append(t)
        if not text:
            empty.append(ref)
    print(f"short clips at volume x{vol}: {len(clips) - len(empty)}/{len(clips)} transcribed "
          f"(median {np.median(ms):.0f} ms){'  MISSED: ' + str(empty) if empty else ''}")
rng = np.random.default_rng(1)
noise = {"silence 2 s": np.zeros(32000, "float32"),
         "room noise 2 s": (rng.normal(0, 0.002, 32000)).astype("float32"),
         "fan hum 3 s": (0.004 * np.sin(2 * np.pi * 120 * np.arange(48000) / 16000) + rng.normal(0, 0.001, 48000)).astype("float32"),
         "key click": np.concatenate([np.zeros(12000), rng.normal(0, 0.2, 400), np.zeros(12000)]).astype("float32"),
         "breath 1.5 s": (rng.normal(0, 0.01, 24000) * np.hanning(24000)).astype("float32")}
for name, a in noise.items():
    text, t = send(a)
    print(f"{name:15s} -> {text!r} ({t:.0f} ms)  {'OK' if not text else 'NOT EMPTY'}")
