"""Does the start beep (picked up by the mic) change what Whisper writes? Same clips with and without the beep mixed
in at the moment it plays (0.4 s pre-roll + 0.12 s), at a loud pickup level. Uses the live service, source=test."""
import glob, io, json, os, sys, urllib.request, uuid, wave
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\wisper\service")
import numpy as np
from faster_whisper import decode_audio
from dictate import make_beep

with wave.open(make_beep(r"D:\wisper\data\beep_test.wav"), "rb") as w:
    b44 = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype("float32") / 32767
beep = np.interp(np.arange(0, len(b44), 44100 / 16000), np.arange(len(b44)), b44).astype("float32")
os.remove(r"D:\wisper\data\beep_test.wav")


def send(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n").encode() + \
        buf.getvalue() + f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"source\"\r\n\r\ntest\r\n--{b}--\r\n".encode()
    req = urllib.request.Request("http://127.0.0.1:47821/v1/audio/transcriptions", body,
                                 {"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.load(urllib.request.urlopen(req, timeout=60))["text"]


files = sorted(glob.glob(r"D:\wisper\lab\clips\*.webm"), key=os.path.getsize)[::4][:15]
same = 0
for f in files:
    a = decode_audio(f)
    pre = np.zeros(int(0.52 * 16000), "float32")
    plain = np.concatenate([pre, a])
    beeped = plain.copy()
    beeped[int(0.52 * 16000) - len(beep) // 2:][:len(beep)] += beep * 0.5   # loud pickup: half the played level
    t1, t2 = send(plain), send(beeped)
    same += t1 == t2
    print("SAME" if t1 == t2 else f"DIFF\n   without: {t1}\n   with   : {t2}")
print(f"{same}/{len(files)} identical with the beep in the recording")
