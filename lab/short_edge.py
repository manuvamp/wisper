"""Short dictations: where do words get lost? Runs your short clips (<= 4 s) through the live settings and variants:
as recorded, quieter (soft voice / far from the mic), first 0.3 s missing (mic just opened), last 0.2 s missing
(released a bit early); each with voice detection (VAD) on and off. Prints what VAD kept and what Whisper wrote."""
import glob, os, sqlite3, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.add_dll_directory(r"D:\wisper\bin\cuda")
os.environ["PATH"] = r"D:\wisper\bin\cuda" + os.pathsep + os.environ["PATH"]
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, r"D:\wisper\service")
import numpy as np
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
from cleanup import Personal

VAD = {"min_silence_duration_ms": 500, "speech_pad_ms": 300}
m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")
prompt = Personal().asr_prompt
db = sqlite3.connect(r"D:\wisper\data\wisper.db")
rows = db.execute("select final_text, audio_file from dictations where audio_s <= 4 and audio_file is not null "
                  "and final_text != '' and coalesce(source,'live')='live'").fetchall()
clips = [(t, decode_audio(f)) for t, f in rows if os.path.exists(f)]
clips += [(os.path.basename(f), decode_audio(f)) for f in glob.glob(r"D:\wisper\lab\clips\*.webm")
          if os.path.getsize(f) < 60000][:20]
clips = [(t, a) for t, a in clips if len(a) <= 16000 * 4.5]


def run(a, vad):
    segs, info = m.transcribe(a, language="en", beam_size=1, initial_prompt=prompt, without_timestamps=True,
                              vad_filter=vad, vad_parameters=VAD if vad else None, condition_on_previous_text=False,
                              no_speech_threshold=0.6, log_prob_threshold=-1.0)
    segs = [s for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
    return " ".join(s.text.strip() for s in segs).strip()


variants = {
    "as recorded": lambda a: a,
    "quiet x0.25": lambda a: a * 0.25,
    "quiet x0.1": lambda a: a * 0.1,
    "start -0.3s": lambda a: a[int(0.3 * 16000):],
    "end -0.2s": lambda a: a[: -int(0.2 * 16000)],
}
run(clips[0][1], True)
print(len(clips), "short clips")
lost = {}
for name, f in variants.items():
    for vad in (True, False):
        empty, t0 = 0, time.perf_counter()
        for ref, a in clips:
            x = f(a).astype("float32")
            if not run(x, vad):
                empty += 1
        lost[(name, vad)] = empty
        print(f"{name:12s} VAD {'on ' if vad else 'off'}: {empty:2d}/{len(clips)} came back empty "
              f"({1000 * (time.perf_counter() - t0) / len(clips):.0f} ms/clip)", flush=True)
# how much speech does VAD find in the quiet versions?
for k in (1.0, 0.25, 0.1):
    found = [sum(s["end"] - s["start"] for s in get_speech_timestamps((a * k).astype("float32"), VadOptions(**VAD))) / 16000
             for _, a in clips]
    print(f"VAD speech found at volume x{k}: median {np.median(found):.2f}s, zero in {sum(x == 0 for x in found)} clips")
