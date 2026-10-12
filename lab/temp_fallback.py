"""Whisper retries a pass at higher temperatures when it looks unsure (low log-prob or repetitive). How often does
that happen on your clips, what does it cost, and does it change the words? Default retries vs temperature 0 only."""
import difflib, glob, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.add_dll_directory(r"D:\wisper\bin\cuda")
os.environ["PATH"] = r"D:\wisper\bin\cuda" + os.pathsep + os.environ["PATH"]
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, r"D:\wisper\service")
from faster_whisper import WhisperModel, decode_audio
from cleanup import Personal

m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")
prompt = Personal().asr_prompt
files = sorted(glob.glob(r"D:\wisper\lab\clips\*.webm")) + sorted(glob.glob(r"D:\wisper\data\audio\*\*.ogg"))
clips = [(f, a) for f in files for a in [decode_audio(f)] if 0.5 < len(a) / 16000 <= 28]


def run(a, temp):
    kw = {} if temp is None else {"temperature": temp}
    t0 = time.perf_counter()
    segs, _ = m.transcribe(a, language="en", beam_size=1, initial_prompt=prompt, without_timestamps=True,
                           no_speech_threshold=0.6, log_prob_threshold=-1.0, vad_filter=True,
                           vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
                           condition_on_previous_text=False, **kw)
    segs = list(segs)
    return " ".join(s.text.strip() for s in segs).strip(), (time.perf_counter() - t0) * 1000, \
        [round(s.temperature, 1) for s in segs]


words = lambda t: re.findall(r"[a-z0-9']+", t.lower())
run(clips[0][1], None)
retried, diffs, slow = 0, 0, []
t_def = t_zero = 0.0
for f, a in clips:
    d, ms_d, temps = run(a, None)
    z, ms_z, _ = run(a, 0.0)
    t_def += ms_d; t_zero += ms_z
    if any(t > 0 for t in temps):
        retried += 1
        sm = difflib.SequenceMatcher(None, words(z), words(d), autojunk=False)
        e = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        diffs += e
        slow.append((ms_d, ms_z))
        print(f"retried ({ms_d:.0f} ms vs {ms_z:.0f} ms at temp 0, {e} words differ)\n   with retries: {d[:150]}\n   temp 0 only : {z[:150]}")
print(f"\n{len(clips)} clips: {retried} needed retries; total time {t_def / 1000:.1f}s with retries vs {t_zero / 1000:.1f}s "
      f"temp 0 only; words that differ in those clips: {diffs}")
