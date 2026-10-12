"""Per-pass costs on real clips (<= 12 s, like the live tail): voice detection on/off, vocabulary hint length."""
import glob, os, statistics as st, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.add_dll_directory(r"D:\wisper\bin\cuda")
os.environ["PATH"] = r"D:\wisper\bin\cuda" + os.pathsep + os.environ["PATH"]
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, r"D:\wisper\service")
from faster_whisper import WhisperModel, decode_audio
from cleanup import Personal

m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")
full = Personal().asr_prompt
short = "Notes on " + ", ".join(Personal().names[:15]) + "."
print("prompt tokens: full", len(m.hf_tokenizer.encode(" " + full).ids), "short", len(m.hf_tokenizer.encode(" " + short).ids))
clips = [a for f in sorted(glob.glob(r"D:\wisper\lab\clips\*.webm"))[:30] for a in [decode_audio(f)] if len(a) < 16000 * 12]
VAD = {"min_silence_duration_ms": 500, "speech_pad_ms": 300}


def t(audio, prompt, vad):
    t0 = time.perf_counter()
    list(m.transcribe(audio, language="en", beam_size=1, initial_prompt=prompt, without_timestamps=True,
                      vad_filter=vad, vad_parameters=VAD if vad else None, condition_on_previous_text=False)[0])
    return (time.perf_counter() - t0) * 1000


t(clips[0], full, True)
for name, prompt, vad in (("live now (full hint, VAD)", full, True), ("no VAD", full, False),
                          ("short hint", short, True), ("no hint", None, True), ("full hint, VAD again", full, True)):
    ms = [t(a, prompt, vad) for a in clips]
    print(f"{name:28s} median {st.median(ms):4.0f} ms  mean {st.mean(ms):4.0f} ms  ({len(clips)} clips)")
from faster_whisper.vad import get_speech_timestamps, VadOptions
ms = []
for a in clips:
    t0 = time.perf_counter(); get_speech_timestamps(a, VadOptions(**VAD)); ms.append((time.perf_counter() - t0) * 1000)
print(f"VAD alone median {st.median(ms):.0f} ms")
