"""Short encoder window: same words as the normal 30 s padded pass? And how much faster?
faster-whisper always pads features to 30 s (pad_or_trim, 3000 frames); the encoder costs ~420 ms for that but ~60 ms
for 5 s. Here the pad target becomes audio length + margin, and every clip is compared word by word with the
normal pass, using the same options as the live service.

usage: python shortwin.py [margins in seconds...]   (default: 1 2 4)
"""
import difflib, glob, json, math, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.add_dll_directory(r"D:\wisper\bin\cuda")
os.environ["PATH"] = r"D:\wisper\bin\cuda" + os.pathsep + os.environ["PATH"]
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, r"D:\wisper\service")
import faster_whisper.transcribe as T
from faster_whisper import WhisperModel, decode_audio
from cleanup import Personal

orig = T.pad_or_trim
MARGIN = [None]


def short_pad(array, length=3000, *, axis=-1):
    if MARGIN[0] is None:
        return orig(array, length, axis=axis)
    n = array.shape[axis]
    target = min(length, int(math.ceil((n + MARGIN[0] * 100) / 100.0)) * 100)
    return orig(array, target, axis=axis)


T.pad_or_trim = short_pad
m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")
prompt = Personal().asr_prompt
files = sorted(glob.glob(r"D:\wisper\lab\clips\*.webm")) + sorted(glob.glob(r"D:\wisper\data\audio\*\*.ogg"))
clips = [(f, a) for f in files for a in [decode_audio(f)] if 0.5 < len(a) / 16000 <= 28]


def run(audio):
    t0 = time.perf_counter()
    segs, _ = m.transcribe(audio, language="en", beam_size=1, initial_prompt=prompt, without_timestamps=True,
                           no_speech_threshold=0.6, log_prob_threshold=-1.0, vad_filter=True,
                           vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
                           condition_on_previous_text=False)
    text = " ".join(s.text.strip() for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)).strip()
    return text, (time.perf_counter() - t0) * 1000


def words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


run(clips[0][1])
margins = [float(x) for x in sys.argv[1:]] or [1.0, 2.0, 4.0]
ref, ref_ms = {}, []
for f, a in clips:
    MARGIN[0] = None
    ref[f], ms = run(a)
    ref_ms.append(ms)
total_words = sum(len(words(t)) for t in ref.values())
print(f"{len(clips)} clips, {total_words} words; normal 30 s window: median {sorted(ref_ms)[len(ref_ms)//2]:.0f} ms")
for mg in margins:
    MARGIN[0] = mg
    errs, ms_all, changed = 0, [], []
    for f, a in clips:
        t, ms = run(a)
        ms_all.append(ms)
        sm = difflib.SequenceMatcher(None, words(ref[f]), words(t), autojunk=False)
        e = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        if e:
            errs += e
            changed.append((os.path.basename(f), ref[f], t))
    print(f"margin {mg:.1f}s: median {sorted(ms_all)[len(ms_all)//2]:.0f} ms, words differing {errs}/{total_words} "
          f"({100*errs/total_words:.1f}%), clips changed {len(changed)}/{len(clips)}")
    for name, a, b in changed[:12]:
        print(f"   {name}\n     30s : {a[:220]}\n     short: {b[:220]}")
