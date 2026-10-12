"""Two cold starts: (1) how long reopening the mic takes after the idle close, (2) a Whisper pass after the GPU idled
15 s, with and without a tiny 'wake-up' encode ~0.5 s earlier (what pressing the hotkey could trigger)."""
import glob, os, sys, time
os.add_dll_directory(r"D:\wisper\bin\cuda")
os.environ["PATH"] = r"D:\wisper\bin\cuda" + os.pathsep + os.environ["PATH"]
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, sounddevice as sd, ctranslate2
from faster_whisper import WhisperModel, decode_audio

dev = next(h["default_input_device"] for h in sd.query_hostapis() if "WASAPI" in h["name"])
for i in range(3):
    t0 = time.perf_counter()
    st = sd.InputStream(device=dev, channels=1, samplerate=16000, dtype="float32", blocksize=800,
                        extra_settings=sd.WasapiSettings(auto_convert=True), callback=lambda *a: None)
    st.start()
    t1 = time.perf_counter()
    first = []
    st2 = None
    st.close()
    print(f"mic open #{i + 1}: {1000 * (t1 - t0):.0f} ms")

m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")
audio = decode_audio(sorted(glob.glob(r"D:\wisper\lab\clips\*.webm"), key=os.path.getsize)[20])[: 16000 * 4]
tiny = ctranslate2.StorageView.from_array(np.zeros((1, 128, 200), "float32"))


def p():
    t0 = time.perf_counter()
    list(m.transcribe(audio, language="en", beam_size=1, without_timestamps=True)[0])
    return 1000 * (time.perf_counter() - t0)


p(); p()
print(f"warm pass: {p():.0f} ms")
for trial in range(3):
    time.sleep(15)
    cold = p()
    time.sleep(15)
    t0 = time.perf_counter(); m.model.encode(tiny); nudge = 1000 * (time.perf_counter() - t0)
    time.sleep(0.5)
    woken = p()
    print(f"after 15 s idle: {cold:.0f} ms  |  with wake-up encode ({nudge:.0f} ms) 0.5 s before: {woken:.0f} ms")
