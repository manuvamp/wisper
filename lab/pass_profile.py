"""Where does one ~0.45 s Whisper pass go? features (CPU) vs encoder (GPU) vs decoding, per compute type."""
import os, sys, time
CUDA_DIR = os.environ.get("CUDA_DIR", r"D:\wisper\bin\cuda")
os.add_dll_directory(CUDA_DIR)
os.environ["PATH"] = CUDA_DIR + os.pathsep + os.environ["PATH"]
os.environ["HF_HUB_OFFLINE"] = "1"
import ctranslate2, numpy as np
from faster_whisper import WhisperModel
print("cuBLAS from", CUDA_DIR)
print("ctranslate2", ctranslate2.__version__, "cuda devices", ctranslate2.get_cuda_device_count(),
      "types", ctranslate2.get_supported_compute_types("cuda"))

audio = np.random.default_rng(0).normal(0, 0.05, 16000 * 5).astype("float32")
for ct in sys.argv[1:] or ["float16"]:
    try:
        m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type=ct)
    except Exception as e:
        print(ct, "failed:", e); continue
    fe = m.feature_extractor
    for _ in range(3):
        t0 = time.perf_counter(); feats = fe(audio); t1 = time.perf_counter()
    for chunk in (None, 5, 10):
        f = feats[:, : fe.nb_max_frames] if chunk is None else feats[:, : chunk * 100]
        f = np.pad(f, ((0, 0), (0, 3000 - f.shape[1]))) if chunk is None and f.shape[1] < 3000 else f
        x = ctranslate2.StorageView.from_array(np.ascontiguousarray(f[None]).astype("float32"))
        try:
            for _ in range(3):
                t2 = time.perf_counter(); enc = m.model.encode(x, to_cpu=True); t3 = time.perf_counter()
            print(f"{ct:14s} features {1000*(t1-t0):4.0f} ms | encoder {f.shape[1]/100:4.0f}s window {1000*(t3-t2):4.0f} ms")
        except Exception as e:
            print(f"{ct:14s} encoder {chunk}s window failed: {str(e)[:80]}")
    for _ in range(3):
        t0 = time.perf_counter()
        list(m.transcribe(audio, language="en", beam_size=1, without_timestamps=True, condition_on_previous_text=False)[0])
        t1 = time.perf_counter()
    print(f"{ct:14s} full transcribe 5s noise {1000*(t1-t0):.0f} ms")
    del m
