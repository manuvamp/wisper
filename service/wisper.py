"""wisper: personal local dictation. One process, one model, no AI cleanup.

  Ctrl+Shift hold-to-talk (dictate.py) or POST /v1/audio/transcriptions (OpenAI-compatible)
  -> Whisper large-v3 turbo on the GPU (English, VAD, your vocabulary as a hint) -> noise/hallucination guard
  -> code-only cleanup (cleanup.py: hesitations, stutters, filler asides, your mishearings, names, punctuation)
Every dictation is logged to SQLite (+ an Opus clip) for the nightly learner (learn/learn.py).
"""
import io, json, os, re, sqlite3, sys, threading, time, logging
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {**json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8")),  # + your own overrides:
       **(json.load(open(os.path.join(ROOT, "config.local.json"), encoding="utf-8"))
          if os.path.exists(os.path.join(ROOT, "config.local.json")) else {})}
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
logging.basicConfig(filename=os.path.join(ROOT, "logs", "wisper.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wisper")

# ctranslate2's CUDA build needs cuBLAS 12 (kept in bin/cuda; importing torch for it would cost RAM).
CUDA_DIR = os.path.join(ROOT, "bin", "cuda")
os.add_dll_directory(CUDA_DIR)
os.environ["PATH"] = CUDA_DIR + os.pathsep + os.environ["PATH"]
try:  # dictation must win against background work
    import win32api, win32process
    win32process.SetPriorityClass(win32api.GetCurrentProcess(), win32process.ABOVE_NORMAL_PRIORITY_CLASS)
except Exception:
    pass
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are cached; never touch the network at startup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I mode doesn't add the script dir
from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio  # noqa: E402
import numpy as np  # noqa: E402
from cleanup import Personal, clean, looks_like_hallucination  # noqa: E402


def voiced_seconds(audio, block=800):
    """Seconds of clearly-above-background sound (50 ms blocks louder than 3x this clip's own noise floor)."""
    n = len(audio) // block
    if n < 4:
        return 0.0
    lv = np.sqrt(np.mean(audio[:n * block].reshape(n, block) ** 2, axis=1))
    floor = max(float(np.percentile(lv, 10)), 0.0005)
    return float(np.sum(lv > max(0.003, 3 * floor))) * block / 16000


class Engine:
    def __init__(self):
        self.personal = Personal()
        t0 = time.perf_counter()
        self.model = WhisperModel(CFG["asr_model"], device="cuda", compute_type=CFG["asr_compute_type"])
        self.batched = BatchedInferencePipeline(self.model)
        log.info("ASR loaded in %.1fs", time.perf_counter() - t0)
        self.gpu = threading.Lock()
        self.dblock = threading.Lock()
        self.db = sqlite3.connect(os.path.join(ROOT, CFG["log_db"]), check_same_thread=False)
        self.db.execute("""create table if not exists dictations (id integer primary key, ts text default
            (datetime('now','localtime')), audio_s real, speech_s real, asr_ms int, llm_ms int, total_ms int,
            asr_text text, rules_text text, final_text text, llm_used int, llm_rejected int, llm_text text,
            source text default 'live', audio_file text)""")
        self.db.commit()
        import ctranslate2
        self._tiny = ctranslate2.StorageView.from_array(np.zeros((1, 128, 200), "float32"))
        self.asr(np.zeros(16000, "float32"), warm=True)

    def asr(self, audio, warm=False, prefix=None):
        """16 kHz float32 numpy -> {"text", "speech_s", "ms"}; noise/hallucinations come back as "".
        prefix: what you said just before this piece (later chunks of one dictation), so the sentence continues."""
        t0 = time.perf_counter()
        self.personal.maybe_reload()
        audio_s = len(audio) / 16000
        if audio_s < 0.2:
            return {"text": "", "speech_s": 0.0, "ms": 0}
        prompt = self.personal.asr_prompt_short + " " + prefix.strip() if prefix else self.personal.asr_prompt
        opts = dict(language=CFG["language"], beam_size=CFG["asr_beam_size"], initial_prompt=prompt,
                    without_timestamps=True, no_speech_threshold=0.6, log_prob_threshold=-1.0,
                    temperature=0.0)  # no slow re-decodes when unsure: 1 in 159 clips, +1.1 s, same words
        with self.gpu:
            if audio_s > 28:  # long audio: VAD-split pieces decoded as one GPU batch
                segs, info = self.batched.transcribe(audio, batch_size=8, vad_filter=True, **opts)
            else:
                segs, info = self.model.transcribe(
                    audio, vad_filter=not warm, vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
                    condition_on_previous_text=False, **opts)
            segs = [s for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
            text = " ".join(s.text.strip() for s in segs).strip()
            speech_s = getattr(info, "duration_after_vad", audio_s) or 0.0
            sound_s = 0.0 if text or warm or audio_s > 28 else voiced_seconds(audio)
            if sound_s >= 0.3:
                # voice detection heard nothing, but the mic clearly did: short or soft dictations ("Do more.")
                # fall below its threshold. Let Whisper itself listen; the noise guard below still drops junk.
                segs, _ = self.model.transcribe(audio, vad_filter=False, condition_on_previous_text=False, **opts)
                segs = [s for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)
                        and s.compression_ratio < 2.4 and re.search(r"[A-Za-z]", s.text)]   # ". . . ." from a breath
                text, speech_s = " ".join(s.text.strip() for s in segs).strip(), sound_s
                log.info("voice detection missed %.1fs of sound; without it: %r", sound_s, text[:80])
        if not warm and looks_like_hallucination(text, speech_s):
            log.info("dropped as noise/hallucination: %r", text[:120])
            text = ""
        return {"text": text, "speech_s": speech_s, "ms": (time.perf_counter() - t0) * 1000}

    def wake_gpu(self):
        """A ~20 ms job that brings the GPU out of its idle clock while you talk, so the real pass isn't run cold
        (after ~15 s idle a pass took up to 2x longer). Skipped if a real pass is using the GPU."""
        if self.gpu.acquire(blocking=False):
            try:
                self.model.model.encode(self._tiny)
            finally:
                self.gpu.release()

    def clean_text(self, asr_text):
        t0 = time.perf_counter()
        final = clean(asr_text, self.personal) if asr_text else ""
        return {"final": final, "ms": (time.perf_counter() - t0) * 1000}

    def log_dictation(self, asr_text, final, speech_s, audio_s, asr_ms, total_ms, source="live", audio=None):
        with self.dblock:
            cur = self.db.execute(
                "insert into dictations (audio_s,speech_s,asr_ms,llm_ms,total_ms,asr_text,rules_text,final_text,"
                "llm_used,llm_rejected,source) values (?,?,?,0,?,?,?,?,0,0,?)",
                (round(audio_s, 2), round(speech_s, 2), int(asr_ms), int(total_ms), asr_text, final, final, source))
            self.db.commit()
            row_id = cur.lastrowid
        if audio is not None and (final or audio_s >= 0.5):   # empty results too: they show what got missed
            threading.Thread(target=self._save_audio, args=(row_id, audio), daemon=True).start()
        log.info("%.1fs audio -> asr %dms total %dms (%s)", audio_s, asr_ms, total_ms, source)

    def _save_audio(self, row_id, audio):
        """Keep the clip (Opus, ~3 KB/s) so the nightly learner can re-check it; pruned after 90 days."""
        try:
            import av
            d = os.path.join(ROOT, "data", "audio", time.strftime("%Y-%m"))
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{row_id}.ogg")
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            with av.open(path, "w", format="ogg") as out:
                st = out.add_stream("libopus", rate=48000)
                st.bit_rate = 24000
                st.layout = "mono"
                frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
                frame.sample_rate = 16000
                resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
                for fr in resampler.resample(frame) + resampler.resample(None):
                    for pkt in st.encode(fr):
                        out.mux(pkt)
                for pkt in st.encode(None):
                    out.mux(pkt)
            with self.dblock:
                self.db.execute("update dictations set audio_file=? where id=?", (path, row_id))
                self.db.commit()
        except Exception as e:
            log.warning("could not save audio: %s", e)

    def transcribe(self, audio_bytes, source="live"):
        """HTTP path (OpenAI-compatible clients, learner backfill)."""
        audio = decode_audio(io.BytesIO(audio_bytes))
        r = self.asr(audio)
        c = self.clean_text(r["text"])
        self.log_dictation(r["text"], c["final"], r["speech_s"], len(audio) / 16000, r["ms"], r["ms"] + c["ms"], source)
        return c["final"]


ENGINE = None
DICTATION = None


class Server(ThreadingHTTPServer):
    allow_reuse_address = False  # fail loudly if the port is taken instead of sharing it

    def server_bind(self):
        import socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._json(200, {"status": "ok"})
        if self.path == "/debug/mic" and DICTATION:  # open the mic now (it closes again after the idle timeout)
            try:
                DICTATION.mic.ensure_open()
                import sounddevice as sd
                return self._json(200, {"mic": sd.query_devices(DICTATION.mic.stream.device)["name"]})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list", "data": [{"id": "wisper", "object": "model", "owned_by": "local"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.startswith("/debug/simulate") and DICTATION:  # test: play a clip through push-to-talk
            from urllib.parse import parse_qs, urlparse
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            opts = json.loads(q["opts"]) if "opts" in q else None   # test other chunking settings
            return self._json(200, DICTATION.simulate(decode_audio(io.BytesIO(body)),
                                                      tail_silence_s=float(q.get("tail", 0.15)), opts=opts))
        if not self.path.rstrip("/").endswith("/audio/transcriptions"):
            return self._json(404, {"error": "not found"})
        try:
            msg = BytesParser(policy=HTTP).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
            fields = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                      for p in msg.iter_parts()}
            if not fields.get("file"):
                return self._json(400, {"error": "no file field"})
            source = (fields.get("source") or b"live").decode("utf-8", "replace")
            self._json(200, {"text": ENGINE.transcribe(fields["file"], source=source)})
        except Exception as e:
            log.exception("transcription failed")
            self._json(500, {"error": {"message": str(e)}})


def main():
    global ENGINE, DICTATION
    srv = Server((CFG["host"], CFG["port"]), Handler)  # bind first: exits fast if another copy is running
    ENGINE = Engine()
    if CFG.get("native_hotkey", True):
        from dictate import Dictation
        DICTATION = Dictation(ENGINE, CFG)
    log.info("wisper listening on http://%s:%d/v1", CFG["host"], CFG["port"])
    srv.serve_forever()


if __name__ == "__main__":
    main()
