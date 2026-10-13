"""wisper on your phone: the wisper-keys keyboards (Android + iOS) stream their mic here over the network.

  ws://<this PC>:47822/v1/stream          token in "Authorization: Bearer ..." or ?token=...
  http://<this PC>:47823/v1/audio/transcriptions   same token as the API key: any OpenAI-compatible phone app
                                                   (or an iOS Shortcut) can use it, one file per request

Reach it on your Wi-Fi (LAN address) or from anywhere with Tailscale (100.x address); `tools/pair.py` shows both
plus the token as a QR code for the phone. The phone sends 16 kHz mono 16-bit PCM while you hold its mic key. That
audio feeds the same Session as Ctrl+Shift (chunks committed at pauses while you talk, an early pass on the tail),
so release -> text is as fast as on the PC plus one network round trip. Same model, vocabulary and cleanup;
dictations are logged with source "phone" so the nightly learner learns from them too.

Messages (text frames are JSON, binary frames are audio):
  server: {"type": "ready"}
  phone:  {"type": "start", "id": 7, "before": "<text before the cursor>"}, then binary PCM frames
  server: {"type": "partial", "id": 7, "text": "..."}   finished pieces so far, while you talk
  phone:  {"type": "stop", "id": 7}  ->  server: {"type": "final", "id": 7, "text": "...", "ms": <PC time>}
  phone:  {"type": "cancel", "id": 7}
  phone:  {"type": "ping", "t": ...} ->  server: {"type": "pong", "t": ...}
"""
import hmac, json, logging, os, re, secrets, socket, threading, time
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from cleanup import is_common
from dictate import BLOCK, SR, Session

log = logging.getLogger("wisper")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_FILE = os.path.join(ROOT, "data", "remote-token.txt")
GIST_FILE = os.path.join(ROOT, "data", "remote-gist.txt")
TUNNEL_FILE = os.path.join(ROOT, "data", "remote-tunnel.json")    # current public address, for tools/pair.py
CLOUDFLARED = os.path.join(ROOT, "bin", "cloudflared.exe")
MIN_AUDIO_S = 0.3


def _kill_with_me_job():
    """A Windows job that kills its processes when wisper exits (crash, restart, logoff): no orphaned tunnels."""
    try:
        import win32job
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
        return job
    except Exception:
        log.exception("no job object; the tunnel may outlive wisper")
        return None


def _kill_stale(exe):
    """A tunnel left over from an older wisper (only this exact program; other apps' cloudflared are untouched)."""
    try:
        import win32api, win32con, win32process
        for pid in win32process.EnumProcesses():
            try:
                h = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ
                                         | win32con.PROCESS_TERMINATE, False, pid)
            except Exception:
                continue
            try:
                if os.path.normcase(win32process.GetModuleFileNameEx(h, None)) == os.path.normcase(exe):
                    win32api.TerminateProcess(h, 1)
                    log.info("stopped a leftover tunnel (pid %d)", pid)
            except Exception:
                pass
            finally:
                win32api.CloseHandle(h)
    except Exception:
        log.exception("could not check for a leftover tunnel")


def remote_token(cfg):
    """config "remote_token", else data/remote-token.txt (made on first start; delete it to issue a new one)."""
    if cfg.get("remote_token"):
        return cfg["remote_token"]
    if os.path.exists(TOKEN_FILE):
        return open(TOKEN_FILE, encoding="utf-8").read().strip()
    tok = secrets.token_urlsafe(18)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(tok)
    return tok


def fit_to_context(text, before, personal):
    """Dictating into the middle of a sentence continues it: "and then we left" instead of "And then we left".
    Names, "I", acronyms and words that aren't ordinary English keep their capital."""
    if not text or not before.strip() or re.search(r"[.!?\n]['\")\]]*\s*$", before):
        return text
    m = re.match(r"([A-Z][a-z']*)\b", text)
    if not m:
        return text
    w = m.group(1)
    if w == "I" or w.startswith("I'") or w in personal.names or not is_common(w.replace("'", "")):
        return text
    return text[0].lower() + text[1:]


class _Feed:
    """Stands in for the PC mic: Session reads `rec` (50 ms float32 blocks); None = released."""
    def __init__(self):
        self.rec = []


class PhoneSession:
    def __init__(self, engine, sid, before, send):
        self.engine, self.id, self.before, self.send = engine, sid, before, send
        self.feed = _Feed()
        self.spare = np.zeros(0, "float32")     # samples not yet filling a 50 ms block
        self.odd = b""                          # a frame ended mid-sample
        self.session = Session(engine, self.feed)
        self.t0 = time.time()
        threading.Thread(target=self._keep_gpu_awake, daemon=True).start()
        threading.Thread(target=self._partials, daemon=True).start()

    def active(self):
        return self.feed.rec is not None and not self.session.cancelled

    def audio(self, data):
        data = self.odd + data
        cut = len(data) - len(data) % 2
        self.odd = data[cut:]
        x = np.concatenate([self.spare, np.frombuffer(data[:cut], "<i2").astype("float32") / 32768.0])
        n = len(x) // BLOCK * BLOCK
        rec = self.feed.rec
        if rec is not None:
            rec.extend(x[i:i + BLOCK] for i in range(0, n, BLOCK))
        self.spare = x[n:]

    def _keep_gpu_awake(self):
        while self.active():
            try:
                self.engine.wake_gpu()
            except Exception:
                return
            time.sleep(1.5)

    def _partials(self):
        """While you talk, send the pieces that are already transcribed (plus the early pass on the tail)."""
        last = ""
        while self.active():
            time.sleep(0.15)
            s, parts = self.session, []
            for th, r in list(s.chunks):
                if th is not None and th.is_alive():
                    break
                if r.get("clean") and r["clean"]["final"]:
                    parts.append(r["clean"]["final"])
            else:
                sp = s.spec
                if sp is not None and sp[0] == s.cut and not sp[2].is_alive() and (sp[3].get("clean") or {}).get("final"):
                    parts.append(sp[3]["clean"]["final"])
            text = fit_to_context(" ".join(parts), self.before, self.engine.personal)
            if text != last and self.active():
                last = text
                self.send({"type": "partial", "id": self.id, "text": text})

    def cancel(self):
        self.session.cancelled = True
        self.feed.rec = None

    def finish(self):
        if len(self.spare):
            self.feed.rec.append(self.spare)
        blocks, self.feed.rec = self.feed.rec, None
        audio_s = len(blocks) * BLOCK / SR
        t_release = time.time()
        if audio_s < MIN_AUDIO_S:
            self.session.cancelled = True
            return {"type": "final", "id": self.id, "text": "", "ms": 0}
        asr_text, final, speech_s, wait_ms, hit = self.session.finish(blocks)
        final = fit_to_context(final, self.before, self.engine.personal)
        total = (time.time() - t_release) * 1000
        self.engine.log_dictation(asr_text, final, speech_s, audio_s, wait_ms, total, "phone", np.concatenate(blocks))
        log.info("phone dictation: %.1fs audio, release->text %dms on the PC, speculation %s, chunks %d",
                 audio_s, total, "hit" if hit else "miss", len(self.session.chunks))
        return {"type": "final", "id": self.id, "text": final, "ms": int(total)}


class Remote:
    def __init__(self, engine, cfg, http_handler):
        self.engine, self.cfg = engine, cfg
        self.token = remote_token(cfg)
        self.tunnel_url = None
        host = cfg.get("remote_host", "0.0.0.0")
        ws_port, http_port = cfg.get("remote_port", 47822), cfg.get("remote_http_port", 47823)
        self.ws = serve(self._handle, host, ws_port, process_request=self._check, compression=None,
                        max_size=2 ** 20, ping_interval=15, ping_timeout=15)
        threading.Thread(target=self.ws.serve_forever, daemon=True, name="remote-ws").start()
        if http_port:
            self.http = ThreadingHTTPServer((host, http_port), self._authed(http_handler))
            threading.Thread(target=self.http.serve_forever, daemon=True, name="remote-http").start()
        log.info("phone access on %s: ws :%d/v1/stream, http :%s/v1 (run tools/pair.py to pair a phone)",
                 host, ws_port, http_port or "-")
        threading.Thread(target=self._announce, args=(ws_port, http_port), daemon=True, name="remote-mdns").start()
        if cfg.get("remote_tunnel", True) and os.path.exists(CLOUDFLARED):
            threading.Thread(target=self._tunnel, args=(ws_port,), daemon=True, name="remote-tunnel").start()

    # ---------------------------------------------------------------- away from home
    def _tunnel(self, ws_port):
        """Reach the PC from anywhere (mobile data, other Wi-Fi) with no app on the phone: a Cloudflare quick tunnel
        (no account) gives a public wss:// address with a real certificate; every connection still needs the token.
        The address is new each time the tunnel starts, so it is published to a secret GitHub gist (via the gh CLI
        you're signed in to) that the phone reads when its saved addresses stop answering. Light: one ~45 MB
        process, idle unless you dictate; it dies with wisper (job object), never orphaned."""
        import subprocess
        job = _kill_with_me_job()
        _kill_stale(CLOUDFLARED)
        delay = 5
        while True:
            t0 = time.time()
            try:
                p = subprocess.Popen([CLOUDFLARED, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{ws_port}"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW, text=True, encoding="utf-8", errors="replace")
                if job:
                    import win32api, win32con, win32job
                    h = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, p.pid)
                    win32job.AssignProcessToJobObject(job, h)
                for line in p.stderr:                 # keep reading: a full pipe would stall it
                    m = re.search(r"https://([a-z0-9-]+\.trycloudflare\.com)", line)
                    if m and not self.tunnel_url:
                        self.tunnel_url = "wss://" + m.group(1)
                        log.info("reachable from anywhere at %s", self.tunnel_url)
                        threading.Thread(target=self._publish, daemon=True).start()
                p.wait()
            except Exception:
                log.exception("tunnel failed")
            self.tunnel_url = None
            delay = 5 if time.time() - t0 > 300 else min(delay * 2, 300)
            log.warning("tunnel stopped; restarting in %ds", delay)
            time.sleep(delay)

    def _publish(self):
        """Write the current addresses to the gist (made on first use; its id is in data/remote-gist.txt)."""
        import hashlib, shutil, subprocess
        gh = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"
        url = self.tunnel_url
        body = json.dumps({"id": hashlib.sha256(self.token.encode()).hexdigest()[:12], "urls": [url],
                           "pc": socket.gethostname(), "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        run = dict(capture_output=True, text=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
        for attempt in range(30):                     # offline right after boot: keep trying for ~30 min
            if url != self.tunnel_url:
                return                                # a newer tunnel took over
            try:
                gid = open(GIST_FILE).read().strip() if os.path.exists(GIST_FILE) else ""
                if not gid:
                    r = subprocess.run([gh, "gist", "create", "--filename", "wisper.json", "-d",
                                        "wisper: where the phone keyboard finds this PC", "-"], input=body, **run)
                    gid = r.stdout.strip().rstrip("/").rsplit("/", 1)[-1] if r.returncode == 0 else ""
                    if gid:
                        open(GIST_FILE, "w").write(gid)
                else:
                    r = subprocess.run([gh, "api", "-X", "PATCH", f"gists/{gid}", "--input", "-"],
                                       input=json.dumps({"files": {"wisper.json": {"content": body}}}), **run)
                if gid and r.returncode == 0:
                    json.dump({"url": url, "rendezvous": f"https://api.github.com/gists/{gid}"},
                              open(TUNNEL_FILE, "w"))
                    return log.info("published the address for the phone (gist %s)", gid)
                log.warning("could not publish the address: %s", (r.stderr or r.stdout).strip()[:200])
            except Exception as e:
                log.warning("could not publish the address: %s", e)
            time.sleep(60)

    def _announce(self, ws_port, http_port):
        """Tell phones on the network where this PC is (mDNS/Bonjour, _wisper._tcp), so pairing is once: when the
        PC gets a new Wi-Fi address after a restart, the phone finds it again. The service only carries a
        fingerprint of the token ("id") so the phone can tell its own PC apart; the token itself never goes out."""
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            return log.warning("zeroconf not installed: phones can't find this PC if its address changes")
        import hashlib, ipaddress
        fid = hashlib.sha256(self.token.encode()).hexdigest()[:12]
        name = re.sub(r"[^A-Za-z0-9-]", "-", socket.gethostname())

        def addrs():
            """The Wi-Fi/LAN address (the one with the default route; not WSL/VPN adapters) + Tailscale."""
            out = set()
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("10.255.255.255", 1))           # no packet is sent; just picks the interface
                out.add(ipaddress.ip_address(s.getsockname()[0]).packed)
                s.close()
            except OSError:
                pass
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = ipaddress.ip_address(info[4][0])
                if ip in ipaddress.ip_network("100.64.0.0/10"):
                    out.add(ip.packed)
            return sorted(out)

        zc, info, last = None, None, None
        while True:
            try:
                cur = addrs()
                if cur != last:          # first start, or the PC's addresses changed (new network, Wi-Fi back)
                    if zc is None:
                        zc = Zeroconf()
                    if info is not None:
                        zc.unregister_service(info)
                    info = ServiceInfo("_wisper._tcp.local.", f"{name}._wisper._tcp.local.", port=ws_port,
                                       addresses=cur, server=f"{name}.local.",
                                       properties={"id": fid, "pc": socket.gethostname(), "http": str(http_port or "")})
                    zc.register_service(info)
                    log.info("announced on the network as %s (%s)", name,
                             ", ".join(str(ipaddress.ip_address(a)) for a in cur))
                    last = cur
            except Exception:
                log.exception("mDNS announce failed (will retry)")
                last = None
            time.sleep(60)

    def ok(self, supplied):
        return bool(supplied) and hmac.compare_digest(supplied.encode(), self.token.encode())

    def _check(self, conn, request):
        url = urlparse(request.path)
        if url.path.rstrip("/") == "/health":
            return conn.respond(200, "ok\n")
        if url.path.rstrip("/") != "/v1/stream":
            return conn.respond(404, "not found\n")
        auth = request.headers.get("Authorization", "")
        supplied = auth[7:].strip() if auth.startswith("Bearer ") else (parse_qs(url.query).get("token") or [""])[0]
        if not self.ok(supplied):
            log.warning("phone connection refused (bad token) from %s", conn.remote_address[0])
            return conn.respond(401, "bad token\n")

    def _authed(self, base):
        remote = self

        class Authed(base):
            def _allowed(self):
                auth = self.headers.get("Authorization", "")
                if self.path.startswith("/health") or remote.ok(auth[7:].strip() if auth.startswith("Bearer ") else ""):
                    return True
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self._json(401, {"error": {"message": "bad token"}})
                return False

            def do_GET(self):
                if self.path.startswith("/debug"):       # PC-only test hooks
                    return self._json(404, {"error": "not found"})
                if self._allowed():
                    super().do_GET()

            def do_POST(self):
                if not self.path.rstrip("/").endswith("/audio/transcriptions"):
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    return self._json(404, {"error": "not found"})
                if self._allowed():
                    super().do_POST()
        return Authed

    def _handle(self, ws):
        try:
            ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # small replies go out at once
        except OSError:
            pass
        lock = threading.Lock()
        peer = ws.remote_address[0]

        def send(obj):
            with lock:
                try:
                    ws.send(json.dumps(obj))
                except Exception:
                    pass

        log.info("phone connected from %s", peer)
        send({"type": "ready", "v": 1})
        cur = None
        try:
            for msg in ws:
                if isinstance(msg, bytes):
                    if cur is not None and cur.active():
                        cur.audio(msg)
                    continue
                m = json.loads(msg)
                kind = m.get("type")
                if kind == "start":
                    if cur is not None and cur.active():
                        cur.cancel()
                    cur = PhoneSession(self.engine, m.get("id"), (m.get("before") or "")[-300:], send)
                elif kind == "stop" and cur is not None and cur.active() and cur.id == m.get("id"):
                    send(cur.finish())
                elif kind == "cancel" and cur is not None and cur.active():
                    cur.cancel()
                    log.info("phone dictation cancelled")
                elif kind == "ping":
                    send({"type": "pong", "t": m.get("t")})
        except ConnectionClosed:
            pass                                   # phone went to sleep / changed network: normal
        except Exception:
            log.exception("phone connection error")
        finally:
            if cur is not None and cur.active():
                cur.cancel()
            log.info("phone disconnected (%s)", peer)


def start_remote(engine, cfg, http_handler):
    try:
        return Remote(engine, cfg, http_handler)
    except Exception:
        log.exception("could not start phone access")
