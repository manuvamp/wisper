"""Native push-to-talk front end for wisper (replaces OpenWhispr).

Hold Ctrl+Shift -> talk -> release: text is typed where your cursor is.
- The mic stays open while you're active (closes after `mic_idle_close_s` without dictation), with a short rolling
  pre-roll so recording starts instantly and the first syllable is never cut.
- Long dictations are transcribed in chunks at natural pauses *while you talk*; on release only the tail is left.
- Another key pressed during the hold (e.g. Ctrl+Shift+T) cancels: it was a shortcut, not dictation.
- Esc during the hold cancels.
Text is typed with Unicode key events, so the clipboard (e.g. a screenshot you're about to paste) is never touched.
"""
import collections, ctypes, ctypes.wintypes as wt, logging, os, queue, threading, time
import numpy as np
import sounddevice as sd

log = logging.getLogger("wisper")
SR = 16000
BLOCK = 800                      # 50 ms
PREROLL_S = 0.4
MIN_HOLD_S = 0.25                # shorter presses are accidental
CHUNK_MIN_S = 5.0                # commit a chunk once this much new audio has piled up...
PAUSE_S = 0.3                    # ...at the next natural pause, so only the last few seconds remain at release

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
VK_LSHIFT, VK_RSHIFT, VK_LCTRL, VK_RCTRL, VK_ESC = 0xA0, 0xA1, 0xA2, 0xA3, 0x1B
CTRL, SHIFT = {VK_LCTRL, VK_RCTRL, 0x11}, {VK_LSHIFT, VK_RSHIFT, 0x10}
WH_KEYBOARD_LL, WM_KEYDOWN, WM_SYSKEYDOWN, WM_KEYUP, WM_SYSKEYUP = 13, 0x100, 0x104, 0x101, 0x105
LLKHF_INJECTED = 0x10


def make_beep(path, hz=880, ms=70, volume=0.18):
    """Short soft tone (faded in/out so it doesn't click) written once as a WAV for async playback."""
    import wave
    rate = 44100
    t = np.arange(int(rate * ms / 1000)) / rate
    env = np.minimum(1, np.minimum(t, t[-1] - t) / 0.012)
    pcm = (np.sin(2 * np.pi * hz * t) * env * volume * 32767).astype("<i2")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
    return path


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wt.DWORD), ("scanCode", wt.DWORD), ("flags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wt.WPARAM, wt.LPARAM)
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wt.HINSTANCE, wt.DWORD]
user32.SetWindowsHookExW.restype = wt.HHOOK
user32.CallNextHookEx.argtypes = [wt.HHOOK, ctypes.c_int, wt.WPARAM, wt.LPARAM]
user32.CallNextHookEx.restype = ctypes.c_ssize_t
user32.GetAsyncKeyState.restype = ctypes.c_short


# ---------------------------------------------------------------- typing (SendInput, Unicode)
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _U)]


KEYEVENTF_UNICODE, KEYEVENTF_KEYUP = 0x4, 0x2


def type_text(text, target=None):
    """Type into `target` (the window you pressed the hotkey in). Returns False if it can't be reached."""
    # wait (max 1.5 s) until Ctrl/Shift are physically released so nothing turns into a shortcut
    t_end = time.time() + 1.5
    while time.time() < t_end and any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in (0x10, 0x11, 0x12)):
        time.sleep(0.01)
    fg = user32.GetForegroundWindow()
    if target and fg != target:
        log.info("focus moved during dictation: %r -> %r", window_title(target)[:60], window_title(fg)[:60])
        if not refocus(target):
            return False
    events = []
    data =text.replace("\r", "").replace("\n", " ").encode("utf-16-le")
    for i in range(0, len(data), 2):
        code = int.from_bytes(data[i:i + 2], "little")
        for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            events.append(INPUT(type=1, ki=KEYBDINPUT(0, code, flags, 0, 0)))
    arr = (INPUT * len(events))(*events)
    user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))
    return True


# ---------------------------------------------------------------- keeping the text going to the right window
user32.GetForegroundWindow.restype = wt.HWND
user32.GetWindowThreadProcessId.argtypes = (wt.HWND, ctypes.POINTER(wt.DWORD))
user32.SetForegroundWindow.argtypes = (wt.HWND,)
user32.IsWindow.argtypes = (wt.HWND,)
user32.GetWindowTextW.argtypes = (wt.HWND, wt.LPWSTR, ctypes.c_int)


def window_title(hwnd):
    buf = ctypes.create_unicode_buffer(200)
    user32.GetWindowTextW(hwnd, buf, 200)
    return buf.value


def refocus(hwnd):
    """Give the keyboard back to the window you were dictating into (something, e.g. a system pop-up, took it).
    Windows only lets the process that owns the foreground do this, so borrow its input queue for a moment."""
    fg = user32.GetForegroundWindow()
    if not hwnd or fg == hwnd or not user32.IsWindow(hwnd):
        return fg == hwnd
    me = kernel32.GetCurrentThreadId()
    theirs = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    if theirs and theirs != me:
        user32.AttachThreadInput(me, theirs, True)
    try:
        user32.SetForegroundWindow(hwnd)
    finally:
        if theirs and theirs != me:
            user32.AttachThreadInput(me, theirs, False)
    time.sleep(0.05)
    return user32.GetForegroundWindow() == hwnd


def copy_to_clipboard(text):
    """Fallback only (when the target window is gone): normally the clipboard is never touched."""
    k = kernel32
    k.GlobalAlloc.restype, k.GlobalLock.restype = ctypes.c_void_p, ctypes.c_void_p
    k.GlobalLock.argtypes = k.GlobalUnlock.argtypes = (ctypes.c_void_p,)
    user32.SetClipboardData.argtypes = (wt.UINT, ctypes.c_void_p)
    data = (text + "\0").encode("utf-16-le")
    for _ in range(10):                       # another app may hold the clipboard for a moment
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        return False
    try:
        user32.EmptyClipboard()
        h = k.GlobalAlloc(0x2, len(data))     # GMEM_MOVEABLE
        ctypes.memmove(k.GlobalLock(h), data, len(data))
        k.GlobalUnlock(h)
        return bool(user32.SetClipboardData(13, h))   # CF_UNICODETEXT
    finally:
        user32.CloseClipboard()


class FILTERKEYS(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("dwFlags", wt.DWORD), ("iWaitMSec", wt.DWORD), ("iDelayMSec", wt.DWORD),
                ("iRepeatMSec", wt.DWORD), ("iBounceMSec", wt.DWORD)]


def disable_filter_keys_shortcut():
    """Holding Shift ~8 s (a long dictation) triggers Windows' Filter Keys shortcut, whose pop-up steals the keyboard
    focus so the text goes nowhere. Turn off only that shortcut (Filter Keys itself stays as you set it)."""
    fk = FILTERKEYS(cbSize=ctypes.sizeof(FILTERKEYS))
    if not user32.SystemParametersInfoW(0x32, fk.cbSize, ctypes.byref(fk), 0):   # SPI_GETFILTERKEYS
        return
    FKF_HOTKEYACTIVE = 0x4
    if fk.dwFlags & FKF_HOTKEYACTIVE:
        fk.dwFlags &= ~FKF_HOTKEYACTIVE
        user32.SystemParametersInfoW(0x33, fk.cbSize, ctypes.byref(fk), 0x3)       # SET, save + broadcast
        log.info("turned off the Windows Filter Keys shortcut (hold Shift 8 s): its pop-up steals focus mid-dictation")


# ---------------------------------------------------------------- floating pill (never takes focus)
LRESULT, WPARAM, LPARAM = ctypes.c_ssize_t, ctypes.c_size_t, ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, WPARAM, LPARAM)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("style", wt.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wt.HINSTANCE), ("hIcon", wt.HICON),
                ("hCursor", wt.HANDLE), ("hbrBackground", wt.HBRUSH), ("lpszMenuName", wt.LPCWSTR),
                ("lpszClassName", wt.LPCWSTR), ("hIconSm", wt.HICON)]


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [("hdc", wt.HDC), ("fErase", wt.BOOL), ("rcPaint", wt.RECT), ("fRestore", wt.BOOL),
                ("fIncUpdate", wt.BOOL), ("rgbReserved", ctypes.c_byte * 32)]


gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
user32.DefWindowProcW.argtypes = (wt.HWND, wt.UINT, WPARAM, LPARAM)
user32.DefWindowProcW.restype = LRESULT
user32.CreateWindowExW.restype = wt.HWND
user32.CreateWindowExW.argtypes = (wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID)
user32.BeginPaint.restype = wt.HDC
user32.BeginPaint.argtypes = (wt.HWND, ctypes.POINTER(PAINTSTRUCT))
user32.EndPaint.argtypes = (wt.HWND, ctypes.POINTER(PAINTSTRUCT))
user32.PostMessageW.argtypes = (wt.HWND, wt.UINT, WPARAM, LPARAM)
user32.ShowWindow.argtypes = (wt.HWND, ctypes.c_int)
user32.SetWindowPos.argtypes = (wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.UINT)
user32.InvalidateRect.argtypes = (wt.HWND, wt.LPVOID, wt.BOOL)
user32.SetLayeredWindowAttributes.argtypes = (wt.HWND, wt.DWORD, ctypes.c_ubyte, wt.DWORD)
user32.FillRect.argtypes = (wt.HDC, ctypes.POINTER(wt.RECT), wt.HBRUSH)
gdi32.CreateFontW.restype = wt.HFONT
gdi32.SelectObject.restype = wt.HGDIOBJ
gdi32.SelectObject.argtypes = (wt.HDC, wt.HGDIOBJ)
gdi32.CreateSolidBrush.restype = wt.HBRUSH
gdi32.SetTextColor.argtypes = gdi32.SetBkColor.argtypes = (wt.HDC, wt.DWORD)
gdi32.TextOutW.argtypes = (wt.HDC, ctypes.c_int, ctypes.c_int, wt.LPCWSTR, ctypes.c_int)
gdi32.GetTextExtentPoint32W.argtypes = (wt.HDC, wt.LPCWSTR, ctypes.c_int, ctypes.POINTER(wt.SIZE))
gdi32.DeleteObject.argtypes = (wt.HGDIOBJ,)
gdi32.GetStockObject.restype = wt.HGDIOBJ
gdi32.Ellipse.argtypes = (wt.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
user32.GetClientRect.argtypes = (wt.HWND, ctypes.POINTER(wt.RECT))
kernel32.GetModuleHandleW.restype = wt.HMODULE
user32.GetDC.restype = wt.HDC
user32.GetDC.argtypes = (wt.HWND,)
user32.ReleaseDC.argtypes = (wt.HWND, wt.HDC)


def _rgb(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return r | (g << 8) | (b << 16)


class Pill:
    """Small "Listening" / "Transcribing" badge at the bottom of the screen. A plain Win32 window created with
    WS_EX_NOACTIVATE from the start: it can never take the keyboard focus (the earlier Tk window grabbed the foreground
    once at service start, so whatever you were typing in lost its cursor)."""
    WM_APP_UPDATE = 0x8001
    BG, FG = "#18181b", "#f4f4f5"

    def __init__(self):
        self.state = None           # (text, color) or None = hidden
        self.hwnd = None
        self.ready = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="pill").start()
        self.ready.wait(5)

    def show(self, text, color):
        self.state = (text, color)
        self._post()

    def hide(self):
        self.state = None
        self._post()

    def _post(self):
        if self.hwnd:
            user32.PostMessageW(self.hwnd, self.WM_APP_UPDATE, 0, 0)

    def _layout(self, text):
        hdc = user32.GetDC(self.hwnd)
        old = gdi32.SelectObject(hdc, self.font)
        size = wt.SIZE()
        gdi32.GetTextExtentPoint32W(hdc, text, len(text), ctypes.byref(size))
        gdi32.SelectObject(hdc, old)
        user32.ReleaseDC(self.hwnd, hdc)
        h = size.cy + 14
        return 12 + 10 + 8 + size.cx + 16, h

    def _proc(self, hwnd, msg, wparam, lparam):
        if msg == self.WM_APP_UPDATE:
            st = self.state
            if st is None:
                user32.ShowWindow(hwnd, 0)                                   # SW_HIDE
            else:
                w, h = self._layout(st[0])
                sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
                SWP_NOACTIVATE, SWP_SHOWWINDOW = 0x10, 0x40
                user32.SetWindowPos(hwnd, wt.HWND(-1), (sw - w) // 2, sh - h - 90, w, h, SWP_NOACTIVATE | SWP_SHOWWINDOW)
                user32.InvalidateRect(hwnd, None, True)
            return 0
        if msg == 0x000F:                                                    # WM_PAINT
            ps = PAINTSTRUCT()
            hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
            st = self.state or ("", self.FG)
            rc = wt.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rc))
            user32.FillRect(hdc, ctypes.byref(rc), self.bg_brush)
            old = gdi32.SelectObject(hdc, self.font)
            gdi32.SetBkColor(hdc, _rgb(self.BG))
            dot = gdi32.CreateSolidBrush(_rgb(st[1]))
            old_brush = gdi32.SelectObject(hdc, dot)
            null_pen = gdi32.SelectObject(hdc, gdi32.GetStockObject(8))      # NULL_PEN
            cy = (rc.bottom - rc.top) // 2
            gdi32.Ellipse(hdc, 12, cy - 5, 22, cy + 5)
            gdi32.SelectObject(hdc, null_pen)
            gdi32.SelectObject(hdc, old_brush)
            gdi32.DeleteObject(dot)
            gdi32.SetTextColor(hdc, _rgb(self.FG))
            size = wt.SIZE()
            gdi32.GetTextExtentPoint32W(hdc, st[0], len(st[0]), ctypes.byref(size))
            gdi32.TextOutW(hdc, 30, cy - size.cy // 2, st[0], len(st[0]))
            gdi32.SelectObject(hdc, old)
            user32.EndPaint(hwnd, ctypes.byref(ps))
            return 0
        if msg == 0x0021:                                                    # WM_MOUSEACTIVATE
            return 3                                                         # MA_NOACTIVATE
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _run(self):
        hinst = kernel32.GetModuleHandleW(None)
        self._wndproc = WNDPROC(self._proc)                                  # keep a reference
        wc = WNDCLASSEXW(cbSize=ctypes.sizeof(WNDCLASSEXW), lpfnWndProc=self._wndproc, hInstance=hinst,
                         lpszClassName="wisper-pill")
        user32.RegisterClassExW(ctypes.byref(wc))
        self.font = gdi32.CreateFontW(-13, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI")   # CLEARTYPE_QUALITY
        self.bg_brush = gdi32.CreateSolidBrush(_rgb(self.BG))
        WS_EX = 0x08000000 | 0x80 | 0x20 | 0x8 | 0x80000   # NOACTIVATE | TOOLWINDOW | TRANSPARENT | TOPMOST | LAYERED
        self.hwnd = user32.CreateWindowExW(WS_EX, "wisper-pill", "wisper", 0x80000000,  # WS_POPUP, not visible
                                           0, 0, 10, 10, None, None, hinst, None)
        user32.SetLayeredWindowAttributes(self.hwnd, 0, 235, 0x2)            # LWA_ALPHA
        self.ready.set()
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))


# ---------------------------------------------------------------- microphone (open while active)
class Mic:
    def __init__(self, idle_close_s):
        self.idle_close_s = idle_close_s
        self.stream = None
        self.lock = threading.Lock()
        self.preroll = collections.deque(maxlen=int(PREROLL_S * SR / BLOCK))
        self.rec = None                     # list of blocks while recording
        self.last_used = time.time()
        wasapi = next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])
        self.wasapi = wasapi
        threading.Thread(target=self._idle_closer, daemon=True, name="mic-idle").start()

    def _cb(self, indata, frames, t, status):
        block = indata[:, 0].copy()
        rec = self.rec
        if rec is not None:
            rec.append(block)
        else:
            self.preroll.append(block)

    def _candidates(self):
        """WASAPI (resampled to 16 kHz by Windows) first, then MME, then whatever PortAudio calls default."""
        out = []
        for name, extra in (("WASAPI", sd.WasapiSettings(auto_convert=True)), ("MME", None)):
            for i, h in enumerate(sd.query_hostapis()):
                if name in h["name"] and h["default_input_device"] >= 0:
                    out.append((h["default_input_device"], extra))
        out.append((None, None))
        return out

    def ensure_open(self):
        with self.lock:
            if self.stream is not None:
                return
            errors = []
            for dev, extra in self._candidates():
                try:
                    st = sd.InputStream(device=dev, channels=1, samplerate=SR, dtype="float32", blocksize=BLOCK,
                                        extra_settings=extra, callback=self._cb)
                    st.start()
                    self.stream = st
                    log.info("mic opened: %s", sd.query_devices(st.device)["name"])
                    return
                except Exception as e:
                    errors.append(f"{dev}: {e}")
            raise RuntimeError("could not open microphone: " + " | ".join(errors))

    def start(self):
        self.ensure_open()
        self.rec = list(self.preroll)
        self.preroll.clear()

    def stop(self):
        rec, self.rec = self.rec, None
        self.last_used = time.time()
        return rec or []

    def _idle_closer(self):
        while True:
            time.sleep(15)
            with self.lock:
                if self.stream is not None and self.rec is None and time.time() - self.last_used > self.idle_close_s:
                    self.stream.close()
                    self.stream = None
                    self.preroll.clear()
                    log.info("mic closed after %ds idle", self.idle_close_s)


def rms(x):
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


# ---------------------------------------------------------------- session: one hold of the hotkey
SPEC_PAUSE_S = 0.2      # silence after speech that triggers a speculative transcription of everything so far
KEYUP_NOISE_S = 0.15    # ignore the key-release click at the very end


class Session:
    """One hold of the hotkey. While you talk:
    - long dictations: at a pause after >= CHUNK_MIN_S, the audio so far is committed as a chunk and transcribed +
      cleaned in the background;
    - at every short pause after new speech, the uncommitted tail is speculatively transcribed + cleaned. If you then
      release without saying anything more, that result is used as-is (release -> text = typing time).
    """

    DEFAULTS = {"chunk_min": CHUNK_MIN_S, "pause": PAUSE_S, "spec_pause": SPEC_PAUSE_S,
                "reuse_running": False, "context": False}

    def __init__(self, engine, mic, opts=None):
        self.engine, self.mic = engine, mic
        self.o = {**self.DEFAULTS, **(opts or {})}
        self.cut = 0                 # blocks before this index are committed chunks
        self.chunks = []             # (thread, result) for committed chunks, in order
        self.spec = None             # (cut, last_speech_idx, thread, result) of the latest speculative pass
        self.cancelled = False
        self.noise = None
        self.levels = []             # rms per block
        self.last_speech = -1
        threading.Thread(target=self._watch, daemon=True, name="chunker").start()

    def _is_speech(self, level):
        """Any sound clearly above this recording's own background level (soft trailing words count)."""
        return level > max(0.003, 3 * (self.noise or 0.001))

    def _scan(self, rec):
        new = rec[len(self.levels):]
        if not new:
            return
        self.levels.extend(rms(b) for b in new)
        nz = [x for x in self.levels if x > 1e-5]               # ignore digital silence
        self.noise = float(np.percentile(nz, 10)) if len(nz) >= 5 else 0.001
        for i in range(len(self.levels) - len(new), len(self.levels)):
            if self._is_speech(self.levels[i]):
                self.last_speech = i

    def _job(self, blocks, accurate=False):
        audio = np.concatenate(blocks) if blocks else np.zeros(0, "float32")
        res = {}
        before = list(self.chunks)

        def run():
            prefix = None
            if self.o["context"] and before:      # continue the previous chunk's sentence, not start a new one
                if before[-1][0]:
                    before[-1][0].join()
                prefix = " ".join(r.get("text", "") for _, r in before if r.get("text"))[-200:] or None
            r = self.engine.asr(audio, prefix=prefix)
            res.update(r)
            if res.get("superseded"):        # a newer speculative pass replaced this one: skip the LLM work
                return
            res["clean"] = self.engine.clean_text(r["text"]) if r["text"] else None

        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th, res

    def _watch(self):
        pause_blocks = int(self.o["spec_pause"] * SR / BLOCK)
        chunk_pause = int(self.o["pause"] * SR / BLOCK)
        while not self.cancelled:
            time.sleep(0.05)
            rec = self.mic.rec
            if rec is None:
                break
            self._scan(rec)
            n = len(self.levels)
            silent_for = n - 1 - self.last_speech
            if self.last_speech < self.cut:
                continue                                   # nothing new said since the last commit
            if (n - self.cut) * BLOCK / SR >= self.o["chunk_min"] and silent_for >= chunk_pause:
                sp = self.spec
                if (sp is not None and sp[0] == self.cut and sp[4] <= n
                        and not any(self._is_speech(x) for x in self.levels[sp[4]:n])):
                    self.chunks.append((sp[2], sp[3]))           # the head-start pass already covers it: reuse
                    self.cut, self.spec = sp[4], None
                else:
                    if sp is not None:
                        sp[3]["superseded"] = True
                    self.chunks.append(self._job(rec[self.cut:n]))   # commit a finished chunk
                    self.cut, self.spec = n, None
            elif (silent_for >= pause_blocks and (self.spec is None or self.spec[1] != self.last_speech)
                  and (self.spec is None or not self.spec[2].is_alive())):   # one speculative pass at a time
                th, res = self._job(rec[self.cut:n])
                self.spec = (self.cut, self.last_speech, th, res, n)

    def finish(self, blocks):
        """-> (asr_text, final_text, speech_s, wait_ms_at_release, speculation_hit) or None if cancelled."""
        if self.cancelled:
            return None
        self._scan(blocks)
        end = max(0, len(self.levels) - int(KEYUP_NOISE_S * SR / BLOCK))
        last = max([i for i in range(self.cut, end) if self._is_speech(self.levels[i])] or [-1])
        t0 = time.perf_counter()
        # use the early pass only if nothing at all was said after it: a soft trailing word ("...all of your
        # working") sits below the speech threshold, so require the uncovered audio to be near the noise floor
        soft = max(0.003, 3 * (self.noise or 0.001))
        hit = (self.spec is not None and self.spec[0] == self.cut and self.spec[1] >= last
               and max(self.levels[self.spec[4]:end] or [0]) <= soft)
        self.diag = {"spec": self.spec is not None, "spec_cut_ok": bool(self.spec and self.spec[0] == self.cut),
                     "spec_last": self.spec[1] if self.spec else None, "last": last, "end": end,
                     "after_max": round(max(self.levels[self.spec[4]:end] or [0]), 4) if self.spec else None,
                     "soft": round(soft, 4), "noise": round(self.noise or 0, 4),
                     "spec_running": bool(self.spec and self.spec[2].is_alive())}
        sp = self.spec
        if (not hit and self.o["reuse_running"] and sp is not None and sp[0] == self.cut and self.cut < sp[4] < end
                and last >= sp[4]):
            # you kept talking after the early pass started: keep it (it ends at a pause) as a chunk and
            # transcribe only what came after, instead of redoing everything
            self.chunks.append((sp[2], sp[3]))
            self.cut, self.spec = sp[4], None
            self.diag["reused_running"] = True
        elif self.spec is not None and not hit:
            self.spec[3]["superseded"] = True
        if hit:
            th, res = self.spec[2], self.spec[3]
        elif last < self.cut and self.chunks:
            th, res = None, {"text": "", "speech_s": 0, "clean": None}  # only silence after the last chunk
        else:
            th, res = self._job(blocks[self.cut:])
        parts = self.chunks + ([(th, res)] if th or res.get("text") else [])
        for t, _ in parts:
            if t:
                t.join()
        asr_ms = (time.perf_counter() - t0) * 1000
        asr_text = " ".join(r.get("text", "") for _, r in parts if r.get("text")).strip()
        joined = " ".join(r["clean"]["final"] for _, r in parts if r.get("clean") and r["clean"]["final"])
        merged = self.engine.clean_text(joined)["final"] if len(parts) > 1 else joined  # fix chunk seams
        speech = sum(r.get("speech_s", 0) for _, r in parts)
        return asr_text, merged, speech, asr_ms, hit


# ---------------------------------------------------------------- hotkey + orchestration
class Dictation:
    def __init__(self, engine, cfg):
        self.engine = engine
        self.mic = Mic(cfg.get("mic_idle_close_s", 600))
        self.pill = Pill()
        self.down = set()
        self.session = None
        self.press_t = 0.0
        self.chord_dirty = False         # another key was pressed during the hold -> shortcut, not dictation
        self.events = queue.Queue()
        self.target = None
        if cfg.get("fix_filter_keys_shortcut", True):
            try:
                disable_filter_keys_shortcut()
            except Exception:
                log.exception("could not check the Filter Keys shortcut")
        self.beep = None
        if cfg.get("start_beep", True):
            self.beep = make_beep(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                               "data", "beep.wav"))
        threading.Thread(target=self._worker, daemon=True, name="dictation").start()
        threading.Thread(target=self._hook_loop, daemon=True, name="kbd-hook").start()

    # --- low-level keyboard hook (must return fast; real work goes to the worker thread)
    def _hook_loop(self):
        self._proc = HOOKPROC(self._hook)
        self.hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, None, 0)
        if not self.hook:
            log.error("keyboard hook failed: %s", ctypes.get_last_error())
            return
        log.info("hotkey ready: hold Ctrl+Shift to dictate")
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _hook(self, n, wparam, lparam):
        if n == 0:
            k = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            if not k.flags & LLKHF_INJECTED:          # ignore our own typing
                vk = k.vkCode
                if wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                    if vk not in self.down:
                        self.down.add(vk)
                        self.events.put(("down", vk))
                elif wparam in (WM_KEYUP, WM_SYSKEYUP):
                    self.down.discard(vk)
                    self.events.put(("up", vk))
        return user32.CallNextHookEx(None, n, wparam, lparam)

    def _chord(self):
        return bool(self.down & CTRL) and bool(self.down & SHIFT)

    def _worker(self):
        while True:
            kind, vk = self.events.get()
            try:
                if kind == "down":
                    if self.session is None and self._chord() and not (self.down - CTRL - SHIFT):
                        self._begin()
                    elif self.session is not None and vk not in CTRL | SHIFT:
                        self._cancel("Esc" if vk == VK_ESC else "shortcut")
                elif kind == "up" and self.session is not None and vk in CTRL | SHIFT:
                    self._end()
            except Exception:
                log.exception("dictation error")
                self.session = None
                self.pill.hide()

    def _begin(self):
        self.press_t = time.time()
        self.target = user32.GetForegroundWindow()   # where the text should go, whatever pops up meanwhile
        self.mic.start()
        self.session = Session(self.engine, self.mic)
        threading.Timer(0.12, self._listening, args=(self.session,)).start()
        threading.Thread(target=self._keep_gpu_awake, args=(self.session,), daemon=True).start()

    def _keep_gpu_awake(self, s):
        """While you hold the keys, keep the GPU clocked up so the pass after release isn't run cold."""
        while self.session is s and not s.cancelled:
            try:
                self.engine.wake_gpu()
            except Exception:
                return
            time.sleep(1.5)

    def _listening(self, s):
        """Shown/played only once the hold is clearly dictation, not a Ctrl+Shift+<key> shortcut."""
        if self.session is not s or s.cancelled:
            return
        self.pill.show("Listening", "#ef4444")
        if self.beep:
            import winsound
            winsound.PlaySound(self.beep, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

    def _cancel(self, why):
        s, self.session = self.session, None
        s.cancelled = True
        self.mic.stop()
        self.pill.hide()
        log.info("dictation cancelled (%s)", why)

    def _end(self):
        s, self.session = self.session, None
        blocks = self.mic.stop()
        held = time.time() - self.press_t
        audio_s = len(blocks) * BLOCK / SR
        if held < MIN_HOLD_S or audio_s < 0.3:
            s.cancelled = True
            self.pill.hide()
            return
        self.pill.show("Transcribing", "#a1a1aa")
        threading.Thread(target=self._finish, args=(s, blocks, audio_s, self.target), daemon=True).start()

    def simulate(self, audio, tail_silence_s=0.15, opts=None):
        """Test hook: play 16 kHz audio (+ the pause people leave before releasing) into a session in real time."""
        class FakeMic:
            rec = []
        fake = FakeMic()
        sess = Session(self.engine, fake, opts)
        audio = np.concatenate([audio.astype("float32"), np.zeros(int(tail_silence_s * SR), "float32")])
        t_next = time.perf_counter()
        for i in range(0, len(audio), BLOCK):
            fake.rec.append(audio[i:i + BLOCK])
            t_next += BLOCK / SR
            time.sleep(max(0.0, t_next - time.perf_counter()))
        blocks, fake.rec = fake.rec, None
        t_release = time.time()
        asr_text, final, speech_s, wait_ms, hit = sess.finish(blocks)
        return {"text": final, "asr_text": asr_text, "release_to_text_ms": int((time.time() - t_release) * 1000),
                "speculation_hit": hit, "chunks": len(sess.chunks), "diag": sess.diag}

    def _finish(self, s, blocks, audio_s, target=None):
        t_release = time.time()
        try:
            res = s.finish(blocks)
            if res is None:
                return
            asr_text, final, speech_s, wait_ms, hit = res
            if final and not type_text(final + " ", target):
                copied = copy_to_clipboard(final + " ")
                log.warning("could not get back to %r: text %s", window_title(target)[:60],
                            "copied to the clipboard" if copied else "lost (clipboard busy)")
                if copied:
                    self.pill.show("Copied - press Ctrl+V", "#f59e0b")
                    time.sleep(4)
            total = (time.time() - t_release) * 1000
            self.engine.log_dictation(asr_text, final, speech_s, audio_s, wait_ms, total, "live",
                                      np.concatenate(blocks) if blocks else None)
            log.info("native dictation: %.1fs audio, release->typed %dms, speculation %s, chunks %d",
                     audio_s, total, "hit" if hit else "miss", len(s.chunks))
        finally:
            self.pill.hide()
