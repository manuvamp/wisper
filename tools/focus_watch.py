"""Log every change of the active (foreground) window: time, program, title. Finds what steals your typing focus.

usage: pythonw -I tools\\focus_watch.py [logfile]      (default logs\\focus.log; stop by ending the process)
"""
import ctypes, ctypes.wintypes as wt, os, sys, time

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "logs", "focus.log")
EVENT_SYSTEM_FOREGROUND, WINEVENT_OUTOFCONTEXT = 0x3, 0x0
WinEventProc = ctypes.WINFUNCTYPE(None, wt.HANDLE, wt.DWORD, wt.HWND, wt.LONG, wt.LONG, wt.DWORD, wt.DWORD)
user32.SetWinEventHook.restype = wt.HANDLE
user32.SetWinEventHook.argtypes = (wt.DWORD, wt.DWORD, wt.HMODULE, WinEventProc, wt.DWORD, wt.DWORD, wt.DWORD)
user32.GetWindowTextW.argtypes = (wt.HWND, wt.LPWSTR, ctypes.c_int)
user32.GetClassNameW.argtypes = (wt.HWND, wt.LPWSTR, ctypes.c_int)
user32.GetWindowThreadProcessId.argtypes = (wt.HWND, ctypes.POINTER(wt.DWORD))


def program(pid):
    h = kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return "?"
    buf, n = ctypes.create_unicode_buffer(400), wt.DWORD(400)
    ok = kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n))
    kernel32.CloseHandle(h)
    return os.path.basename(buf.value) if ok else "?"


def on_change(hook, event, hwnd, obj, child, thread, ms):
    title, cls, pid = ctypes.create_unicode_buffer(200), ctypes.create_unicode_buffer(100), wt.DWORD()
    user32.GetWindowTextW(hwnd, title, 200)
    user32.GetClassNameW(hwnd, cls, 100)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {program(pid.value):24s} pid {pid.value:<6d} [{cls.value[:30]}] {title.value[:90]!r}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


proc = WinEventProc(on_change)
hook = user32.SetWinEventHook(EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, None, proc, 0, 0, WINEVENT_OUTOFCONTEXT)
on_change(None, 0, user32.GetForegroundWindow(), 0, 0, 0, 0)
msg = wt.MSG()
while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
    user32.TranslateMessage(ctypes.byref(msg))
    user32.DispatchMessageW(ctypes.byref(msg))
