"""Pair a phone with wisper: shows a QR code that the wisper-keys app (Android/iOS) reads with the phone camera.

  python -I tools\\pair.py            opens the pairing page in your browser (and prints the same details)

The link holds every address the phone can try, in order: this PC on your Wi-Fi, then Tailscale (100.x) if it's
installed, for when you're away from home. The keyboard uses whichever answers first.
"""
import html, ipaddress, json, os, socket, sys, webbrowser
from urllib.parse import urlencode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
CFG = {**json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8")),
       **(json.load(open(os.path.join(ROOT, "config.local.json"), encoding="utf-8"))
          if os.path.exists(os.path.join(ROOT, "config.local.json")) else {})}


def addresses():
    """[(label, ip)]: the Wi-Fi/LAN address (the one with the default route), then any Tailscale address."""
    out = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))          # no packet is sent; just picks the outgoing interface
        out.append(("Wi-Fi / LAN", s.getsockname()[0]))
        s.close()
    except OSError:
        pass
    tailnet = ipaddress.ip_network("100.64.0.0/10")
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        ip = info[4][0]
        if ipaddress.ip_address(ip) in tailnet and ("Tailscale", ip) not in out:
            out.append(("Tailscale", ip))
    return out


def main():
    if not CFG.get("remote"):
        sys.exit('Phone access is off: set "remote": true in config.local.json and restart wisper.')
    from remote import remote_token
    token = remote_token(CFG)
    ws_port, http_port = CFG.get("remote_port", 47822), CFG.get("remote_http_port", 47823)
    addrs = addresses()
    if not addrs:
        sys.exit("No network address found: connect to Wi-Fi (or Tailscale) first.")
    link = "wisperkeys://pair?" + urlencode([("u", f"ws://{ip}:{ws_port}") for _, ip in addrs]
                                            + [("t", token), ("n", socket.gethostname())])
    import segno
    svg = segno.make(link, error="m").svg_inline(scale=7, dark="#f4f4f5", light="#18181b")
    rows = "".join(f"<tr><td>{html.escape(lbl)}</td><td><code>ws://{ip}:{ws_port}</code></td>"
                   f"<td><code>http://{ip}:{http_port}/v1</code></td></tr>" for lbl, ip in addrs)
    page = f"""<!doctype html><meta charset="utf-8"><title>Pair a phone with wisper</title>
<style>body{{background:#0f0f11;color:#e4e4e7;font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 20px}}
code{{background:#27272a;padding:2px 6px;border-radius:4px}} td{{padding:4px 12px 4px 0}} .qr{{background:#18181b;
display:inline-block;padding:18px;border-radius:14px}} h1{{font-size:22px}} .dim{{color:#a1a1aa}}</style>
<h1>Pair a phone with wisper</h1>
<p>Open the <b>wisper keys</b> app on your phone, tap <b>Scan pairing code</b> (or point the phone camera here).</p>
<div class="qr">{svg}</div>
<h3>Addresses</h3><table><tr class="dim"><td></td><td>keyboard (streaming)</td><td>OpenAI-compatible API</td></tr>{rows}</table>
<p>Token (also the API key): <code>{html.escape(token)}</code></p>
<p class="dim">Away from home? Install Tailscale on this PC and the phone (same account), then run this again so the
code includes the 100.x address. New token: delete <code>data/remote-token.txt</code> and restart wisper.</p>"""
    path = os.path.join(ROOT, "data", "pair.html")
    open(path, "w", encoding="utf-8").write(page)
    print("pairing link:", link)
    for lbl, ip in addrs:
        print(f"  {lbl:12} ws://{ip}:{ws_port}/v1/stream   http://{ip}:{http_port}/v1")
    print("token:", token)
    if "--no-open" not in sys.argv:
        webbrowser.open("file:///" + path.replace("\\", "/"))


if __name__ == "__main__":
    main()
