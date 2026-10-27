"""The training viewer: a local web page that polls /data.json while
experiments train, or a self-contained HTML file with the runs embedded.

    rlab.show(exp1, exp2)                 # serve and block until Ctrl-C
    rlab.save_html("runs/gaps.html", exp1, exp2)
"""
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).parent / "web"
_server = None
_runs = []


def _payload(runs):
    return json.dumps({"runs": [r.to_dict() for r in runs]}, separators=(",", ":"))


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def do_GET(self):
        if self.path.startswith("/data.json"):
            body = _payload(_runs).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, *a):
        pass


def serve(*experiments, port=8765, open_browser=True):
    """Add experiments to the live viewer, starting it on first use."""
    global _server
    for e in experiments:
        if e not in _runs:
            _runs.append(e)
    if _server is None:
        _server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        threading.Thread(target=_server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/"
        print(f"rlab viewer: {url}")
        if open_browser:
            webbrowser.open(url)
    return _server


def block():
    print("Viewer running - press Ctrl-C to quit.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


def show(*experiments, port=8765, wait=True):
    serve(*experiments, port=port)
    if wait:
        block()


def save_html(path, *experiments):
    """Write a single self-contained HTML file with the runs embedded."""
    html = (WEB / "index.html").read_text()
    for name in ("style.css",):
        html = html.replace(f'<link rel="stylesheet" href="{name}">', f"<style>{(WEB / name).read_text()}</style>")
    for name in ("charts.js", "app.js"):
        html = html.replace(f'<script src="{name}"></script>', f"<script>{(WEB / name).read_text()}</script>")
    data = _payload(experiments).replace("</", "<\\/")
    html = html.replace("<!--DATA-->", f"<script>window.RLAB_DATA = {data};</script>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    print(f"saved {path}")
    return path
