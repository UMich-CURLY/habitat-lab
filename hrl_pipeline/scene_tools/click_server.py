"""Host-side server for the live click tool. Serves the click page (GET /) and
the rendered videos (GET /vid_*.mp4) from hrl_pipeline/manual/, and handles
POST /run: writes the 4 clicked points to a request file, runs the expert inside
the container via `docker exec live_run.sh`, and returns the verdict + video
name. Runs on the HOST (so it's reachable through VSCode port forwarding); the
sim work happens in the container. Runs are serialized (sim is single-GPU).

Usage: python3 hrl_pipeline/scene_tools/click_server.py <click.html> [port]
"""
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading

DIR = "/home/xinyuan/habicrowd/Simulator/habitat-lab/hrl_pipeline/manual"
PAGE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(DIR, "click.html")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
LOCK = threading.Lock()
N = [0]


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=DIR, **k)

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(PAGE, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()   # static (videos) from DIR

    def do_POST(self):
        if self.path != "/run":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        with LOCK:
            N[0] += 1
            tag = "r%d" % N[0]
            json.dump(req, open(os.path.join(DIR, "_live_%s.req.json" % tag), "w"))
            print(f"[run {tag}] scene={req['scene_id'].split('/')[-1][:16]} ...", flush=True)
            try:
                cp = subprocess.run(
                    ["docker", "exec", "-u", "root", "wxinyuan", "bash",
                     "/habitat-lab/scripts/live_run.sh",
                     "/habitat-lab/hrl_pipeline/manual/_live_%s.req.json" % tag, tag],
                    capture_output=True, text=True, timeout=300)
                res = None
                for line in cp.stdout.splitlines():
                    if line.startswith("RESULT:"):
                        res = json.loads(line[len("RESULT:"):])
                        break
                if res is None:
                    res = {"ok": False, "errors": ["no RESULT (docker/eval failed)"]}
            except subprocess.TimeoutExpired:
                res = {"ok": False, "errors": ["timeout (>5min)"]}
            print(f"[run {tag}] -> {res.get('verdict', res.get('errors'))}", flush=True)
        self._json(res)


if __name__ == "__main__":
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), H) as httpd:
        print(f"click_server on http://127.0.0.1:{PORT}/  (page={PAGE})", flush=True)
        httpd.serve_forever()
