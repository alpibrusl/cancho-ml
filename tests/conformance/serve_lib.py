"""Shared pieces of the ml-serve gates (docs/serving-http.md section 9): start the binary on a free port, speak HTTP to it over real sockets with exact control of the bytes,
read the answers with number tokens kept as text."""

import json
import os
import pathlib
import re
import socket
import subprocess
import tempfile
import time

from harness import ROOT, Sandbox

SERVE_BIN = os.environ.get("ML_SERVE_BIN") or str(ROOT / "build" / "ml-serve")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    """ml-serve on a free port. `flags` are the flags after --model-file; a token is given with token="..." (written to a file in the root)."""

    def __init__(self, root, model, *flags, token=None, no_token=True, stdin=subprocess.DEVNULL):
        self.root = pathlib.Path(root)
        self.model = model
        self.flags = list(flags)
        self.token = token
        self.no_token = no_token and token is None
        self.stdin = stdin
        self.proc = None
        self.port = None
        self.log = None
        self.conns = []

    def argv(self, port):
        a = [SERVE_BIN, "--root", str(self.root), "--model-file", self.model, "--port", str(port)]
        if self.token is not None:
            (self.root / "token.txt").write_text(self.token + "\n")
            a += ["--token-file", "token.txt"]
        elif self.no_token:
            a += ["--no-token"]
        return a + self.flags

    def start(self, wait=True):
        last = None
        for attempt in range(5):
            self.port = free_port()
            if self.log is not None:
                self.log.close()
            self.log = tempfile.TemporaryFile()
            self.proc = subprocess.Popen(self.argv(self.port), stdout=subprocess.PIPE, stderr=self.log, stdin=self.stdin)
            if not wait:
                return self
            deadline = time.time() + 10
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    break
                try:
                    c = socket.create_connection(("127.0.0.1", self.port), timeout=1)
                    c.close()
                    return self
                except OSError:
                    time.sleep(0.02)
            last = (self.proc.poll(), self.stderr())
            self.stop()
            if last[0] is not None and last[0] != 5:
                break
        raise RuntimeError("ml-serve did not start: %r" % (last,))

    def stderr(self):
        self.log.seek(0)
        return self.log.read().decode("utf-8", "replace")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        for c in self.conns:
            c.close()
        self.conns = []
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            if self.proc.stdout:
                self.proc.stdout.close()
        self.proc = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    def conn(self, timeout=10):
        c = Conn(self.port, timeout)
        self.conns.append(c)
        return c

    def request(self, method, path, body=None, headers=None, json_body=None, timeout=10):
        c = Conn(self.port, timeout)
        try:
            return c.request(method, path, body=body, headers=headers, json_body=json_body)
        finally:
            c.close()


class Response:
    def __init__(self, status, reason, headers, body, raw_head):
        self.status = status
        self.reason = reason
        self.headers = headers
        self.body = body
        self.raw_head = raw_head

    def text(self):
        return self.body.decode("utf-8")

    def json(self):
        return loads_tokens(self.text())

    def ctype(self):
        return self.headers.get("content-type", "")

    def __repr__(self):
        return "<%d %s %r>" % (self.status, self.ctype(), self.body[:200])


def loads_tokens(text):
    """JSON with every number kept as its text token (str): the comparison of docs/serving-http.md section 4.1 is `str == str`."""
    return json.loads(text, parse_float=str, parse_int=str, parse_constant=_no_constant)


def _no_constant(name):
    raise ValueError("not JSON: " + name)


class Conn:
    def __init__(self, port, timeout=10, host="127.0.0.1"):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.buf = b""

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def send(self, data):
        self.sock.sendall(data)

    def _fill(self):
        d = self.sock.recv(65536)
        if not d:
            raise EOFError("closed")
        self.buf += d

    def read_response(self):
        while b"\r\n\r\n" not in self.buf:
            self._fill()
        head, rest = self.buf.split(b"\r\n\r\n", 1)
        lines = head.split(b"\r\n")
        m = re.match(rb"HTTP/1\.[01] (\d{3}) ?(.*)", lines[0])
        status, reason = int(m.group(1)), m.group(2).decode()
        headers = {}
        for l in lines[1:]:
            k, v = l.split(b":", 1)
            headers[k.decode().lower()] = v.strip().decode()
        n = int(headers.get("content-length", "0"))
        self.buf = rest
        while len(self.buf) < n:
            self._fill()
        body, self.buf = self.buf[:n], self.buf[n:]
        return Response(status, reason, headers, body, head)

    @staticmethod
    def build(method, path, body=None, headers=None, json_body=None, close=False):
        hs = {"Host": "localhost"}
        if json_body is not None:
            body = json_body if isinstance(json_body, bytes) else json.dumps(json_body).encode()
            hs["Content-Type"] = "application/json"
        if body is not None:
            hs["Content-Length"] = str(len(body))
        if close:
            hs["Connection"] = "close"
        hs.update(headers or {})
        head = "%s %s HTTP/1.1\r\n" % (method, path) + "".join("%s: %s\r\n" % (k, v) for k, v in hs.items() if v is not None) + "\r\n"
        return head.encode() + (body or b"")

    def request(self, method, path, body=None, headers=None, json_body=None, close=False):
        self.send(self.build(method, path, body, headers, json_body, close))
        return self.read_response()


def corpus():
    """(sandbox, models, inputs) of serving_corpus: every kind of model, 19 of them, and their inputs."""
    import serving_corpus as C

    sb = Sandbox()
    models, inputs = C.build(sb)
    return sb, models, inputs
