"""A protocol client for `ml-mcp` (docs/mcp.md gates A to D): standard library only.

It starts the server over standard input and output, speaks newline-delimited JSON-RPC 2.0 (revision 2025-06-18) and exposes the pieces a test needs: the handshake, the
list of tools, a tool call, raw writes (a byte at a time, a message cut short), and the raw bytes of what the server answered, so that `structuredContent` can be compared
with the line the command line prints byte for byte (`raw_structured`).
"""

import json
import os
import pathlib
import select
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = os.environ.get("ML_MCP_BIN") or str(ROOT / "build" / "ml-mcp")
REVISION = "2025-06-18"


class Timeout(Exception):
    pass


class Client:
    def __init__(self, *flags, cwd=None, binary=None, env=None):
        self.argv = [binary or BIN, *[str(f) for f in flags]]
        self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env, bufsize=0)
        self.buf = b""
        self.next_id = 1
        self.raw_lines = []

    # ------------------------------------------------------------------ writing
    def write(self, data):
        self.proc.stdin.write(data)

    def write_slowly(self, data, chunk=1):
        for i in range(0, len(data), chunk):
            self.proc.stdin.write(data[i:i + chunk])
            self.proc.stdin.flush()

    def send_line(self, text):
        self.write((text if isinstance(text, bytes) else text.encode("utf-8")) + b"\n")

    def send(self, obj):
        self.send_line(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))

    def close_input(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass

    # ------------------------------------------------------------------ reading
    def read_line(self, timeout=60):
        """The next line the server wrote, without its line feed (bytes); None at the end of its output."""
        deadline = time.time() + timeout
        while b"\n" not in self.buf:
            left = deadline - time.time()
            if left <= 0:
                raise Timeout("no line from the server in %ds (so far %r)" % (timeout, self.buf[:200]))
            ready, _, _ = select.select([self.proc.stdout], [], [], left)
            if not ready:
                continue
            chunk = os.read(self.proc.stdout.fileno(), 1 << 20)
            if not chunk:
                rest, self.buf = self.buf, b""
                return rest if rest else None
            self.buf += chunk
        line, _, self.buf = self.buf.partition(b"\n")
        self.raw_lines.append(line)
        return line

    def read(self, timeout=60):
        line = self.read_line(timeout)
        return None if line is None else json.loads(line)

    def read_response(self, timeout=60):
        """The next response; (parsed, raw line)."""
        line = self.read_line(timeout)
        if line is None:
            return None, None
        return json.loads(line), line

    def expect_silence(self, timeout=0.3):
        """True when the server writes nothing for `timeout` seconds."""
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if self.buf:
            return False
        return not ready

    # ------------------------------------------------------------------ requests
    def request(self, method, params=None, id=None, timeout=60):
        if id is None:
            id = self.next_id
            self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        return self.read(timeout)

    def request_raw(self, method, params=None, id=None, timeout=60):
        """(parsed, raw line) of the response."""
        if id is None:
            id = self.next_id
            self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        return self.read_response(timeout)

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)

    def initialize(self, version=REVISION):
        r = self.request("initialize", {"protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "mcp_client.py", "version": "1"}})
        self.notify("notifications/initialized")
        return r

    def list_tools(self):
        return self.request("tools/list")["result"]["tools"]

    def call(self, name, arguments=None, timeout=300):
        params = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return self.request("tools/call", params, timeout=timeout)

    def call_raw(self, name, arguments=None, timeout=300):
        params = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return self.request_raw("tools/call", params, timeout=timeout)

    # ------------------------------------------------------------------ the end
    def finish(self, timeout=30):
        """Close the input, wait for the server to exit: (status, stderr)."""
        self.close_input()
        try:
            status = self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            status = self.proc.wait()
            raise Timeout("the server did not exit when its input ended")
        err = self.proc.stderr.read()
        return status, err

    def alive(self):
        return self.proc.poll() is None

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            pass
        for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                f.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.kill()


def raw_structured(raw_line):
    """The exact bytes of the `structuredContent` member of a tool-result line."""
    text = raw_line.decode("utf-8")
    at = text.index('"structuredContent":') + len('"structuredContent":')
    _, end = json.JSONDecoder().raw_decode(text, at)
    return text[at:end].encode("utf-8")


def started(*flags, cwd=None):
    """A client whose handshake is done."""
    c = Client(*flags, cwd=cwd)
    c.initialize()
    return c


if __name__ == "__main__":
    with started(*sys.argv[1:]) as c:
        print(json.dumps([t["name"] for t in c.list_tools()]))
