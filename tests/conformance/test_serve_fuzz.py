"""S4 (docs/serving-http.md sections 5.1 and 9): fuzz of request bodies, paths and headers. No input traps the server and none hangs it.

Seeded random bytes, truncations, byte flips, insertions and deletions of valid requests, paths and headers drawn from a list of hostile shapes (`..`, percent escapes, NUL, a path of 20 KiB,
headers by the hundred, a `Content-Length` that is negative, overflowing or twice), HTTP versions that do not exist, and JSON bodies of hostile shape (deep nesting, a long string, huge
numbers, lone surrogates, a million-digit number). Every case runs on its own connection, many at once; the server is `--idle-seconds 2`, so a request that never ends is closed in a few seconds.
After the cases the process is alive (a trap would have ended it), a normal request is answered at once, and the answers to the cases are all either a well-formed HTTP response or a closed
connection. `SERVE_FUZZ_N` sets the number of cases (default 3000).
"""

import json
import os
import random
import re
import resource
import socket
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from serve_lib import Conn, Server, corpus

N = int(os.environ.get("SERVE_FUZZ_N", "3000"))
SB = None
JSON = {"Content-Type": "application/json"}
REC = b'{"a":3,"b":2.50,"c":0.5}'


def setUpModule():
    global SB
    SB, _, _ = corpus()


def tearDownModule():
    SB.cleanup()


def valid_requests():
    return [
        Conn.build("POST", "/predict", body=REC, headers=JSON),
        Conn.build("POST", "/predict/batch", body=b'{"rows":[' + REC + b"," + REC + b"]}", headers=JSON),
        Conn.build("GET", "/health"),
        Conn.build("GET", "/model"),
        Conn.build("GET", "/openapi.json"),
        Conn.build("DELETE", "/predict"),
        Conn.build("GET", "/nope?x=1"),
    ]


def hostile_paths():
    return ["/", "//", "/..", "/../../etc/passwd", "/predict/..", "/%00", "/%2e%2e/", "/predict%00", "/pre%64ict", "/predict/", "/predict?x=%ff", "/predict?" + "a=b&" * 2000, "/" + "a" * 20000, "/é",
            "/predict#frag", "*", "http://x/predict", "/predict/batch/", "/model/../health", "/health\x00", "/%", "/%zz", "/ ", "/predict HTTP/1.1"]


def hostile_headers():
    hs = [b"Content-Length: -1", b"Content-Length: 99999999999999999999", b"Content-Length: 1e3", b"Content-Length: 0x10", b"Content-Length: 5\r\nContent-Length: 6", b"Content-Length:",
          b"Transfer-Encoding: chunked\r\nContent-Length: 5", b"Transfer-Encoding: gzip", b"Transfer-Encoding: chunked, chunked", b"Expect: 100-continue", b"Connection: upgrade\r\nUpgrade: websocket",
          b"Host:", b"Host: " + b"a" * 5000, b"X: " + b"\xff" * 100, b"X-A\x00: b", b": novalue", b"NoColon", b" leading: space", b"Content-Type: " + b"a" * 9000, b"Authorization: Bearer " + b"a" * 9000,
          b"Content-Type: application/json\r\nContent-Type: text/plain", b"Cookie: " + b"a=b; " * 3000, b"Connection: close\r\nConnection: keep-alive"]
    return hs


def hostile_chunked():
    return [b"0\r\n\r\n", b"ffffffffffffffff\r\nx", b"-1\r\nx\r\n0\r\n\r\n", b"5\r\nabc", b"5\r\nabcdefgh\r\n0\r\n\r\n", b"zz\r\n", b"5;ext=" + b"a" * 5000 + b"\r\nhello\r\n0\r\n\r\n", b"1\r\n{\r\n0\r\n\r\n",
            (b"1\r\n{\r\n" * 3000) + b"0\r\n\r\n"]


def hostile_bodies():
    big_digits = b"1" * 3500
    return [b"[" * 3000, b"{" * 3000, b'{"a":' * 700, b'{"a":"' + b"x" * 3500 + b'"}', b'{"a":' + big_digits + b',"b":1,"c":1}', b'{"a":1,"b":' + big_digits + b'.5,"c":1}', b'{"a":1,"b":1,"c":0.' + big_digits + b"}",
            b'{"a":1,"b":1,"c":1e' + big_digits + b"}", b'{"a":1,"b":1,"c":1e-' + big_digits + b"}", b'{"a":"\\ud800","b":1,"c":1}', b'{"a":"\\udc00\\ud800","b":1,"c":1}', b'{"a\\u0000":1}',
            b'{"a":1,"b":1,"c":1,"y":"\\u0000\\ud83d\\ude00"}', b'{"rows":[' + b"{}," * 1000 + b"{}]}", b'{"rows":' + b"[" * 100 + b"]" * 100 + b"}", b"\xff\xfe\x00{", b'{"\xc3\x28":1}',
            b'{"a":-0,"b":-0.0,"c":-0e0}', b'{"a":1,"b":1,"c":1}' * 200, b'{"a":1,"b":1,"c":1,"a":1,"b":1,"c":1}', b"1" * 3500, b'"' + b"\\\\" * 1700 + b'"', b"\x00" * 3000, b"null", b"{}", b"[]"]


def cases(rng, n):
    reqs = valid_requests()
    out = []
    for i in range(n):
        kind = rng.randrange(10)
        if kind == 0:
            out.append(("random bytes", bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 600)))))
        elif kind == 1:
            r = bytearray(rng.choice(reqs))
            for _ in range(rng.randint(1, 8)):
                if r:
                    r[rng.randrange(len(r))] = rng.getrandbits(8)
            out.append(("flipped", bytes(r)))
        elif kind == 2:
            r = rng.choice(reqs)
            out.append(("truncated", r[: rng.randrange(len(r) + 1)]))
        elif kind == 3:
            r = bytearray(rng.choice(reqs))
            for _ in range(rng.randint(1, 4)):
                at = rng.randrange(len(r) + 1)
                if rng.random() < 0.5:
                    r[at:at] = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 20)))
                else:
                    del r[at: at + rng.randint(1, 20)]
            out.append(("edited", bytes(r)))
        elif kind == 4:
            p = rng.choice(hostile_paths())
            m = rng.choice(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH", "TRACE", "CONNECT", "GE\x00T", "get", ""])
            out.append(("path", ("%s %s HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n" % (m, p)).encode("utf-8", "surrogatepass")))
        elif kind == 5:
            h = rng.choice(hostile_headers())
            out.append(("header", b"POST /predict HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n" + h + b"\r\n\r\n" + REC))
        elif kind == 6:
            v = rng.choice(["HTTP/0.9", "HTTP/1.0", "HTTP/1.2", "HTTP/2.0", "HTTP/9.9", "http/1.1", "HTTP/1.1 ", "HTTP", ""])
            out.append(("version", ("POST /predict %s\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (v, len(REC))).encode() + REC))
        elif kind == 7:
            b = rng.choice(hostile_bodies())
            path = rng.choice(["/predict", "/predict/batch"])
            out.append(("body", Conn.build("POST", path, body=b, headers=JSON)))
        elif kind == 8:
            c = rng.choice(hostile_chunked())
            out.append(("chunked", b"POST /predict HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n" + c))
        else:
            # headers by the hundred
            hs = b"".join(b"X-H%d: %s\r\n" % (j, b"v" * rng.randint(0, 50)) for j in range(rng.randint(50, 400)))
            out.append(("many headers", b"POST /predict HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n" % len(REC) + hs + b"\r\n" + REC))
    out.append(("pipelined garbage", b"".join(rng.choice(reqs) for _ in range(20)) + bytes(rng.getrandbits(8) for _ in range(100))))
    return out


HEAD = re.compile(rb"HTTP/1\.[01] \d{3} ")


def one(port, data, deadline):
    """Send `data`, read until the server closes, answers and stays quiet for a moment, or the deadline. ('answered' | 'closed' | 'quiet', first status or None)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=3)
    s.settimeout(1.0)
    got = b""
    try:
        try:
            s.sendall(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
        t0 = time.time()
        while time.time() - t0 < deadline:
            try:
                d = s.recv(65536)
            except socket.timeout:
                if got:
                    return ("answered", got)
                continue
            except ConnectionResetError:
                return ("closed", got)
            if not d:
                return ("closed", got)
            got += d
            if len(got) > 4 and b"\r\n\r\n" in got and not data.startswith(b"x"):
                # an answer has come; a pipeline may bring more, the next recv times out
                pass
        return ("quiet", got)
    finally:
        s.close()


class Fuzz(unittest.TestCase):
    def test_no_trap_no_hang(self):
        rng = random.Random(int(os.environ.get("SERVE_FUZZ_SEED", "20261015")))
        todo = cases(rng, N)
        with Server(SB.dir, "linear.cml", "--idle-seconds", "2", "--max-body-bytes", "4096", "--max-batch-rows", "5") as srv:
            rss0 = rss(srv.proc.pid)
            tally = {}
            bad = []

            def work(case):
                what, data = case
                try:
                    res, got = one(srv.port, data, 6.0)
                except OSError as e:
                    return (what, "error", repr(e))
                if got and not HEAD.match(got):
                    return (what, "malformed", got[:80])
                return (what, res, None)

            with ThreadPoolExecutor(max_workers=48) as pool:
                for what, res, extra in pool.map(work, todo):
                    tally[(what, res)] = tally.get((what, res), 0) + 1
                    if res in ("malformed", "error", "quiet"):
                        bad.append((what, res, extra))
            self.assertTrue(srv.alive(), "the server ended: a trap? exit %r\n%s" % (srv.proc.poll(), srv.stderr()[-500:]))
            # a request after all of it is answered at once
            t0 = time.time()
            r = srv.request("POST", "/predict", body=REC, headers=JSON, timeout=3)
            self.assertEqual(r.status, 200)
            self.assertLess(time.time() - t0, 1.0)
            self.assertEqual(srv.request("GET", "/health").status, 200)
            self.assertEqual(bad, [], bad[:5])
            rss1 = rss(srv.proc.pid)
            print("serve fuzz: %d cases, no trap, no hang; outcomes %s; resident set %d KiB before, %d KiB after" % (len(todo), dict(sorted((k[0] + "/" + k[1], v) for k, v in tally.items())), rss0, rss1))
            # the memory of the server did not grow with the cases (a leak per request would show: 3000 cases of up to 20 KiB)
            self.assertLess(rss1, rss0 + 200 * 1024)

    def test_a_connection_that_sends_nothing_is_closed(self):
        with Server(SB.dir, "linear.cml", "--idle-seconds", "1") as srv:
            s = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
            t0 = time.time()
            s.settimeout(6)
            try:
                d = s.recv(10)
            except socket.timeout:
                d = None
            self.assertEqual(d, b"", "an idle connection must be closed by the server")
            self.assertLess(time.time() - t0, 5)
            s.close()
            # half a head, then nothing
            s = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
            s.sendall(b"POST /predict HTTP/1.1\r\nHost: x\r\n")
            s.settimeout(6)
            t0 = time.time()
            self.assertEqual(s.recv(10), b"")
            self.assertLess(time.time() - t0, 5)
            s.close()

    def test_many_connections_at_once(self):
        with Server(SB.dir, "linear.cml", "--idle-seconds", "3", "--max-body-bytes", "4096") as srv:
            socks = []
            try:
                for _ in range(300):
                    socks.append(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
                r = srv.request("GET", "/health")
                self.assertEqual(r.status, 200)
                for s in socks[:50]:
                    s.sendall(Conn.build("GET", "/health"))
                for s in socks[:50]:
                    s.settimeout(5)
                    self.assertIn(b"200 OK", s.recv(1000))
            finally:
                for s in socks:
                    s.close()
            self.assertTrue(srv.alive())

    def test_the_number_of_connections_is_bounded_by_the_buffers(self):
        # docs/serving-http.md 5.1: the loop keeps one buffer of (--max-body-bytes + 16 KiB) for each connection within 256 MiB, so the default 1 MiB body allows about 240
        with Server(SB.dir, "linear.cml", "--idle-seconds", "5") as srv:
            socks = []
            served = 0
            try:
                for _ in range(400):
                    try:
                        socks.append(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
                    except OSError:
                        break
                for s in socks:
                    try:
                        s.sendall(Conn.build("GET", "/health"))
                    except OSError:
                        pass
                for s in socks:
                    s.settimeout(3)
                    try:
                        if b"200 OK" in s.recv(1000):
                            served += 1
                    except OSError:
                        pass
            finally:
                for s in socks:
                    s.close()
            self.assertTrue(srv.alive())
            self.assertEqual(srv.request("GET", "/health").status, 200)
            print("serve fuzz: %d of %d simultaneous connections served at the default --max-body-bytes" % (served, len(socks)))
            self.assertGreaterEqual(served, 200)
            self.assertLess(served, 400)

    def test_a_client_that_does_not_read(self):
        # 50 pipelined requests for the largest answer, never read: the server may close that connection, and goes on serving the others
        with Server(SB.dir, "linear.cml", "--idle-seconds", "2", "--max-batch-rows", "1000") as srv:
            rows = b'{"rows":[' + b",".join([REC] * 1000) + b"]}"
            s = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
            req = Conn.build("POST", "/predict/batch", body=rows, headers=JSON)
            try:
                s.sendall(req * 50)
            except OSError:
                pass
            time.sleep(0.5)
            for _ in range(3):
                self.assertEqual(srv.request("GET", "/health").status, 200)
            s.close()
            self.assertTrue(srv.alive())


def rss(pid):
    import subprocess

    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) if out else 0


if __name__ == "__main__":
    unittest.main()
