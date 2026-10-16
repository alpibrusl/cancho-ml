"""S10 (docs/serving-http.md section 5.4): where `ml-serve` listens, on real sockets.

* the default (and `--bind loopback`) accepts on 127.0.0.1 and is REFUSED (ConnectionRefusedError) on the non-loopback address of the machine, with no token;
* `--bind all` without `--token-file` is refused at start (serve.no-token, exit 2), `--bind all --no-token` is refused (args.conflict, exit 2), a value other than loopback and all is
  args.bad-value;
* `--bind all --token-file` answers on both addresses, and the token is enforced on both;
* a token on the default (loopback) bind is enforced too.

The non-loopback address is the source address the OS routes to a public address (a UDP socket is connected and nothing is sent). A machine with none skips the tests that need it
and says so; the skip is printed, so a result of "skipped" is never read as a pass.
"""

import json
import socket
import subprocess
import unittest

from harness import ROOT
from serve_lib import SERVE_BIN, Conn, Server, corpus

SB = None


def setUpModule():
    global SB
    SB, _, _ = corpus()


def tearDownModule():
    SB.cleanup()


def outside_address():
    """The machine's non-loopback IPv4 address, or None."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))
        addr = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    return None if addr.startswith("127.") or addr == "0.0.0.0" else addr


OUTSIDE = outside_address()
NEEDS_OUTSIDE = unittest.skipIf(OUTSIDE is None, "this machine has no non-loopback IPv4 address: the refusal on it is not tested here")


def get(addr, port, path="/health", headers=None, timeout=5):
    c = Conn(port, timeout, host=addr)
    try:
        return c.request("GET", path, headers=headers, close=True)
    finally:
        c.close()


def refused(addr, port):
    try:
        socket.create_connection((addr, port), timeout=3).close()
    except ConnectionRefusedError:
        return True
    except OSError as e:
        raise AssertionError("expected ConnectionRefusedError on %s:%d, got %r" % (addr, port, e))
    return False


def start(*args):
    p = subprocess.run([SERVE_BIN, *args], capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
    out = p.stdout.decode()
    return p.returncode, (json.loads(out.splitlines()[0]) if out.strip() else None), p.stderr.decode()


class Loopback(unittest.TestCase):
    def check(self, *flags):
        with Server(SB.dir, "linear.cml", *flags, no_token=False) as s:
            self.assertEqual(get("127.0.0.1", s.port).status, 200)
            self.assertIn("127.0.0.1", s.stderr())
            self.assertNotIn("0.0.0.0", s.stderr())
            if OUTSIDE is not None:
                self.assertTrue(refused(OUTSIDE, s.port), "the default server accepted a connection on %s" % OUTSIDE)
                self.assertTrue(s.alive())

    @NEEDS_OUTSIDE
    def test_the_default_listens_on_127_0_0_1_only(self):
        self.check()

    @NEEDS_OUTSIDE
    def test_bind_loopback_is_the_default(self):
        self.check("--bind", "loopback")

    def test_the_default_needs_no_token_and_answers(self):
        with Server(SB.dir, "linear.cml", no_token=False) as s:
            r = get("127.0.0.1", s.port, "/model")
            self.assertEqual(r.status, 200)

    def test_a_token_on_loopback_is_enforced(self):
        with Server(SB.dir, "linear.cml", token="loop-tok_1") as s:
            self.assertEqual(get("127.0.0.1", s.port, "/model").status, 401)
            self.assertEqual(get("127.0.0.1", s.port, "/model", {"Authorization": "Bearer loop-tok_1"}).status, 200)
            self.assertEqual(get("127.0.0.1", s.port, "/health").status, 200)
            if OUTSIDE is not None:
                self.assertTrue(refused(OUTSIDE, s.port))

    def test_no_token_is_still_accepted_on_loopback(self):
        with Server(SB.dir, "linear.cml", "--bind", "loopback") as s:  # no_token=True: --no-token is given
            self.assertEqual(get("127.0.0.1", s.port, "/model").status, 200)


class Everywhere(unittest.TestCase):
    def test_all_without_a_token_is_refused_at_start(self):
        code, d, err = start("--root", str(SB.dir), "--model-file", "linear.cml", "--bind", "all", "--port", "0")
        self.assertEqual((code, d["error"]["rule"]), (2, "serve.no-token"))
        self.assertNotIn("listening", err)
        code, d, err = start("--root", str(SB.dir), "--model-file", "linear.cml", "--bind", "all")
        self.assertEqual((code, d["error"]["rule"]), (2, "serve.no-token"))
        self.assertNotIn("listening", err)

    def test_all_with_no_token_is_refused_at_start(self):
        code, d, err = start("--root", str(SB.dir), "--model-file", "linear.cml", "--bind", "all", "--no-token")
        self.assertEqual((code, d["error"]["rule"]), (2, "args.conflict"))
        self.assertNotIn("listening", err)

    def test_the_value_of_bind(self):
        for v in ("everywhere", "", "ALL", "127.0.0.1", "0.0.0.0", "localhost"):
            code, d, _ = start("--root", str(SB.dir), "--model-file", "linear.cml", "--no-token", "--bind", v)
            self.assertEqual((code, d["error"]["rule"]), (2, "args.bad-value"), v)
        code, d, _ = start("--root", str(SB.dir), "--model-file", "linear.cml", "--no-token", "--bind")
        self.assertEqual((code, d["error"]["rule"]), (2, "args.missing-value"))

    def test_all_with_a_token_answers_on_both_addresses(self):
        tok = "all-tok_1"
        with Server(SB.dir, "linear.cml", "--bind", "all", token=tok, no_token=False) as s:
            self.assertIn("0.0.0.0", s.stderr())
            addrs = ["127.0.0.1"] + ([OUTSIDE] if OUTSIDE else [])
            for a in addrs:
                self.assertEqual(get(a, s.port, "/health").status, 200, a)
                self.assertEqual(get(a, s.port, "/model").status, 401, a)
                self.assertEqual(get(a, s.port, "/model", {"Authorization": "Bearer wrong"}).status, 401, a)
                self.assertEqual(get(a, s.port, "/model", {"Authorization": "Bearer " + tok}).status, 200, a)
            if OUTSIDE is None:
                print("\nnote: no non-loopback address on this machine: --bind all was tested on 127.0.0.1 only")

    def test_the_flag_is_in_the_spec(self):
        p = subprocess.run([SERVE_BIN, "introspect"], capture_output=True)
        d = json.loads(p.stdout)
        flags = {f["name"]: f for f in d["flags"]}
        self.assertIn("--bind", flags)
        self.assertEqual(flags["--bind"].get("default"), "loopback")
        self.assertIn("loopback/all", json.dumps(flags["--bind"]))


if __name__ == "__main__":
    unittest.main()
