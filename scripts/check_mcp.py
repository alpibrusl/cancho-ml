#!/usr/bin/env python3
"""The fast MCP gate (docs/mcp.md gates A to D, docs/ci.md section 5): the protocol client starts build/ml-mcp, does the handshake, lists the tools and calls every one on the
bundled data against the command line, then the refusals as data, the schemas against the real answers and the hostile input.

    python3 scripts/check_mcp.py           # run the four modules of tests/conformance that speak the protocol; exit 1 on a failure

The modules: test_mcp (the fixtures of the command line through the tools, byte for byte, and the tool tests), test_mcp_schema (the schemas against the server),
test_mcp_hostile (malformed input) and test_authority's ml-mcp class. They are also part of `scripts/run_conformance.py`; this script is the quick entry the fast job names.
"""
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONF = ROOT / "tests" / "conformance"
MODULES = ["test_mcp", "test_mcp_schema", "test_mcp_hostile", "test_authority.AuthorityMcp"]


def main():
    if not (ROOT / "build" / "ml-mcp").exists() and not __import__("os").environ.get("ML_MCP_BIN"):
        print("check_mcp: build/ml-mcp does not exist (cancho build --bin ml-mcp)", file=sys.stderr)
        return 2
    t = time.time()
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", *MODULES], cwd=CONF, capture_output=True, text=True)
    dt = time.time() - t
    tail = p.stderr.strip().splitlines()
    ran = [l for l in tail if l.startswith("Ran ")]
    if p.returncode != 0:
        print(p.stdout + p.stderr)
        print("check_mcp: FAILED after %.1f s" % dt)
        return 1
    print("check_mcp: %s in %.1f s (%s)" % (ran[-1] if ran else "ok", dt, ", ".join(MODULES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
