#!/usr/bin/env python3
"""G-mutants for the sums (docs/spikes.md section 15.3, correctness side): each mutant is one source edit of spike_q.cho, rebuilt; the oracle
gates of oracle_q.py (n = 100000) must FAIL on it. The unmutated control must pass first. Exit 1 if a mutant survives."""
import os, pathlib, subprocess, sys, tempfile, shutil
import qlib

SRC = qlib.HERE / "spike_q.cho"
CANCHO = os.environ.get("CANCHO", str(qlib.ROOT.parent / "cancho-compiler-a4572ea" / "target" / "release" / "cancho"))
def build(src_text, out, order_text=None, tmp=None):
    d = pathlib.Path(tmp)
    (d / "spike_q.cho").write_text(src_text)
    (d / "facc.cho").write_text((qlib.HERE / "facc.cho").read_text())
    (d / "order.cho").write_text(order_text if order_text is not None else (qlib.ROOT / "lib/linalg/order.cho").read_text())
    subprocess.run([CANCHO, "build", "--std", str(d / "order.cho"), str(d / "facc.cho"), str(d / "spike_q.cho"), "-o", out], check=True, capture_output=True)


def oracle(binary):
    env = dict(os.environ, Q_BIN=binary)
    p = subprocess.run([sys.executable, str(qlib.HERE / "oracle_q.py"), "100000"], env=env, capture_output=True, text=True)
    return p.returncode == 0, p.stdout


def main():
    src = SRC.read_text()
    order = (qlib.ROOT / "lib/linalg/order.cho").read_text()
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m")
        build(src, out, None, tmp)
        ok, txt = oracle(out)
        print(f"control (unmutated): oracle gates pass = {ok}")
        if not ok:
            print(txt); sys.exit(1)
        muts = [
            ("C-block-255", src, "var hi = (b + 1) * 256;", "var hi = (b + 1) * 255;", None),
            ("C-tree-leftfold (order.cho: merge the two top slots whatever their level)", src, None, None, ("while sp >= 2 && lv[sp - 1] == lv[sp - 2] {", "while sp >= 2 {")),
            ("B-key-sign", src, "        bits = key ^ 0x7fffffffffffffff;\n    }\n    return decode(bits, tab);", "        bits = key;\n    }\n    return decode(bits, tab);", None),
            ("B-wrong-accumulator", src, "facc.add_bits(av, 252 * 73, bits_of(loss));", "facc.add_bits(av, 251 * 73, bits_of(loss));", None),
            ("B-term-perturbed", src, "facc.add_bits(av, (lo + b) * 73, bits_of(th));", "facc.add_bits(av, (lo + b) * 73, bits_of(th * 1.0000000000000002));", None),
            ("A-hessian-t-order", src, "hrow[b] = hrow[b] + t2 * rs[b];", "hrow[b] = hrow[b] + rs[b] * t2 * 1.0000000000000002;", None),
            ("A-loss-term-perturbed", src, "acc[252] = acc[252] + loss;\n        var a = 0;", "acc[252] = acc[252] + (loss * 1.0000000000000002);\n        var a = 0;", None),
        ]
        survivors = []
        for name, text, old, new, ordedit in muts:
            t = text
            o = order
            if old is not None:
                if old not in t:
                    print(f"{name}: edit site not found"); sys.exit(2)
                t = t.replace(old, new, 1)
            if ordedit is not None:
                if ordedit[0] not in o:
                    print(f"{name}: order edit site not found"); sys.exit(2)
                o = o.replace(ordedit[0], ordedit[1])
            build(t, out, o, tmp)
            ok, txt = oracle(out)
            killed = not ok
            first = [l for l in txt.splitlines() if (l.startswith("G-") and ("253/253" not in l) and "True" not in l)]
            print(f"mutant {name}: {'KILLED' if killed else 'SURVIVED'}  {first[:2]}")
            if not killed:
                survivors.append(name)
    print("survivors:", survivors if survivors else "none")
    sys.exit(1 if survivors else 0)


main()
