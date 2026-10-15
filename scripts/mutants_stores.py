#!/usr/bin/env python3
"""Mutants of the store gates (docs/stores.md section 6, PS1 to PS6). Each mutant is applied to a COPY of this tree (never to the tree itself) and the
gate that must catch it is run; the mutant is killed if the gate exits non-zero. A control (the unmutated copy) must pass first.

    python3 scripts/mutants_stores.py --fast      # the gates that need no compiler: the closure, the copy and circularity checks, the consumer check
    python3 scripts/mutants_stores.py             # all of them (about 15 minutes: every --check publishes 23 stores, every consumption installs and builds)
    python3 scripts/mutants_stores.py NAME...     # the mutants whose name contains one of the words

Exit 1 if a control fails or a mutant survives. The compiler is $CANCHO or `cancho` on PATH (the pin of cancho.toml).
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
# another compiler, next to the pinned one (…/cancho-compiler-a4572ea/target/release/cancho): the check must refuse it
_CANCHO = pathlib.Path(os.environ.get("CANCHO") or "/nonexistent/a/b/c/d")
E47D = _CANCHO.parents[3] / "cancho-compiler-e47d647" / "target" / "release" / "cancho" if len(_CANCHO.parents) > 3 else pathlib.Path("/nonexistent")
GIT = ["git", "-c", "user.name=mutants", "-c", "user.email=mutants@example.invalid", "-c", "commit.gpgsign=false"]


def run(cmd, cwd, env=None, timeout=1800):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return r.returncode, r.stdout


def copy_tree(dst):
    """the files of this tree a store gate reads (not .git, build, the other binaries)"""
    for d in ("lib", "scripts", "docs", ".cancho-vcs"):
        shutil.copytree(ROOT / d, dst / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("cancho.toml",):
        shutil.copy(ROOT / f, dst / f)
    (dst / "tools" / "ml").mkdir(parents=True)


def edit(path, old, new):
    t = path.read_text()
    assert t.count(old) == 1, f"{path}: pattern occurs {t.count(old)} times"
    path.write_text(t.replace(old, new))


def first(p, pattern):
    return sorted(p.glob(pattern))[0]


def commit(d, msg):
    subprocess.run(GIT + ["add", "-A"], cwd=d, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(GIT + ["commit", "-q", "-m", msg], cwd=d, check=True, stdout=subprocess.DEVNULL)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, stdout=subprocess.PIPE, text=True, check=True).stdout.strip()


def flip(path):
    b = bytearray(path.read_bytes())
    i = b.index(b"fn ") if b"fn " in b else len(b) // 2
    b[i] = b[i] ^ 1
    path.write_bytes(bytes(b))


# ---- mutants of the repository gates: (name, speed, mutate(copy), gate(copy) -> (cmd, env))
def pub(copy, *a, env=None):
    return [PY, str(copy / "scripts" / "publish_stores.py")] + list(a), env or {}


def pin(copy, env=None):
    return [PY, str(copy / "scripts" / "check_stores_pin.py")], dict({"ROOT": str(copy)}, **(env or {}))


WHY = {
    "closure-": "not published",
    "closure-file-declares-another-module": "does not declare module",
    "pin-copy": "a copy",
    "pin-circular": "does not consume its own stores",
    "pin-doc": "docs/stores.md lists modules",
    "pin-api": "stores_api.txt is missing",
    "consumer-two-revs": "different revs",
    "consumer-short-rev": "not a full commit hash",
    "consumer-unlisted": "not .cancho-vcs/<a published module>",
    "consumer-wrong-path": "not .cancho-vcs/<a published module>",
    "consumer-copies": "a copy",
    "check-other-compiler": "cancho.toml pins",
    "check-": "are not what a fresh publish gives",
    "consume-store-removed-checked": "not what a fresh publish gives",
    "consume-source-changed": "not what a fresh publish gives",
    "consume-behaviour": "differs from the definitions",
    "consume-rev-without": "no .cancho-vcs",
    "consume-store-removed": "cancho",
    "consume-store-byte": "cancho",
}


def why(name):
    """the longest key that prefixes the name"""
    keys = [k for k in WHY if name.startswith(k)]
    return WHY[max(keys, key=len)] if keys else ""


def mutants():
    M = []

    def add(name, speed, mutate, gate, control=False):
        M.append((name, speed, mutate, gate))

    # PS4: the closure
    add("closure-matrix-without-tensor", "fast", lambda c: None, lambda c: pub(c, "--closure", "--modules", "ml.matrix,ml.limits,ml.refuse"))
    add("closure-chol-without-order", "fast", lambda c: None, lambda c: pub(c, "--closure", "--modules", "linalg.chol,linalg.tensor,ml.refuse"))
    add("closure-splits-without-rng", "fast", lambda c: None, lambda c: pub(c, "--closure", "--modules", "metrics.splits,ml.limits,ml.refuse,metrics.rules"))
    add("closure-source-gains-an-import", "fast", lambda c: edit(c / "lib/core/rng.cho", "module ml.rng;\n", "module ml.rng;\n\nimport ml.dict;\n"),
        lambda c: pub(c, "--closure"))
    add("closure-file-declares-another-module", "fast", lambda c: edit(c / "lib/core/fsq.cho", "module ml.fsq;", "module ml.fsqq;"),
        lambda c: pub(c, "--closure"))
    # PS5: no copy, no circle, the list agrees with the document
    add("pin-copy-of-rng-in-tools", "fast", lambda c: shutil.copy(c / "lib/core/rng.cho", c / "tools/ml/rngcopy.cho"), lambda c: pin(c))
    add("pin-copy-of-chol-in-models", "fast", lambda c: (c / "lib/models").mkdir(exist_ok=True) or shutil.copy(c / "lib/linalg/chol.cho", c / "lib/models/chol2.cho"),
        lambda c: pin(c))
    add("pin-circular-dependency", "fast",
        lambda c: (c / "cancho.toml").write_text((c / "cancho.toml").read_text() + '\n[dependencies.rng]\ngit = "https://github.com/alpibrusl/cancho-ml"\n'
                                                 'rev = "' + "a" * 40 + '"\npath = ".cancho-vcs/ml.rng"\n'), lambda c: pin(c))
    add("pin-doc-lists-an-extra-module", "fast", lambda c: edit(c / "docs/stores.md", "`metrics.rules`, `metrics.ratio`", "`metrics.rules`, `metrics.logloss`, `metrics.ratio`"),
        lambda c: pin(c))
    add("pin-doc-misses-a-module", "fast", lambda c: edit(c / "docs/stores.md", "`ml.fbits`, ", ""), lambda c: pin(c))
    add("pin-api-file-missing", "fast", lambda c: (c / "docs/stores_api.txt").unlink(), lambda c: pin(c))

    def consumer(files, toml):
        def mutate(c):
            d = c / "consumer"
            d.mkdir()
            (d / "cancho.toml").write_text(toml)
            for name, text in files.items():
                (d / name).parent.mkdir(parents=True, exist_ok=True)
                (d / name).write_text(text)
        return mutate

    rev1, rev2 = "1" * 40, "2" * 40

    def block(label, module, rev, path=None, git="https://github.com/alpibrusl/cancho-ml"):
        return f'[dependencies.{label}]\ngit = "{git}"\nrev = "{rev}"\npath = "{path or ".cancho-vcs/" + module}"\n\n'

    head = '[package]\nname = "s"\ncancho = "a4572ea3b506e4bcec1c4be43deda1043c11ba9f"\n\n'
    cgate = lambda c: pin(c, {"CONSUMER": str(c / "consumer")})
    add("consumer-two-revs", "fast", consumer({}, head + block("a", "ml.rng", rev1) + block("b", "ml.exact", rev2)), cgate)
    add("consumer-short-rev", "fast", consumer({}, head + block("a", "ml.rng", "abc123")), cgate)
    add("consumer-unlisted-module", "fast", consumer({}, head + block("a", "ml.dict", rev1)), cgate)
    add("consumer-wrong-path", "fast", consumer({}, head + block("a", "ml.rng", rev1, "lib/core")), cgate)
    add("consumer-copies-a-module", "fast", consumer({"src/rng.cho": "module ml.rng;\n"}, head + block("a", "ml.rng", rev1)), cgate)
    # PS2, PS3: --check
    chk = lambda c: pub(c, "--check")
    add("check-source-comment-byte", "slow", lambda c: edit(c / "lib/core/rng.cho", "// SplitMix64 and derived streams", "// SplitMix64 and derived streamz"), chk)
    add("check-source-code-byte", "slow", lambda c: edit(c / "lib/core/rng.cho", "z ^ z >> 31 & 0x1FFFFFFFF", "z ^ z >> 30 & 0x1FFFFFFFF"), chk)
    add("check-source-in-linalg", "slow", lambda c: edit(c / "lib/linalg/order.cho", "module linalg.order;", "module linalg.order; "), chk)
    add("check-store-source-byte", "slow", lambda c: flip(first(c / ".cancho-vcs/ml.exact/sources", "*.cho")), chk)
    add("check-store-op-byte", "slow", lambda c: flip(first(c / ".cancho-vcs/metrics.rank/ops", "*.json")), chk)
    add("check-store-manifest-byte", "slow", lambda c: flip(c / ".cancho-vcs/linalg.chol/manifest.json"), chk)
    add("check-store-requires-byte", "slow", lambda c: flip(c / ".cancho-vcs/metrics.splits/requires/0.json"), chk)
    add("check-store-file-missing", "slow", lambda c: first(c / ".cancho-vcs/ml.fbits/ops", "*.json").unlink(), chk)
    add("check-store-missing", "slow", lambda c: shutil.rmtree(c / ".cancho-vcs/ml.fsq"), chk)
    add("check-store-extra-dir", "slow", lambda c: (c / ".cancho-vcs/ml.extra").mkdir(), chk)
    add("check-store-extra-file", "slow", lambda c: (c / ".cancho-vcs/ml.rng/notes.txt").write_text("x"), chk)
    add("check-stores-dir-missing", "slow", lambda c: shutil.rmtree(c / ".cancho-vcs"), chk)
    add("check-api-file-edited", "slow", lambda c: edit(c / "docs/stores_api.txt", "pub fn gold() -> [] int", "pub fn gold() -> [] float"), chk)
    add("check-api-file-missing", "slow", lambda c: (c / "docs/stores_api.txt").unlink(), chk)
    add("check-pub-signature-changed", "slow", lambda c: edit(c / "lib/core/rng.cho", "pub fn gold() -> [] int", "pub fn gold() -> [] int "), chk)
    if E47D.exists():
        add("check-other-compiler", "slow", lambda c: None, lambda c: pub(c, "--check", env={"CANCHO": str(E47D)}))
    return M


def consume_mutants():
    """each: (name, how to change a git copy of the tree, flags of consume_stores.py)"""
    def drop_store(c):
        shutil.rmtree(c / ".cancho-vcs/ml.rng")

    def byte_store(c):
        flip(first(c / ".cancho-vcs/ml.exact/sources", "*.cho"))

    def stale(c):
        edit(c / "lib/core/rng.cho", "z ^ z >> 31 & 0x1FFFFFFFF", "z ^ z >> 30 & 0x1FFFFFFFF")

    def changed_and_republished(c):
        stale(c)
        code, out = run([PY, str(c / "scripts/publish_stores.py")], c)
        assert code == 0, out

    return [
        ("consume-store-removed", drop_store, ["--no-check"]),
        ("consume-store-removed-checked", drop_store, []),
        ("consume-store-byte", byte_store, ["--no-check"]),
        ("consume-source-changed-stores-stale", stale, []),
        ("consume-behaviour-changed-and-republished", changed_and_republished, []),
    ]


def main():
    words = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv
    wanted = lambda n, speed="slow": (not words or any(w in n for w in words)) and not (fast and speed == "slow")
    failures = []
    with tempfile.TemporaryDirectory(prefix="stores-mutants-") as tmp:
        tmp = pathlib.Path(tmp)
        control = tmp / "control"
        control.mkdir()
        copy_tree(control)
        for name, gate in (("closure", pub(control, "--closure")), ("pin", pin(control))):
            code, out = run(gate[0], control, gate[1])
            print(f"control {name}: {'passes' if code == 0 else 'FAILS'}", flush=True)
            if code != 0:
                print(out[-800:])
                sys.exit(1)
        if not fast:
            code, out = run(pub(control, "--check")[0], control)
            print(f"control --check: {'passes' if code == 0 else 'FAILS'}", flush=True)
            if code != 0:
                print(out[-800:])
                sys.exit(1)
        killed = 0
        for i, (name, speed, mutate, gate) in enumerate(mutants()):
            if not wanted(name, speed):
                continue
            c = tmp / f"m{i}"
            c.mkdir()
            copy_tree(c)
            mutate(c)
            cmd, env = gate(c)
            code, out = run(cmd, c, env)
            status = "killed" if code != 0 else "SURVIVED"
            if code != 0 and why(name) not in out:
                status = "KILLED FOR ANOTHER REASON"
            print(f"{name}: {status}", flush=True)
            if status != "killed":
                if status != "SURVIVED":
                    print("   " + out.strip()[-300:])
                failures.append(name)
            else:
                killed += 1
            shutil.rmtree(c)
        if not fast:
            repo = tmp / "repo"
            repo.mkdir()
            copy_tree(repo)
            shutil.rmtree(repo / "tools")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            for name, mutate, flags in consume_mutants():
                if not wanted(name):
                    continue
                c = tmp / "k"
                shutil.copytree(repo, c, symlinks=True)
                base = commit(c, "base")
                mutate(c)
                rev = commit(c, "mutant")
                code, out = run([PY, str(ROOT / "scripts/consume_stores.py"), "--url", "file://" + str(c), "--rev", rev] + flags, ROOT)
                status = "killed" if code != 0 else "SURVIVED"
                if code != 0 and why(name) not in out:
                    status = "KILLED FOR ANOTHER REASON"
                print(f"{name}: {status}" + ("" if code == 0 else "  <- " + (out.strip().splitlines() or [""])[-1][:110]), flush=True)
                if status != "killed":
                    failures.append(name)
                else:
                    killed += 1
                shutil.rmtree(c)
            if wanted("consume-rev-without-stores"):
                c = tmp / "k"
                shutil.copytree(repo, c, symlinks=True)
                shutil.rmtree(c / ".cancho-vcs")
                bare = commit(c, "no stores")
                (c / ".cancho-vcs").mkdir()
                shutil.copytree(repo / ".cancho-vcs", c / ".cancho-vcs", dirs_exist_ok=True)
                commit(c, "stores")
                code, out = run([PY, str(ROOT / "scripts/consume_stores.py"), "--url", "file://" + str(c), "--rev", bare], ROOT)
                ok = code != 0 and why("consume-rev-without") in out
                print(f"consume-rev-without-stores: {'killed' if ok else 'SURVIVED OR OTHER REASON'}", flush=True)
                if not ok:
                    failures.append("consume-rev-without-stores")
                else:
                    killed += 1
                shutil.rmtree(c)
            if wanted("publish-twice"):
                a, b = tmp / "pa", tmp / "pb"
                a.mkdir()
                b.mkdir()
                other = tmp / "elsewhere"
                other.mkdir()
                c1, o1 = run([PY, str(ROOT / "scripts/publish_stores.py"), "--out", str(a)], ROOT)
                c2, o2 = run([PY, str(ROOT / "scripts/publish_stores.py"), "--out", str(b)], other)
                sys.path.insert(0, str(ROOT / "scripts"))
                import publish_stores
                same = c1 == 0 and c2 == 0 and not publish_stores.differences(str(a / ".cancho-vcs"), str(b / ".cancho-vcs")) \
                    and not publish_stores.differences(str(a / ".cancho-vcs"), str(ROOT / ".cancho-vcs"))
                print(f"publish-twice (two cwd, two out dirs, and the committed stores): {'identical' if same else 'DIFFERENT'}", flush=True)
                if not same:
                    failures.append("publish-twice")
    print(f"{killed} killed, {len(failures)} survived" + (": " + ", ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
