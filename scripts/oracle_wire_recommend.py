#!/usr/bin/env python3
"""The oracle of the wiring of the recommender (docs/wiring-recommend.md gates WR2, WR3, WR5, WR6, WR11): a Python implementation that fits the same log and the same ALS as
`ml fit --model recommend` and writes the model file's bytes, so that the file the real binary wrote is compared byte for byte; and that answers `describe`, `predict --for`, `predict --like`
and `eval` from the definitions of the document, so that the real answers are compared to it.

What it reuses, and why that is not circular: the log (the numbering of the ids, the sums of a repeated implicit pair), the ALS (the initial factors of stream kind 12, the sums in the order of the
document, the Cholesky of oracle_linalg.py), the payload and the metrics are `oracle_recommend.py`'s, each compared with the library by `gate_recommend.sh` against its own driver. What is written here
is the glue of this stage: the CSV, the numbering of the ids by the bytewise order of the distinct cells, the canonical plan, the schema with no target, the sections, the meta and the SHA-256 trailer,
the held-out split and the renumbering of the items that follows it, and the answers of the verbs.

    python3 scripts/oracle_wire_recommend.py --suite [N]     N seeded cases against build/ml (default 40)
    python3 scripts/oracle_wire_recommend.py --controls      the oracle fails where it must (a wrong order of the ids, of the sums, of a tie)
Exit 1 on any mismatch. ML_BIN names the binary.
"""

import csv
import hashlib
import io
import json
import math
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_pipe as op  # noqa: E402
import oracle_recommend as orc  # noqa: E402
import oracle_wire as ow  # noqa: E402

ROOT = HERE.parent
BIN = os.environ.get("ML_BIN", str(ROOT / "build" / "ml"))
PIN = ow.PIN
SUM_ORDER = 3

USER_POOL = ["a", "b", "c", "Z", "z", "007", "7", "1e3", "10", "9", "é", "b c", "p,q", 'x"y', "u1", "u2", "u10", "u3", "~", "!", "A", "B", "zz", "aa", "ab", "0", "00", "000", "1", "2", "3", "x", "y", "w", "v", "t", "s", "r", "q", "p", "n", "m", "l", "k", "j", "i", "h", "g", "f", "e", "d"]
ITEM_POOL = ["i%d" % k for k in range(25)] + ["007", "7", "1e3", "é", "item with space", "A", "a", "Z"]


# ---------------------------------------------------------------- the cases
class Case:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.seed = seed
        self.rng = rng
        self.nu = rng.choice([1, 2, 3, 5, 12, 25])
        self.ni = rng.choice([2, 3, 6, 15, 25])
        self.users = rng.sample(USER_POOL, self.nu)
        self.items = rng.sample(ITEM_POOL, self.ni)
        self.rated = rng.random() < 0.6
        self.implicit = (not self.rated) or rng.random() < 0.35
        self.f = rng.choice([1, 2, 3, 4, 6])
        self.iters = rng.choice([1, 2, 3, 5])
        self.fseed = rng.choice([0, 1, 7, 12345])
        self.lam = rng.choice(["0.1", "0.5", "1", "2.5", None])
        self.alpha = rng.choice(["1", "10", "40", None]) if self.implicit else None
        self.top = rng.randint(1, max(1, self.ni - 1))
        rows = []
        for u in self.users:
            k = rng.randint(1, min(self.ni, 8))
            for it in rng.sample(self.items, k):
                rows.append([u, it, self.rating(), rng.randint(0, 40)])
                if self.implicit and rng.random() < 0.15:
                    rows.append([u, it, self.rating(), rng.randint(0, 40)])
        rng.shuffle(rows)
        self.rows = rows
        self.names = ("user", "item", "rating", "ts")
        self.split_last = rng.choice([None, None, 1, 2])
        self.threads = rng.choice([1, 1, 2, 4])

    def rating(self):
        if not self.rated:
            return ""
        return self.rng.choice(["1", "2", "3", "4", "5", "3.5", "0.5", "4.25", "10", "1e1", "2.0"])

    def csv_text(self, rows=None, header=None):
        rows = self.rows if rows is None else rows
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(header or (self.names[0], self.names[1], self.names[2], self.names[3]))
        for r in rows:
            w.writerow(r)
        return buf.getvalue()

    def fit_args(self, out, log_name, split=False):
        a = ["fit", "--model", "recommend", "--user", self.names[0], "--item", self.names[1]]
        if self.rated:
            a += ["--rating", self.names[2]]
        if self.implicit and self.rated:
            a += ["--feedback", "implicit"]
        a += ["--factors", self.f, "--iterations", self.iters, "--seed", self.fseed, "--threads", self.threads]
        if self.lam is not None:
            a += ["--l2", self.lam]
        if self.alpha is not None:
            a += ["--alpha", self.alpha]
        if split:
            a += ["--time", self.names[3], "--split-user-last", self.split_last, "--top", self.top]
        a += ["--output", out, log_name]
        return a

    def lam_f(self):
        return float(self.lam) if self.lam is not None else 0.1

    def alpha_f(self):
        return float(self.alpha) if self.alpha is not None else 40.0


# ---------------------------------------------------------------- the model, from the definitions
def number(cells):
    """the distinct cells in bytewise order and the number of each cell"""
    distinct = sorted(set(cells), key=lambda c: c.encode())
    at = {c: k for k, c in enumerate(distinct)}
    return distinct, at


def parse_value(text):
    return float(text)


def canon(x):
    return op.canon_float(x)


def plan_text(c):
    return "model kind=recommend factors=%d l2=%s iterations=%d alpha=%s feedback=%s\n" % (c.f, canon(c.lam_f()), c.iters, canon(c.alpha_f()), "implicit" if c.implicit else "explicit")


def fit_model(c, rows):
    """-> ("ok", model dict) or ("refused", tag). rows: [user, item, rating text, time] in file order."""
    users, uat = number([r[0] for r in rows])
    items, iat = number([r[1] for r in rows])
    uid = [uat[r[0]] for r in rows]
    iid = [iat[r[1]] for r in rows]
    vals = [parse_value(r[2]) if c.rated else 1.0 for r in rows]
    res = orc.log_oracle(uid, iid, vals, len(users), len(items), c.implicit, len(rows))
    if res[0] != "ok":
        return ("refused", res)
    w = res[1]
    out = orc.als_oracle(w, len(users), len(items), c.f, c.lam_f(), c.alpha_f() if c.implicit else 0.0, c.implicit, c.iters, c.fseed, False)
    if out[0] != "ok":
        return ("refused", out[1])
    return ("ok", {"users": users, "items": items, "uat": uat, "iat": iat, "w": w, "p": out[1], "q": out[2], "loss": out[3][-1]})


def file_bytes(c, m):
    users = [u.encode() for u in m["users"]]
    items = [i.encode() for i in m["items"]]
    payload = orc.payload_oracle(m["w"], len(users), len(items), c.implicit, c.f, c.lam_f(), c.alpha_f() if c.implicit else 0.0, c.iters, c.fseed, users, items, m["p"], m["q"])
    ncols = 3 if c.rated else 2
    types = [(0, 0), (0, 0)] + ([(2, 0)] if c.rated else [])
    names = list(c.names[:ncols])
    schema = ow.schema_bytes(names, types, ncols)
    meta = struct.pack("<QQQII", m["w"]["nnz"], ncols, c.fseed, 0, len(PIN)) + PIN.encode() + struct.pack("<I", SUM_ORDER)
    body = b"CML1" + struct.pack("<II", 1, 5) + ow.section(1, plan_text(c).encode()) + ow.section(2, schema) + ow.section(3, struct.pack("<I", 0)) + ow.section(4, payload) + ow.section(5, meta)
    return body + hashlib.sha256(body).digest()


def dot(p, q):
    s = 0.0
    for a, b in zip(p, q):
        s = s + a * b
    return s


def history(m, u):
    w = m["w"]
    return w["uitem"][w["uoff"][u]:w["uoff"][u + 1]]


def ranked(keys, excluded, top):
    """the items not excluded by score descending, ties by item number; the first `top`"""
    ex = set(excluded)
    order = sorted((i for i in range(len(keys)) if i not in ex), key=lambda i: -keys[i])
    return order[:top]


def predict_rows(m, ids, by_user, top, keep):
    out = []
    ni = len(m["items"])
    if by_user:
        for who in ids:
            u = m["uat"][who]
            keys = [dot(m["p"][u], m["q"][i]) for i in range(ni)]
            for r, i in enumerate(ranked(keys, [] if keep else history(m, u), top)):
                out.append([who, r + 1, m["items"][i], keys[i]])
    else:
        norms = [math.sqrt(dot(q, q)) for q in m["q"]]
        for who in ids:
            j = m["iat"][who]
            keys = []
            for i in range(ni):
                den = norms[j] * norms[i]
                keys.append(dot(m["q"][j], m["q"][i]) / den if den > 0.0 else 0.0)
            for r, i in enumerate(ranked(keys, [j], top)):
                out.append([who, r + 1, m["items"][i], keys[i]])
    return out


def eval_expect(m, test_rows, k):
    """the answer of eval on the rows [user, item] of a test log"""
    nu, ni = len(m["users"]), len(m["items"])
    known_u = {u for u in {r[0] for r in test_rows} if u in m["uat"]}
    cold = len({r[0] for r in test_rows}) - len(known_u)
    nuser = nitem = inh = 0
    tests = [set() for _ in range(nu)]
    for u, i in test_rows:
        if u not in m["uat"]:
            nuser += 1
        elif i not in m["iat"]:
            nitem += 1
        elif m["iat"][i] in history(m, m["uat"][u]):
            inh += 1
            tests[m["uat"][u]]  # a pair of the history is counted, not scored
        else:
            tests[m["uat"][u]].add(m["iat"][i])
    ntest = sum(len(t) for t in tests)
    users = [u for u in range(nu) if tests[u]]
    scores = [[dot(m["p"][u], m["q"][i]) for i in range(ni)] for u in users]
    res = orc.evaluate(scores, [history(m, u) for u in users], [sorted(tests[u]) for u in users], k)
    return {"rows": len(test_rows), "k": k, "users": len({r[0] for r in test_rows}), "users_scored": res["scored"], "users_cold": cold, "rows_user_unknown": nuser, "rows_item_unknown": nitem, "rows_in_history": inh,
            "test_items": ntest, "metrics": res}


# ---------------------------------------------------------------- the binary
def ml(*args, cwd=None):
    p = subprocess.run([BIN, *[str(a) for a in args]], capture_output=True, cwd=cwd)
    return p.returncode, p.stdout.decode()


def same_float(text, x):
    return struct.pack("<d", float(text)) == struct.pack("<d", x)


def check_case(seed):
    c = Case(seed)
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        pathlib.Path(tmp, "log.csv").write_text(c.csv_text())
        want = fit_model(c, c.rows)
        rc, out = ml("--root", tmp, *c.fit_args("m.cml", "log.csv"))
        if want[0] != "ok":
            doc = json.loads(out)
            if rc != 8 or doc.get("error", {}).get("rule") != "ml.recommend.duplicate-pair":
                problems.append("seed %d: expected a duplicate-pair refusal, got exit %d %s" % (seed, rc, out[:200]))
            return problems
        if rc != 0:
            return ["seed %d: fit exit %d %s" % (seed, rc, out[:300])]
        got = pathlib.Path(tmp, "m.cml").read_bytes()
        exp = file_bytes(c, want[1])
        if got != exp:
            at = next((k for k in range(min(len(got), len(exp))) if got[k] != exp[k]), min(len(got), len(exp)))
            problems.append("seed %d: the file differs from the oracle's at byte %d (sizes %d and %d)" % (seed, at, len(got), len(exp)))
            return problems
        doc = json.loads(out)["data"]
        if not same_float(doc["summary"]["loss"], want[1]["loss"]):
            problems.append("seed %d: the loss %s differs from the oracle's %r" % (seed, doc["summary"]["loss"], want[1]["loss"]))
        m = want[1]
        # describe
        rc, out = ml("--root", tmp, "describe", "--model-file", "m.cml")
        d = json.loads(out)["data"]
        if rc != 0 or d["users"] != len(m["users"]) or d["items"] != len(m["items"]) or d["interactions"] != m["w"]["nnz"] or d["factors"] != c.f or d["feedback"] != ("implicit" if c.implicit else "explicit"):
            problems.append("seed %d: describe %s" % (seed, out[:300]))
        # predict --for, with and without the history
        rng = random.Random(seed * 7 + 1)
        ids = [rng.choice(m["users"]) for _ in range(rng.randint(1, 4))]
        pathlib.Path(tmp, "ids.csv").write_text("x,who\n" + "".join("1,%s\n" % csv_cell(i) for i in ids))
        for keep in (False, True):
            top = min(c.top, len(m["items"]))
            args = ["--root", tmp, "predict", "--model-file", "m.cml", "--for", "who", "--top", top] + (["--keep-seen"] if keep else []) + ["ids.csv"]
            rc, out = ml(*args)
            if rc != 0:
                problems.append("seed %d: predict exit %d %s" % (seed, rc, out[:300]))
                continue
            rows = json.loads(out)["data"]["rows"]
            exp_rows = predict_rows(m, ids, True, top, keep)
            if len(rows) != len(exp_rows) or any(r[0] != e[0] or r[1] != e[1] or r[2] != e[2] or not same_float(r[3], e[3]) for r, e in zip(rows, exp_rows)):
                problems.append("seed %d: predict --for (keep-seen %s) differs: %s vs %s" % (seed, keep, rows[:2], exp_rows[:2]))
        # predict --like
        if len(m["items"]) >= 2:
            qs = [rng.choice(m["items"]) for _ in range(rng.randint(1, 3))]
            pathlib.Path(tmp, "qs.csv").write_text("it\n" + "".join("%s\n" % csv_cell(i) for i in qs))
            top = min(c.top, len(m["items"]) - 1)
            rc, out = ml("--root", tmp, "predict", "--model-file", "m.cml", "--like", "it", "--top", top, "qs.csv")
            if rc != 0:
                problems.append("seed %d: predict --like exit %d %s" % (seed, rc, out[:300]))
            else:
                rows = json.loads(out)["data"]["rows"]
                exp_rows = predict_rows(m, qs, False, top, False)
                if len(rows) != len(exp_rows) or any(r[0] != e[0] or r[1] != e[1] or r[2] != e[2] or not same_float(r[3], e[3]) for r, e in zip(rows, exp_rows)):
                    problems.append("seed %d: predict --like differs: %s vs %s" % (seed, rows[:2], exp_rows[:2]))
        # eval on a test log with users and items the model has, has not, and pairs of its history
        test = []
        for _ in range(rng.randint(3, 12)):
            u = rng.choice(m["users"] + ["unknown-user"])
            i = rng.choice(m["items"] + ["unknown-item"])
            test.append([u, i])
        pathlib.Path(tmp, "test.csv").write_text("user,item\n" + "".join(",".join(csv_cell(x) for x in r) + "\n" for r in test))
        k = min(c.top, len(m["items"]))
        rc, out = ml("--root", tmp, "eval", "--model-file", "m.cml", "--user", "user", "--item", "item", "--top", k, "test.csv")
        if rc != 0:
            problems.append("seed %d: eval exit %d %s" % (seed, rc, out[:300]))
        else:
            d = json.loads(out)["data"]
            e = eval_expect(m, test, k)
            for key in ("rows", "k", "users", "users_scored", "users_cold", "rows_user_unknown", "rows_item_unknown", "rows_in_history", "test_items"):
                if d[key] != e[key]:
                    problems.append("seed %d: eval %s %r != %r" % (seed, key, d[key], e[key]))
            for key in ("hit", "recall", "ndcg", "mrr"):
                want_v = e["metrics"][key]
                if want_v is None:
                    if d[key] is not None and float(d[key]) != 0.0:
                        problems.append("seed %d: eval %s %r with no user scored" % (seed, key, d[key]))
                elif not same_float(d[key], want_v):
                    problems.append("seed %d: eval %s %s != %r" % (seed, key, d[key], want_v))
        # the held-out split: the file of the fit on the training rows, and the metrics of the held-out part
        if c.split_last is not None:
            problems += check_split(c, tmp)
    return problems


def csv_cell(x):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow([x])
    return buf.getvalue()


def check_split(c, tmp):
    seed = c.seed
    rc, out = ml("--root", tmp, *c.fit_args("s.cml", "log.csv", split=True))
    rows = c.rows
    # the held-out rows: each user's latest `last` by (dense time rank, row), a user with more than `last` rows
    mask = [0] * len(rows)
    for u in {r[0] for r in rows}:
        mine = sorted((k for k in range(len(rows)) if rows[k][0] == u), key=lambda k: (rows[k][3], k))
        if len(mine) > c.split_last:
            for k in mine[len(mine) - c.split_last:]:
                mask[k] = 1
    train = [rows[k] for k in range(len(rows)) if mask[k] == 0]
    want = fit_model(c, train)
    if want[0] != "ok":
        if rc == 8 and json.loads(out).get("error", {}).get("rule") == "ml.recommend.duplicate-pair":
            return []
        return ["seed %d: split fit expected a refusal, got %d" % (seed, rc)]
    if c.top > len(want[1]["items"]):
        doc = json.loads(out)
        if rc == 2 and doc.get("error", {}).get("rule") == "ml.param.out-of-range":
            return []
        return ["seed %d: --top above the items of the training rows expected ml.param.out-of-range, got exit %d" % (seed, rc)]
    if rc != 0:
        return ["seed %d: split fit exit %d %s" % (seed, rc, out[:300])]
    got = pathlib.Path(tmp, "s.cml").read_bytes()
    exp = file_bytes(c, want[1])
    problems = []
    if got != exp:
        problems.append("seed %d: the split file differs from the oracle's (sizes %d and %d)" % (seed, len(got), len(exp)))
        return problems
    m = want[1]
    held = [[rows[k][0], rows[k][1]] for k in range(len(rows)) if mask[k] == 1]
    k = min(c.top, len(m["items"]))
    if c.top > len(m["items"]):
        return problems
    # the held-out items the model has no number for are counted, not scored; the pairs in the history are not test items
    e = eval_expect(m, held, k)
    d = json.loads(out)["data"]["summary"]
    h = d["holdout"]
    if h["users_scored"] != e["users_scored"] or h["rows_item_unseen"] != e["rows_item_unknown"] or h["rows_in_history"] != e["rows_in_history"]:
        problems.append("seed %d: split counts %s vs %s" % (seed, h, e))
    for key in ("hit", "recall", "ndcg", "mrr"):
        want_v = e["metrics"][key]
        if want_v is not None and not same_float(h[key], want_v):
            problems.append("seed %d: split %s %s != %r" % (seed, key, h[key], want_v))
    if d["split"] != {"user_last": c.split_last, "train": m["w"]["nnz"], "test": e["test_items"]}:
        problems.append("seed %d: split sizes %s" % (seed, d["split"]))
    return problems


# ---------------------------------------------------------------- controls: the oracle must fail where it should
def controls():
    bad = []
    c = Case(3)
    want = fit_model(c, c.rows)
    if want[0] != "ok":
        c = Case(4)
        want = fit_model(c, c.rows)
    good = file_bytes(c, want[1])
    # the ids numbered by first appearance instead of the bytewise order give another file
    users = list(dict.fromkeys(r[0] for r in c.rows))
    items = list(dict.fromkeys(r[1] for r in c.rows))
    if users != sorted(users, key=lambda s: s.encode()) or items != sorted(items, key=lambda s: s.encode()):
        m2 = dict(want[1])
        m2["users"], m2["items"] = users, items
        if file_bytes(c, m2) == good:
            bad.append("the order of the ids does not reach the bytes")
    # a different factor count gives another file
    c2 = Case(3)
    c2.f = c.f + 1
    w2 = fit_model(c2, c2.rows)
    if w2[0] == "ok" and file_bytes(c2, w2[1]) == good:
        bad.append("the factors do not reach the bytes")
    # ties by the larger item number rank differently
    keys = [1.0, 1.0, 0.5]
    if ranked(keys, [], 3) != [0, 1, 2] or sorted(range(3), key=lambda i: (-keys[i], -i)) == [0, 1, 2]:
        bad.append("the tie rule is not what is defined")
    return bad


def main():
    argv = sys.argv[1:]
    if "--controls" in argv:
        bad = controls()
        for b in bad:
            print("FAIL control:", b)
        print("oracle_wire_recommend controls: %s" % ("all as defined" if not bad else "%d failed" % len(bad)))
        return 1 if bad else 0
    if "--suite" not in argv:
        print(__doc__)
        return 2
    at = argv.index("--suite") + 1
    n = int(argv[at]) if len(argv) > at and argv[at].isdigit() else 40
    bad = 0
    for seed in range(1, n + 1):
        for p in check_case(seed):
            bad += 1
            print("MISMATCH", p)
        if bad > 12:
            break
    print("oracle_wire_recommend: %d cases, %s" % (n, "every file equals the oracle's byte for byte, and every answer agrees" if not bad else "%d mismatches" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
