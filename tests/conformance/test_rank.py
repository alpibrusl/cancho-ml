"""R3 -- learning to rank at the command line (docs/ranking.md): `--task ranking --group COL` on `fit`, `cv`, `eval` and `predict --rank` for `--model boost`.

What is judged from outside: the group column is not a feature (not in the schema of the file, not needed by predict), the file is the same bytes at any thread count, a group is wholly on
one side of a split and of every fold, every other split strategy is ml.split.group-required, the metrics of `eval` are those of an independent implementation of the definitions of section 3
over the scores `predict` prints, the column `rank` of `predict --rank` is the position in the group by score (ties in file order), unit weights give the bytes of the unweighted fit and
integer weights the predictions of the file with the rows repeated (pairs of copies), every hostile file is a rule tag, and the damaged files of a ranking model are refused. The byte-for-byte
comparison with the Python oracle is test_wire_rank_oracle.py."""

import json
import random
import struct
import sys
import unittest

from harness import ROOT, Sandbox, fix_checksum, sections, set_bytes, validate

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_rank as orank  # noqa: E402

FIT = ["--model", "boost", "--task", "ranking", "--group", "qid", "--target", "rel", "--rounds", "8", "--max-depth", "3", "--min-leaf", "2"]


def rank_csv(groups=24, seed=3, weights=None, grid=False, header="qid,a,b,rel", shuffle=True, ceiling=3):
    """qid, a, b, rel: the rows of a group are not next to each other; the relevance follows a - b/2 with noise"""
    r = random.Random(seed)
    rows = []
    for g in range(groups):
        for _ in range(r.randint(2, 9)):
            if grid:
                a, b = r.randint(0, 9), r.randint(0, 9)
            else:
                a, b = r.uniform(-1, 1), r.uniform(-1, 1)
            z = a - 0.5 * b + r.gauss(0, 0.4 if grid else 0.2)
            rel = max(0, min(ceiling, int(z * (1 if grid else 2) + 1.5)))
            rows.append(["q%d" % g, a, b, rel])
    if shuffle:
        r.shuffle(rows)
    out = [header]
    for i, (q, a, b, rel) in enumerate(rows):
        cells = [q, ("%d" % a) if grid else "%.4f" % a, ("%d" % b) if grid else "%.4f" % b, str(rel)]
        if weights is not None:
            cells.insert(1, str(weights(i)))
        out.append(",".join(cells))
    return "\n".join(out) + "\n"


class Rank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("rk.csv", rank_csv())

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def ml(self, *args):
        return self.sb.run(*args)

    def ok(self, r):
        self.assertEqual(r.status, 0, r)
        self.assertEqual(validate(r), [], r)
        return r.data()

    def rule(self, r, rule, status=None):
        self.assertEqual(validate(r), [], r)
        self.assertNotEqual(r.status, 0, r)
        self.assertEqual(r.doc()["error"]["rule"], rule, r)
        if status is not None:
            self.assertEqual(r.status, status, r)
        return r.doc()["error"]

    def fit(self, out, csv="rk.csv", *extra):
        return self.ml("fit", *FIT, "--output", out, "--overwrite", *extra, csv)

    def scores(self, model, csv="rk.csv", *drop):
        r = self.ml("predict", "--model-file", model, "--limit", "100000", *drop, csv)
        return [float(c[0]) for c in self.ok(r)["rows"]]

    # ------------------------------------------------------------------ the column and the answer
    def test_the_group_column_is_not_a_feature(self):
        d = self.ok(self.fit("m.cml"))
        self.assertEqual(d["task"], "ranking")
        self.assertEqual(d["summary"]["features"], 2)
        desc = self.ok(self.ml("describe", "--model-file", "m.cml"))
        self.assertEqual(desc["task"], "ranking")
        names = [c["name"] for c in desc["columns"]]
        self.assertNotIn("qid", names)
        # predict needs the columns of the schema only
        self.sb.write("p.csv", "a,b\n0.1,0.2\n-0.3,0.4\n")
        p = self.ok(self.ml("predict", "--model-file", "m.cml", "p.csv"))
        self.assertEqual(p["task"], "ranking")
        self.assertEqual(p["columns"], ["prediction"])

    def test_the_file_is_the_same_bytes_at_any_thread_count(self):
        self.sb.write("big.csv", rank_csv(groups=150, seed=9))
        seen = set()
        for t in (1, 2, 4, 8):
            self.ok(self.fit("t%d.cml" % t, "big.csv", "--threads", str(t)))
            seen.add(self.sb.read("t%d.cml" % t))
        self.assertEqual(len(seen), 1)

    # ------------------------------------------------------------------ the splits
    def test_a_group_is_wholly_on_one_side(self):
        d = self.ok(self.fit("s.cml", "rk.csv", "--split", "train=0.7", "--seed", "3", "--emit-folds"))
        assign = d["summary"]["split"]["assignment"]
        groups = [ln.split(",")[0] for ln in self.sb.read("rk.csv").decode().strip().split("\n")[1:]]
        side = {}
        for g, a in zip(groups, assign):
            side.setdefault(g, set()).add(a)
        self.assertTrue(all(len(v) == 1 for v in side.values()), side)
        self.assertEqual({a for v in side.values() for a in v}, {0, 1})
        self.assertEqual(d["summary"]["split"]["spec"], "holdout train=0.7 group=qid seed=3")

    def test_every_fold_of_cv_holds_whole_groups(self):
        d = self.ok(self.ml("cv", "--model", "boost", "--task", "ranking", "--group", "qid", "--target", "rel", "--rounds", "4", "--min-leaf", "2", "--folds", "4", "--seed", "2", "--emit-folds", "rk.csv"))
        self.assertEqual(d["strategy"], "group")
        folds = d["assignment"]["folds"][0]
        groups = [ln.split(",")[0] for ln in self.sb.read("rk.csv").decode().strip().split("\n")[1:]]
        side = {}
        for g, f in zip(groups, folds):
            side.setdefault(g, set()).add(f)
        self.assertTrue(all(len(v) == 1 for v in side.values()), side)
        self.assertEqual(len(d["results"]), 4)
        for res in d["results"]:
            self.assertIn("ndcg", res["metrics"])
        self.assertIn("ndcg", d["summary"]["metrics"])
        self.assertNotIn("k", d["summary"]["metrics"])

    def test_other_split_strategies_are_refused(self):
        base = ["--split", "train=0.7", "--seed", "1"]
        for extra in (["--stratify"], ["--split-by", "time:a"], ["--split-group", "a"]):
            e = self.rule(self.fit("x.cml", "rk.csv", *base, *extra), "ml.split.group-required", 2)
            self.assertIn("flag", e["detail"])
        # the same column is redundant, not wrong
        self.ok(self.fit("x.cml", "rk.csv", *base, "--split-group", "qid"))

    # ------------------------------------------------------------------ the metrics
    def test_eval_equals_the_definitions_over_the_printed_scores(self):
        self.ok(self.fit("e.cml"))
        sc = self.scores("e.cml", "rk.csv", "--drop", "qid,rel")
        rows = [ln.split(",") for ln in self.sb.read("rk.csv").decode().strip().split("\n")[1:]]
        gid = [r[0] for r in rows]
        rel = [int(r[3]) for r in rows]
        for k, gain in ((10, "exp"), (3, "linear"), (0, "exp"), (1, "linear")):
            want = orank.evaluate(gid, rel, sc, k, gain)
            d = self.ok(self.ml("eval", "--model-file", "e.cml", "--group", "qid", "--rank-k", str(k), "--gain", gain, "rk.csv"))
            m = d["metrics"]
            self.assertEqual(d["task"], "ranking")
            for key, name in (("ndcg", "ndcg"), ("map", "map"), ("mrr", "mrr"), ("precision", "prec"), ("recall", "recall")):
                w = want[name]
                if w is None:
                    self.assertIsNone(m[key], (k, gain, key))
                else:
                    self.assertEqual(float(m[key]), w, (k, gain, key))
            self.assertEqual((m["groups"], m["groups_scored"], m["k"], m["gain"], m["ties"]), (want["groups"], want["scored"], k, gain, "file order"))

    def test_groups_with_no_relevant_row_are_left_out_and_counted(self):
        rows = ["qid,a,b,rel", "x,1,1,0", "x,2,2,0", "y,1,1,2", "y,2,3,0", "z,3,1,1", "z,0,0,0"]
        self.sb.write("few.csv", "\n".join(rows) + "\n")
        self.ok(self.ml("fit", "--model", "boost", "--task", "ranking", "--group", "qid", "--target", "rel", "--rounds", "3", "--min-leaf", "1", "--output", "f.cml", "--overwrite", "few.csv"))
        m = self.ok(self.ml("eval", "--model-file", "f.cml", "--group", "qid", "few.csv"))["metrics"]
        self.assertEqual((m["groups"], m["groups_scored"]), (3, 2))
        # every group empty: undefined, not a refusal
        self.sb.write("none.csv", "qid,a,b,rel\nx,1,1,0\nx,2,2,0\ny,3,1,0\ny,0,0,0\n")
        m = self.ok(self.ml("eval", "--model-file", "f.cml", "--group", "qid", "none.csv"))["metrics"]
        self.assertEqual((m["groups_scored"], m["ndcg"], m["map"], m["mrr"], m["recall"]), (0, None, None, None, None))
        self.assertEqual(float(m["precision"]), 0.0)

    # ------------------------------------------------------------------ predict --rank
    def test_the_rank_column(self):
        self.ok(self.fit("r.cml"))
        sc = self.scores("r.cml", "rk.csv", "--drop", "qid,rel")
        rows = [ln.split(",") for ln in self.sb.read("rk.csv").decode().strip().split("\n")[1:]]
        d = self.ok(self.ml("predict", "--model-file", "r.cml", "--rank", "--group", "qid", "--drop", "rel", "--limit", "100000", "rk.csv"))
        self.assertEqual(d["columns"], ["prediction", "rank"])
        ranks = [int(c[1]) for c in d["rows"]]
        for lab in {r[0] for r in rows}:
            idx = [i for i, r in enumerate(rows) if r[0] == lab]
            order = orank.ranking([sc[i] for i in idx])
            for pos, k in enumerate(order):
                self.assertEqual(ranks[idx[k]], pos + 1)
        # ties in file order
        self.sb.write("tie.csv", "qid,a,b\nx,1,1\nx,1,1\nx,1,1\n")
        t = self.ok(self.ml("predict", "--model-file", "r.cml", "--rank", "--group", "qid", "tie.csv"))
        self.assertEqual([c[1] for c in t["rows"]], ["1", "2", "3"])

    def test_rank_and_group_go_together_on_a_ranking_model(self):
        self.ok(self.fit("r.cml"))
        self.rule(self.ml("predict", "--model-file", "r.cml", "--rank", "--drop", "qid,rel", "rk.csv"), "args.conflict", 2)
        self.rule(self.ml("predict", "--model-file", "r.cml", "--group", "qid", "--drop", "rel", "rk.csv"), "args.conflict", 2)
        self.rule(self.ml("eval", "--model-file", "r.cml", "rk.csv"), "args.conflict", 2)
        self.sb.write("reg.csv", "a,b,y\n1,2,3\n2,3,5\n3,5,8\n4,1,2\n")
        self.ok(self.ml("fit", "--model", "boost", "--target", "y", "--rounds", "2", "--min-leaf", "1", "--output", "g.cml", "--overwrite", "reg.csv"))
        self.rule(self.ml("predict", "--model-file", "g.cml", "--rank", "--group", "a", "reg.csv"), "args.conflict", 2)
        self.rule(self.ml("eval", "--model-file", "g.cml", "--group", "a", "reg.csv"), "args.conflict", 2)

    # ------------------------------------------------------------------ the refusals of the flags and the cells
    def test_the_flags_go_together(self):
        self.rule(self.ml("fit", "--model", "boost", "--task", "ranking", "--target", "rel", "--output", "x.cml", "--overwrite", "rk.csv"), "args.conflict", 2)
        self.rule(self.ml("fit", "--model", "boost", "--target", "rel", "--group", "qid", "--output", "x.cml", "--overwrite", "rk.csv"), "args.conflict", 2)
        self.rule(self.ml("fit", "--model", "forest", "--task", "ranking", "--group", "qid", "--target", "rel", "--output", "x.cml", "--overwrite", "rk.csv"), "args.conflict", 2)
        self.rule(self.ml("fit", "--model", "linear", "--group", "qid", "--target", "rel", "--output", "x.cml", "--overwrite", "rk.csv"), "args.conflict", 2)
        for extra in (["--drop", "qid"], ["--cat", "qid"]):
            e = self.rule(self.fit("x.cml", "rk.csv", *extra), "args.conflict", 2)
            self.assertEqual(e["detail"]["flag"], "group")
        self.rule(self.ml("fit", "--model", "boost", "--task", "ranking", "--group", "rel", "--target", "rel", "--output", "x.cml", "--overwrite", "rk.csv"), "args.conflict", 2)
        e = self.rule(self.ml("fit", "--model", "boost", "--task", "ranking", "--group", "nosuch", "--target", "rel", "--rounds", "2", "--output", "x.cml", "--overwrite", "rk.csv"), "ml.input.unknown-column", 3)
        self.assertEqual(e["detail"]["flag"], "--group")
        # early stopping validates on a random tenth of the rows, which splits groups
        self.rule(self.fit("x.cml", "rk.csv", "--early-stopping", "3", "--split", "train=0.8"), "ml.split.combination", 2)

    def test_the_cells(self):
        text = self.sb.read("rk.csv").decode().split("\n")
        for rel, rule in (("31", "ml.rank.bad-relevance"), ("-1", "ml.rank.bad-relevance"), ("1.5", "ml.rank.bad-relevance")):
            bad = list(text)
            f = bad[3].split(",")
            f[3] = rel
            bad[3] = ",".join(f)
            self.sb.write("bad.csv", "\n".join(bad))
            e = self.rule(self.fit("x.cml", "bad.csv"), rule, 8)
            self.assertEqual(e["detail"]["row"], 3)
        bad = list(text)
        f = bad[5].split(",")
        f[0] = ""
        bad[5] = ",".join(f)
        self.sb.write("bad.csv", "\n".join(bad))
        e = self.rule(self.fit("x.cml", "bad.csv"), "ml.rank.group-missing", 8)
        self.assertEqual(e["detail"]["row"], 5)
        # a group over the ceiling, and pairs over the ceiling
        self.sb.write("big.csv", "qid,a,rel\n" + "".join("g,%d,%d\n" % (i % 13, i % 2) for i in range(4097)))
        e = self.rule(self.ml("fit", "--model", "boost", "--task", "ranking", "--group", "qid", "--target", "rel", "--rounds", "1", "--output", "x.cml", "--overwrite", "big.csv"), "ml.limit.group-size", 8)
        self.assertEqual((e["detail"]["rows"], e["detail"]["max"]), (4097, 4096))
        self.sb.write("pairs.csv", "qid,a,rel\n" + "".join("g%d,%d,%d\n" % (i // 4096, i % 13, i % 2) for i in range(4096 * 65)))
        e = self.rule(self.ml("fit", "--model", "boost", "--task", "ranking", "--group", "qid", "--target", "rel", "--rounds", "1", "--output", "x.cml", "--overwrite", "pairs.csv"), "ml.limit.pairs", 8)
        self.assertEqual(e["detail"]["max"], 268435456)
        self.assertGreater(e["detail"]["pairs"], 268435456)

    def test_degenerate_files_are_fitted(self):
        # one row a group: no pair, a model of single leaves worth 0; all relevance equal in every group: the same
        self.sb.write("one.csv", "qid,a,b,rel\n" + "".join("g%d,%d,%d,%d\n" % (i, i % 5, i % 3, i % 4) for i in range(30)))
        self.ok(self.fit("one.cml", "one.csv"))
        self.assertEqual(set(self.scores("one.cml", "one.csv", "--drop", "qid,rel")), {0.0})
        self.sb.write("flat.csv", "qid,a,b,rel\n" + "".join("g%d,%d,%d,2\n" % (i // 5, i % 5, i % 3) for i in range(30)))
        self.ok(self.fit("flat.cml", "flat.csv"))
        self.assertEqual(set(self.scores("flat.cml", "flat.csv", "--drop", "qid,rel")), {0.0})

    # ------------------------------------------------------------------ the weights
    def test_unit_weights_are_the_unweighted_fit_as_bytes(self):
        self.sb.write("w1.csv", rank_csv(weights=lambda i: 1, header="qid,w,a,b,rel"))
        self.ok(self.ml("fit", *FIT, "--weight", "w", "--output", "a.cml", "--overwrite", "w1.csv"))
        self.ok(self.fit("b.cml", "w1.csv", "--drop", "w"))
        self.assertEqual(self.sb.read("a.cml"), self.sb.read("b.cml"))

    def test_integer_weights_give_the_predictions_of_the_file_with_the_rows_repeated(self):
        wts = [1 + (i * 7) % 3 for i in range(400)]
        text = rank_csv(groups=12, seed=5, grid=True, weights=lambda i: wts[i], header="qid,w,a,b,rel")
        self.sb.write("wt.csv", text)
        lines = text.strip().split("\n")
        dup = ["qid,a,b,rel"]
        for ln in lines[1:]:
            q, w, a, b, rel = ln.split(",")
            dup += [",".join([q, a, b, rel])] * int(w)
        self.sb.write("dup.csv", "\n".join(dup) + "\n")
        args = ["--model", "boost", "--task", "ranking", "--group", "qid", "--target", "rel", "--rounds", "6", "--max-depth", "3", "--min-leaf", "1"]
        self.ok(self.ml("fit", *args, "--weight", "w", "--output", "a.cml", "--overwrite", "wt.csv"))
        self.ok(self.ml("fit", *args, "--output", "b.cml", "--overwrite", "dup.csv"))
        self.sb.write("grid.csv", "a,b\n" + "".join("%d,%d\n" % (a, b) for a in range(10) for b in range(10)))
        sa, sb_ = self.scores("a.cml", "grid.csv"), self.scores("b.cml", "grid.csv")
        close = sum(1 for x, y in zip(sa, sb_) if abs(x - y) <= 1e-9 * max(1.0, abs(y)))
        self.assertGreaterEqual(close, 95, close)

    # ------------------------------------------------------------------ the file
    def edit(self, at, new):
        data = self.sb.read("fresh.cml")
        start = sections(data)[3][1]
        return fix_checksum(set_bytes(data, start + at, new))

    def test_a_damaged_ranking_file_is_refused(self):
        self.ok(self.fit("fresh.cml"))
        data = self.sb.read("fresh.cml")
        start = sections(data)[3][1]
        self.assertEqual(struct.unpack_from("<q", data, start + 20)[0], 2)
        self.sb.write("p.csv", "a,b\n0.1,0.2\n")
        for label, at, new in (("a task above 2", 20, struct.pack("<q", 3)), ("early stopping on a ranking file", 76, struct.pack("<q", 2)),
                               ("a base score above 0", 116, struct.pack("<d", 0.5)), ("a base score of -0.0", 116, struct.pack("<d", -0.0))):
            self.sb.write("d.cml", self.edit(at, new))
            r = self.ml("predict", "--model-file", "d.cml", "p.csv")
            self.assertNotEqual(r.status, 0, label)
            self.assertEqual(validate(r), [], label)
            self.assertIn(r.doc()["error"]["rule"], ("ml.model.malformed", "ml.model.version"), label)
        # the same file with task 1 is a regression file whose plan says ranking: a file that disagrees with itself
        self.sb.write("d.cml", self.edit(20, struct.pack("<q", 1)))
        r = self.ml("predict", "--model-file", "d.cml", "p.csv")
        self.assertEqual(r.doc()["error"]["rule"], "ml.model.malformed")

    # ------------------------------------------------------------------ the run record
    def test_a_tracked_ranking_fit_is_replayed(self):
        (self.sb.dir / "runs").mkdir(exist_ok=True)
        d = self.ok(self.fit("tr.cml", "rk.csv", "--track", "runs"))
        rec = json.loads((self.sb.dir / "runs" / (d["run"]["id"] + ".json")).read_text())
        self.assertEqual(rec["inputs"]["group"], "qid", rec)
        v = self.ml("runs", "verify", "runs", d["run"]["id"])
        self.assertEqual(v.status, 0, v)


if __name__ == "__main__":
    unittest.main()
