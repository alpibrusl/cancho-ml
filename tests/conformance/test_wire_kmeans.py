"""WK2, WK3, WK4, WK6, WK7, WK8 -- k-means through `ml fit | predict | eval` (docs/wiring-kmeans.md).

What is judged from outside: a fit needs no target and refuses one; the answers of the three verbs with numbers known by hand (the inertia, the sizes and the mean distances
of a six-point set, the tie of two equal distances going to the lowest cluster); the file (sections, kind 6, the schema whose target index is the number of columns, the sum
order 2) and the refusals of a file that disagrees with itself; missing cells refused with their repair; the bounds of k, of the features and of the diagnostics, each refused
before anything is allocated; the model file at 1, 2, 4, 8 and 16 threads; a split. The byte-for-byte comparison with the Python oracle is test_wire_kmeans_oracle.py."""

import random
import struct
import unittest

from harness import Sandbox, fix_checksum, sections, set_bytes, validate
from test_model_file import payloads


def u32(b, at):
    return struct.unpack_from("<I", b, at)[0]


def blobs_csv(n=90, d=3, seed=1, groups=3, missing=False, label=True):
    rng = random.Random(seed)
    head = ",".join("f%d" % j for j in range(d)) + (",lab" if label else "")
    rows = [head]
    for i in range(n):
        g = i % groups
        cells = []
        for j in range(d):
            if missing and (i * 7 + j) % 11 == 0:
                cells.append("")
            else:
                cells.append("%.3f" % (g * 20.0 + rng.gauss(0, 1.0) + j))
        rows.append(",".join(cells) + ("," + str(g % 2) if label else ""))
    return "\n".join(rows) + "\n"


SIX = "x\n0.0\n1.0\n2.0\n3.0\n10.0\n11.0\n"


def centroids(data):
    """(k, d, the k x d doubles) of the centroids record of a kind 6 file"""
    m = payloads(data)[4]
    assert u32(m, 0) == 6
    dtype, rank = struct.unpack_from("<II", m, 4)
    assert (dtype, rank) == (2, 2)
    k, d = struct.unpack_from("<QQ", m, 12)
    vals = struct.unpack_from("<%dd" % (k * d), m, 28)
    assert len(m) == 28 + 8 * k * d
    return k, d, vals


class KMeansWire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("blobs.csv", blobs_csv())
        cls.sb.write("full.csv", blobs_csv(label=False))
        cls.sb.write("miss.csv", blobs_csv(missing=True, label=False))
        cls.sb.write("six.csv", SIX)

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def ml(self, *args):
        return self.sb.run(*args)

    def fit(self, csv, out, *extra):
        r = self.ml("fit", "--model", "kmeans", "--output", out, "--overwrite", *extra, csv)
        self.assertEqual(r.status, 0, r)
        return r

    # ------------------------------------------------------------------ WK2: no target
    def test_a_fit_needs_no_target_and_says_what_it_did(self):
        r = self.fit("full.csv", "m.cml", "--k", "3")
        self.assertEqual(validate(r), [])
        d = r.data()
        self.assertEqual((d["model"], d["task"], d["rows"], d["columns"], d["target"]), ("kmeans", "clustering", 90, 3, None))
        c = d["summary"]["clusters"]
        self.assertEqual((c["k"], c["iterations"], c["init"], c["rows"]), (3, 10, "kmeans++", 90))
        self.assertEqual(sorted(c["metrics"]), ["inertia", "mean_distance", "sizes"])
        self.assertEqual(sum(c["metrics"]["sizes"]), 90)
        self.assertEqual(d["summary"]["features"], 3)

    def test_a_target_is_refused_for_kmeans_and_still_needed_for_the_others(self):
        r = self.ml("fit", "--model", "kmeans", "--k", "2", "--target", "lab", "--output", "m.cml", "--overwrite", "blobs.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)
        self.assertEqual(r.error()["detail"]["flag"], "target")
        for model in ("linear", "logistic", "tree", "forest", "constant"):
            r = self.ml("fit", "--model", model, "--output", "m.cml", "--overwrite", "blobs.csv")
            self.assertEqual((r.status, r.first_rule(), r.error()["detail"]["flag"]), (2, "args.required-flag", "--target"), (model, r))

    def test_a_plan_decides_after_it_is_read(self):
        self.sb.write("km.plan", "scale column=* kind=standard\nmodel kind=kmeans k=3\n")
        self.sb.write("lin.plan", "model kind=linear\n")
        r = self.ml("fit", "--plan", "km.plan", "--output", "p.cml", "--overwrite", "full.csv")
        self.assertEqual((r.status, r.data()["task"]), (0, "clustering"), r)
        r = self.ml("fit", "--plan", "km.plan", "--target", "f0", "--output", "p.cml", "--overwrite", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)
        r = self.ml("fit", "--plan", "lin.plan", "--output", "p.cml", "--overwrite", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.required-flag"), r)
        r = self.ml("fit", "--plan", "lin.plan", "--target", "f0", "--output", "p.cml", "--overwrite", "full.csv")
        self.assertEqual(r.status, 0, r)

    def test_the_flags_k_means_does_not_take(self):
        for flag, value in (("--task", "regression"), ("--task", "classification"), ("--positive", "1"), ("--l2", "0.5"), ("--trees", "3"), ("--max-depth", "3")):
            r = self.ml("fit", "--model", "kmeans", "--k", "2", flag, value, "--output", "m.cml", "--overwrite", "full.csv")
            self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), (flag, r))
        # --k and --init mean something to k-means only
        for model, extra in (("linear", ["--target", "f0"]), ("tree", ["--target", "f0"]), ("constant", ["--target", "f0"])):
            for flag, value in (("--k", "3"), ("--init", "random")):
                r = self.ml("fit", "--model", model, flag, value, *extra, "--output", "m.cml", "--overwrite", "full.csv")
                self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), (model, flag, r))
        r = self.ml("fit", "--model", "kmeans", "--k", "2", "--init", "bogus", "--output", "m.cml", "--overwrite", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.bad-value"), r)

    def test_a_fit_without_k_is_a_missing_plan_key(self):
        r = self.ml("fit", "--model", "kmeans", "--output", "m.cml", "--overwrite", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.plan.missing-key"), r)

    def test_the_label_column_of_a_labelled_file_is_dropped(self):
        r = self.ml("fit", "--model", "kmeans", "--k", "3", "--output", "m.cml", "--overwrite", "blobs.csv")
        self.assertEqual(r.status, 0, r)                       # the label is a column like the others: a feature
        self.assertEqual(r.data()["columns"], 4)
        r = self.fit("blobs.csv", "m.cml", "--k", "3", "--drop", "lab")
        self.assertEqual(r.data()["columns"], 3)
        p = self.ml("predict", "--model-file", "m.cml", "--limit", "2", "blobs.csv")
        self.assertEqual((p.status, p.first_rule()), (8, "ml.schema.mismatch"), p)
        self.assertEqual(p.error()["detail"]["extra"], ["lab"])
        self.assertEqual(self.ml("predict", "--model-file", "m.cml", "--limit", "2", "--drop", "lab", "blobs.csv").status, 0)

    # ------------------------------------------------------------------ WK3: the verbs, with numbers known by hand
    def test_six_points_by_hand(self):
        # 0 1 2 3 | 10 11 : the centres 1.5 and 10.5 (whichever is cluster 0); inertia 5 + 0.5 = 5.5; mean distances 1 and 0.5
        r = self.fit("six.csv", "m.cml", "--k", "2", "--seed", "3")
        m = r.data()["summary"]["clusters"]["metrics"]
        self.assertEqual(m["inertia"], "5.5")
        self.assertEqual(sorted(m["sizes"]), [2, 4])
        pairs = sorted(zip(m["sizes"], m["mean_distance"]))
        self.assertEqual(pairs, [(2, "0.5"), (4, "1.0")])
        k, d, vals = centroids(self.sb.read("m.cml"))
        self.assertEqual((k, d, sorted(vals)), (2, 1, [1.5, 10.5]))
        p = self.ml("predict", "--model-file", "m.cml", "--distance", "--limit", "10", "six.csv").data()
        self.assertEqual(p["columns"], ["cluster", "distance"])
        big = vals.index(10.5)
        small = vals.index(1.5)
        self.assertEqual([int(c[0]) for c in p["rows"]], [small] * 4 + [big] * 2)
        self.assertEqual([float(c[1]) for c in p["rows"]], [2.25, 0.25, 0.25, 2.25, 0.25, 0.25])
        e = self.ml("eval", "--model-file", "m.cml", "six.csv").data()
        self.assertEqual((e["model"], e["task"], e["rows"], e["clusters"]), ("kmeans", "clustering", 6, 2))
        self.assertEqual(e["metrics"]["inertia"], "5.5")
        self.assertEqual(e["metrics"]["sizes"], [4, 2] if small == 0 else [2, 4])

    def test_predict_has_one_column_without_distance_and_two_with(self):
        self.fit("full.csv", "m.cml", "--k", "3")
        one = self.ml("predict", "--model-file", "m.cml", "--limit", "5", "full.csv").data()
        self.assertEqual(one["columns"], ["cluster"])
        two = self.ml("predict", "--model-file", "m.cml", "--distance", "--limit", "5", "full.csv").data()
        self.assertEqual(two["columns"], ["cluster", "distance"])
        self.assertEqual([c[0] for c in two["rows"]], [c[0] for c in one["rows"]])
        self.assertTrue(all(float(c[1]) >= 0.0 for c in two["rows"]))
        self.assertEqual((one["row_count"], one["total_rows"], one["truncated"]), (5, 90, True))
        self.assertEqual(self.ml("predict", "--model-file", "m.cml", "--proba", "full.csv").first_rule(), "args.conflict")

    def test_the_tie_of_two_equal_distances_goes_to_the_lowest_cluster(self):
        self.fit("six.csv", "m.cml", "--k", "2", "--seed", "3")
        k, d, vals = centroids(self.sb.read("m.cml"))
        mid = (vals[0] + vals[1]) / 2.0
        self.sb.write("tie.csv", "x\n%r\n" % mid)
        p = self.ml("predict", "--model-file", "m.cml", "--distance", "tie.csv").data()["rows"]
        self.assertEqual(p[0][0], "0")
        self.assertEqual(float(p[0][1]), (vals[0] - mid) ** 2)

    def test_distance_is_for_kmeans_only(self):
        self.sb.write("reg.csv", "a,y\n1,2\n2,4\n3,6.5\n4,8\n")
        self.assertEqual(self.ml("fit", "--model", "linear", "--target", "y", "--output", "lin.cml", "--overwrite", "reg.csv").status, 0)
        r = self.ml("predict", "--model-file", "lin.cml", "--distance", "reg.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)
        r = self.ml("fit", "--target", "y", "--output", "const.cml", "--overwrite", "reg.csv")
        self.assertEqual(r.status, 0, r)
        r = self.ml("predict", "--model-file", "const.cml", "--distance", "reg.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)

    def test_eval_has_no_label_to_score_and_takes_none(self):
        self.fit("full.csv", "m.cml", "--k", "3")
        r = self.ml("eval", "--model-file", "m.cml", "--positive", "1", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)
        e = self.ml("eval", "--model-file", "m.cml", "full.csv").data()
        self.assertEqual(sorted(e["metrics"]), ["inertia", "mean_distance", "sizes"])
        self.assertEqual(sum(e["metrics"]["sizes"]), 90)
        # the text form: one line for each of the three numbers
        t = self.ml("--format", "text", "eval", "--model-file", "m.cml", "full.csv")
        lines = t.stdout.decode().splitlines()
        self.assertEqual([l.split(" ")[0] for l in lines], ["on", "inertia", "sizes", "mean_distance"])

    def test_an_empty_cluster_has_a_null_mean_distance(self):
        # fitted on the six points, scored on rows that are all near the first centre: the other cluster is empty
        self.fit("six.csv", "m.cml", "--k", "2", "--seed", "3")
        self.sb.write("near.csv", "x\n1.0\n2.0\n1.5\n")
        e = self.ml("eval", "--model-file", "m.cml", "near.csv").data()["metrics"]
        self.assertEqual(sorted(e["sizes"]), [0, 3])
        empty = e["sizes"].index(0)
        self.assertIsNone(e["mean_distance"][empty])
        self.assertEqual(e["inertia"], "0.5")

    def test_k_equal_to_the_rows_gives_zero_inertia(self):
        r = self.fit("six.csv", "m.cml", "--k", "6", "--init", "random")
        m = r.data()["summary"]["clusters"]["metrics"]
        self.assertEqual((m["inertia"], m["sizes"]), ("0.0", [1] * 6))

    def test_the_seed_and_the_initialisation_decide_the_file(self):
        self.fit("full.csv", "a.cml", "--k", "5", "--seed", "1")
        self.fit("full.csv", "b.cml", "--k", "5", "--seed", "1")
        self.fit("full.csv", "c.cml", "--k", "5", "--seed", "2")
        self.fit("full.csv", "d.cml", "--k", "5", "--seed", "1", "--init", "random")
        self.assertEqual(self.sb.read("a.cml"), self.sb.read("b.cml"))
        self.assertNotEqual(self.sb.read("a.cml"), self.sb.read("c.cml"))
        self.assertNotEqual(self.sb.read("a.cml"), self.sb.read("d.cml"))
        self.assertEqual(struct.unpack_from("<Q", payloads(self.sb.read("c.cml"))[5], 16)[0], 2)
        # the default is kmeans++ with 10 iterations
        self.fit("full.csv", "e.cml", "--k", "5", "--seed", "1", "--init", "kmeans++", "--iterations", "10")
        self.assertEqual(self.sb.read("a.cml"), self.sb.read("e.cml"))

    def test_the_held_out_rows_are_scored_and_eval_with_the_same_split_agrees(self):
        r = self.fit("full.csv", "s.cml", "--k", "3", "--split", "train=0.8", "--seed", "5")
        s = r.data()["summary"]
        self.assertEqual((s["split"]["train"], s["split"]["test"], s["clusters"]["rows"]), (72, 18, 72))
        hold = s["holdout"]
        self.assertEqual(hold["rows"], 18)
        e = self.ml("eval", "--model-file", "s.cml", "--split", "train=0.8", "--seed", "5", "full.csv").data()
        self.assertEqual(e["metrics"], hold["metrics"])
        self.assertEqual(e["rows"], 18)
        self.assertEqual(sum(e["metrics"]["sizes"]), 18)

    # ------------------------------------------------------------------ WK4: the file
    def test_the_file(self):
        self.fit("full.csv", "m.cml", "--k", "4", "--iterations", "7", "--init", "random", "--seed", "11", "--step", "scale column=f0 kind=standard")
        d = self.sb.read("m.cml")
        p = payloads(d)
        self.assertEqual(p[1], b"scale column=f0 kind=standard ddof=0\nmodel kind=kmeans k=4 iterations=7 init=random\n")
        self.assertEqual(u32(p[3], 0), 1)                                            # one step
        k, dim, vals = centroids(d)
        self.assertEqual((k, dim), (4, 3))
        self.assertTrue(all(abs(v) < 1e6 for v in vals))
        # the schema: three columns and no target, the target index being the number of columns
        sc = p[2]
        self.assertEqual(struct.unpack_from("<II", sc, 0), (3, 3))
        meta = p[5]
        self.assertEqual(struct.unpack_from("<QQQ", meta, 0), (90, 3, 11))           # rows, columns, the seed
        self.assertEqual(u32(meta, len(meta) - 4), 2)                                # the lane order of the decisive sums

    def test_the_schema_and_the_model_must_agree_on_the_target(self):
        self.fit("full.csv", "m.cml", "--k", "3")
        d = self.sb.read("m.cml")
        sec = sections(d)
        at = sec[1][1] + 4                                                           # the target index of the schema section
        for target in (0, 1, 2, 4, 1 << 31):
            bad = fix_checksum(set_bytes(d, at, struct.pack("<I", target)))
            self.sb.write("bad.cml", bad)
            for verb in ("predict", "eval"):
                r = self.ml(verb, "--model-file", "bad.cml", "full.csv")
                self.assertEqual((r.status, r.first_rule()), (8, "ml.model.malformed"), (target, verb, r))
        # and the other way: a linear file whose schema says it has no target
        self.sb.write("reg.csv", "a,b,y\n1,2,3\n2,1,4\n3,5,9\n4,3,8\n5,7,12\n")
        self.assertEqual(self.ml("fit", "--model", "linear", "--target", "y", "--output", "lin.cml", "--overwrite", "reg.csv").status, 0)
        lin = self.sb.read("lin.cml")
        at = sections(lin)[1][1] + 4
        bad = fix_checksum(set_bytes(lin, at, struct.pack("<I", 3)))
        self.sb.write("bad.cml", bad)
        r = self.ml("predict", "--model-file", "bad.cml", "reg.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.model.malformed"), r)

    def test_a_consistent_file_that_disagrees_about_the_target_is_malformed(self):
        # a kind 6 file whose schema marks the last column as the target, with the payload rewritten for the two features that leaves: every size agrees, only the schema and the kind do not
        from test_model_file import write_file
        self.fit("full.csv", "m.cml", "--k", "3")
        d = self.sb.read("m.cml")
        secs = payloads(d)
        secs.pop(7, None)  # the provenance would disagree with the new schema first; only the schema and the kind are to disagree here
        k, dim, vals = centroids(d)
        secs[2] = secs[2][:4] + struct.pack("<I", 2) + secs[2][8:]
        keep = [vals[i * dim + j] for i in range(k) for j in range(dim - 1)]
        secs[4] = struct.pack("<I", 6) + struct.pack("<IIQQ", 2, 2, k, dim - 1) + struct.pack("<%dd" % len(keep), *keep)
        self.sb.write("target.cml", write_file(secs))
        r = self.ml("predict", "--model-file", "target.cml", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.model.malformed"), r)
        # and a linear file whose schema says it has no target, with the payload written for all three columns
        self.sb.write("reg.csv", "a,b,y\n1,2,3\n2,1,4\n3,5,9\n4,3,8\n5,7,12\n")
        self.assertEqual(self.ml("fit", "--model", "linear", "--target", "y", "--output", "lin.cml", "--overwrite", "reg.csv").status, 0)
        lin = payloads(self.sb.read("lin.cml"))
        lin.pop(7, None)
        lin[2] = lin[2][:4] + struct.pack("<I", 3) + lin[2][8:]
        w = [0.5, 0.25, 1.0]
        lin[4] = struct.pack("<I", 4) + struct.pack("<IIQ", 2, 1, 3) + struct.pack("<3d", *w) + struct.pack("<IIQ", 2, 1, 1) + struct.pack("<d", 0.0)
        self.sb.write("notarget.cml", write_file(lin))
        r = self.ml("predict", "--model-file", "notarget.cml", "reg.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.model.malformed"), r)

    def test_the_plan_and_the_payload_must_agree_on_k(self):
        self.fit("full.csv", "m.cml", "--k", "3")
        d = self.sb.read("m.cml")
        plan = payloads(d)[1]
        self.assertIn(b"k=3", plan)
        # the same file with k=4 in the plan (same length, so the sections keep their places)
        bad = fix_checksum(set_bytes(d, sections(d)[0][1] + plan.index(b"k=3") + 2, b"4"))
        self.sb.write("bad.cml", bad)
        r = self.ml("predict", "--model-file", "bad.cml", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.model.malformed"), r)
        # the iterations and the initialisation of the plan only describe the fit
        good = fix_checksum(set_bytes(d, sections(d)[0][1] + plan.index(b"iterations=10"), b"iterations=11"))
        self.sb.write("good.cml", good)
        self.assertEqual(self.ml("predict", "--model-file", "good.cml", "full.csv").status, 0)

    def test_the_sum_order_is_two_for_kmeans(self):
        self.fit("full.csv", "m.cml", "--k", "3")
        d = self.sb.read("m.cml")
        sec = sections(d)
        end = sec[4][1] + sec[4][2] - 4
        for order in (0, 1, 3, 4, 255):
            bad = fix_checksum(set_bytes(d, end, struct.pack("<I", order)))
            self.sb.write("bad.cml", bad)
            r = self.ml("predict", "--model-file", "bad.cml", "full.csv")
            self.assertEqual(r.first_rule(), "ml.model.version", (order, r))
            self.assertEqual((r.error()["detail"]["sum_order"], r.error()["detail"]["known"]), (order, 2))

    def test_the_payload_is_checked_before_it_is_used(self):
        self.fit("full.csv", "m.cml", "--k", "3")
        d = self.sb.read("m.cml")
        m0 = sections(d)[3][1]
        edits = {
            "dtype": (m0 + 4, struct.pack("<I", 1)),
            "rank": (m0 + 8, struct.pack("<I", 1)),
            "k zero": (m0 + 12, struct.pack("<Q", 0)),
            "k huge": (m0 + 12, struct.pack("<Q", (1 << 63) - 1)),
            "k negative": (m0 + 12, struct.pack("<q", -3)),
            "width": (m0 + 20, struct.pack("<Q", 4)),
            "width huge": (m0 + 20, struct.pack("<Q", 1 << 62)),
            "a nan centroid": (m0 + 28, struct.pack("<Q", 0x7FF8000000000000)),
            "an infinite centroid": (m0 + 36, struct.pack("<Q", 0x7FF0000000000000)),
            "the kind": (m0, struct.pack("<I", 5)),
        }
        for name, (at, new) in edits.items():
            self.sb.write("bad.cml", fix_checksum(set_bytes(d, at, new)))
            r = self.ml("predict", "--model-file", "bad.cml", "full.csv")
            self.assertEqual((r.status, r.first_rule()), (8, "ml.model.malformed"), (name, r))

    # ------------------------------------------------------------------ WK6: missing cells
    def test_a_missing_cell_is_refused_with_the_impute_repair(self):
        r = self.ml("fit", "--model", "kmeans", "--k", "3", "--output", "m.cml", "--overwrite", "miss.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.input.missing-value"), r)
        self.assertEqual(r.error()["repair"]["kind"], "retry")
        self.assertIn("impute column=", r.error()["repair"]["argv"][-1])
        self.assertEqual(validate(r), [])
        # the repair works
        r = self.fit("miss.csv", "m.cml", "--k", "3", "--step", "impute column=* strategy=mean")
        self.assertEqual(r.data()["rows"], 90)
        # and predict and eval refuse a file the stored steps leave missing cells in: a column the impute step does not cover
        self.fit("miss.csv", "m2.cml", "--k", "3", "--step", "impute column=f0,f1,f2 strategy=median")
        self.assertEqual(self.ml("predict", "--model-file", "m2.cml", "miss.csv").status, 0)
        self.sb.write("one.csv", "f0,f1,f2\n1,2,3\n4,5,6\n7,8,9\n3,1,2\n9,9,9\n")
        self.fit("one.csv", "m3.cml", "--k", "2")
        self.sb.write("holes.csv", "f0,f1,f2\n1,,3\n4,5,6\n")
        for verb in ("predict", "eval"):
            r = self.ml(verb, "--model-file", "m3.cml", "holes.csv")
            self.assertEqual((r.status, r.first_rule()), (8, "ml.input.missing-value"), (verb, r))

    def test_the_held_out_rows_of_a_split_fit_are_refused_too(self):
        # the training rows are complete after the steps, the held-out rows keep a hole that a step does not fill: scale passes it through
        rows = ["a,b"] + ["%d,%d" % (i, 2 * i) for i in range(1, 41)]
        rows[5] = "5,"
        rows[17] = "17,"
        rows[33] = "33,"
        self.sb.write("holes40.csv", "\n".join(rows) + "\n")
        r = self.ml("fit", "--model", "kmeans", "--k", "2", "--split", "train=0.5", "--output", "h.cml", "--overwrite", "holes40.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.input.missing-value"), r)

    # ------------------------------------------------------------------ WK7: the bounds, before any allocation
    def test_the_range_of_k_and_of_the_iterations(self):
        for flag, value in (("--k", "0"), ("--k", "1025"), ("--iterations", "0"), ("--iterations", "1001")):
            other = ["--iterations", "5"] if flag == "--k" else ["--k", "2"]
            r = self.ml("fit", "--model", "kmeans", *other, flag, value, "--output", "m.cml", "--overwrite", "full.csv")
            self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), (flag, value, r))
            self.assertEqual(validate(r), [], (flag, value))
            d = r.error()["detail"]
            self.assertEqual((d["flag"], d["value"], d["min"], d["max"]), (flag[2:], int(value), 1, 1024 if flag == "--k" else 1000), (flag, value, r))
        self.assertEqual(self.ml("fit", "--model", "kmeans", "--k", "1024", "--iterations", "1000", "--output", "m.cml", "--overwrite", "full.csv").first_rule(), "ml.param.out-of-range")

    def test_k_above_the_rows_is_refused(self):
        r = self.ml("fit", "--model", "kmeans", "--k", "7", "--output", "m.cml", "--overwrite", "six.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)
        self.assertEqual((r.error()["detail"]["k"], r.error()["detail"]["rows"]), (7, 6))
        self.assertFalse((self.sb.dir / "never.cml").exists())
        # the rows are the training rows
        r = self.ml("fit", "--model", "kmeans", "--k", "4", "--split", "train=0.5", "--output", "m.cml", "--overwrite", "six.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)

    def test_k_times_the_features_is_bounded(self):
        # 1024 clusters of 128 features: 1024 * 128 + 1024 + 1 = 132097 > 131072
        d = 128
        n = 1030
        self.sb.write("wide.csv", ",".join("c%d" % j for j in range(d)) + "\n" + "".join(",".join(str((i * (j + 3)) % 17) for j in range(d)) + "\n" for i in range(n)))
        r = self.ml("fit", "--model", "kmeans", "--k", "1024", "--iterations", "1", "--output", "w.cml", "--overwrite", "wide.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)
        self.assertEqual(r.error()["detail"]["features"], d)
        self.assertIn("131072", r.error()["message"])
        self.assertFalse((self.sb.dir / "w.cml").exists())

    def test_the_diagnostics_of_the_iterations_are_bounded(self):
        # k = 1024 and 126 features: 130049 sums a slot; 129 slots pass 2^24 floats
        d = 126
        n = 1030
        self.sb.write("wide2.csv", ",".join("c%d" % j for j in range(d)) + "\n" + "".join(",".join(str((i * (j + 3)) % 17) for j in range(d)) + "\n" for i in range(n)))
        r = self.ml("fit", "--model", "kmeans", "--k", "1024", "--iterations", "129", "--output", "w.cml", "--overwrite", "wide2.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)
        self.assertIn("diagnostics", r.error()["message"])
        r = self.ml("fit", "--model", "kmeans", "--k", "1024", "--iterations", "1", "--output", "w.cml", "--overwrite", "wide2.csv")
        self.assertEqual(r.status, 0, r)

    def test_the_rows_are_bounded_by_the_reader(self):
        r = self.ml("fit", "--model", "kmeans", "--k", "2", "--max-rows", "5", "--output", "m.cml", "--overwrite", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.input.too-many-rows"), r)

    def test_threads_outside_the_range_are_refused(self):
        for t in ("0", "65"):
            r = self.ml("fit", "--model", "kmeans", "--k", "2", "--threads", t, "--output", "m.cml", "--overwrite", "full.csv")
            self.assertEqual((r.status, r.first_rule()), (2, "ml.param.threads"), (t, r))

    def test_sums_that_overflow(self):
        # the sums of the update overflow a double: a fit is refused; a score that overflows has no value (null), as a metric with none does everywhere
        self.sb.write("huge.csv", "x\n1e200\n-1e200\n3e200\n5e200\n")
        r = self.ml("fit", "--model", "kmeans", "--k", "2", "--output", "h.cml", "--overwrite", "huge.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.fit.non-finite"), r)
        self.sb.write("mid.csv", "x\n1e154\n-1e154\n2e154\n")
        self.fit("mid.csv", "mid.cml", "--k", "2")
        e = self.ml("eval", "--model-file", "mid.cml", "huge.csv")
        self.assertEqual(e.status, 0, e)
        self.assertIsNone(e.data()["metrics"]["inertia"])
        self.assertEqual(e.data()["metrics"]["sizes"], [4, 0])
        self.assertEqual(e.data()["metrics"]["mean_distance"], [None, None])

    def test_a_refused_fit_writes_nothing(self):
        self.sb.write("keep.cml", b"x")
        r = self.ml("fit", "--model", "kmeans", "--k", "7", "--output", "keep.cml", "--overwrite", "six.csv")
        self.assertEqual(r.first_rule(), "ml.param.out-of-range")
        self.assertEqual(self.sb.read("keep.cml"), b"x")

    def test_a_file_with_no_column_left_is_empty(self):
        self.sb.write("one.csv", "x\n1\n2\n3\n")
        r = self.ml("fit", "--model", "kmeans", "--k", "2", "--drop", "x", "--output", "m.cml", "--overwrite", "one.csv")
        self.assertEqual(r.status, 8, r)
        self.assertEqual(r.first_rule(), "ml.input.empty", r)

    # ------------------------------------------------------------------ WK8: threads
    def test_the_model_file_is_the_same_at_1_2_4_8_and_16_threads(self):
        self.sb.write("big.csv", blobs_csv(n=600, d=4, groups=5, label=False))
        cases = (["--k", "5", "--seed", "9"], ["--k", "7", "--init", "random", "--seed", "4"], ["--k", "3", "--split", "train=0.8", "--seed", "2"],
                 ["--k", "6", "--iterations", "20", "--step", "scale column=* kind=standard"])
        for extra in cases:
            seen = {}
            for t in (1, 2, 4, 8, 16):
                r = self.ml("fit", "--model", "kmeans", "--output", "th.cml", "--overwrite", "--threads", str(t), *extra, "big.csv")
                self.assertEqual(r.status, 0, r)
                seen[t] = (self.sb.read("th.cml"), r.stdout.replace(b"th.cml", b"M"))
            self.assertEqual({t: v[0] for t, v in seen.items() if v[0] != seen[1][0]}, {}, extra)
            self.assertEqual({t: 1 for t, v in seen.items() if v[1] != seen[1][1]}, {}, extra)

    def test_eval_accepts_threads_and_changes_nothing(self):
        self.fit("full.csv", "m.cml", "--k", "3")
        a = self.ml("eval", "--model-file", "m.cml", "full.csv").stdout
        b = self.ml("eval", "--model-file", "m.cml", "--threads", "7", "full.csv").stdout
        self.assertEqual(a, b)

    # ------------------------------------------------------------------ the columns and the steps
    def test_columns_are_matched_by_name(self):
        self.fit("full.csv", "m.cml", "--k", "3")
        rev = self.ml("predict", "--model-file", "m.cml", "--distance", "--limit", "1000", "full.csv").data()["rows"]
        lines = self.sb.read("full.csv").decode().splitlines()
        swapped = "\n".join(",".join(reversed(l.split(","))) for l in lines) + "\n"
        self.sb.write("swapped.csv", swapped)
        self.assertEqual(self.ml("predict", "--model-file", "m.cml", "--distance", "--limit", "1000", "swapped.csv").data()["rows"], rev)
        self.sb.write("two.csv", "f0,f1\n1,2\n")
        self.assertEqual(self.ml("predict", "--model-file", "m.cml", "two.csv").first_rule(), "ml.schema.mismatch")

    def test_categorical_columns_reach_k_means_through_the_encoders(self):
        rng = random.Random(7)
        rows = ["x,city"] + ["%.2f,%s" % (rng.gauss(0, 1) + (5 if c == "b" else 0), c) for c in rng.choices("abc", k=60)]
        self.sb.write("cat.csv", "\n".join(rows) + "\n")
        r = self.ml("fit", "--model", "kmeans", "--k", "3", "--cat", "city", "--step", "encode column=city kind=onehot", "--output", "c.cml", "--overwrite", "cat.csv")
        self.assertEqual((r.status, r.data()["task"]), (0, "clustering"), r)
        self.assertIn("string_table", r.data()["summary"])
        p = self.ml("predict", "--model-file", "c.cml", "--distance", "--limit", "3", "cat.csv")
        self.assertEqual((p.status, p.data()["columns"]), (0, ["cluster", "distance"]), p)
        e = self.ml("eval", "--model-file", "c.cml", "cat.csv")
        self.assertEqual((e.status, e.data()["clusters"]), (0, 3), e)
        self.assertEqual(self.ml("fit", "--model", "kmeans", "--k", "3", "--cat", "city", "--output", "c.cml", "--overwrite", "cat.csv").first_rule(), "ml.input.categorical-not-encoded")
        # a numeric step beside the encoder reads the numeric column by its name: the schema of the numeric columns marks no target
        r = self.ml("fit", "--model", "kmeans", "--k", "2", "--cat", "city", "--step", "encode column=city kind=label;scale column=x kind=standard", "--output", "c2.cml", "--overwrite", "cat.csv")
        self.assertEqual((r.status, r.data()["summary"]["steps"]), (0, 2), r)

    def test_a_dry_run_writes_nothing(self):
        r = self.ml("fit", "--model", "kmeans", "--k", "3", "--dry-run", "--output", "dry.cml", "full.csv")
        self.assertEqual(r.status, 9, r)
        self.assertFalse((self.sb.dir / "dry.cml").exists())


if __name__ == "__main__":
    unittest.main()
