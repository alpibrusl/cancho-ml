"""I4, I7 -- scikit-learn, with the bounds of docs/interpret.md section 8 fixed before the run.

A module of its own because its import needs numpy and scikit-learn (scripts/run_conformance.py --no-oracles lists it as NOT RUN where they are not installed).

I4a  the structure of OUR trees (bootstrap off, so that a node's rows are the rows that reach it) is loaded into scikit-learn's `Tree`; the impurity of every node is computed here
     from the data routed down our tree; scikit-learn's own `compute_feature_importances` is compared with the table `ml` prints: relative 1e-12 (classification) and
     1e-9 (regression, which uses the stored leaf means).
I4b  `RandomForest*.feature_importances_` on the same data is a DIFFERENT algorithm (exact thresholds, its own streams): the top feature is the same, the Spearman correlation of the
     two rankings is at least 0.9, and no share differs by more than 0.1; the control (scikit-learn's importances of a forest fitted on shuffled labels) is outside those bounds.
I7   `sklearn.inspection.permutation_importance` on the SAME fitted model (an estimator whose `predict` is `ml predict` on the permuted data, 30 repeats, its own stream) against
     `ml importance --repeats 30`: for every column |ours - theirs| <= 4 sqrt((std_o^2 + std_s^2) / 30) + 1e-3 |baseline|, and two columns whose means differ by more than four
     standard errors keep their order. Aligned: the model, the rows, the metric, the positive label. Not aligned: the permutations. The control (the same comparison with a
     model that has not seen the target) is outside the bounds."""

import json
import math
import sys
import unittest

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree._tree import Tree

from harness import ROOT, Sandbox

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_interp as O  # noqa: E402

FEATURES = ["x0", "x1", "x2", "NOISE", "x3", "x4", "x5"]


def data_of(name, n, seed):
    header, rows = O.gen_dataset(name, n, seed)
    X = np.array([[float(c) for c in r[:-1]] for r in rows])
    y = np.array([float(r[-1]) for r in rows])
    return O.to_csv(header, rows), X, y


def sk_tree(M, t, X, y, flip=False):
    """scikit-learn's Tree for tree t of our model M: the structure and the thresholds are ours, the impurities are computed here from the rows routed down our tree"""
    T = M["tree"]
    lo, hi = T["off"][t], T["off"][t + 1]
    n = hi - lo
    cls = T["task"] == 0
    dt = np.dtype({"names": ["left_child", "right_child", "feature", "threshold", "impurity", "n_node_samples", "weighted_n_node_samples", "missing_go_to_left"],
                   "formats": ["<i8", "<i8", "<i8", "<f8", "<f8", "<i8", "<f8", "u1"], "offsets": [0, 8, 16, 24, 32, 40, 48, 56], "itemsize": 64})
    nodes = np.zeros(n, dtype=dt)
    rows_of = {0: np.arange(len(y))}
    values = np.zeros((n, 1, 2 if cls else 1))
    for i in range(n):
        g = lo + i
        r = rows_of[i]
        yy = y[r]
        if len(yy) == 0:
            imp = 0.0
        elif cls:
            p1 = float((yy == 1).sum()) / len(yy)
            imp = 1.0 - p1 * p1 - (1 - p1) * (1 - p1)
            values[i, 0, :] = [len(yy) - (yy == 1).sum(), (yy == 1).sum()]
        else:
            imp = float(np.var(yy))
            values[i, 0, 0] = yy.mean()
        nodes[i]["impurity"] = imp
        nodes[i]["n_node_samples"] = len(yy)
        nodes[i]["weighted_n_node_samples"] = float(len(yy))
        if T["feat"][g] < 0:
            nodes[i]["left_child"] = -1
            nodes[i]["right_child"] = -1
            nodes[i]["feature"] = -2
            nodes[i]["threshold"] = -2.0
        else:
            f = T["feat"][g]
            nodes[i]["left_child"] = T["left"][g]
            nodes[i]["right_child"] = T["right"][g]
            nodes[i]["feature"] = f
            nodes[i]["threshold"] = T["val"][g]
            go = (X[r, f] < T["val"][g]) != flip
            rows_of[T["left"][g]] = r[go]
            rows_of[T["right"][g]] = r[~go]
    tree = Tree(len(FEATURES), np.array([2 if cls else 1], dtype=np.intp), 1)
    tree.__setstate__({"max_depth": T["max_depth"], "node_count": n, "nodes": nodes, "values": values})
    return tree


def sk_importances_of_our_model(M, X, y, flip=False):
    per = []
    for t in range(M["tree"]["trees"]):
        v = sk_tree(M, t, X, y, flip).compute_feature_importances(normalize=True)
        per.append(v)
    avg = sum(per) / len(per)
    return avg / avg.sum()


class OurTreesInScikitLearn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def fit(self, name, model, task, extra, n=1500):
        text, X, y = data_of(name, n, 21)
        self.sb.write(name + ".csv", text)
        r = self.sb.run("fit", "--model", model, "--task", task, "--target", "y", "--overwrite", "--output", name + model + ".ml", name + ".csv", *extra)
        self.assertEqual(r.status, 0, r)
        M = O.read_model(self.sb.read(name + model + ".ml"))
        got = self.sb.run("importance", "--model-file", name + model + ".ml").data()["importance"]
        return M, X, y, np.array([float(c["importance"]) for c in got["columns"]])

    def test_i4a_a_decision_tree_classification(self):
        M, X, y, ours = self.fit("cls", "tree", "classification", ["--max-depth", "7"])
        theirs = sk_importances_of_our_model(M, X, y)
        self.assertLessEqual(float(np.max(np.abs(ours - theirs) / np.maximum(theirs, 1e-300) * (theirs > 0))), 1e-12, (ours, theirs))
        self.assertTrue(np.all(ours[theirs == 0] == 0))

    def test_i4a_a_decision_tree_regression(self):
        M, X, y, ours = self.fit("reg", "tree", "regression", ["--max-depth", "7"])
        theirs = sk_importances_of_our_model(M, X, y)
        self.assertLessEqual(float(np.max(np.abs(ours - theirs) / np.maximum(theirs, 1e-300) * (theirs > 0))), 1e-9, (ours, theirs))

    def test_i4a_a_forest_without_bootstrap_classification(self):
        M, X, y, ours = self.fit("cls", "forest", "classification", ["--trees", "20", "--max-depth", "6", "--bootstrap", "0", "--seed", "2"])
        theirs = sk_importances_of_our_model(M, X, y)
        self.assertLessEqual(float(np.max(np.abs(ours - theirs) / np.maximum(theirs, 1e-300) * (theirs > 0))), 1e-12, (ours, theirs))

    def test_i4a_a_forest_without_bootstrap_regression(self):
        M, X, y, ours = self.fit("reg", "forest", "regression", ["--trees", "20", "--max-depth", "6", "--bootstrap", "0", "--seed", "2"])
        theirs = sk_importances_of_our_model(M, X, y)
        self.assertLessEqual(float(np.max(np.abs(ours - theirs) / np.maximum(theirs, 1e-300) * (theirs > 0))), 1e-9, (ours, theirs))

    def test_i4a_the_check_can_fail(self):
        # the same comparison with the rows routed down the wrong side of every threshold (so that the node impurities no longer belong to the stored counts) is outside the bound
        M, X, y, ours = self.fit("cls", "tree", "classification", ["--max-depth", "7"])
        theirs = sk_importances_of_our_model(M, X, y, flip=True)
        self.assertGreater(float(np.max(np.abs(ours - theirs))), 1e-3, (ours, theirs))


def rank_corr(a, b):
    return spearmanr(a, b)[0]


class RandomForestImportances(unittest.TestCase):
    """I4b: a different algorithm, so only the ranking and the order of magnitude are compared"""

    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def compare(self, name, task, shuffled=False):
        text, X, y = data_of(name, 2000, 31)
        self.sb.write(name + ".csv", text)
        r = self.sb.run("fit", "--model", "forest", "--task", task, "--trees", "100", "--max-depth", "8", "--seed", "1", "--target", "y", "--overwrite", "--output", name + ".ml", name + ".csv")
        self.assertEqual(r.status, 0, r)
        got = self.sb.run("importance", "--model-file", name + ".ml").data()["importance"]
        ours = np.array([float(c["importance"]) for c in got["columns"]])
        ys = y.copy()
        if shuffled:
            ys = np.random.RandomState(0).permutation(y)
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        sk = cls(n_estimators=100, max_depth=8, random_state=0, n_jobs=1).fit(X, ys)
        return ours, sk.feature_importances_

    def within(self, ours, theirs):
        return (int(np.argmax(ours)) == int(np.argmax(theirs)) and rank_corr(ours, theirs) >= 0.9 and float(np.max(np.abs(ours - theirs))) <= 0.1)

    def test_i4b_classification(self):
        ours, theirs = self.compare("cls", "classification")
        self.assertTrue(self.within(ours, theirs), (ours, theirs, rank_corr(ours, theirs)))

    def test_i4b_regression(self):
        ours, theirs = self.compare("reg", "regression")
        self.assertTrue(self.within(ours, theirs), (ours, theirs, rank_corr(ours, theirs)))

    def test_i4b_the_control_is_outside_the_bounds(self):
        for name, task in (("cls", "classification"), ("reg", "regression")):
            ours, theirs = self.compare(name, task, shuffled=True)
            self.assertFalse(self.within(ours, theirs), (name, ours, theirs))


class MLBase(BaseEstimator):
    """an estimator whose predict is `ml predict` on the rows it is given"""

    def __init__(self, sb=None, model=None, classification=False, columns=None):
        self.sb, self.model, self.classification, self.columns = sb, model, classification, columns

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def _run(self, X, proba):
        n = len(self.columns)
        text = ",".join(self.columns) + "\n" + "".join(",".join(repr(float(v)) for v in row) + "\n" for row in X)
        self.sb.write("sk_in.csv", text)
        r = self.sb.run("predict", "--model-file", self.model, *(["--proba"] if proba else []), "--limit", "1000000", "sk_in.csv")
        assert r.status == 0, r
        rows = r.data()["rows"]
        return rows

    def predict(self, X):
        rows = self._run(X, self.classification)
        return np.array([float(r[0]) for r in rows])

    def predict_proba(self, X):
        rows = self._run(X, True)
        p = np.array([float(r[1]) for r in rows])
        return np.column_stack([1.0 - p, p])


class MLRegressor(RegressorMixin, MLBase):
    pass


class MLClassifier(ClassifierMixin, MLBase):
    pass


class PermutationAgainstScikitLearn(unittest.TestCase):
    R = 30

    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def prepare(self, name, model, task):
        text, X, y = data_of(name, 1500, 41)
        text2, X2, y2 = data_of(name, 400, 42)
        self.sb.write(name + ".csv", text)
        self.sb.write(name + "_new.csv", text2)
        extra = ["--max-depth", "7", "--trees", "40", "--seed", "1"] if model == "forest" else (["--step", "scale column=* kind=standard"] if model in ("linear", "logistic") else [])
        r = self.sb.run("fit", "--model", model, *(["--task", task] if model in ("forest", "tree") else []), "--target", "y", "--overwrite", "--output", name + model + ".ml", name + ".csv", *extra)
        self.assertEqual(r.status, 0, r)
        return X2, y2

    def ours(self, name, model, metric):
        d = self.sb.run("importance", "--model-file", name + model + ".ml", "--data", name + "_new.csv", "--repeats", str(self.R), "--seed", "5", "--metric", metric).data()["permutation"]
        return d, np.array([float(c["importance"]) for c in d["columns"]]), np.array([float(c["std"]) for c in d["columns"]])

    def theirs(self, name, model, X2, y2, scoring, classification, drop_signal=False):
        est = (MLClassifier if classification else MLRegressor)(self.sb, name + model + ".ml", classification, FEATURES)
        est.fit(X2, y2)
        yy = y2
        if drop_signal:
            yy = np.random.RandomState(1).permutation(y2)
        res = permutation_importance(est, X2, yy, scoring=scoring, n_repeats=self.R, random_state=0)
        return res.importances_mean, res.importances_std

    def inside(self, d, mo, so, mt, st, baseline):
        bound = 4.0 * np.sqrt((so ** 2 + st ** 2) / self.R) + 1e-3 * abs(baseline)
        ok = bool(np.all(np.abs(mo - mt) <= bound))
        order_ok = True
        for i in range(len(mo)):
            for j in range(len(mo)):
                if abs(mt[i] - mt[j]) > 4.0 * math.sqrt((st[i] ** 2 + st[j] ** 2) / self.R) and (mt[i] > mt[j]) != (mo[i] > mo[j]) and abs(mo[i] - mo[j]) > 0:
                    order_ok = False
        return ok, order_ok

    def run_case(self, name, model, task, metric, scoring, classification):
        X2, y2 = self.prepare(name, model, task)
        d, mo, so = self.ours(name, model, metric)
        mt, st = self.theirs(name, model, X2, y2, scoring, classification)
        ok, order_ok = self.inside(d, mo, so, mt, st, float(d["baseline"]))
        self.assertTrue(ok, (name, model, metric, list(mo), list(mt)))
        self.assertTrue(order_ok, (name, model, metric))
        # the control: scikit-learn's importances with the target shuffled are not ours
        mc, sc = self.theirs(name, model, X2, y2, scoring, classification, drop_signal=True)
        ok_c, _ = self.inside(d, mo, so, mc, sc, float(d["baseline"]))
        self.assertFalse(ok_c, (name, model, "control", list(mo), list(mc)))

    def test_i7_a_forest_regression_rmse(self):
        self.run_case("reg", "forest", "regression", "rmse", "neg_root_mean_squared_error", False)

    def test_i7_a_linear_model_r2(self):
        self.run_case("reg", "linear", "regression", "r2", "r2", False)

    def test_i7_a_forest_classification_auc(self):
        self.run_case("cls", "forest", "classification", "roc_auc", "roc_auc", True)

    def test_i7_a_logistic_model_accuracy(self):
        self.run_case("cls", "logistic", "classification", "accuracy", "accuracy", True)


if __name__ == "__main__":
    unittest.main()
