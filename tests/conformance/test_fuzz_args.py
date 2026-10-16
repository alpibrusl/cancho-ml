"""N1 -- no input reaches a trap. A seeded fuzz of argument vectors, and an exhaustive sweep of one-flag mistakes.

Spike F found `SIGILL` on a missing argument and on `..`. Here every vector must end in a status the tool declares, never a trap
(132, 134, 136, 139 or a negative status), and its answer must be one valid JSON line unless the vector asked for text. The fuzz is
seeded (FUZZ_SEED, FUZZ_N) and runs in a scratch directory that is the working directory and the --root; the vocabulary holds no
absolute directory that could be written to."""

import os
import random
import unittest

from fixtures import CASES, REG, CLS, LCLS, LIN, LINMIS
from harness import TRAPS, Sandbox, introspect, run, spec, validate

T = spec.tables()
VERBS = ["fit", "predict", "eval", "introspect", "skill", "frobnicate", "", "FIT", "fit predict", "importance", "dump", "describe", "cv", "runs"]
FLAGS = ["--" + f["name"] for f in T["flags"]] + ["--", "-", "-x", "---", "--=", "--root=", "--bogus", "-h", "--help", "--version"]
VALUES = ["1", "0", "-1", "1000", "99999999999999999999", "18446744073709551616", "", "..", "../x", "a/../b", "./x", "a//b", "a/",
          "/etc/passwd", "/dev/null", "/nonexistent", "a" * 5000, "reg.csv", "cls.csv", "m.cml", "c.cml", "bad.cml", "sub", "sub/reg.csv",
          "missing", "y", "lab", "id", "x", "id,x", "y,y", ",", ",,", "\n", "\x01", "\xe9", "\xff", "constant", "regression", "classification",
          "linear", "logistic", "kmeans", "kmeans++", "random", "tree", "forest", "31", "4194305", "999999999", "4096", "1025", "256", "train=0.8", "train=2", "train=.5", "train=0.123456789", "train=", "0.5", "-0.5", "1e400", "nan", "1e-5", "p.txt", "lin.csv",
          "lcls.csv", "lin.cml", "log.cml", "impute column=* strategy=mean", "scale column=a kind=standard;impute column=b strategy=median",
          "scale column=zzz kind=minmax", "model kind=linear", ";", ";;;", "encode column=a kind=onehot max-categories=0", "bucket column=a kind=quantile bins=300",
          "impute column=a strategy=constant", "scale column= kind=minmax", "a=b=c", "# c", "\n",
          "rmse", "roc_auc", "log_loss", "r2", "f1", "100", "101", "tc.cml", "json", "text", "tab", "log.ndjson", "Y", "a b", "--x", "-1e5", "0x10", "1_0", "\ud800".encode("utf-8", "surrogatepass").decode("latin-1")]


def sandbox():
    # the working directory is one level inside a private directory, so that a path that escaped it by `..` would land
    # in a place nothing else uses
    sb = Sandbox()
    sb.top = sb.dir
    sb.dir = sb.top / "work"
    sb.dir.mkdir()
    sb.write("reg.csv", REG)
    sb.write("cls.csv", CLS)
    (sb.dir / "sub").mkdir()
    sb.write("sub/reg.csv", REG)
    sb.write("lin.csv", LIN)
    sb.write("lcls.csv", LCLS)
    sb.write("p.txt", "impute column=* strategy=mean\nscale column=* kind=standard\nmodel kind=linear l2=0.1\n")
    assert sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv").status == 0
    assert sb.run("fit", "--model", "linear", "--target", "y", "--output", "lin.cml", "lin.csv").status == 0
    assert sb.run("fit", "--model", "logistic", "--step", "scale column=* kind=standard", "--target", "lab", "--output", "log.cml", "lcls.csv").status == 0
    assert sb.run("fit", "--task", "classification", "--target", "lab", "--output", "c.cml", "cls.csv").status == 0
    assert sb.run("fit", "--model", "tree", "--target", "y", "--output", "tree.cml", "lin.csv").status == 0
    assert sb.run("fit", "--model", "forest", "--task", "classification", "--trees", "5", "--target", "lab", "--output", "forest.cml", "lcls.csv").status == 0
    assert sb.run("fit", "--model", "kmeans", "--k", "3", "--seed", "2", "--output", "km.cml", "lin.csv").status == 0
    sb.write("bad.cml", sb.read("m.cml")[:100])
    return sb


def judge(argv, r):
    problems = []
    if r.status in TRAPS or r.status < 0:
        return ["TRAP status %d" % r.status]
    declared = {c["code"] for c in introspect()["exit_codes"]} | {0}
    if r.status not in declared:
        problems.append("undeclared status %d" % r.status)
    verb = next((a for a in argv if a in ("introspect", "skill")), None)
    if argv and argv[0] in ("introspect", "skill") and r.status == 0:
        return problems
    wants_text = "--format" in argv and "text" in argv
    if not r.stdout:
        if not wants_text and r.status not in (1,):
            problems.append("no output, status %d" % r.status)
        return problems
    try:
        r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return problems + ["stdout is not UTF-8"]
    if wants_text and r.status in (0, 9):
        return problems
    return problems + validate(r, expect_stderr_empty=not wants_text)


class Fuzz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = sandbox()

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.sb.top, ignore_errors=True)

    def go(self, argv, rooted=True):
        bare = argv[:1] in (["introspect"], ["skill"])
        if bare:
            return run(*argv, cwd=self.sb.dir)
        if not rooted:
            # no --root: paths are the working directory's, and `..` is the tool's to refuse, not the builtins' to trap on
            return run(*argv, cwd=self.sb.dir)
        return run("--root", self.sb.dir, *argv, cwd=self.sb.dir)

    def check(self, argv, rooted=True):
        r = self.go(argv, rooted)
        problems = judge(argv, r)
        self.assertEqual(problems, [], "argv %r (rooted %s) -> %r" % (argv, rooted, r))

    def test_no_arguments_at_all(self):
        # `ml` alone, and with only an empty argument: argv[1] does not exist (spike F: a program that reads an argument before
        # counting them is ended by SIGILL)
        self.check([], rooted=False)
        self.check([""], rooted=False)
        self.check([""], rooted=True)
        self.check([])

    def test_every_flag_every_verb_every_one_mistake(self):
        for verb in ("fit", "predict", "eval"):
            base = {"fit": ["--target", "y", "--output", "o.cml", "reg.csv"], "predict": ["--model-file", "m.cml", "reg.csv"],
                    "eval": ["--model-file", "m.cml", "reg.csv"]}[verb]
            for flag in FLAGS:
                self.check([verb, *base, flag])
                for v in ("", "..", "../x", "99999999999999999999", "-1", "0", "a" * 5000, "reg.csv", "\xe9", "sub", "/etc/passwd"):
                    self.check([verb, *base, flag, v])
                    self.check([verb, *base, flag + "=" + v])
                    self.check([verb, flag, v, *base])
                self.check([verb, flag])
                self.check([verb, *base, flag, flag])
            for drop in range(len(base)):
                self.check([verb, *(base[:drop] + base[drop + 1:])])
            self.check([verb])
            self.check([verb, "--"])
            self.check([verb, "--", *base])

    def test_every_flag_of_the_planned_models_every_mistake(self):
        for base in (["fit", "--model", "linear", "--target", "y", "--output", "o.cml", "lin.csv"],
                     ["fit", "--model", "logistic", "--step", "scale column=* kind=standard", "--target", "lab", "--output", "o.cml", "lcls.csv"],
                     ["fit", "--plan", "p.txt", "--target", "y", "--output", "o.cml", "lin.csv"],
                     ["predict", "--model-file", "lin.cml", "lin.csv"], ["predict", "--model-file", "log.cml", "--proba", "lcls.csv"],
                     ["eval", "--model-file", "lin.cml", "lin.csv"], ["eval", "--model-file", "log.cml", "lcls.csv"]):
            for flag in ("--plan", "--step", "--l2", "--iterations", "--positive", "--split", "--seed", "--threads", "--proba", "--model", "--task"):
                self.check([*base, flag])
                self.check([*base, flag, flag])
                for v in VALUES[:0] + ["", "1", "0", "-1", "65", "99999999999999999999", "0.5", "-0.5", "1e400", "nan", "linear", "logistic", "train=0.8", "train=2",
                                       "train=0.01", "p.txt", "..", "impute column=* strategy=mean", "scale column=zzz kind=minmax", ";", "model kind=linear"]:
                    self.check([*base, flag, v])
                    self.check([*base, flag + "=" + v])

    def test_every_flag_of_the_tree_and_the_forest_every_mistake(self):
        bases = (["fit", "--model", "tree", "--target", "y", "--output", "o.cml", "lin.csv"],
                 ["fit", "--model", "forest", "--task", "classification", "--target", "lab", "--output", "o.cml", "lcls.csv"],
                 ["fit", "--model", "forest", "--target", "y", "--output", "o.cml", "lin.csv"],
                 ["fit", "--plan", "p.txt", "--target", "y", "--output", "o.cml", "lin.csv"],
                 ["predict", "--model-file", "tree.cml", "lin.csv"], ["predict", "--model-file", "forest.cml", "--proba", "lcls.csv"],
                 ["eval", "--model-file", "tree.cml", "lin.csv"], ["eval", "--model-file", "forest.cml", "lcls.csv"])
        values = ("", "0", "1", "2", "-1", "5", "30", "31", "255", "256", "1024", "1025", "4096", "4097", "4194304", "4194305", "999999999", "1000000000", "99999999999999999999",
                  "0.5", "1e3", "nan", "classification", "regression", "tree", "forest", "..")
        for base in bases:
            for flag in ("--trees", "--max-depth", "--min-leaf", "--bins", "--mtry", "--bootstrap", "--max-nodes", "--task", "--model", "--seed", "--threads", "--positive"):
                self.check([*base, flag])
                self.check([*base, flag, flag])
                for v in values:
                    self.check([*base, flag, v])
                    self.check([*base, flag + "=" + v])

    def test_every_flag_of_importance_and_dump_every_mistake(self):
        # docs/interpret.md: the models of the sandbox, with and without a data file, and every flag with a hostile value
        bases = (["importance", "--model-file", "lin.cml"], ["importance", "--model-file", "log.cml", "--data", "lcls.csv"], ["importance", "--model-file", "tree.cml", "--data", "lin.csv"],
                 ["importance", "--model-file", "forest.cml", "--data", "lcls.csv", "--repeats", "2"], ["importance", "--model-file", "km.cml"], ["importance", "--model-file", "m.cml"],
                 ["dump", "--model-file", "tree.cml"], ["dump", "--model-file", "forest.cml", "--tree", "1"], ["dump", "--model-file", "lin.cml"], ["dump", "--model-file", "km.cml"],
                 ["dump", "--model-file", "m.cml"])
        values = ("", "0", "1", "2", "-1", "4", "5", "100", "101", "1023", "1024", "4194304", "4194305", "99999999999999999999", "rmse", "roc_auc", "f1", "nope", "lin.csv", "..", "\xe9")
        for base in bases:
            for flag in ("--data", "--metric", "--repeats", "--seed", "--tree", "--max-nodes", "--max-work", "--threads", "--drop", "--format", "--model-file", "--max-bytes"):
                self.check([*base, flag])
                self.check([*base, flag, flag])
                for v in values:
                    self.check([*base, flag, v])
                    self.check([*base, flag + "=" + v])
            for drop in range(len(base)):
                self.check(base[:drop] + base[drop + 1:])
            self.check([*base, "extra.csv"])

    def test_every_flag_of_kmeans_every_mistake(self):
        # the 4 bases: a fit by flags, a fit by a plan, predict and eval; the flags of k-means and the flags that mean nothing to it
        self.sb.write("km.txt", "scale column=* kind=standard\nmodel kind=kmeans k=3 iterations=4 init=random\n")
        bases = (["fit", "--model", "kmeans", "--k", "3", "--output", "o.cml", "lin.csv"], ["fit", "--plan", "km.txt", "--output", "o.cml", "lin.csv"],
                 ["fit", "--model", "kmeans", "--output", "o.cml", "lcls.csv"], ["predict", "--model-file", "km.cml", "lin.csv"], ["eval", "--model-file", "km.cml", "lin.csv"])
        values = ("", "0", "1", "2", "3", "8", "9", "-1", "1024", "1025", "1000", "1001", "4097", "131072", "99999999999999999999", "0.5", "nan", "kmeans++", "random", "kmeans", "plusplus",
                  "train=0.5", "lab", "y", "..")
        for base in bases:
            for flag in ("--k", "--init", "--iterations", "--distance", "--target", "--task", "--positive", "--proba", "--seed", "--threads", "--split", "--drop", "--model", "--l2", "--trees"):
                self.check([*base, flag])
                self.check([*base, flag, flag])
                for v in values:
                    self.check([*base, flag, v])
                    self.check([*base, flag + "=" + v])

    def test_seeded_vectors_with_kmeans(self):
        rng = random.Random(int(os.environ.get("FUZZ_SEED", "20261012")) + 2)
        flags = ["--k", "--init", "--iterations", "--distance", "--target", "--task", "--positive", "--proba", "--seed", "--threads", "--split", "--drop", "--step", "--plan", "--cat", "--max-rows"]
        values = ["1", "2", "3", "7", "8", "9", "1024", "1025", "1000", "1001", "0", "-1", "kmeans++", "random", "y", "lab", "a", "train=0.5", "km.txt", "impute column=* strategy=mean",
                  "scale column=* kind=standard", "bucket column=a kind=width bins=3", "", "..", "99999999999999999999"]
        self.sb.write("km.txt", "scale column=* kind=standard\nmodel kind=kmeans k=3 iterations=4 init=random\n")
        for _ in range(int(os.environ.get("FUZZ_N", "2500")) // 2):
            verb = rng.choice(["fit", "fit", "predict", "eval"])
            if verb == "fit":
                argv = ["fit", "--model", "kmeans", "--k", rng.choice(["2", "3", "9"]), "--output", "o.cml", rng.choice(["lin.csv", "lcls.csv", "reg.csv"])]
            else:
                argv = [verb, "--model-file", "km.cml", rng.choice(["lin.csv", "lcls.csv", "reg.csv"])]
            for _ in range(rng.randint(1, 5)):
                argv.insert(rng.randrange(1, len(argv)), rng.choice(values) if rng.random() < 0.5 else rng.choice(flags))
                if rng.random() < 0.7:
                    i = rng.randrange(1, len(argv))
                    argv.insert(i, rng.choice(flags))
                    argv.insert(i + 1, rng.choice(values))
            self.check(argv, rooted=rng.random() < 0.6)

    def test_seeded_vectors_with_the_tree_and_the_forest(self):
        rng = random.Random(int(os.environ.get("FUZZ_SEED", "20261012")) + 1)
        flags = ["--trees", "--max-depth", "--min-leaf", "--bins", "--mtry", "--bootstrap", "--max-nodes", "--threads", "--seed", "--task", "--positive", "--split", "--proba", "--step", "--plan"]
        values = ["1", "2", "3", "7", "30", "31", "64", "255", "256", "1000", "1025", "4096", "4194305", "0", "-1", "classification", "regression", "train=0.5", "p.txt",
                  "impute column=* strategy=mean", "scale column=* kind=standard", "bucket column=a kind=width bins=3", "", "..", "99999999999999999999"]
        for _ in range(int(os.environ.get("FUZZ_N", "2500")) // 2):
            verb = rng.choice(["fit", "fit", "predict", "eval"])
            if verb == "fit":
                argv = ["fit", "--model", rng.choice(["tree", "forest"]), "--target", rng.choice(["y", "lab"]), "--output", "o.cml", rng.choice(["lin.csv", "lcls.csv"])]
            else:
                argv = [verb, "--model-file", rng.choice(["tree.cml", "forest.cml"]), rng.choice(["lin.csv", "lcls.csv"])]
            for _ in range(rng.randint(1, 5)):
                argv.insert(rng.randrange(1, len(argv)), rng.choice(values) if rng.random() < 0.5 else rng.choice(flags))
                if rng.random() < 0.7:
                    i = rng.randrange(1, len(argv))
                    argv.insert(i, rng.choice(flags))
                    argv.insert(i + 1, rng.choice(values))
            self.check(argv, rooted=rng.random() < 0.6)

    def test_seeded_vectors(self):
        rng = random.Random(int(os.environ.get("FUZZ_SEED", "20261012")))
        n = int(os.environ.get("FUZZ_N", "2500"))
        seeds = [c["argv"] for c in CASES if not c.get("noroot") and not c.get("close_stdin") and not c.get("close_stdout")]
        for _ in range(n):
            mode = rng.randrange(3)
            if mode == 0:
                argv = [rng.choice(VERBS)] + [rng.choice(FLAGS + VALUES) for _ in range(rng.randint(0, 9))]
            else:
                argv = list(rng.choice(seeds))
                for _ in range(rng.randint(1, 3)):
                    op = rng.randrange(5)
                    if op == 0 and argv:
                        del argv[rng.randrange(len(argv))]
                    elif op == 1 and argv:
                        argv.insert(rng.randrange(len(argv) + 1), rng.choice(argv))
                    elif op == 2 and argv:
                        argv[rng.randrange(len(argv))] = rng.choice(VALUES + FLAGS)
                    elif op == 3 and len(argv) > 1:
                        i, j = rng.randrange(len(argv)), rng.randrange(len(argv))
                        argv[i], argv[j] = argv[j], argv[i]
                    else:
                        argv.insert(rng.randrange(len(argv) + 1), rng.choice(FLAGS))
            argv = [a for a in argv if "\x00" not in a]
            if any(a.startswith("--root") for a in argv):
                argv = [a for a in argv if not a.startswith("--root")]
            if "--output" in argv or "--log" in argv:
                # never let a vector name an absolute or parent path as something to write
                argv = [("o.cml" if a.startswith("/") or a.startswith("..") else a) for a in argv]
            self.check(argv, rooted=rng.random() < 0.6)

    def test_a_missing_value_and_a_dotdot_in_every_path_position(self):
        for flag in ("--model-file", "--output", "--log", "--root"):
            self.check(["fit", "reg.csv", flag])
            self.check(["fit", "--target", "y", "--output", "o.cml", "reg.csv", flag, ".."])
            self.check(["predict", "--model-file", "m.cml", flag, "a/../..", "reg.csv"])
        for p in ("..", "../reg.csv", "sub/..", "sub/../reg.csv", "./..", "a/../../b", ".../x", "..x", "x.."):
            for rooted in (True, False):
                self.check(["fit", "--target", "y", "--output", "o.cml", p], rooted)
                self.check(["predict", "--model-file", p, "reg.csv"], rooted)
                self.check(["eval", "--model-file", "m.cml", "--log", p, "reg.csv"], rooted)
                self.check(["fit", "--target", "y", "--output", p, "reg.csv"], rooted)


if __name__ == "__main__":
    unittest.main()
