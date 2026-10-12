"""N1 -- no input reaches a trap. A seeded fuzz of argument vectors, and an exhaustive sweep of one-flag mistakes.

Spike F found `SIGILL` on a missing argument and on `..`. Here every vector must end in a status the tool declares, never a trap
(132, 134, 136, 139 or a negative status), and its answer must be one valid JSON line unless the vector asked for text. The fuzz is
seeded (FUZZ_SEED, FUZZ_N) and runs in a scratch directory that is the working directory and the --root; the vocabulary holds no
absolute directory that could be written to."""

import os
import random
import unittest

from fixtures import CASES, REG, CLS
from harness import TRAPS, Sandbox, introspect, run, spec, validate

T = spec.tables()
VERBS = ["fit", "predict", "eval", "introspect", "skill", "frobnicate", "", "FIT", "fit predict"]
FLAGS = ["--" + f["name"] for f in T["flags"]] + ["--", "-", "-x", "---", "--=", "--root=", "--bogus", "-h", "--help", "--version"]
VALUES = ["1", "0", "-1", "1000", "99999999999999999999", "18446744073709551616", "", "..", "../x", "a/../b", "./x", "a//b", "a/",
          "/etc/passwd", "/dev/null", "/nonexistent", "a" * 5000, "reg.csv", "cls.csv", "m.cml", "c.cml", "bad.cml", "sub", "sub/reg.csv",
          "missing", "y", "lab", "id", "x", "id,x", "y,y", ",", ",,", "\n", "\x01", "\xe9", "\xff", "constant", "regression", "classification",
          "json", "text", "tab", "log.ndjson", "Y", "a b", "--x", "-1e5", "0x10", "1_0", "\ud800".encode("utf-8", "surrogatepass").decode("latin-1")]


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
    assert sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv").status == 0
    assert sb.run("fit", "--task", "classification", "--target", "lab", "--output", "c.cml", "cls.csv").status == 0
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
