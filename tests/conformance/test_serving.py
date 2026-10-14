"""V1 to V4, P1, P2, P5, D3 and T1 of docs/serving.md: `predict` from standard input, the binary `ml-predict`, and the one-flag-table gate for both.

The corpus (serving_corpus.py) is every kind of stored model, 1 to 7, with categorical steps, missing cells, unseen categories and hash steps, and inputs of every
shape the reader meets. Each case runs a real process and compares bytes.
"""

import json
import os
import re
import stat
import subprocess
import sys
import unittest

from harness import BIN, PREDICT_BIN, ROOT, Sandbox, TRAPS, introspect, run, spec

import serving_corpus as C

sys.path.insert(0, str(ROOT / "scripts"))

SB = None
MODELS = {}
INPUTS = {}


def setUpModule():
    global SB, MODELS, INPUTS
    SB = Sandbox()
    MODELS, INPUTS = C.build(SB)


def tearDownModule():
    SB.cleanup()


def ml(*args, stdin=None):
    return SB.run(*args, stdin=stdin)


def mp(*args, stdin=None):
    return SB.run(*args, stdin=stdin, binary_path=PREDICT_BIN)


def as_predict(r):
    """The retry of a repair starts with the name the binary was run by: `ml-predict` for the other binary."""
    return r.stdout.replace(BIN.encode(), PREDICT_BIN.encode())


def same_bytes(file_run, other, name):
    """The refusal of a file names the file in places (detail.path, the argv of a retry repair): the file's name is the stdin's `-` there."""
    want = file_run.stdout.replace(name.encode(), b"-").replace(b'"path":"-"', b'"path":"-"')
    return want == other.stdout


def pairs(only_ok=False):
    for mid, (mf, kind, fam) in MODELS.items():
        for iid, (name, ifam) in INPUTS.items():
            yield mid, mf, iid, name


class StdinEqualsFile(unittest.TestCase):
    """V1 and V2: the answer for the bytes on standard input is the answer for the same bytes in a file."""

    def test_every_model_and_every_input_give_the_same_bytes(self):
        n_ok = n_refused = 0
        for mid, mf, iid, name in pairs():
            f = ml("predict", name, "--model-file", mf)
            s = ml("predict", "-", "--model-file", mf, stdin=(SB.dir / name).read_bytes())
            self.assertNotIn(f.status, TRAPS, (mid, iid))
            self.assertEqual((f.status, f.stderr), (s.status, s.stderr), (mid, iid, f, s))
            if f.status == 0:
                self.assertEqual(f.stdout, s.stdout, (mid, iid))
                n_ok += 1
            else:
                self.assertTrue(same_bytes(f, s, name), (mid, iid, f.stdout, s.stdout))
                n_refused += 1
        # the corpus has both kinds of case in numbers (a corpus of refusals only would prove nothing)
        self.assertGreater(n_ok, 60)
        self.assertGreater(n_refused, 100)

    def test_an_input_of_many_chunks(self):
        # the scan reads 64 KiB at a time: a chunk boundary inside a row, in a file and in memory
        rows = "".join("%d,%d.%02d,%d.5\n" % (i % 97, i % 61, i % 100, i % 13) for i in range(14000))
        data = ("a,b,c\n" + rows).encode()
        self.assertGreater(len(data), 2 * 65536)
        SB.write("many_chunks.csv", data)
        for mid in ("linear", "tree_cls", "const_reg_dec"):
            mf = MODELS[mid][0]
            f = ml("predict", "many_chunks.csv", "--model-file", mf, "--limit", "1000000")
            s = ml("predict", "-", "--model-file", mf, "--limit", "1000000", stdin=data)
            self.assertEqual(f.status, 0, f)
            self.assertEqual(f.data()["total_rows"], 14000)
            self.assertEqual((f.status, f.stdout), (s.status, s.stdout), mid)
        # one byte more or less would change a row at a boundary; so does the last byte of a file with no final newline
        s = ml("predict", "-", "--model-file", MODELS["linear"][0], stdin=data[:-1])
        f = ml("predict", "many_chunks.csv", "--model-file", MODELS["linear"][0], "--limit", "5", "--from", "13995")
        self.assertEqual(s.data()["total_rows"], 14000)

    def test_no_operand_is_standard_input(self):
        for mid, mf, iid, name in pairs():
            if not INPUTS[iid][0].startswith(("num_plain", "cat_seen", "cat0_seen")):
                continue
            data = (SB.dir / name).read_bytes()
            a = ml("predict", "-", "--model-file", mf, stdin=data)
            b = ml("predict", "--model-file", mf, stdin=data)
            self.assertEqual(a.status, b.status, (mid, iid))
            if a.status == 0:
                self.assertEqual(a.stdout, b.stdout, (mid, iid))
            else:
                # the retry of a repair names the operand it was given: `-`, or none
                self.assertEqual(a.first_rule(), b.first_rule(), (mid, iid))

    def test_the_text_format_the_probability_the_page_and_the_dropped_columns(self):
        data = (SB.dir / "num_plain.csv").read_bytes()
        for mid in ("linear", "logistic", "tree_cls", "forest_reg", "const_cls", "const_reg_dec"):
            mf = MODELS[mid][0]
            for extra in (["--format", "text"], ["--limit", "3", "--from", "2"], ["--limit", "2", "--format", "text"], ["--from", "50"], ["--proba"],
                          ["--proba", "--limit", "4", "--from", "1"]):
                f = ml("predict", "num_plain.csv", "--model-file", mf, *extra)
                s = ml("predict", "-", "--model-file", mf, *extra, stdin=data)
                self.assertEqual((f.status, f.stdout, f.stderr), (s.status, s.stdout, s.stderr), (mid, extra))
        extra_col = b"a,b,c,zz\n1,2,3,4\n"
        f = ml("predict", "num_extra.csv", "--model-file", MODELS["linear"][0], "--drop", "zz") if SB.write("num_extra.csv", extra_col) else None
        s = ml("predict", "-", "--model-file", MODELS["linear"][0], "--drop", "zz", stdin=extra_col)
        self.assertEqual((f.status, f.stdout), (s.status, s.stdout))
        self.assertEqual(f.status, 0, f)

    def test_the_reader_limits_apply_to_standard_input_as_to_a_file(self):
        mf = MODELS["linear"][0]
        for flag, value, rule in (("--max-rows", "5", "ml.input.too-many-rows"), ("--max-columns", "2", "ml.input.too-many-columns"), ("--max-cells", "30", "ml.input.too-many-cells"),
                                  ("--max-line-bytes", "8", "limit.line-too-long")):
            f = ml("predict", "num_big.csv", "--model-file", mf, flag, value)
            s = ml("predict", "-", "--model-file", mf, flag, value, stdin=(SB.dir / "num_big.csv").read_bytes())
            self.assertEqual(f.first_rule(), rule, f)
            self.assertTrue(same_bytes(f, s, "num_big.csv"), (flag, f.stdout, s.stdout))

    def test_a_byte_order_mark_a_missing_final_newline_blank_lines_and_crlf(self):
        # the inputs of the corpus hold each; this asserts the corpus does (a mutant that drops one is killed by the equality above)
        for iid in ("num_crlf_bom", "num_no_final_newline", "num_blank_lines", "cat_bom_crlf"):
            self.assertIn(iid, INPUTS)
        data = (SB.dir / INPUTS["num_crlf_bom"][0]).read_bytes()
        self.assertTrue(data.startswith(b"\xef\xbb\xbf") and b"\r\n" in data)
        r = ml("predict", "-", "--model-file", MODELS["linear"][0], stdin=data)
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["total_rows"], 12)


class StdinBounds(unittest.TestCase):
    """V3."""

    def input_of_length(self, n):
        """A valid CSV of exactly n bytes: rows of 12 bytes, the last one stretched by digits of the float column."""
        head = b"a,b,c\n"
        row = b"1,2.50,0.5\n"
        k = (n - len(head) - 11) // len(row)
        rows = row * k
        rest = n - len(head) - len(rows)
        data = head + rows + b"1,2.50,0." + b"5" * (rest - len(b"1,2.50,0.\n")) + b"\n"
        self.assertEqual(len(data), n)
        return data

    def test_the_bound_is_exact(self):
        mf = MODELS["const_reg_dec"][0]
        size = len((SB.dir / mf).read_bytes())
        data = self.input_of_length(size + 300)
        n = len(data)
        ok = ml("predict", "-", "--model-file", mf, "--max-bytes", str(n), stdin=data)
        self.assertEqual(ok.status, 0, ok)
        bad = ml("predict", "-", "--model-file", mf, "--max-bytes", str(n - 1), stdin=data)
        self.assertEqual(bad.first_rule(), "limit.input-too-large", bad)
        self.assertEqual(bad.error()["detail"], {"size": n, "max": n - 1})
        # standard input much longer than the bound: the size found is the bound plus one, not the length of the stream
        huge = ml("predict", "-", "--model-file", mf, "--max-bytes", str(size + 10), stdin=data * 3)
        self.assertEqual(huge.error()["detail"], {"size": size + 11, "max": size + 10})
        # the same bytes in a file are read: --max-bytes bounds standard input, and a model file, not a CSV file
        SB.write("sized.csv", data)
        f = ml("predict", "sized.csv", "--model-file", mf, "--max-bytes", str(n - 1))
        self.assertEqual(f.status, 0, f)

    def test_an_empty_standard_input_and_a_closed_one_are_an_empty_file(self):
        mf = MODELS["linear"][0]
        SB.write("empty.csv", b"")
        f = ml("predict", "empty.csv", "--model-file", mf)
        for s in (ml("predict", "-", "--model-file", mf, stdin=b""), ml("predict", "--model-file", mf, stdin=b""), SB.run("predict", "-", "--model-file", mf, close_stdin=True)):
            self.assertEqual(f.first_rule(), "ml.input.empty")
            self.assertEqual((s.status, s.stdout), (f.status, f.stdout), s)

    def test_a_cell_that_is_not_utf8_and_a_nul_byte_are_the_files_refusals(self):
        for iid in ("bad_invalid_utf8", "bad_nul_byte"):
            name = INPUTS[iid][0]
            for mid in ("cat_linear_onehot", "linear"):
                f = ml("predict", name, "--model-file", MODELS[mid][0])
                s = ml("predict", "-", "--model-file", MODELS[mid][0], stdin=(SB.dir / name).read_bytes())
                self.assertTrue(same_bytes(f, s, name), (iid, mid))

    def test_a_refused_command_line_does_not_wait_for_standard_input(self):
        # standard input is an open pipe that nobody writes to or closes: a command line that is refused answers anyway
        for argv in (["predict", "-", "--model-file", MODELS["linear"][0], "--bogus"], ["predict", "--model-file"], ["predict", "-", "--model-file", MODELS["linear"][0], "--max-rows", "0"]):
            p = subprocess.Popen([BIN, "--root", str(SB.dir), *argv], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                p.wait(timeout=10)
                self.assertNotEqual(p.returncode, 0, argv)
            except subprocess.TimeoutExpired:
                p.kill()
                self.fail("waited for standard input after a refused command line: %r" % (argv,))
            finally:
                p.stdin.close()
                p.stdout.close()
                p.stderr.close()
                p.wait()

    def test_the_command_line_is_checked_before_standard_input_is_read(self):
        # a refused command line answers with no input at all: nothing to wait for
        r = SB.run("predict", "-", "--model-file", MODELS["linear"][0], "--bogus", close_stdin=True)
        self.assertEqual(r.first_rule(), "args.unknown-flag")


class Operand(unittest.TestCase):
    """V4."""

    def test_a_file_named_dash_is_dot_slash_dash(self):
        mf = MODELS["linear"][0]
        data = (SB.dir / "num_plain.csv").read_bytes()
        SB.write("-", data)
        f = ml("predict", "num_plain.csv", "--model-file", mf)
        dash = ml("predict", "./-", "--model-file", mf)
        self.assertEqual(f.status, 0)
        self.assertEqual(dash.stdout, f.stdout)
        # `-` reads standard input: with another content on standard input the answer is that content's
        other = ml("predict", "-", "--model-file", mf, stdin=b"a,b,c\n1,2.50,0.5\n")
        self.assertEqual(other.data()["total_rows"], 1)
        none = ml("predict", "-", "--model-file", mf, stdin=b"")
        self.assertEqual(none.first_rule(), "ml.input.empty")
        os.unlink(SB.dir / "-")

    def test_fit_and_eval_read_a_file_named_dash(self):
        SB.write("-", (SB.dir / C.NUM).read_bytes())
        r = ml("fit", "-", "--target", "y", "--drop", "lab", "--output", "dash.cml", "--overwrite")
        self.assertEqual(r.status, 0, r)
        e = ml("eval", "-", "--model-file", "dash.cml", "--drop", "lab")
        self.assertEqual(e.status, 0, e)
        # with no operand they still need the file
        self.assertEqual(ml("fit", "--target", "y", "--output", "dash2.cml").first_rule(), "args.missing-operand")
        self.assertEqual(ml("eval", "--model-file", "dash.cml").first_rule(), "args.missing-operand")
        os.unlink(SB.dir / "-")

    def test_root_does_not_apply_to_standard_input_and_is_still_checked(self):
        mf = MODELS["linear"][0]
        data = b"a,b,c\n1,2.50,0.5\n"
        ok = run("--root", SB.dir, "predict", "-", "--model-file", mf, stdin=data)
        self.assertEqual(ok.status, 0, ok)
        bad = run("--root", "..", "predict", "-", "--model-file", mf, stdin=data)
        self.assertEqual(bad.first_rule(), "path.dotdot", bad)

    def test_too_many_operands_and_a_second_dash(self):
        mf = MODELS["linear"][0]
        self.assertEqual(ml("predict", "num_plain.csv", "-", "--model-file", mf).first_rule(), "args.too-many-operands")
        self.assertEqual(ml("describe", "x.csv", "--model-file", mf).first_rule(), "args.too-many-operands")


class MlPredict(unittest.TestCase):
    """P1: the bytes of `ml-predict` are the bytes of `ml`."""

    def test_every_model_and_every_input_from_a_file_and_from_standard_input(self):
        for mid, mf, iid, name in pairs():
            data = (SB.dir / name).read_bytes()
            a = ml("predict", name, "--model-file", mf)
            b = mp("predict", name, "--model-file", mf)
            self.assertEqual((a.status, as_predict(a), a.stderr), (b.status, b.stdout, b.stderr), (mid, iid))
            c = mp("predict", "-", "--model-file", mf, stdin=data)
            d = ml("predict", "-", "--model-file", mf, stdin=data)
            self.assertEqual((c.status, c.stdout, c.stderr), (d.status, as_predict(d), d.stderr), (mid, iid))

    def test_the_flags_of_predict(self):
        data = (SB.dir / "num_plain.csv").read_bytes()
        for mid in ("linear", "logistic", "tree_cls", "const_cls", "forest_reg"):
            mf = MODELS[mid][0]
            for extra in (["--format", "text"], ["--limit", "3", "--from", "2"], ["--proba"], ["--max-rows", "4"], ["--max-line-bytes", "5"], ["--max-bytes", "10"]):
                a = ml("predict", "-", "--model-file", mf, *extra, stdin=data)
                b = mp("predict", "-", "--model-file", mf, *extra, stdin=data)
                self.assertEqual((a.status, as_predict(a), a.stderr), (b.status, b.stdout, b.stderr), (mid, extra))

    def test_describe_prints_the_same_bytes(self):
        for mid, (mf, kind, fam) in MODELS.items():
            a = ml("describe", "--model-file", mf)
            b = mp("describe", "--model-file", mf)
            self.assertEqual(a.status, 0, a)
            self.assertEqual((a.status, a.stdout), (b.status, b.stdout), mid)
            a = ml("describe", "--model-file", mf, "--limit", "2", "--format", "text")
            b = mp("describe", "--model-file", mf, "--limit", "2", "--format", "text")
            self.assertEqual((a.status, a.stdout), (b.status, b.stdout), mid)

    def test_a_missing_model_a_directory_and_a_damaged_model(self):
        SB.write("junk.cml", b"CML1" + bytes(100))
        for argv in (["predict", "num_plain.csv", "--model-file", "absent.cml"], ["predict", "num_plain.csv", "--model-file", "junk.cml"],
                     ["describe", "--model-file", "absent.cml"], ["describe", "--model-file", "junk.cml"], ["predict", "num_plain.csv", "--model-file", "."]):
            a, b = ml(*argv), mp(*argv)
            self.assertNotEqual(a.status, 0, a)
            self.assertEqual((a.status, as_predict(a)), (b.status, b.stdout), argv)


class MlPredictSurface(unittest.TestCase):
    """P2: predict, describe, introspect and skill, and nothing else."""

    def test_the_verbs_of_fit_and_eval_are_unknown_commands(self):
        for verb in ("fit", "eval", "frobnicate"):
            r = mp(verb, "num_plain.csv", "--model-file", MODELS["linear"][0])
            self.assertEqual(r.first_rule(), "args.unknown-command", r)
            self.assertEqual(r.status, 2)

    def test_the_flags_of_fit_eval_and_the_log_are_unknown_flags(self):
        mf = MODELS["linear"][0]
        for flag in ("--target", "--output", "--model", "--plan", "--step", "--threads", "--seed", "--split", "--positive", "--overwrite", "--dry-run", "--log", "--log-max-bytes",
                     "--cat", "--float", "--trees", "--l2", "--task", "--bins"):
            r = mp("predict", "num_plain.csv", "--model-file", mf, flag, "1")
            self.assertEqual(r.first_rule(), "args.unknown-flag", (flag, r))

    def test_the_flags_it_has(self):
        tool = {f["name"] for f in json.loads(subprocess.run([PREDICT_BIN, "introspect"], capture_output=True, stdin=subprocess.DEVNULL).stdout)["flags"]}
        self.assertEqual(tool, {"--" + n for n in ("root", "proba", "drop", "model-file", "limit", "from", "max-rows", "max-columns", "max-cells", "max-line-bytes", "max-bytes", "format")})

    def test_introspect_and_skill_name_the_binary(self):
        i = json.loads(subprocess.run([PREDICT_BIN, "introspect"], capture_output=True, stdin=subprocess.DEVNULL).stdout)
        self.assertEqual(i["tool"], "ml-predict")
        self.assertEqual({r["rule"] for r in i["rules"]}, {r["rule"] for r in introspect()["rules"]})
        s = subprocess.run([PREDICT_BIN, "skill"], capture_output=True, stdin=subprocess.DEVNULL).stdout.decode()
        self.assertIn("ml-predict", s)
        for flag in i["flags"]:
            self.assertIn("`%s" % flag["name"], s)
        self.assertNotIn("`--target", s)

    def test_the_authority_row_has_no_write_and_no_thread(self):
        i = json.loads(subprocess.run([PREDICT_BIN, "introspect"], capture_output=True, stdin=subprocess.DEVNULL).stdout)
        labels = {l["name"] for l in i["authority"]["labels"]}
        self.assertEqual(labels, {"args", "dir_read", "err_write", "file_read", "fs_read", "heap", "io_read", "io_write"})
        self.assertEqual(labels & {"conc", "file_write", "dir_write", "fs_write", "ffi", "net_out", "net_in", "clock"}, set())
        self.assertEqual(i["authority"], json.loads((ROOT / "manifests" / "ml-predict.authority.json").read_text()))

    def test_it_writes_nothing(self):
        # P5: a directory nobody can write to, a model and a CSV in it; the run leaves the listing as it found it
        sb = Sandbox()
        try:
            sb.write("m.cml", (SB.dir / MODELS["linear"][0]).read_bytes())
            sb.write("in.csv", (SB.dir / "num_plain.csv").read_bytes())
            before = sorted((p.name, p.read_bytes()) for p in sb.dir.iterdir())
            os.chmod(sb.dir, stat.S_IRUSR | stat.S_IXUSR)
            try:
                for argv, data in ((["predict", "in.csv", "--model-file", "m.cml"], None), (["predict", "-", "--model-file", "m.cml"], b"a,b,c\n1,2,3\n"), (["describe", "--model-file", "m.cml"], None)):
                    r = sb.run(*argv, stdin=data, binary_path=PREDICT_BIN)
                    self.assertEqual(r.status, 0, r)
            finally:
                os.chmod(sb.dir, stat.S_IRWXU)
            self.assertEqual(sorted((p.name, p.read_bytes()) for p in sb.dir.iterdir()), before)
        finally:
            sb.cleanup()


class FlagTable(unittest.TestCase):
    """T1 for the second binary: one definition of every flag."""

    def test_every_flag_of_ml_predict_is_a_flag_of_ml_with_the_same_kind_role_and_default(self):
        mlf = {f["name"]: f for f in introspect()["flags"]}
        i = json.loads(subprocess.run([PREDICT_BIN, "introspect"], capture_output=True, stdin=subprocess.DEVNULL).stdout)
        for f in i["flags"]:
            g = mlf[f["name"]]
            self.assertEqual({k: f[k] for k in ("kind", "role", "default")}, {k: g[k] for k in ("kind", "role", "default")}, f["name"])
        self.assertEqual({l["name"] for l in i["limits"]} - {f["name"][2:] for f in i["flags"]}, set())

    def test_the_verbs_the_binary_takes_are_the_flags_of_predict_and_describe(self):
        T = spec.tables()
        want = (set(T["allowed"]["predict"]) | set(T["allowed"]["describe"])) - {"log", "log-max-bytes"}
        i = json.loads(subprocess.run([PREDICT_BIN, "introspect"], capture_output=True, stdin=subprocess.DEVNULL).stdout)
        self.assertEqual({f["name"][2:] for f in i["flags"]}, want)

    def test_describe_is_a_verb_of_ml_with_its_own_flags(self):
        T = spec.tables()
        self.assertEqual(sorted(T["allowed"]["describe"]), sorted(["root", "model-file", "limit", "max-bytes", "format"]))
        for flag in (["--proba"], ["--drop", "x"], ["--from", "1"], ["--max-rows", "5"]):
            r = ml("describe", "--model-file", MODELS["linear"][0], *flag)
            self.assertEqual(r.first_rule(), "args.conflict", (flag, r))


if __name__ == "__main__":
    unittest.main()
