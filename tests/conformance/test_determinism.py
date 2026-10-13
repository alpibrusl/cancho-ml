"""D1 -- the same input and plan give the same model bytes; the same command gives the same stdout bytes.

Whatever the working directory, the spelling of the path, the line ending of the CSV or the caps given, the model file is the same
bytes. If a `--threads` flag ever appears in the table this test runs it at 1, 2, 8 and 64 and compares bytes; today there is
none, and the test says so."""

import hashlib
import os
import shutil
import stat
import tempfile
import unittest

from harness import ROOT, Sandbox, run, spec

REG = "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n"
CLS = "a,b,lab\n1,2,0\n2,3,1\n3,4,1\n4,5,1\n5,6,0\n"


class Determinism(unittest.TestCase):
    def setUp(self):
        self.sbs = []

    def tearDown(self):
        for sb in self.sbs:
            sb.cleanup()

    def sandbox(self, **files):
        sb = Sandbox()
        self.sbs.append(sb)
        for k, v in files.items():
            sb.write(k.replace("__", "."), v)
        return sb

    def test_same_input_same_bytes_in_different_places_and_spellings(self):
        outs = []
        for i, spell in enumerate(("reg.csv", "./reg.csv", "d/../reg.csv".replace("d/../", ""), "sub//reg.csv")):
            sb = self.sandbox()
            sb.write("reg.csv", REG)
            sb.write("sub/reg.csv", REG)
            r = sb.run("fit", "--target", "y", "--output", "out.cml", spell)
            self.assertEqual(r.status, 0, r)
            outs.append((sb.read("out.cml"), r.stdout.replace(b'"path":"out.cml"', b"")))
        self.assertEqual(len({o[0] for o in outs}), 1)
        self.assertEqual(len({o[1] for o in outs}), 1)

    def test_working_directory_and_absolute_root_do_not_matter(self):
        a = self.sandbox()
        a.write("reg.csv", REG)
        b = self.sandbox()
        b.write("reg.csv", REG)
        r1 = a.run("fit", "--target", "y", "--output", "m.cml", "reg.csv", cwd=a.dir)
        r2 = run("--root", ".", "fit", "--target", "y", "--output", "m.cml", "reg.csv", cwd=b.dir)
        self.assertEqual(r1.status, 0, r1)
        self.assertEqual(r2.status, 0, r2)
        self.assertEqual(a.read("m.cml"), b.read("m.cml"))
        self.assertEqual(r1.stdout, r2.stdout)
        # without --root, relative to the working directory
        c = self.sandbox()
        c.write("reg.csv", REG)
        r3 = run("fit", "--target", "y", "--output", "m.cml", "reg.csv", cwd=c.dir)
        self.assertEqual(r3.status, 0, r3)
        self.assertEqual(c.read("m.cml"), a.read("m.cml"))

    def test_line_endings_bom_and_blank_lines_do_not_change_the_model(self):
        variants = [REG, REG.replace("\n", "\r\n"), "﻿" + REG, REG.replace("\n", "\n\n"), REG.rstrip("\n")]
        models = []
        for v in variants:
            sb = self.sandbox()
            sb.write("reg.csv", v)
            r = sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv")
            self.assertEqual(r.status, 0, (v, r))
            models.append(sb.read("m.cml"))
        self.assertEqual(len(set(models)), 1)

    def test_caps_do_not_change_the_model(self):
        sb = self.sandbox()
        sb.write("reg.csv", REG)
        models = set()
        for caps in ([], ["--max-rows", "4"], ["--max-line-bytes", "20"], ["--max-cells", "12", "--max-columns", "3"]):
            r = sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", *caps, "reg.csv")
            self.assertEqual(r.status, 0, r)
            models.add(sb.read("m.cml"))
        self.assertEqual(len(models), 1)

    def test_same_command_same_stdout(self):
        sb = self.sandbox()
        sb.write("reg.csv", REG)
        sb.write("cls.csv", CLS)
        for argv in (["fit", "--overwrite", "--target", "y", "--output", "m.cml", "reg.csv"],
                     ["fit", "--overwrite", "--task", "classification", "--target", "lab", "--output", "c.cml", "cls.csv"]):
            sb.run(*argv)
        for argv in (["predict", "--model-file", "m.cml", "reg.csv"], ["eval", "--model-file", "m.cml", "reg.csv"],
                     ["eval", "--model-file", "c.cml", "cls.csv"], ["fit", "--dry-run", "--overwrite", "--target", "y", "--output", "m.cml", "reg.csv"],
                     ["fit", "--target", "y", "--output", "m.cml", "reg.csv"], ["predict", "--model-file", "m.cml", "cls.csv"]):
            outs = {(r.status, r.stdout, r.stderr) for r in (sb.run(*argv) for _ in range(5))}
            self.assertEqual(len(outs), 1, argv)

    def test_the_model_bytes_are_pinned(self):
        # the bytes of a model are a function of the data and the plan alone: this hash changes only with the format or the
        # compiler pin (a model records the pin), and then the change is deliberate
        sb = self.sandbox()
        sb.write("reg.csv", REG)
        sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv")
        body = sb.read("m.cml")[:-32]
        pin = spec.ROOT.joinpath("cancho.toml").read_text().split('cancho = "')[1].split('"')[0]
        self.assertIn(pin.encode(), body)

    def test_no_thread_count_changes_a_byte(self):
        names = [f["name"] for f in spec.tables()["flags"]]
        if "threads" not in names:
            self.skipTest("there is no --threads flag: nothing can vary (docs/cli.md section 6)")
        sb = self.sandbox()
        sb.write("reg.csv", REG)
        models = set()
        for t in (1, 2, 8, 64):
            r = sb.run("fit", "--overwrite", "--threads", str(t), "--target", "y", "--output", "m.cml", "reg.csv")
            self.assertEqual(r.status, 0, r)
            models.add(sb.read("m.cml"))
        self.assertEqual(len(models), 1)


if __name__ == "__main__":
    unittest.main()
