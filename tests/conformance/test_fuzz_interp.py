"""N1 for `importance` and `dump` -- no input reaches a trap. Every flag of the two verbs with a hostile value, on every kind of model of the sandbox of test_fuzz_args.

A module of its own so that it runs beside test_fuzz_args instead of after it: the two together were the longest task of the suite (docs/ci.md section 5.2)."""

import shutil
import unittest

import test_fuzz_args as fa


class FuzzInterp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = fa.sandbox()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.sb.top, ignore_errors=True)

    go = fa.Fuzz.go
    check = fa.Fuzz.check

    def test_every_flag_of_importance_and_dump_every_mistake(self):
        # docs/interpret.md: the models of the sandbox, with and without a data file, and every flag with a hostile value
        bases = (["importance", "--model-file", "lin.cml"], ["importance", "--model-file", "log.cml", "--data", "lcls.csv"], ["importance", "--model-file", "tree.cml", "--data", "lin.csv"],
                 ["importance", "--model-file", "forest.cml", "--data", "lcls.csv", "--repeats", "2"], ["importance", "--model-file", "km.cml"], ["importance", "--model-file", "m.cml"],
                 ["dump", "--model-file", "tree.cml"], ["dump", "--model-file", "forest.cml", "--tree", "1"], ["dump", "--model-file", "lin.cml"], ["dump", "--model-file", "km.cml"],
                 ["dump", "--model-file", "m.cml"], ["importance", "--model-file", "boost.cml"], ["dump", "--model-file", "boost.cml"])
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


if __name__ == "__main__":
    unittest.main()
