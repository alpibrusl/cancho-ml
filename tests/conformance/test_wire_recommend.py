"""WR2, WR8, WR10 -- the log is numbered by the bytewise rank of the distinct ids, so a shuffled log gives the same file byte for byte and `007`, `7` and `1e3` are three ids; the serial stage
accepts --threads and changes no byte; the flags that do not apply are args.conflict."""

import random
import unittest

from harness import Sandbox, validate

LOG = "user,item,rating\n" + "".join("%s,%s,%d\n" % (u, i, 1 + (k * 5 + j * 3) % 5) for k, u in enumerate(["007", "7", "1e3", "ann", "Bob", "é"]) for j, i in enumerate(["b", "a", "10", "9"][: 2 + k % 3]))


class RecommendLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def fit(self, name, text, *extra):
        self.sb.write(name + ".csv", text)
        r = self.sb.run("fit", "--model", "recommend", "--user", "user", "--item", "item", "--rating", "rating", "--factors", "3", "--iterations", "3", "--output", name + ".cml", *extra, name + ".csv")
        self.assertEqual(r.status, 0, r)
        return self.sb.read(name + ".cml")

    def test_a_shuffled_log_is_the_same_file(self):
        lines = LOG.split("\n")[1:-1]
        base = self.fit("a", LOG)
        for seed in range(5):
            rows = list(lines)
            random.Random(seed).shuffle(rows)
            self.assertEqual(self.fit("s%d" % seed, "user,item,rating\n" + "\n".join(rows) + "\n"), base, seed)

    def test_ids_that_look_like_numbers_are_ids(self):
        self.fit("a", LOG)
        r = self.sb.run("describe", "--model-file", "a.cml")
        self.assertEqual(r.data()["users"], 6)
        self.assertEqual(r.data()["items"], 4)

    def test_threads_change_no_byte(self):
        base = self.fit("t1", LOG, "--threads", "1")
        for t in ("2", "4", "8", "16"):
            self.assertEqual(self.fit("t" + t, LOG, "--threads", t), base, t)

    def test_the_seed_changes_the_file(self):
        self.assertNotEqual(self.fit("s0", LOG, "--seed", "0"), self.fit("s1", LOG, "--seed", "1"))

    def test_the_header_may_have_other_columns_and_any_order(self):
        wide = "x,rating,y,item,user\n" + "".join("1,%s,2,%s,%s\n" % (r, i, u) for u, i, r in (row.split(",") for row in LOG.split("\n")[1:-1]))
        self.assertEqual(self.fit("w", wide), self.fit("a", LOG))

    def test_a_missing_rating_column_is_refused_with_its_name(self):
        self.sb.write("n.csv", LOG)
        r = self.sb.run("fit", "--model", "recommend", "--user", "user", "--item", "item", "--rating", "stars", "--output", "n.cml", "n.csv")
        self.assertEqual(r.first_rule(), "ml.input.unknown-column", r)
        self.assertEqual(r.error()["detail"]["flag"], "rating")
        self.assertEqual(validate(r), [])

    def test_every_flag_that_does_not_apply_is_a_conflict(self):
        self.sb.write("n.csv", LOG)
        for flag, value in (("--target", "rating"), ("--drop", "x"), ("--cat", "user"), ("--float", "rating"), ("--positive", "1"), ("--stratify", None), ("--split-by", "time:ts"), ("--split-group", "user"),
                            ("--na-values", "NA"), ("--step", "scale column=* kind=standard"), ("--max-columns", "10"), ("--max-cells", "100"), ("--max-line-bytes", "100"), ("--track", "runs"), ("--weight", "rating")):
            argv = ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--rating", "rating", "--output", "n.cml", flag] + ([value] if value else []) + ["n.csv"]
            r = self.sb.run(*argv)
            self.assertEqual(r.first_rule(), "args.conflict", (flag, r))
            self.assertEqual(validate(r), [], flag)

    def test_the_flags_of_a_recommender_with_another_model_are_a_conflict(self):
        self.sb.write("n.csv", "a,y\n1,2\n2,3\n3,5\n")
        for flag, value in (("--user", "a"), ("--item", "a"), ("--rating", "a"), ("--time", "a"), ("--split-user-last", "1"), ("--top", "3")):
            r = self.sb.run("fit", "--model", "linear", "--target", "y", "--output", "n.cml", flag, value, "n.csv")
            self.assertEqual(r.first_rule(), "args.conflict", (flag, r))

    def test_cv_is_a_conflict_that_names_the_split(self):
        self.sb.write("n.csv", LOG)
        r = self.sb.run("cv", "--model", "recommend", "--user", "user", "--item", "item", "n.csv")
        self.assertEqual(r.first_rule(), "args.conflict", r)


if __name__ == "__main__":
    unittest.main()
