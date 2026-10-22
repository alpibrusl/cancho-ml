"""RK12 -- `ml-serve` with a ranking model (docs/ranking.md section 7): the records of a batch are one group, the answer of `POST /predict/batch` has the columns `prediction` and `rank`, and
its cells are the text `ml predict --rank --group COL` writes for the same rows with one group value; `POST /predict` answers the score alone."""

import json
import unittest

from harness import Sandbox
from serve_lib import Server, loads_tokens

JSON = {"Content-Type": "application/json"}
FIT = ["--model", "boost", "--target", "rel", "--group", "qid", "--rounds", "8", "--max-depth", "3", "--min-leaf", "2"]


def rank_csv():
    import random
    r = random.Random(4)
    out = ["qid,a,b,rel"]
    for g in range(20):
        for _ in range(r.randint(3, 9)):
            a, b = r.uniform(-1, 1), r.uniform(-1, 1)
            out.append("q%d,%.4f,%.4f,%d" % (g, a, b, max(0, min(3, int(2 * (a - b / 2) + 1.5)))))
    return "\n".join(out) + "\n"


class ServeRank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("rk.csv", rank_csv())
        for task, out in (("ranking", "r.cml"), ("lambdarank", "l.cml")):
            r = cls.sb.run("fit", *FIT, "--task", task, "--output", out, "rk.csv")
            assert r.status == 0, r
        r = cls.sb.run("fit", "--model", "boost", "--target", "rel", "--drop", "qid", "--rounds", "4", "--max-depth", "2", "--min-leaf", "2", "--output", "g.cml", "rk.csv")
        assert r.status == 0, r
        cls.servers = {m: Server(cls.sb.dir, m).start() for m in ("r.cml", "l.cml", "g.cml")}

    @classmethod
    def tearDownClass(cls):
        for s in cls.servers.values():
            s.stop()
        cls.sb.cleanup()

    def rows_of(self, group):
        lines = self.sb.read("rk.csv").decode().splitlines()[1:]
        return [l.split(",") for l in lines if l.split(",")[0] == group]

    def test_the_batch_is_one_group_and_equals_predict_rank(self):
        for model, srv in list(self.servers.items())[:2]:
            for group in ("q0", "q7", "q19"):
                rows = self.rows_of(group)
                body = json.dumps({"rows": [{"a": float(x[1]), "b": float(x[2])} for x in rows]}).encode()
                resp = srv.request("POST", "/predict/batch", body=body, headers=JSON)
                self.assertEqual(resp.status, 200, resp)
                got = loads_tokens(resp.text())
                self.assertEqual(got["columns"], ["prediction", "rank"], (model, group))
                self.sb.write("g.csv", "qid,a,b\n" + "\n".join("%s,%s,%s" % (x[0], x[1], x[2]) for x in rows) + "\n")
                cli = self.sb.run("predict", "--model-file", model, "--rank", "--group", "qid", "--limit", "100000", "g.csv").data()
                self.assertEqual(cli["columns"], ["prediction", "rank"])
                self.assertEqual([[str(c) for c in row] for row in got["rows"]], [[str(c) for c in row] for row in cli["rows"]], (model, group))

    def test_the_ranks_are_a_permutation_with_ties_by_order(self):
        srv = self.servers["r.cml"]
        same = {"a": 0.25, "b": 0.5}
        resp = srv.request("POST", "/predict/batch", body=json.dumps({"rows": [same, same, same]}).encode(), headers=JSON)
        self.assertEqual(resp.status, 200, resp)
        got = loads_tokens(resp.text())
        self.assertEqual([row[1] for row in got["rows"]], ["1", "2", "3"])

    def test_a_model_that_does_not_rank_has_no_rank_column(self):
        rows = [{"a": 0.1, "b": 0.2}, {"a": 0.3, "b": 0.4}]
        resp = self.servers["g.cml"].request("POST", "/predict/batch", body=json.dumps({"rows": rows}).encode(), headers=JSON)
        self.assertEqual(resp.status, 200, resp)
        self.assertEqual(loads_tokens(resp.text())["columns"], ["prediction"])

    def test_one_record_answers_the_score_alone(self):
        for srv in self.servers.values():
            resp = srv.request("POST", "/predict", body=json.dumps({"a": 0.1, "b": 0.2}).encode(), headers=JSON)
            self.assertEqual(resp.status, 200, resp)
            self.assertEqual(loads_tokens(resp.text())["columns"], ["prediction"])


if __name__ == "__main__":
    unittest.main()
