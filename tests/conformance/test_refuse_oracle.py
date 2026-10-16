"""R48 -- the oracle of separation: an exact test of separability of integer points and an independent numpy Newton iteration (`scripts/oracle_refuse.py sep`).

A module of its own, because it needs numpy: the fast job of the CI workflow, which installs none, lists it as not run, and the other tests of issue 48 (test_refuse_fit.py) run there.
`scripts/mutants_refusals.py` runs this module with test_refuse_fit.
"""

import os
import subprocess
import sys
import unittest

import numpy  # noqa: F401  (the oracle imports it when it runs; listing it here makes a missing numpy "not run" and not an error)

from harness import BIN, ROOT


def oracle(*args):
    env = dict(os.environ, ML_BIN=BIN)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "oracle_refuse.py"), *args, "--ml", BIN], capture_output=True, text=True, env=env)


class RefuseOracle(unittest.TestCase):
    def test_oracle_of_separation(self):
        r = oracle("sep", "--n", "90")
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-500:])


if __name__ == "__main__":
    unittest.main()
