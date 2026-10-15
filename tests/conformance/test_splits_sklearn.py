"""SP S1 -- scikit-learn on the folds of --emit-folds (docs/splits.md section 9).

A module of its own because its import needs numpy and scikit-learn: scripts/run_conformance.py --no-oracles lists such a module as NOT RUN
(the fast job of the workflow installs neither), and the full gate runs it. The gate itself is `gate_sklearn` of scripts/oracle_split.py."""

import os
import sys
import tempfile
import unittest

import numpy  # noqa: F401  (the import is the declaration of what this module needs)
import sklearn  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import oracle_split as O  # noqa: E402

QUICK = os.environ.get("SPLIT_QUICK") == "1"


class Sklearn(unittest.TestCase):
    @unittest.skipIf(QUICK, "scikit-learn comparison runs in the full gate")
    def test_s1_scikit_learn_on_the_same_folds(self):
        O.FAILS.clear()
        with tempfile.TemporaryDirectory() as tmp:
            O.WORK[0] = tmp
            O.gate_sklearn(tmp, QUICK)
        self.assertEqual(O.FAILS, [], "\n".join(O.FAILS[:12]))
