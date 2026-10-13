#!/usr/bin/env python3
"""Write docs/bench/mutants_<model>.json, the record that scripts/admit.py check Q10 reads (docs/modules.md section 5).

    python3 scripts/mutants_record.py MODEL SCRIPT=LOG [SCRIPT=LOG ...] [--machine TEXT]

LOG is the output of a run of the mutant script (`python3 scripts/mutants_wire.py > docs/bench/mutants_wire.log`). The record holds, per script, the number of
patterns (read from the log's summary line), how many were killed and each survivor with the reason the script printed. Q10 checks that the pattern count
is still the script's live count (`SCRIPT --check`), so a record made before a pattern was added or removed is stale and fails the admission.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse(log):
    text = pathlib.Path(log).read_text()
    m = re.search(r"(\d+) mutants: (\d+) killed, (\d+) survived, (\d+) invalid", text)
    if not m:
        raise SystemExit("%s: no summary line `N mutants: K killed, S survived, I invalid`" % log)
    total, killed, survived, invalid = map(int, m.groups())
    if invalid:
        raise SystemExit("%s: %d invalid mutants" % (log, invalid))
    surv = [{"id": i, "why": w.strip()} for i, w in re.findall(r"^SURVIVOR (\S+?): (.*)$", text, re.M)]
    if len(surv) != survived:
        raise SystemExit("%s: %d survivors counted, %d listed" % (log, survived, len(surv)))
    return total, killed, surv


def main(argv):
    model = argv[0]
    machine = "Mac arm64 (Darwin 25.2.0), cancho a4572ea"
    if "--machine" in argv:
        machine = argv[argv.index("--machine") + 1]
    suites = []
    for a in argv[1:]:
        if "=" not in a or a.startswith("--"):
            continue
        script, log = a.split("=", 1)
        total, killed, surv = parse(log)
        suites.append({"script": script, "log": log, "patterns": total, "killed": killed, "survivors": surv})
    out = {"model": model, "machine": machine, "suites": suites}
    path = ROOT / "docs" / "bench" / ("mutants_%s.json" % model)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote %s: %s" % (path.relative_to(ROOT), ", ".join("%s %d patterns %d killed %d survived" % (s["script"], s["patterns"], s["killed"], len(s["survivors"])) for s in suites)))


if __name__ == "__main__":
    main(sys.argv[1:])
