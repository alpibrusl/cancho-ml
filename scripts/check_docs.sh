#!/bin/sh
# The documentation checks, which need the built binary (build/ml: `cancho install && cancho build --bin ml`) and nothing else:
#   1. the program output in README.md and the project pages is what the binary prints now (scripts/site.py --check)
#   2. the links of the README, docs/status.md and the pages resolve to files of this repository (scripts/check_links.py)
#   3. the basics of the pages, and the sitemap and robots.txt agree with the list of pages (scripts/seo_check.py)
#   4. the README, the pages and docs/status.md describe the current state and hold no phrase of history (scripts/check_prose.py), and the
#      lint is shown able to fail (its mutants)
#   5. every docs/ path that README.md and docs/*.md name is a file of this repository, and none names a local clone (scripts/check_doc_refs.py),
#      and the check is shown able to fail (its mutants)
#   6. the landing page: its figures equal the sources, the header is the same on the five pages, the three replays (the demo of scripts/demo.sh as it runs now, a refusal with its own repair), the experiments and the server in words, the runners of the workflow, the build
#      commands are the README's, the limits are stated, the size budget holds (scripts/check_landing.py); the contrast of every colour pair of site.css
#      (scripts/check_contrast.py) and its mutants; every mutant of scripts/mutants_site.py applies (the run itself, about a minute, is on request)
# scripts/check.sh runs this script at its end.
set -e
cd "$(dirname "$0")/.."
python3 scripts/site.py --check
python3 scripts/check_links.py
python3 scripts/seo_check.py
python3 scripts/check_prose.py
python3 scripts/check_prose.py --mutants
python3 scripts/check_doc_refs.py
python3 scripts/check_doc_refs.py --mutants
python3 scripts/check_landing.py --demo
python3 scripts/check_contrast.py
python3 scripts/check_contrast.py --mutants
python3 scripts/mutants_site.py --check
