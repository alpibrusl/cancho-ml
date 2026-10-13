#!/bin/sh
# The documentation checks, which need the built binary (build/ml: `cancho install && cancho build --bin ml`) and nothing else:
#   1. the program output in README.md and the project pages is what the binary prints now (scripts/site.py --check)
#   2. the links of the README, docs/status.md and the pages resolve to files of this repository (scripts/check_links.py)
#   3. the basics of the pages, and the sitemap and robots.txt agree with the list of pages (scripts/seo_check.py)
#   4. the README, the pages and docs/status.md describe the current state and hold no phrase of history (scripts/check_prose.py), and the
#      lint is shown able to fail (its mutants)
# scripts/check.sh runs this script at its end.
set -e
cd "$(dirname "$0")/.."
python3 scripts/site.py --check
python3 scripts/check_links.py
python3 scripts/seo_check.py
python3 scripts/check_prose.py
python3 scripts/check_prose.py --mutants
