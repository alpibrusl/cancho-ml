#!/bin/sh
# The documentation checks, which need the built binary (build/ml: `cancho install && cancho build --bin ml`) and nothing else:
#   1. the program output in README.md and docs/index.html is what the binary prints now (scripts/site.py --check)
#   2. the links of the README, docs/status.md and the pages resolve to files of this repository (scripts/check_links.py)
#   3. the basics of the pages, and the sitemap and robots.txt agree with the list of pages (scripts/seo_check.py)
# Not part of scripts/check.sh; add `sh scripts/check_docs.sh` at its end to make it part of what CI runs.
set -e
cd "$(dirname "$0")/.."
python3 scripts/site.py --check
python3 scripts/check_links.py
python3 scripts/seo_check.py
