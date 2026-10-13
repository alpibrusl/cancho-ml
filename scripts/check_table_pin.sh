#!/bin/sh
# RD1 (docs/reader.md section 5): cancho-table's numeric modules are a pinned package dependency and are not copied here.
#   - cancho.toml has a [dependencies.X] for each of pow5, dec, flt, facc and reader, with git = cancho-table, a path of
#     .cancho-vcs/X, and one rev, 40 hex digits, the same in all five
#   - cancho.toml has no other [dependencies.X] whose git is cancho-table (a sixth, such as query, is refused: not in the documented list)
#   - that rev is the one docs/reader.md documents (the line "cancho-table rev: <hash>"; bumping the pin edits both)
#   - the RD1 row of the DOC names the same five modules and no other
#   - no `sources` entry of cancho.toml reaches outside this tree (no ../cancho-table-main)
#   - no .cho file under lib/, tools/ or tests/ declares one of the five modules
#   - if build/deps exists, the installed sources declare the five modules (a rev without the stores installs none)
#   - CHECK_REMOTE=1: the rev has .cancho-vcs/X for the five on GitHub (needs gh and the network; skipped, not failed, when offline)
# TOML and DOC override the files read (the mutants). Exit 1 with the reason when any of that is false.
set -e
cd "$(dirname "$0")/.."
TOML=${TOML:-cancho.toml}
DOC=${DOC:-docs/reader.md}
MODS="pow5 dec flt facc reader"
URL=https://github.com/alpibrusl/cancho-table
fail() { echo "check_table_pin: $1" >&2; exit 1; }
[ -f "$TOML" ] || fail "no $TOML"
want=$(sed -n 's/^cancho-table rev: `\([0-9a-f]*\)`.*/\1/p' "$DOC" | head -1)
[ ${#want} -eq 40 ] || fail "$DOC has no line 'cancho-table rev: \`<40 hex digits>\`'"
for m in $MODS; do
  blk=$(awk -v h="[dependencies.$m]" '$0==h{f=1;next} /^\[/{f=0} f' "$TOML")
  [ -n "$blk" ] || fail "$TOML has no [dependencies.$m]"
  git=$(echo "$blk" | sed -n 's/^git *= *"\(.*\)".*/\1/p')
  rev=$(echo "$blk" | sed -n 's/^rev *= *"\(.*\)".*/\1/p')
  path=$(echo "$blk" | sed -n 's/^path *= *"\(.*\)".*/\1/p')
  [ "$git" = "$URL" ] || fail "[dependencies.$m]: git is '$git', not $URL"
  [ "$path" = ".cancho-vcs/$m" ] || fail "[dependencies.$m]: path is '$path', not .cancho-vcs/$m"
  case "$rev" in *[!0-9a-f]*|"") fail "[dependencies.$m]: rev is not a full commit hash" ;; esac
  [ ${#rev} -eq 40 ] || fail "[dependencies.$m]: rev is not a full commit hash (${#rev} digits)"
  [ "$rev" = "$want" ] || fail "[dependencies.$m]: rev $rev is not the documented $want ($DOC)"
done
# every dependency on cancho-table is one of the five: a store that ml does not import is not pinned
for m in $(sed -n 's/^\[dependencies\.\(.*\)\]$/\1/p' "$TOML"); do
  blk=$(awk -v h="[dependencies.$m]" '$0==h{f=1;next} /^\[/{f=0} f' "$TOML")
  if echo "$blk" | grep -q "^git *= *\"$URL\""; then
    case " $MODS " in *" $m "*) ;; *) fail "[dependencies.$m] is a cancho-table store that is not in the documented list ($MODS)" ;; esac
  fi
done
row=$(grep '^| RD1 ' "$DOC" | head -1)
[ -n "$row" ] || fail "$DOC has no RD1 row"
for m in $MODS; do echo "$row" | grep -qw "$m" || fail "the RD1 row of $DOC does not name $m"; done
if echo "$row" | grep -qw query; then fail "the RD1 row of $DOC names query, which is not in the documented list ($MODS)"; fi
if grep -q '^sources *=.*\.\./' "$TOML"; then fail "a sources entry of $TOML reaches outside this tree"; fi
for f in $(find lib tools tests -name '*.cho' 2>/dev/null); do
  for m in $MODS; do
    if grep -q "^module $m;" "$f"; then fail "$f declares module $m, which is cancho-table's"; fi
  done
done
if [ -d build/deps ] && ls build/deps/*.cho >/dev/null 2>&1; then
  for m in $MODS; do
    grep -qs "^module $m;" build/deps/*.cho || fail "build/deps has no installed module $m (run cancho install)"
  done
fi
if [ -n "$CHECK_REMOTE" ]; then
  if command -v gh >/dev/null 2>&1 && gh api repos/alpibrusl/cancho-table >/dev/null 2>&1; then
    for m in $MODS; do
      gh api "repos/alpibrusl/cancho-table/contents/.cancho-vcs/$m?ref=$want" >/dev/null 2>&1 || fail "$want has no .cancho-vcs/$m on GitHub"
    done
    echo "check_table_pin: remote ok"
  else
    echo "check_table_pin: remote check skipped (offline or no gh)"
  fi
fi
echo "check_table_pin: ok (5 dependencies at ${want%"${want#???????}"}, none copied)"
