#!/bin/sh
# RD1 (docs/reader.md section 5): the cancho-table modules `ml` compiles are the pinned ones and are not copied here.
#   - the clone ($CANCHO_TABLE, default ../cancho-table-main) is at the commit of table.pin and has no change to tracked files
#   - each of the six sources has the pinned SHA-256
#   - no file under lib/, tools/ or tests/ declares one of the six modules, or has the bytes of one of them
# Exit 1 with the reason when any of that is false. LOCAL-DEV: it ends when the modules are a [dependencies.*] (docs/reader.md 1.2).
set -e
cd "$(dirname "$0")/.."
PIN=${TABLE_PIN:-table.pin}
CLONE=${CANCHO_TABLE:-../cancho-table-main}
fail() { echo "check_table_pin: $1" >&2; exit 1; }
[ -f "$PIN" ] || fail "no $PIN"
[ -d "$CLONE/tools/table" ] || fail "no clone of cancho-table at $CLONE (set CANCHO_TABLE)"
rev=$(sed -n 's/^rev //p' "$PIN")
[ ${#rev} -eq 40 ] || fail "$PIN: rev is not a full commit hash"
head=$(git -C "$CLONE" rev-parse HEAD) || fail "$CLONE is not a git checkout"
[ "$head" = "$rev" ] || fail "the clone is at $head, the pin is $rev"
[ -z "$(git -C "$CLONE" status --porcelain --untracked-files=no)" ] || fail "the clone has changes to tracked files"
n=0
while read -r kind hash path; do
  [ "$kind" = sha256 ] || continue
  n=$((n + 1))
  got=$(shasum -a 256 "$CLONE/$path" | cut -d' ' -f1)
  [ "$got" = "$hash" ] || fail "$path has sha256 $got, the pin says $hash"
  mod=$(basename "$path" .cho)
  # a copy: the same bytes, or the same module declaration, anywhere in this repository
  for f in $(find lib tools tests -name '*.cho' 2>/dev/null); do
    if [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" = "$hash" ]; then fail "$f is a copy of $path"; fi
    if grep -q "^module $mod;" "$f"; then fail "$f declares module $mod, which is cancho-table's"; fi
  done
done < "$PIN"
[ "$n" -eq 6 ] || fail "$PIN lists $n sources, not 6"
echo "check_table_pin: ok ($n sources at ${rev%"${rev#???????}"}, none copied)"
