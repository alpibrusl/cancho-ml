"""The fixtures of the recommender (docs/wiring-recommend.md gate WR1): every flag it adds and every rule it adds is reached by a case here, with the exit code of the tables. Read by fixtures.py,
which appends them to CASES."""

# six users, five items; user k has the items k, k + 1 and k + 2 (mod 5), so each item is in three histories; times grow along a user's row
REC = "user,item,rating,ts\n" + "".join(
    "%s,%s,%d,%d\n" % (u, "pqrst"[(k + j) % 5], 1 + (k * 3 + j * 2) % 5, 100 + 3 * k + j)
    for k, u in enumerate(["ann", "bob", "cy", "di", "ed", "flo"])
    for j in range(3)
)
FIT_REC = ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--rating", "rating", "--factors", "3", "--iterations", "4", "--output", "r.cml", "rec.csv"]
RBASE = [("write", "rec.csv", REC)]
WITH_REC = RBASE + [("fit", *FIT_REC[1:])]


def _diagonal(sb):
    # 8200 users and 8200 items, a user and an item that share a number: 8200 * 8200 scores are over the 2^26 cells of the table
    n = 8200
    sb.write("diag.csv", "user,item\n" + "".join("u%05d,i%05d\n" % (k, k) for k in range(n)))
    sb.write("shift.csv", "user,item\n" + "".join("u%05d,i%05d\n" % (k, (k + 1) % n) for k in range(n)))


def _users(sb):
    sb.write("big.csv", "user,item\n" + "".join("u%07d,a\n" % k for k in range(1048577)))


def _items(sb):
    sb.write("big.csv", "user,item\n" + "".join("a,i%07d\n" % k for k in range(1048577)))


def _rows(sb):
    sb.write("big.csv", "user,item\n" + "a,b\n" * 33554433)


def cases(ok, bad, case):
    return [
        case("fit-recommend-explicit-every-flag", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--rating", "rating", "--feedback", "explicit", "--factors", "3", "--l2", "0.5", "--iterations", "4", "--seed", "2", "--threads", "2",
                                                  "--output", "r.cml", "rec.csv"],
             ok({"model": "recommend", "task": "recommendation", "rows": 18, "columns": 3, "target": None, "written": True, "path": "r.cml", "summary": {"method": "als", "feedback": "explicit", "users": 6, "factors": 3, "iterations": 4, "holds_log": True}}),
             setup=RBASE),
        case("fit-recommend-implicit", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--rating", "rating", "--feedback", "implicit", "--alpha", "10", "--factors", "2", "--output", "r.cml", "rec.csv"],
             ok({"model": "recommend", "columns": 3, "summary": {"feedback": "implicit", "factors": 2, "iterations": 10}}), setup=RBASE),
        case("fit-recommend-no-rating-is-implicit", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--output", "r.cml", "rec.csv"], ok({"model": "recommend", "columns": 2, "summary": {"feedback": "implicit", "factors": 32}}), setup=RBASE),
        case("fit-recommend-split-user-last", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--rating", "rating", "--time", "ts", "--split-user-last", "1", "--top", "3", "--factors", "2", "--output", "r.cml", "rec.csv"],
             ok({"model": "recommend", "summary": {"split": {"user_last": 1, "train": 12, "test": 6}, "holdout": {"k": 3, "users": 6}}}), setup=RBASE),
        case("fit-recommend-split-default-top", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--rating", "rating", "--time", "ts", "--split-user-last", "1", "--factors", "2", "--output", "r.cml", "rec.csv"],
             ok({"model": "recommend", "summary": {"holdout": {"k": 5, "users": 6}}}), setup=RBASE),
        case("fit-recommend-dry-run", FIT_REC[:-1] + ["--dry-run", "rec.csv"], ok({"model": "recommend", "written": False}, exit=9, dry_run=True, no_file="r.cml"), setup=RBASE),
        case("fit-recommend-text", ["--format", "text"] + FIT_REC, ok(None, stdout_starts="fit: recommend recommendation, 18 rows, wrote r.cml"), setup=RBASE),
        case("describe-recommend", ["describe", "--model-file", "r.cml"], ok({"model": "recommend", "task": "recommendation", "method": "als", "feedback": "explicit", "users": 6, "factors": 3, "columns": {"user": "user", "item": "item", "rating": "rating"}, "holds_log": True}), setup=WITH_REC),
        case("predict-recommend-for", ["predict", "--model-file", "r.cml", "--for", "who", "--top", "2", "who.csv"],
             ok({"model": "recommend", "columns": ["user", "rank", "item", "score"], "row_count": 4, "total_rows": 4}), setup=WITH_REC + [("write", "who.csv", "who\nann\nflo\n")]),
        case("predict-recommend-for-keep-seen", ["predict", "--model-file", "r.cml", "--for", "who", "--top", "5", "--keep-seen", "who.csv"], ok({"row_count": 5}), setup=WITH_REC + [("write", "who.csv", "who\nann\n")]),
        case("predict-recommend-like", ["predict", "--model-file", "r.cml", "--like", "what", "--top", "2", "what.csv"], ok({"columns": ["item", "rank", "neighbour", "score"], "row_count": 2}), setup=WITH_REC + [("write", "what.csv", "what\np\n")]),
        case("predict-recommend-page", ["predict", "--model-file", "r.cml", "--for", "who", "--top", "2", "--limit", "2", "--from", "1", "who.csv"], ok({"row_count": 2, "total_rows": 4, "truncated": True}), setup=WITH_REC + [("write", "who.csv", "who\nann\nbob\n")]),
        case("eval-recommend", ["eval", "--model-file", "r.cml", "--user", "user", "--item", "item", "--top", "3", "test.csv"],
             ok({"model": "recommend", "task": "recommendation", "k": 3, "users": 4, "users_scored": 2, "users_cold": 1, "rows_user_unknown": 1, "rows_item_unknown": 1, "rows_in_history": 0, "test_items": 2}), setup=WITH_REC + [("write", "test.csv", "user,item\nann,s\nbob,p\nnew,p\ncy,zz\n")]),
        case("eval-recommend-text", ["--format", "text", "eval", "--model-file", "r.cml", "--user", "user", "--item", "item", "--top", "3", "test.csv"], ok(None, stdout_starts="eval: recommend, test rows 4"),
             setup=WITH_REC + [("write", "test.csv", "user,item\nann,s\nbob,p\nnew,p\ncy,zz\n")]),
        # ---- refusals
        case("ml-recommend-missing-id", FIT_REC, bad("ml.recommend.missing-id"), setup=[("write", "rec.csv", REC.replace("ann,p", ",p", 1))]),
        case("ml-recommend-id-too-long", FIT_REC, bad("ml.recommend.id-too-long"), setup=[("write", "rec.csv", REC.replace("ann,p", "a" * 256 + ",p", 1))]),
        case("ml-recommend-bad-rating", FIT_REC, bad("ml.recommend.bad-rating"), setup=[("write", "rec.csv", REC.replace(",p,", ",p,high,", 1).replace(",high,", ",high"))]),
        case("ml-recommend-bad-rating-range", FIT_REC, bad("ml.recommend.bad-rating"), setup=[("write", "rec.csv", REC.replace("ann,p,", "ann,p,2000000,", 1).replace("2000000,", "2000000,", 1))]),
        case("ml-recommend-duplicate-pair", FIT_REC, bad("ml.recommend.duplicate-pair"), setup=[("write", "rec.csv", REC + REC.split("\n")[1] + "\n")]),
        case("ml-recommend-unknown-user", ["predict", "--model-file", "r.cml", "--for", "who", "who.csv"], bad("ml.recommend.unknown-user"), setup=WITH_REC + [("write", "who.csv", "who\nann\nnobody\n")]),
        case("ml-recommend-unknown-item", ["predict", "--model-file", "r.cml", "--like", "what", "what.csv"], bad("ml.recommend.unknown-item"), setup=WITH_REC + [("write", "what.csv", "what\nzz\n")]),
        case("ml-limit-users", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--output", "r.cml", "big.csv"], bad("ml.limit.users"), setup=[("raw", _users)]),
        case("ml-limit-items", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--output", "r.cml", "big.csv"], bad("ml.limit.items"), setup=[("raw", _items)]),
        case("ml-limit-interactions", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--max-rows", "40000000", "--max-bytes", "268435456", "--output", "r.cml", "big.csv"], bad("ml.limit.interactions"), setup=[("raw", _rows)]),
        case("ml-limit-eval-work", ["eval", "--model-file", "d.cml", "--user", "user", "--item", "item", "shift.csv"], bad("ml.limit.eval-work"),
             setup=[("raw", _diagonal), ("fit", "--model", "recommend", "--user", "user", "--item", "item", "--factors", "1", "--iterations", "1", "--output", "d.cml", "diag.csv")]),
        # ---- the command line
        case("args-required-recommend-user", ["fit", "--model", "recommend", "--item", "item", "--output", "r.cml", "rec.csv"], bad("args.required-flag"), setup=RBASE),
        case("args-required-recommend-time", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--split-user-last", "1", "--output", "r.cml", "rec.csv"], bad("args.required-flag"), setup=RBASE),
        case("args-required-predict-for-or-like", ["predict", "--model-file", "r.cml", "who.csv"], bad("args.required-flag"), setup=WITH_REC + [("write", "who.csv", "who\nann\n")]),
        case("args-conflict-recommend-target", FIT_REC[:-1] + ["--target", "rating", "rec.csv"], bad("args.conflict"), setup=RBASE),
        case("args-conflict-recommend-step", FIT_REC[:-1] + ["--step", "scale column=* kind=standard", "rec.csv"], bad("args.conflict"), setup=RBASE),
        case("args-conflict-recommend-time-alone", FIT_REC[:-1] + ["--time", "ts", "rec.csv"], bad("args.conflict"), setup=RBASE),
        case("args-conflict-recommend-explicit-without-rating", ["fit", "--model", "recommend", "--user", "user", "--item", "item", "--feedback", "explicit", "--output", "r.cml", "rec.csv"], bad("args.conflict"), setup=RBASE),
        case("args-conflict-recommend-same-column-twice", ["fit", "--model", "recommend", "--user", "user", "--item", "user", "--output", "r.cml", "rec.csv"], bad("args.conflict"), setup=RBASE),
        case("args-conflict-recommend-top-without-split", FIT_REC[:-1] + ["--top", "3", "rec.csv"], bad("args.conflict"), setup=RBASE),
        case("args-conflict-recommend-alpha-with-explicit", FIT_REC[:-1] + ["--alpha", "5", "rec.csv"], bad("args.conflict"), setup=RBASE),
        case("args-conflict-recommend-flags-with-linear", ["fit", "--model", "linear", "--target", "y", "--user", "user", "--output", "m.cml", "reg.csv"], bad("args.conflict")),
        case("args-conflict-recommend-cv", ["cv", "--model", "recommend", "--user", "user", "--item", "item", "rec.csv"], bad("args.conflict"), setup=RBASE),
        case("args-conflict-predict-for-with-like", ["predict", "--model-file", "r.cml", "--for", "who", "--like", "who", "who.csv"], bad("args.conflict"), setup=WITH_REC + [("write", "who.csv", "who\nann\n")]),
        case("args-conflict-predict-proba-on-recommend", ["predict", "--model-file", "r.cml", "--for", "who", "--proba", "who.csv"], bad("args.conflict"), setup=WITH_REC + [("write", "who.csv", "who\nann\n")]),
        case("args-conflict-recommend-flag-on-a-linear-model", ["predict", "--model-file", "m.cml", "--for", "x", "reg.csv"], bad("args.conflict"), setup=None),
        case("ml-param-out-of-range-recommend-top", ["predict", "--model-file", "r.cml", "--for", "who", "--top", "99", "who.csv"], bad("ml.param.out-of-range"), setup=WITH_REC + [("write", "who.csv", "who\nann\n")]),
        case("ml-param-out-of-range-recommend-l2", FIT_REC[:-1] + ["--l2", "0", "rec.csv"], bad("ml.param.out-of-range"), setup=RBASE),
        case("ml-input-unknown-column-recommend", ["fit", "--model", "recommend", "--user", "who", "--item", "item", "--output", "r.cml", "rec.csv"], bad("ml.input.unknown-column"), setup=RBASE),
    ]
