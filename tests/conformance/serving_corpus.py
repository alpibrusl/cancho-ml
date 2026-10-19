"""The corpus of the serving gates (docs/serving.md section 1): stored models of every kind, 1 to 8 (k-means is 6, boost 8), with categorical steps, missing cells, unseen categories and hash
steps, and the inputs to predict from. Everything is written in a sandbox by `ml fit`, from data a seeded generator makes, so a gate can rebuild it.

    build(sb)  ->  MODELS (id -> file name in the sandbox), INPUTS (id -> (file name, model ids it suits))
"""

import random

# id -> (the `fit` command line without the output and the file, the csv it reads, the kind of the model section)
NUM = "num.csv"
CAT = "cat.csv"
WIDE = "wide.csv"

FITS = {
    # kinds 1 to 3: the constant models
    "const_reg_dec": (["--target", "y", "--drop", "lab"], NUM, 1),
    "const_reg_float": (["--target", "y", "--float", "y", "--drop", "lab"], NUM, 3),
    "const_cls": (["--task", "classification", "--target", "lab", "--drop", "y"], NUM, 2),
    # kind 4: linear, kind 5: logistic, kind 7: tree and forest, each through the pipeline
    "linear": (["--model", "linear", "--l2", "0.5", "--step", "impute column=* strategy=mean;scale column=a,b kind=standard", "--target", "y", "--drop", "lab"], NUM, 4),
    "logistic": (["--model", "logistic", "--iterations", "30", "--l2", "0.5", "--step", "impute column=* strategy=mean;scale column=* kind=minmax", "--target", "lab", "--drop", "y"], NUM, 5),
    "tree_reg": (["--model", "tree", "--task", "regression", "--target", "y", "--drop", "lab"], NUM, 7),
    "tree_cls": (["--model", "tree", "--task", "classification", "--target", "lab", "--drop", "y"], NUM, 7),
    "forest_reg": (["--model", "forest", "--task", "regression", "--trees", "5", "--seed", "3", "--target", "y", "--drop", "lab"], NUM, 7),
    "forest_cls": (["--model", "forest", "--task", "classification", "--trees", "5", "--seed", "3", "--target", "lab", "--drop", "y"], NUM, 7),
    # kind 8: boosted trees (docs/models-boost.md): numeric with a row sample, a classification, and a categorical plan with a string table
    "boost_reg": (["--model", "boost", "--task", "regression", "--rounds", "8", "--min-leaf", "3", "--subsample", "0.7", "--seed", "3", "--target", "y", "--drop", "lab"], NUM, 8),
    "boost_cls": (["--model", "boost", "--task", "classification", "--rounds", "8", "--min-leaf", "3", "--seed", "3", "--target", "lab", "--drop", "y"], NUM, 8),
    "cat_boost_hash": (["--model", "boost", "--task", "regression", "--rounds", "6", "--min-leaf", "2", "--seed", "5", "--cat", "c0,c1", "--max-categories", "9", "--step",
                        "hash column=c0 buckets=8;encode column=c1 kind=onehot unknown=refuse", "--target", "y", "--drop", "lab"], CAT, 8),
    # kind 6: k-means, which has no target (docs/wiring-kmeans.md): numeric with steps, numeric with the other initialisation and no step, categorical with a string table
    "kmeans": (["--model", "kmeans", "--k", "3", "--seed", "2", "--step", "impute column=* strategy=mean;scale column=* kind=standard", "--drop", "y,lab"], NUM, 6),
    "kmeans_random": (["--model", "kmeans", "--k", "4", "--init", "random", "--iterations", "7", "--seed", "9", "--drop", "y,lab"], NUM, 6),
    "cat_kmeans_onehot": (["--model", "kmeans", "--k", "3", "--seed", "4", "--cat", "c0,c1", "--na-values", "NA", "--step",
                           "impute column=n strategy=mean;encode column=c0,c1 kind=onehot missing=zero", "--drop", "y,lab"], CAT, 6),
    # categorical steps: a string table (six sections)
    "cat_linear_onehot": (["--model", "linear", "--l2", "1", "--cat", "c0,c1", "--na-values", "NA,?", "--step",
                           "impute column=n strategy=mean;encode column=c0,c1 kind=onehot missing=zero", "--target", "y", "--drop", "lab"], CAT, 4),
    "cat_logistic_label": (["--model", "logistic", "--iterations", "30", "--l2", "1", "--cat", "c0,c1", "--na-values", "NA", "--step",
                            "impute column=n strategy=median;encode column=c0 kind=label missing=zero;encode column=c1 kind=onehot missing=zero", "--target", "lab", "--drop", "y"], CAT, 5),
    "cat_tree_freq": (["--model", "tree", "--task", "classification", "--cat", "c0,c1", "--step", "freq column=c0,c1", "--target", "lab", "--drop", "y"], CAT, 7),
    "cat_forest_hash": (["--model", "forest", "--task", "regression", "--trees", "4", "--seed", "5", "--cat", "c0,c1", "--max-categories", "9", "--step",
                         "hash column=c0 buckets=8;encode column=c1 kind=onehot unknown=refuse", "--target", "y", "--drop", "lab"], CAT, 7),
    "cat_tree_ignore": (["--model", "tree", "--task", "regression", "--cat", "c0", "--step", "encode column=c0 kind=onehot unknown=ignore", "--target", "y", "--drop", "lab,c1"], CAT, 7),
    # no numeric feature at all, and a vocabulary larger than a small --limit
    "cat_only_forest": (["--model", "forest", "--task", "classification", "--trees", "3", "--cat", "c0,c1", "--step", "encode column=c0,c1 kind=onehot", "--target", "lab", "--drop", "y,n"], CAT, 7),
    "cat_wide_tree": (["--model", "tree", "--task", "classification", "--cat", "w0,w1,w2", "--max-categories", "64", "--na-values", "-,none", "--step", "freq column=*", "--target", "lab", "--drop", "y"], WIDE, 7),
}

C0 = ["red", "green", "blue", "yellow", "a,b", "q\"uote", "multi\nline", "café"]
C1 = ["x", "y", "z", "w", "NA", "?"]


def num_csv(rng, n=60):
    rows = ["a,b,c,y,lab"]
    for i in range(n):
        a = rng.randint(-20, 40)
        b = rng.randint(-5000, 5000) / 100.0
        c = rng.uniform(-3, 3)
        y = round(1.5 * a - 0.7 * b + 4 * c + rng.uniform(-2, 2), 1)
        lab = 1 if a + b > 10 else 0
        rows.append("%d,%.2f,%r,%.1f,%d" % (a, b, c, y, lab))
    return "\n".join(rows) + "\n"


def wide_csv(rng, n=90):
    rows = ["w0,w1,w2,y,lab"]
    for i in range(n):
        w0 = "s%02d" % rng.randint(0, 29)
        w1 = "t%d" % rng.randint(0, 11)
        w2 = rng.choice(["p", "q", "-", "none", "r"])
        y = rng.randint(0, 20)
        rows.append("%s,%s,%s,%d,%d" % (w0, w1, w2, y, 1 if y > 10 else 0))
    return "\n".join(rows) + "\n"


def quoted(s):
    if any(ch in s for ch in ',"\n\r'):
        return '"' + s.replace('"', '""') + '"'
    return s


def cat_csv(rng, n=80):
    rows = ["c0,c1,n,y,lab"]
    for i in range(n):
        c0 = rng.choice(C0[:6])
        c1 = rng.choice(C1[:4] + ["NA"] * 1)
        nn = rng.randint(0, 50)
        y = round(nn * 0.3 + (3 if c0 == "red" else 0) + (2 if c1 == "x" else 0) + rng.uniform(-1, 1), 1)
        lab = 1 if y > 9 else 0
        n_cell = "" if rng.random() < 0.08 else str(nn)
        rows.append("%s,%s,%s,%.1f,%d" % (quoted(c0), quoted(c1), n_cell, y, lab))
    return "\n".join(rows) + "\n"


def inputs(rng):
    """id -> (file name, text or bytes, the models it suits: 'num', 'cat', 'cat0')."""
    out = {}
    # numeric inputs: with and without the target columns (the dropped lab, y are not columns of the models; the constant ones with lab or y keep the other)
    rows = ["a,b,c"]
    for i in range(12):
        rows.append("%d,%.2f,%r" % (rng.randint(-30, 50), rng.randint(-6000, 6000) / 100.0, rng.uniform(-4, 4)))
    out["num_plain"] = ("num_plain.csv", "\n".join(rows) + "\n", "num")
    out["num_crlf_bom"] = ("num_crlf_bom.csv", b"\xef\xbb\xbf" + "\r\n".join(rows).encode() + b"\r\n", "num")
    out["num_no_final_newline"] = ("num_nofinal.csv", "\n".join(rows), "num")
    out["num_blank_lines"] = ("num_blank.csv", "\n\n" + "\n\n".join(rows[:5]) + "\n\n", "num")
    out["num_with_y"] = ("num_with_y.csv", "a,b,c,y\n1,2.50,0.5,3.0\n4,-1.25,1.5,not a number\n", "num")
    out["num_reordered"] = ("num_reordered.csv", "c,a,b\n0.5,1,2.50\n1.5,4,-1.25\n", "num")
    out["num_missing"] = ("num_missing.csv", "a,b,c\n1,,0.5\n,2.25,\n3,4.50,1.5\n", "num")
    out["num_big"] = ("num_big.csv", "a,b,c\n" + "".join("%d,%d.%02d,%d.5\n" % (i % 37, i % 91, i % 100, i % 7) for i in range(400)), "num")
    # categorical inputs: seen, unseen, missing, markers, quoted, reordered, with the target
    out["cat_seen"] = ("cat_seen.csv", "c0,c1,n\nred,x,10\ngreen,y,20\nblue,z,5\n", "cat")
    out["cat_unseen"] = ("cat_unseen.csv", "c0,c1,n\nviolet,x,10\nred,never,20\n\"a,b\",w,3\n", "cat")
    out["cat_missing"] = ("cat_missing.csv", "c0,c1,n\n,x,10\nred,,\nNA,NA,7\n?,y,8\n", "cat")
    out["cat_reordered"] = ("cat_reordered.csv", "n,c1,c0,y\n10,x,red,9.5\n5,y,green,2\n", "cat")
    out["cat_unicode_quoted"] = ("cat_quoted.csv", "c0,c1,n\n\"café\",x,1\n\"multi\nline\",y,2\n\"q\"\"uote\",z,3\n", "cat")
    out["cat_bom_crlf"] = ("cat_bom.csv", b"\xef\xbb\xbfc0,c1,n\r\nred,x,10\r\nblue,y,20\r\n", "cat")
    out["wide_seen"] = ("wide_seen.csv", "w0,w1,w2,y\ns01,t3,p,5\ns99,t0,none,1\n,-,q,2\n", "wide")
    out["catonly_seen"] = ("catonly_seen.csv", "c0,c1\nred,x\nviolet,y\n,\n", "catonly")
    out["cat0_seen"] = ("cat0_seen.csv", "c0,n\nred,10\nblue,2\nnew,3\n", "cat0")
    return out


# damaged inputs: id -> (file name, bytes, what the rule is expected to be, or None when only the equality matters)
def bad_inputs():
    return {
        "ragged": ("bad_ragged.csv", b"a,b,c\n1,2,3\n4,5\n"),
        "bad_quote": ("bad_quote.csv", b'a,b,c\n1,"2"x,3\n'),
        "unterminated": ("bad_unterminated.csv", b'a,b,c\n1,"2,3\n'),
        "empty": ("bad_empty.csv", b""),
        "header_only": ("bad_header.csv", b"a,b,c\n"),
        "not_a_number": ("bad_nan.csv", b"a,b,c\n1,two,3\n"),
        "missing_column": ("bad_missing.csv", b"a,b\n1,2\n"),
        "extra_column": ("bad_extra.csv", b"a,b,c,zz\n1,2,3,4\n"),
        "mistyped": ("bad_typed.csv", b"a,b,c\n1.5,2,3\n"),
        "duplicate": ("bad_dup.csv", b"a,a,c\n1,2,3\n"),
        "invalid_utf8": ("bad_utf8.csv", b"c0,c1,n\n\xff,x,1\n"),
        "nul_byte": ("bad_nul.csv", b"a,b,c\n1,2,\x003\n"),
        "long_line": ("bad_long.csv", b"a,b,c\n" + b"1," * 2000 + b"2,3\n"),
    }


def build(sb, seed=17):
    """Write the training data, fit every model, write the inputs. Returns (models, inputs): model id -> (file name, kind, csv family),
    input id -> (file name, family)."""
    rng = random.Random(seed)
    sb.write(NUM, num_csv(rng))
    sb.write(CAT, cat_csv(rng))
    sb.write(WIDE, wide_csv(rng))
    models = {}
    for mid, (args, csv, kind) in FITS.items():
        r = sb.run("fit", *args, "--output", mid + ".cml", csv)
        assert r.status == 0, (mid, r)
        family = {NUM: "num", WIDE: "wide"}.get(csv) or {"cat_tree_ignore": "cat0", "cat_only_forest": "catonly"}.get(mid, "cat")
        models[mid] = (mid + ".cml", kind, family)
    ins = {}
    for iid, (name, data, family) in inputs(rng).items():
        sb.write(name, data)
        ins[iid] = (name, family)
    for iid, (name, data) in bad_inputs().items():
        sb.write(name, data)
        ins["bad_" + iid] = (name, "bad")
    return models, ins
