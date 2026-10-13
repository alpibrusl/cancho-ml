"""What the worked examples share (docs/examples.md): the list of examples with the `ml` flags of each run, the
tolerances of the gate (section 3 of that document), and the comparison itself. Standard library only.

scripts/examples_data.py (needs scikit-learn) reads EXAMPLES to write the CSVs and docs/examples/reference.json, using
the same --l2 as the `ml` command, so that the two programs cannot be given different parameters. scripts/site.py reads
EXAMPLES to run the `ml` commands and compare them with reference.json.
"""

# One entry per example. A run is one `ml fit` on the train file; "scale" is the --step of the fit, or None.
#   ref=True       a scikit-learn reference exists for the run (its parameters are derived from l2, see sklearn_params)
#   same_as        the run whose coefficients this run is compared with (more iterations: nothing may change)
#   shown_only     the run is shown (its output is the point) and is not compared with scikit-learn
#   eval_refusal   the fit works and `ml eval` on the test file is refused (exit status 8)
#   float          --float NAMES of the fit
#   refusal        the run is expected to be refused (exit status 8)
STD = "scale column=* kind=standard"

EXAMPLES = [
    dict(
        key="breast_cancer", task="classification", target="benign",
        title="Breast Cancer Wisconsin (Diagnostic)", short="Breast cancer",
        runs=[
            # the obvious command: no penalty. The classes are separable, so there is no optimum; what comes out depends on --iterations
            dict(id="naive", model="logistic", l2="0", iterations=25, scale=STD, shown_only=True, eval_train=True),
            dict(id="refused", model="logistic", l2="0", iterations=50, scale=STD, shown_only=True, refusal=True),
            dict(id="fit", model="logistic", l2="0.01", iterations=25, scale=STD, ref=True),
            dict(id="more", model="logistic", l2="0.01", iterations=100, scale=STD, same_as="fit"),
        ],
    ),
    dict(
        key="diabetes", task="regression", target="progression",
        title="Diabetes progression", short="Diabetes",
        runs=[
            dict(id="ols", model="linear", l2="0", iterations=None, scale=None, ref=True),
            dict(id="ridge", model="linear", l2="0.05", iterations=None, scale=None, ref=True),
        ],
    ),
    dict(
        key="iris", task="classification", target="virginica",
        title="Iris, versicolor against virginica", short="Iris",
        runs=[
            dict(id="fit", model="logistic", l2="0", iterations=25, scale=None, ref=True, eval_train=True),
            dict(id="more", model="logistic", l2="0", iterations=100, scale=None, same_as="fit"),
        ],
    ),
    dict(
        key="wine", task="classification", target="class_0",
        title="Wine, class 0 against the rest", short="Wine",
        runs=[
            # the column types are read from each file; color_intensity has two decimals in the train file and six in the test file
            dict(id="typed", model="logistic", l2="0.01", iterations=25, scale=STD, shown_only=True, eval_refusal=True),
            dict(id="naive", model="logistic", l2="0", iterations=25, scale=STD, float="color_intensity", shown_only=True, eval_train=True),
            dict(id="naive_more", model="logistic", l2="0", iterations=50, scale=STD, float="color_intensity", same_as="naive"),
            dict(id="refused", model="logistic", l2="0", iterations=100, scale=STD, float="color_intensity", shown_only=True, refusal=True),
            dict(id="fit", model="logistic", l2="0.01", iterations=25, scale=STD, float="color_intensity", ref=True),
            dict(id="more", model="logistic", l2="0.01", iterations=100, scale=STD, float="color_intensity", same_as="fit"),
        ],
    ),
]

DATA_DIR = "docs/examples/data"
REFERENCE = "docs/examples/reference.json"
PROBA_ROWS = 5


def fit_argv(ex, run):
    """The argv of `ml fit` for a run (what is typed is shlex.join of it, with the step quoted)."""
    a = ["ml", "fit", "--model", run["model"], "--target", ex["target"]]
    if run["scale"]:
        a += ["--step", run["scale"]]
    if run.get("float"):
        a += ["--float", run["float"]]
    if run["iterations"] is not None:
        a += ["--iterations", str(run["iterations"])]
    a += ["--l2", run["l2"], "--output", "%s_%s.ml" % (ex["key"], run["id"]), "%s_train.csv" % ex["key"]]
    return a


def eval_argv(ex, run):
    return ["ml", "eval", "--model-file", "%s_%s.ml" % (ex["key"], run["id"]), "%s_test.csv" % ex["key"]]


def proba_argv(ex, run):
    return ["ml", "predict", "--model-file", "%s_%s.ml" % (ex["key"], run["id"]), "--proba",
            "%s_test.csv" % ex["key"], "--limit", str(PROBA_ROWS)]


# --- the gate of docs/examples.md section 3 -----------------------------------------------------------------------
# kind -> (absolute tolerance, relative to max(1, |value|)?)
COEF_TOL = {"linear": (1e-9, True), "logistic": (1e-6, True)}
METRIC_TOL = {
    "rmse": (1e-9, True), "mae": (1e-9, True), "r2": (1e-9, True),
    "log_loss": (1e-6, False), "roc_auc": (1e-6, False),
    "accuracy": (0.0, False), "precision": (0.0, False), "recall": (0.0, False), "f1": (0.0, False),
}
CLASS_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]
REG_ORDER = ["rmse", "mae", "r2"]


def allowed(tol, value):
    absolute, relative = tol
    return absolute * max(1.0, abs(value)) if relative else absolute


def within(tol, a, b):
    """The difference between two numbers, and whether it is inside the tolerance."""
    d = abs(a - b)
    return d, d <= allowed(tol, b)
