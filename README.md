<p align="center"><img src="docs/assets/cancho-ml-logo.jpg" alt="cancho-ml" width="220"></p>

# cancho-ml

**Classical machine learning on tabular data, as a command line an agent can read.** `ml` fits, applies and scores a model on a CSV with explicit semantics, exact sums where a sum decides the result, bounded resources, structured refusals and a machine-verifiable authority. It is a command line, not a chat box: you (or an agent) give it flags, it gives back one JSON line, or a named refusal with a repair. One small binary, written in [cancho](https://github.com/alpibrusl/cancho), built the way [cancho-table](https://github.com/alpibrusl/cancho-table) is. The [project page](https://alpibrusl.github.io/cancho-ml/) has the same in pages.

**Status: early, alpha, not for production.** The command line builds three models. `--model constant` is the baseline: the training mean for a regression, the majority class for a classification. `--model linear` (least squares, ridge with `--l2`) and `--model logistic` (binary, Newton's method) go through a pipeline of steps (`impute`, `scale`, `encode`, `bucket`) that `--step` or `--plan` describes and that is fitted on the training rows and stored, with the model, in **one** file: `predict` and `eval` apply the stored steps, and no flag can apply another transformation. A float and decimal CSV reader, an empty cell as a missing value that the plan must impute, a seeded `--split`, `--threads` for `fit`, `predict --proba` and the metrics of `eval` (accuracy, precision, recall, F1, ROC-AUC, log-loss, RMSE, MAE, R2) are in. k-means, decision trees and random forests, and text columns, are **not** reachable yet: the first two exist as tested libraries, and text categories are the next stage. A model is a module (a manifest, a fixed adapter, generated glue, an admission script) so that the next one cannot add a float order, a capability or an unregistered refusal: [docs/modules.md](docs/modules.md). One pre-registered gate is red (tree regression against scikit-learn, below). The whole list, with what each item was measured as: [docs/status.md](docs/status.md).

There is no CI workflow in this repository yet. `scripts/check.sh` is what one would run, and `scripts/check_docs.sh` checks this README and the project page (see Contributing).

## One complete flow

Fit, predict, score, get refused with a repair, run the repair. The repair is the corrected command, and the output below is what the built program prints, on the small files shown under "What you can do now".

<!-- gen:flow -->
```console
# Fit the training mean of price. area has a decimal point: it is a float column, read exactly
$ ml fit --target price --output houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"columns":4,"target":"price","path":"houses.ml","bytes":340,"sha256":"299879cc4b9be8afb5c6029aa8c635693162dc7a6d6ef564a4a1a66b83f4f1d8","written":true,"summary":{"mean":"218000.000000"}},"meta":{"version":"0.1.0"}}
# Predict for two new rows, which have no price
$ ml predict --model-file houses.ml new.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["218000.000000"],["218000.000000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
# Score the model on the file it was fitted on
$ ml eval --model-file houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"metrics":{"rmse":"71035.202541","sse":"25230000000.000000000000","prediction":"218000.000000"}},"meta":{"version":"0.1.0"}}
# New rows with a column of text are refused
$ ml predict --model-file houses.ml listings.csv
{"rule": "ml.input.non-numeric", "hint": "drop the column with --drop, or fix the cell", "repair": {"kind": "retry", "argv": ["ml", "predict", "--model-file", "houses.ml", "listings.csv", "--drop", "city"]}}
# exit status 8; the rule, hint, repair of the JSON line it prints
# The tool offered the corrected command. Run it:
$ ml predict --model-file houses.ml listings.csv --drop city
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["218000.000000"],["218000.000000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
```
<!-- /gen:flow -->

## A model through the pipeline

A float CSV with an empty cell. A plan has to say what to do with it, or the fit is refused with a repair. Then the plan, written out: impute, standardise, a logistic regression; predict with probabilities (the empty cell of the new rows is filled with the mean **stored in the model**); score. The output is what the built program prints, on the files in `scripts/site.py`.

<!-- gen:pflow -->
```console
# A float CSV with an empty cell: the plan must say what to do with it. The refusal offers a repair
$ ml fit --model logistic --target bought --output shop2.ml shop2.csv
{"rule": "ml.input.missing-value", "hint": "add an impute step for the columns, for example impute column=a,b strategy=mean.", "repair": {"kind": "retry", "argv": ["ml", "fit", "--model", "logistic", "--target", "bought", "--output", "shop2.ml", "shop2.csv", "--step", "impute column=spend strategy=mean"]}}
# exit status 8; the rule, hint, repair of the JSON line it prints
# The repair works, but a plan can say more: impute, standardise, then a logistic regression (50 Newton iterations, ridge 0.1)
$ ml fit --model logistic --overwrite --step 'impute column=spend strategy=mean;scale column=visits,spend kind=standard' --iterations 50 --l2 0.1 --target bought --output shop2.ml shop2.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":12,"columns":4,"target":"bought","path":"shop2.ml","bytes":689,"sha256":"ad4d3f293465d2952d15e47dc1451944f944ee8ae8a9e008a52610824dae5a74","written":true,"summary":{"plan":"impute column=spend strategy=mean\nscale column=spend,visits kind=standard ddof=0\nmodel kind=logistic iterations=50 l2=0.1\n","steps":2,"features":3,"coefficients":[{"name":"id","value":"-0.08218688808973369"},{"name":"visits","value":"0.25761586355528143"},{"name":"spend","value":"1.1474347278761494"}],"intercept":"0.6023363690462818","classes":[0,1],"positive":1}},"meta":{"version":"0.1.0"}}
# Predict with probabilities for new rows (the empty cell is filled with the mean STORED in the model)
$ ml predict --model-file shop2.ml --proba new2.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","columns":["label","probability"],"rows":[["0","0.36645031865861677"],["1","0.8502846389524489"],["0","0.0906646088784972"]],"row_count":3,"total_rows":3,"truncated":false,"next":null,"unknown_categories":0},"meta":{"version":"0.1.0"}}
# Score it: accuracy, precision, recall, F1, ROC-AUC, log-loss
$ ml eval --model-file shop2.ml shop2.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":12,"positive":1,"metrics":{"accuracy":"0.8333333333333334","precision":"0.8333333333333334","recall":"0.8333333333333334","f1":"0.8333333333333334","roc_auc":"0.9444444444444444","log_loss":"0.4012542324769021"},"unknown_categories":0},"meta":{"version":"0.1.0"}}
```
<!-- /gen:pflow -->

## Why cancho-ml?

Most ML tools optimise for how many models they offer. `ml` is built to be predictable first, and so far offers three models and a pipeline.

* **Explicit types.** A column is an integer, a decimal (an exact mantissa and a scale) or a float (the nearest double, ties to even, read by cancho-table's exact reader). `nan`, `inf`, an empty cell and a cell that is not a number are refused, with the line and the column; `--drop` ignores a column. [docs/reader.md](docs/reader.md), [docs/cli.md](docs/cli.md)
* **Exact where a sum decides.** The mean of an integer or decimal target and the sum of squared errors are exact integers, kept in limbs, not floats; a float target is summed exactly and rounded once. A test compares every digit with Python's `fractions.Fraction`. The one step that is not exact, the `rmse` (three IEEE operations after the exact sum), is said so in the document. [docs/cli.md](docs/cli.md) section 3
* **The same bytes.** A model file holds no timestamp, host or path. The same data, plan and seed give the same model bytes whatever the working directory, the spelling of the path or the repetition, and at every `--threads` from 1 to 16 (tested on one Mac, for linear and logistic); across machines it is **not tested**. [docs/cli.md](docs/cli.md) section 6, [docs/linalg.md](docs/linalg.md) section 5.4
* **Bounded resource use.** Rows, columns, cells, line length, model file size and log size each have a flag, a default and a ceiling, checked before anything sized by the input is allocated. No gate measures memory use. `ml introspect` lists the limits. [docs/cli.md](docs/cli.md) section 2.2
* **No implicit network access.** The authority row, derived by `cancho authority`, lists what the program can reach: no `net_out`, `net_in`, `ffi` or `clock`, no standard input, and no write by path. `scripts/manifest.py --check` fails if the binary differs from the committed [`manifests/ml.authority.json`](manifests/ml.authority.json). [docs/cli.md](docs/cli.md) section 6
* **Machine-readable errors.** One JSON line against a schema (`ml.v1`); a refusal is `{code, rule, message, hint, repair, detail}` with a stable `<area>.<name>` rule tag. [docs/design.md](docs/design.md) section 8
* **Repair hints.** A refusal says how to fix the request, as an argv to run, or why there is none. [docs/cli.md](docs/cli.md)
* **A model file that checks itself.** `CML1`: sections, tensor records and a SHA-256 trailer. Every truncation and every flipped byte of a small model is refused, and no damaged file is accepted. [docs/cli.md](docs/cli.md) section 4
* **Capability introspection.** `ml introspect` prints the flags, limits, rules and authority; `ml skill` prints a guide for an agent. Both come from the tables the parser runs on.

## Two-minute quick start

You need `git`, Rust, `clang` and `python3`, and the machine needs network access for the clones and for `cancho install`, which fetches the pinned modules into `build/deps`: the `cancho-tools` contract package and cancho-table's six numeric modules (`pow5`, `dec`, `flt`, `query`, `facc`, `reader`, the stores that [cancho-table#44](https://github.com/alpibrusl/cancho-table/pull/44) committed), all `[dependencies.*]` of `cancho.toml`, each at one commit. The cancho-table rev is `202b214372a0`; nothing of cancho-table is cloned or copied, and `scripts/check_table_pin.sh` checks that the pin agrees with [docs/reader.md](docs/reader.md) section 1.3, which also says how to move it.

```sh
git clone https://github.com/alpibrusl/cancho                           # the compiler
git clone https://github.com/alpibrusl/cancho-ml && cd cancho-ml
REV=$(sed -n 's/^cancho *= *"\([0-9a-f]*\)".*/\1/p' cancho.toml)        # the compiler these sources need
(cd ../cancho && git fetch -q origin && git checkout "$REV" && cargo build --release -p cancho)
sh scripts/check_table_pin.sh                                           # the pin agrees with docs/reader.md
export PATH=$PWD/../cancho/target/release:$PWD/build:$PATH
cancho install && cancho build --bin ml                                 # install fetches the pins (network); builds build/ml

cat > houses.csv <<'EOF'
id,rooms,area,price
1,2,48.5,180000
2,3,71.25,240000
3,3,64,215000
4,4,98.5,335000
5,1,31,120000
EOF
printf 'id,rooms,area\n6,2,52.5\n7,3,80\n' > new.csv
printf 'id,rooms,area,city\n6,2,52.5,Lyon\n7,3,80,Porto\n' > listings.csv
```

On a 16-core Mac, with the compiler already built, `cancho build --bin ml` took 2.9 seconds once, after `cancho install` had fetched the modules (that fetch, from GitHub, took 1.4 seconds once). The compiler build and the clones are extra.

Fit, predict, score, and a refusal:

<!-- gen:qs -->
```console
$ ml fit --target price --output houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"columns":4,"target":"price","path":"houses.ml","bytes":340,"sha256":"299879cc4b9be8afb5c6029aa8c635693162dc7a6d6ef564a4a1a66b83f4f1d8","written":true,"summary":{"mean":"218000.000000"}},"meta":{"version":"0.1.0"}}
$ ml predict --model-file houses.ml new.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["218000.000000"],["218000.000000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
$ ml eval --model-file houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"metrics":{"rmse":"71035.202541","sse":"25230000000.000000000000","prediction":"218000.000000"}},"meta":{"version":"0.1.0"}}
$ ml predict --model-file houses.ml listings.csv
{"rule": "ml.input.non-numeric", "hint": "drop the column with --drop, or fix the cell", "repair": {"kind": "retry", "argv": ["ml", "predict", "--model-file", "houses.ml", "listings.csv", "--drop", "city"]}}
# exit status 8; the rule, hint, repair of the JSON line it prints
```
<!-- /gen:qs -->

## What you can do now

Every example runs on the small files above and the ones shown with it, in this order, and the output is what the program prints. The first examples use the constant model, whose prediction is the same for every row; the last two are the models that go through the pipeline.

### Fit, predict and score a regression

`fit` reads the CSV, takes `--target` as the column to predict and writes the model with `--output`. `price` is a whole-number column, so its mean is an exact decimal with six more fractional digits than the column has. `predict` needs the same columns as the fit, minus the target; `eval` scores a model on a CSV that has the target.

<!-- gen:t-fit -->
```console
$ ml fit --target price --output houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"columns":4,"target":"price","path":"houses.ml","bytes":340,"sha256":"299879cc4b9be8afb5c6029aa8c635693162dc7a6d6ef564a4a1a66b83f4f1d8","written":true,"summary":{"mean":"218000.000000"}},"meta":{"version":"0.1.0"}}
$ ml predict --model-file houses.ml new.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["218000.000000"],["218000.000000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
$ ml eval --model-file houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"metrics":{"rmse":"71035.202541","sse":"25230000000.000000000000","prediction":"218000.000000"}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:t-fit -->

### Fit and score a classification

`--task classification` needs an integer target with exactly two values. The prediction is the class with more rows (the smaller label on a tie) and the probability is the share of the larger label's rows, to six digits. `predict` comes in pages: `--limit` and `--from`, and `next` is where to continue. This second example uses `shop.csv`.

```csv
id,visits,spend,bought
1,3,12.50,1
2,1,0.00,0
3,7,40.25,1
4,2,5.00,0
5,5,22.75,1
6,4,18.00,1
```

<!-- gen:t-class -->
```console
$ ml fit --task classification --target bought --output shop.ml shop.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"classification","rows":6,"columns":4,"target":"bought","path":"shop.ml","bytes":363,"sha256":"dc81e1ce7ff30ea56c12ec49a39cd808e55f558d049f0858ab136f213f385dbd","written":true,"summary":{"classes":[0,1],"counts":[2,4],"majority":1,"probability":"0.666667"}},"meta":{"version":"0.1.0"}}
$ ml predict --model-file shop.ml shop.csv --limit 2 --from 1
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"classification","columns":["label","probability"],"rows":[["1","0.666667"],["1","0.666667"]],"row_count":2,"total_rows":6,"truncated":true,"next":{"from":3}},"meta":{"version":"0.1.0"}}
$ ml eval --model-file shop.ml shop.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"classification","rows":6,"metrics":{"accuracy":"0.666667","correct":4,"total":6,"prediction":1}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:t-class -->

### Read floats exactly

A cell with an exponent, a leading or trailing point or more than 12 digits is a float without being named (`--float NAMES` forces it). It is read to the nearest double, ties to even, by cancho-table's reader. The mean of a float target is the exact sum of the doubles over the count, rounded once, and written as the shortest decimal that reads back to the same double. This example uses `sensor.csv`.

```csv
id,x,reading
1,1.5,1e1
2,2.5,2.75
3,0.1,3e-1
4,4,5.5
```

<!-- gen:t-float -->
```console
$ ml fit --target reading --output sensor.ml sensor.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":4,"columns":3,"target":"reading","path":"sensor.ml","bytes":306,"sha256":"182d75563021836f542c0542daf74d2ddfcd9608b6e2ac769beb969d2b52ff44","written":true,"summary":{"mean":"4.6375"}},"meta":{"version":"0.1.0"}}
$ ml eval --model-file sensor.ml sensor.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":4,"metrics":{"rmse":"3.6012801543340114","sse":"51.876875","prediction":"4.6375"}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:t-float -->

### Check before you write

An existing output is refused unless `--overwrite` is given. `--dry-run` reads, checks and fits, runs every check the write would, and answers with the file it would write: exit 9, and nothing created or changed. The file is written to a temporary name beside the final one and renamed over it, so a reader never sees half a model.

<!-- gen:t-dry -->
```console
$ ml fit --target price --output houses.ml houses.csv
{"rule": "conflict.exists", "hint": "use --overwrite to replace it, or name another file", "detail": {"path": "houses.ml"}}
# exit status 5; the rule, hint, detail of the JSON line it prints
$ ml fit --dry-run --overwrite --target price --output houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"columns":4,"target":"price","path":"houses.ml","bytes":340,"sha256":"299879cc4b9be8afb5c6029aa8c635693162dc7a6d6ef564a4a1a66b83f4f1d8","written":false,"summary":{"mean":"218000.000000"}},"dry_run":true,"planned_actions":[{"op":"replace","path":"houses.ml","bytes":340,"sha256":"299879cc4b9be8afb5c6029aa8c635693162dc7a6d6ef564a4a1a66b83f4f1d8"}],"meta":{"version":"0.1.0"}}
# exit status 9; a dry run: nothing was written
```
<!-- /gen:t-dry -->

### A damaged model is refused

The model file ends with the SHA-256 of every byte before it. Here one byte of `houses.ml` was changed and the file saved as `damaged.ml`.

<!-- gen:t-damaged -->
```console
$ ml eval --model-file damaged.ml houses.csv
{"rule": "ml.model.checksum", "hint": "the file changed after it was written"}
# exit status 8; the rule, hint of the JSON line it prints
```
<!-- /gen:t-damaged -->

### It says what it does not do

The `--model` list is the list of what is built. There are no stand-in models, and a test pins that.

<!-- gen:t-honest -->
```console
$ ml fit --model forest --target price --output forest.ml houses.csv
{"rule": "args.bad-value", "detail": {"flag": "--model", "value": "forest", "expected": "constant/linear/logistic"}}
# exit status 2; the rule, detail of the JSON line it prints
$ ml fit --target prize --output typo.ml houses.csv
{"rule": "ml.input.unknown-column", "hint": "names are compared exactly, byte for byte", "detail": {"flag": "--target", "name": "prize"}}
# exit status 3; the rule, hint, detail of the JSON line it prints
```
<!-- /gen:t-honest -->

### Hold rows out, with a seed

`--split train=0.8 --seed 3` chooses the rows by the seeded train/test split of the metrics library (the stream registry, kind 4): the test side is the rows the fit does not see, `ceil(n * 0.2)` of them. The fit reports the held-out rows' metrics, scored through the stored steps; `eval --split train=0.8 --seed 3` on the same file scores the same rows. With 5 rows the test side is 1 row, so this is only to show the shape of the answer.

<!-- gen:t-split -->
```console
$ ml fit --model linear --l2 0.1 --split train=0.8 --seed 3 --target price --output houses-lin.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"linear","task":"regression","rows":4,"columns":4,"target":"price","path":"houses-lin.ml","bytes":349,"sha256":"5ccedf38b34a2a4b24edcae27ce13c569c6bcbce4fb1b5238ad419d354ecf5b6","written":true,"summary":{"plan":"model kind=linear l2=0.1\n","steps":0,"features":3,"coefficients":[{"name":"id","value":"-1905.5649303915627"},{"name":"rooms","value":"-15667.071024324228"},{"name":"area","value":"3843.571373934476"}],"intercept":"25324.69546154733","split":{"train":4,"test":1,"seed":3},"holdout":{"rows":1,"metrics":{"rmse":"8366.812920622935","mae":"8366.812920622935","r2":null}}}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:t-split -->

## Built to be read by an agent

`ml introspect` and `ml skill` come from the same tables the code parses its flags with, so they cannot disagree with the program: the interface, the limits, the rules and the authority can be learned from the binary itself. A test checks that every flag and every rule is in the table, in `introspect`, in `skill`, in the schema and in a fixture. [docs/cli.md](docs/cli.md) section 1.

<!-- gen:agentio -->
```console
$ ml introspect          # one JSON document; excerpts, long lines cut

tool        ml 0.1.0   (compiler a4572ea3)
flags       35, for example
  --root        path             resolve FILE, --model-file, --output and --log bene…
  --target      text             fit: the name of the column to predict, as the head…
  --output      path             fit: where to write the model file. The directory m…
  --model-file  path             predict, eval: the model file to read
  --drop        text             columns to ignore, header names separated by commas…
  --dry-run     bool             fit: read, check and fit, then report what would be…
limits      13, for example
  --max-rows         default 10,000,000   ceiling 1,000,000,000
  --max-columns      default 4,096        ceiling 8,192
  --max-line-bytes   default 1,048,576    ceiling 16,777,216
  --max-bytes        default 16,777,216   ceiling 268,435,456
rules       75, for example
  args.unknown-flag        exit 2   a repair: sometimes
  ml.input.non-numeric     exit 8   a repair: sometimes
  ml.schema.mismatch       exit 8   a repair: sometimes
  ml.model.checksum        exit 8   a repair: never
  conflict.exists          exit 5   a repair: never
  limit.line-too-long      exit 8   a repair: sometimes
reads environment: false   reads clock: false
authority   args, conc, dir_read, dir_write, err_write, file_read, file_write, fs_read(""), heap, io_write
            bounded: true   foreign symbols: 0
```
<!-- /gen:agentio -->

* **Rule tags.** Match on `error.rule`, not on the message. A rule is `<area>.<name>`; the table in `introspect` says each rule's exit code and whether a repair can exist (`never`, `sometimes`, `always`). The nine tags of the libraries that no input can make `ml` answer (descriptors, a solve the pivot test never lets fail, trees, the stratified split) are registered in [docs/design.md](docs/design.md) section 8.1 and [docs/wiring.md](docs/wiring.md) section 5, and are not in `introspect`.
* **Repairs.** `repair` is `null`, or `{"kind":"retry","argv":[...]}`: an argv that can be run as it stands and that never widens what the tool may touch.
* **`--dry-run`.** Only `fit` writes; `--dry-run` answers what it would write (`planned_actions`) and exits 9. It was not run under `strace`, so "no mutating system call" is the contract package's claim here and not re-measured ([docs/cli.md](docs/cli.md) section 2).
* **Exit codes.** One per class, from cancho's agent-toolbox; with several errors the first decides the code and `errors` lists them all.

<!-- gen:exit-codes -->
| exit | `code` | meaning |
|---:|---|---|
| 0 | `SUCCESS` | what was asked was done; zero matches is success |
| 1 | `GENERAL_ERROR` | the operating system failed mid-operation, or the tool caught a bug in itself |
| 2 | `INVALID_ARGS` | the invocation is malformed or names something of the wrong kind; nothing was read |
| 3 | `NOT_FOUND` | a named input does not exist |
| 4 | `PERMISSION_DENIED` | the operating system refused, or the tool's own confinement did |
| 5 | `CONFLICT` | state is not what the caller said it would be |
| 8 | `PRECONDITION_FAILED` | the answer is no, or a limit was reached |
| 9 | `DRY_RUN` | a --dry-run completed; planned_actions lists what an apply would do |
<!-- /gen:exit-codes -->

* **The authority row.** What the compiler says the program can reach. It reads the files named on its command line, creates, appends to, renames and removes only beneath a directory it opened itself, and keeps its working data on the heap. `conc` is in the row since the wiring stage: `fit --threads N` calls the threaded kernels of `lib/linalg`; `tools.toml` states the reason, and the test that no thread count changes a byte is in the same diff ([docs/wiring.md](docs/wiring.md) section 6).

<!-- gen:authority -->
```console
# manifests/ml.authority.json
ml    args, conc, dir_read, dir_write, err_write, file_read, file_write, fs_read(""), heap, io_write
# bounded: true; foreign symbols: none; net_out, net_in, ffi, clock: absent
```
<!-- /gen:authority -->

* **Where the output is the contract package's text, not a measurement for `ml`.** `introspect.guarantees` and parts of `skill` are written by the pinned `cancho-tools` package and are wrong or too narrow for `ml` in six places, for example "read-only: it never writes" although `fit` writes. They are listed in [docs/cli.md](docs/cli.md) section 8 so that nobody reads them as claims.
* **An MCP server is not built.** `docs/cli.md` section 9 says how it would be derived from `introspect`.

## What it cannot do yet

* **Three models are reachable.** `constant`, `linear` and `logistic`. k-means, decision trees and random forests exist as libraries (`lib/models`, [docs/models-linear.md](docs/models-linear.md), [docs/models-tree.md](docs/models-tree.md)), gated against independent oracles, with a manifest and `wired = false`: a plan naming one is `ml.plan.model-not-built`. Wiring one means writing its adapter and flipping `wired`; the admission script is what checks it ([docs/modules.md](docs/modules.md)).
* **No text columns yet.** The pipeline has `impute`, `scale`, `encode` (of numbers) and `bucket`. A column of text must be dropped; text categories are the next stage, and the hook for them is in [docs/wiring.md](docs/wiring.md) section 3.
* **Binary classification only.** More than two labels is `ml.label.not-binary`.
* **`--threads` is for `fit` only, in effect.** `eval` accepts and checks the flag, and nothing in it is threaded. The scores of `linear` and `logistic` are single-threaded.
* **One red gate.** The regression forest against scikit-learn's: RMSE 8.1% apart where the tolerance was 3% (`docs/models-tree.md` section 10.8). The algorithms differ (histogram against exact splits) and the stage attributed most of the gap to that; the tolerance was not moved and the maintainer has not decided. [docs/models-tree.md](docs/models-tree.md)
* **Not tested across machines.** The bytes are verified on one Mac (arm64), at 1 to 16 threads. A Linux x86-64 run is not done.
* **No CI workflow yet.** The reader is a pinned dependency on cancho-table's stores ([docs/reader.md](docs/reader.md) section 1), so the build no longer needs a sibling clone, but `cancho install` needs the network and nothing in this repository runs `scripts/check.sh` automatically.
* **No serving, no MCP server.** `ml-serve` (issue #16) and an MCP server are not started.
* **No neural networks and no GPU.** They need matrix kernels, a GPU path and automatic differentiation, which cancho does not have. Inference of small exported models may come later; it is not planned in v1.
* **Not built, not promised here:** multiclass, boosting, naive Bayes, k-NN, feature importance, out-of-bag scores, a model file for k-means and trees (kinds 6 and 7 are written by the libraries, not by `ml`).

## How fast

**No speed is claimed for `ml`: the command line has not been timed.** The libraries under it were timed, on one Mac (arm64, 16 cores, macOS 26.2; the documents that name the chip say Apple M4 Max) that other users and other builds were also using. **The load average was 2.4 or more in every table, so every cell is flagged `load >= 2` and none is a quiet-machine number. A serial rerun on a quiet machine is pending; until it is done, read the numbers below as conditions of one afternoon, not as claims.**

Fitting 1,000,000 rows of 20 features on one thread, in milliseconds, minimum of 7 interleaved samples (spread 1.01 to 1.07), load average 2.4 to 3.2 at the start and end of each cell, and 6.6 to 7.1 as the 5- and 15-minute averages ([docs/bench/models-bench.txt](docs/bench/models-bench.txt), [docs/models-linear.md](docs/models-linear.md) section 6.5):

| model | cancho | the same algorithm in numpy (Accelerate) | scikit-learn |
|---|---:|---:|---:|
| logistic regression, Newton, 6 passes | 236 | 170 | 126 (lbfgs, defaults), 223 (`tol=1e-12`) |
| linear regression, normal equations | 32.0 | 16.1 | 182 (`LinearRegression`, a different algorithm) |
| k-means, k = 8, 10 iterations | 306 | 1,148 | 330 (same initial centroids) |

**Where it is slower.** Dense float linear algebra: on the 29 shapes of the gate, one thread, against the calls a user writes (`np.dot`, `scipy.linalg.cholesky`), 8 match and 21 do not, from 1.05 times slower to 14.9 times slower (`syrk` at 100,000 by 256 is the worst), and square `gemm` is 13.8 to 16.5 times slower than Accelerate. The same algorithm written in C, with fused multiply-add off, takes 0.95 to 1.16 times as long as cancho on dot, axpy, gemv, syrk and Cholesky at 256 columns, so the gap is the vendor library and the language (no SIMD vector type), not the compiler's code generation ([docs/linalg.md](docs/linalg.md) section 5.2 and 5.3; load average 3.8 to 10). Logistic regression is 1.4 to 1.5 times slower than numpy's same algorithm and 1.9 to 2.2 times slower than scikit-learn's default call. A log-loss over a million probabilities is 3.0 times slower than a thin numpy version ([docs/metrics.md](docs/metrics.md) section 5.4).

**Where it is not.** The triangular solves, and Cholesky at 21 and 64 columns, match; k-means is level with scikit-learn (306 against 330 ms); the AUC over a million scores took 24 ms against 86 ms for a thin numpy version and 147 ms for `sklearn.roc_auc_score`. The libraries are threaded and keep their bytes: logistic regression on 1,000,000 rows takes 240 ms on one thread and 59 ms on sixteen, linear regression gains 1.4 times, because every thread owns a copy of its rows ([docs/bench/models-bench.txt](docs/bench/models-bench.txt)).

**Forests.** A forest of 10 trees on 100,000 rows fit in 0.0516 s on one thread and 0.0100 s on sixteen, where scikit-learn's `RandomForestClassifier` took 1.06 s and 0.132 s. **This is not a speed claim:** scikit-learn's forest looks for exact thresholds on the raw values and this one uses quantile bins (64 in this benchmark), so they are different algorithms making models of the same family. The quality of the classification forests agrees to 0.001 AUC; the regression forest does not (the red gate above).

The tables, the load of each cell and how to rerun them: [project page](https://alpibrusl.github.io/cancho-ml/benchmarks.html).

## Learn more

| | |
|---|---|
| [project page](https://alpibrusl.github.io/cancho-ml/) | the flow, the agent view and the quick start, in pages |
| [evidence](https://alpibrusl.github.io/cancho-ml/evidence.html) | the gates, the oracles and the mutants, and what each proves and does not prove |
| [benchmarks](https://alpibrusl.github.io/cancho-ml/benchmarks.html) | what was timed, with machine, load and minimum of N; not claims |
| [docs/status.md](docs/status.md) | what is built, what is left, what was measured, with the source document of each |
| [docs/design.md](docs/design.md) | the plan, the promises in tiers, the model file, the refusals and the registries |
| [docs/cli.md](docs/cli.md) | the command line: gates, reference, arithmetic of the constant model, model file, log, authority |
| [docs/wiring.md](docs/wiring.md), [docs/modules.md](docs/modules.md), [docs/models.md](docs/models.md) | linear and logistic through the pipeline; the module contract and the admission script; the table of models |
| [docs/integration.md](docs/integration.md) | how the six stages were merged, every gate on the merged tree, what is not wired |
| [docs/linalg.md](docs/linalg.md), [docs/rng.md](docs/rng.md) | the tensor and linear-algebra package and its summation order; the random generator |
| [docs/reader.md](docs/reader.md) | the float and decimal CSV reader and the dependency on cancho-table |
| [docs/metrics.md](docs/metrics.md) | metrics and splits |
| [docs/models-linear.md](docs/models-linear.md), [docs/models-tree.md](docs/models-tree.md) | linear and logistic regression and k-means; trees and forests; the red gate |
| [docs/upstream/](docs/upstream/) | reproducers for what cancho lacks, and the draft for cancho-table |

## Contributing

Design before code, in `docs/design.md`, with claims measured; a claim that turns out false is corrected in place, in the document that made it. Gates are written before the code they judge and must be able to fail: a mutant per gate, all killed or each survivor explained. See [CLAUDE.md](CLAUDE.md).

`scripts/check.sh` runs the format check, the unit tests, every library gate, the registry checks, the command line's conformance tests, the generator's drift check and the admission script on the three wired models, with the mutant modules it must refuse. It needs the compiler at the pinned revision, the network for `cancho install` and the files of the tree stage's spike at `../ml-data`. The mutant runs (`scripts/mutants_*.py`) rebuild the tool many times and run on request.

This README and the project page carry program output between paired `gen` comment markers. `python3 scripts/site.py` rewrites them from the built `build/ml`; `scripts/check_docs.sh` runs `site.py --check`, which fails if either file differs from what the binary prints, and `scripts/check_links.py`, which fails on a link to a file that is not in the repository. The prose, and every benchmark and gate figure, is written by hand next to its source document.

## Licence

[EUPL-1.2](LICENSE). Open source, provided as is, without warranty.
