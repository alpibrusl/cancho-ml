# Continuous integration: what it runs, on which machines, and what it does not show

**Status: the design of `.github/workflows/ci.yml`, written before it. A number appears in it only if a script of this repository printed it, with the
conditions (section 8). Tags: [choice] a decision the maintainer can overturn, [measured] printed by a script here, [open] not decided or not tested.**

The question: the claim that `ml` writes the same bytes everywhere (README, "The same bytes") was verified on one Mac. What does a workflow have to run so
that the claim names the machines it holds on, and fails when it stops holding on one of them?

## 1. What the workflow is for

1. **Build** the pinned compiler and the three binaries `ml`, `ml-predict` and `ml-serve` on Linux x86-64 and macOS arm64, from nothing but the repository
   and the pins in `cancho.toml`.
2. **Determinism gate**: on each of those two platforms, the fingerprint of the built `ml` equals one committed file (section 2).
3. **Fast gates**: the checks that need no numerical oracle (section 5).
4. **Full gate** on request and nightly: `scripts/check.sh` (section 7).
5. **Release** on a tag `v*`: the three binaries of each platform, with SHA-256 files (section 9).

It is not a benchmark: no time is asserted by a gate, and nothing here measures speed.

## 2. The determinism gate

`scripts/xplat_check.py` fits 22 models (every kind: constant, linear, logistic, tree, forest, k-means; one-hot, label, frequency and hash steps; 1 and 8 threads) on
the files of `docs/examples/data`, then predicts and scores each with the file it wrote. It prints one line per artefact: the SHA-256 of the whole model
file, and the exit status and SHA-256 of the bytes of each `predict` and `eval` answer, 88 lines in all (22 times `fit-exit`, `model-file`, `predict`, `eval`).

The **golden** is `docs/bench/xplat_fp.txt`. Each platform of the matrix compares its own fingerprint to the golden and fails on any difference. That is the
whole cross-platform comparison [choice]: two platforms that each equal the same committed file are equal to each other, and the job needs no artefact passed
from one runner to the other, so a skipped, cancelled or re-run job cannot leave a comparison undone. A platform that is not in the matrix is not covered.

The script fails (exit 1, a diff on standard output) when

* any fit, predict or eval exits non-zero, or hangs for 300 seconds, or a fit that exited 0 wrote no model file: a failed fit is never only printed;
* a line differs from the golden, a case is missing from the run, a case is missing from the golden, or the number of lines differs;
* the cases that differ only in `--threads` (`-t1` and `-t8`) do not print the same lines;
* `ml-predict` prints other bytes than `ml` for a predict (it has no `eval`); the workflow passes `--require-predict`, so a missing `ml-predict` is a failure.

`--mutants` runs the check against 15 deliberate faults and requires each to fail (a hex digit of a stored model hash and of a stored eval hash changed; a case,
a line missing from the run or from the golden; an extra case; two lines swapped; an empty golden; a fit that exits 1; one byte of a stored tree model altered; a
predict answer altered; an `ml-predict` that prints another answer; a forest fit that writes another byte at 8 threads, caught by the thread check itself; a
binary that cannot start). The workflow runs it on every platform.

### 2.1 What it shows and what it does not

It shows that, for these 22 fits on these inputs, the compiler `a4572ea` and the sources at the commit under test produce the same model bytes and the same
printed answers on macOS arm64 and on Linux x86-64. It does not show the same for any other input, platform, compiler revision or operating-system image;
the arithmetic's reasons (IEEE binary64, no fused multiply-add in the generated code, no platform `libm` in a result) are in `docs/design.md`, and a platform that
disagrees is found by this gate or not at all. A hash of a printed answer includes the printed text, so a change of the output format changes the golden too.

## 3. The golden and how it changes

`docs/bench/xplat_fp.txt` was written by `python3 scripts/xplat_check.py --update` on the Mac (macOS arm64, compiler `a4572ea3b506`, sources at `cb08d5b`), and the
same file was reproduced byte for byte on Linux x86-64 (section 8). `--update` refuses to write when the run has any fault (a failed fit, a thread difference), so a
golden is only written from a clean run; it prints the lines that changed.

The golden changes only when model bytes or printed answers change **on purpose**: in a commit that says why, as `tests/conformance/golden_numeric.json` does. A
red determinism job is not repaired by `--update`; it is first explained (a real platform difference is a bug of the library or the compiler, not of the golden).

## 4. The platforms and the pins

| job | runner label | architecture | what the label points to |
|---|---|---|---|
| Linux | `ubuntu-latest` | x86-64 | an Ubuntu image of GitHub-hosted runners |
| macOS | `macos-latest` | arm64 (Apple silicon) | a macOS image of GitHub-hosted runners |

The label is a moving target [open]: the workflow prints `uname -a`, `sw_vers` (macOS), the C compiler version and the Rust version into the log of every run, so the
log says which image ran. The macOS runner is not the Mac of the earlier measurements (macOS 26.2, Darwin 25.2.0, Apple M4 Max); its operating-system version differs,
and its Xcode, clang and CPU generation may too. That the golden holds on both is the evidence that the bytes do not depend on those.

Pins, none duplicated:

* **the compiler revision** is read from `cancho.toml` (`cancho = "..."` under `[package]`) by a step with `sed`, as the README does. The workflow holds no copy. The
  compiler is built from that revision with `cargo build --release -p cancho`, with the Rust channel read from the compiler's own `rust-toolchain.toml`; the binary is
  cached under a key made of the revision, the operating system and the architecture.
* **the packages** (`cancho-tools`, `cancho-table`, `cancho-web`, `cancho-schema`, the compiler's `http.server`) are the `[dependencies.*]` of `cancho.toml`, fetched by
  `cancho install` at their pinned commits.
* **third-party actions** are pinned by full commit SHA with the release tag in a comment: `actions/checkout` v7.0.1, `actions/cache` v6.1.0,
  `actions/setup-python` v7.0.0, `actions/upload-artifact` v7.0.2, `actions/download-artifact` v8.0.2. `scripts/check_workflow.py` fails on any `uses:` that is not a 40-hex SHA.
* **Python** is 3.14 (the Mac has 3.14.5, the Linux machine of the measurements 3.14.4) and its packages are `ci/requirements-fast.txt` and `ci/requirements-full.txt`: exact
  versions, as on the maintainer's machine.

## 5. The jobs

| job | triggers | permissions | runs |
|---|---|---|---|
| `build-test` (matrix: Linux x86-64, macOS arm64) | push to `main`, pull request, manual | `contents: read` | the fast gates |
| `full-gate` (Linux x86-64) | manual, nightly schedule | `contents: read` | `scripts/check.sh` |
| `release` | a tag `v*` | `contents: write`, this job only | `gh release create` and upload |

The top-level `permissions` is `contents: read`. No secret is used; the default `GITHUB_TOKEN` is the only credential and only `release` can write with it. There is
no `pull_request_target`.

The fast gates (`build-test`, in this order; each can fail, and the first failure ends the job):

1. platform and tool versions printed; the compiler revision derived from `cancho.toml`;
2. the compiler: cache restore, and on a miss clone at the revision and `cargo build --release -p cancho`;
3. `scripts/check_table_pin.sh`, `scripts/check_web_pin.py` (the pins agree with the documents);
4. `cancho install`, `cancho fmt --check lib tests tools`;
5. `cancho build --bin ml`, `--bin ml-predict`, `--bin ml-serve`, then `scripts/check_bins.py` and `scripts/check_web_pin.py --bins` (the three are built from the same sources);
6. **the determinism gate**: `scripts/xplat_check.py --require-predict` and `--mutants`;
7. the conformance tests (`tests/conformance`) by `scripts/run_conformance.py --no-oracles`, with the packages of `ci/requirements-fast.txt`: one process per test module, four at a time. The five
   modules whose import needs numpy (`test_enc`, `test_planned_baseline`, `test_wire_kmeans_oracle`, `test_wire_oracle`, `test_wire_tree_oracle`) are listed as not run; the full gate runs
   them. The `ml-serve` conformance tests are among those that run: they start the server on an ephemeral port and compare one POST per model kind with the command line's cell text
   (`docs/serving-http.md` section 9), and `serve-check`, the driver of their number gate, is built for them;
8. the documentation checks that need only the built binary: `scripts/check_docs.sh` (program output in README and pages, `check_links.py`, `check_prose.py`, `check_doc_refs.py`, their mutants);
9. `scripts/check_workflow.py` (section 6);
10. the binaries, the fingerprint and the logs are uploaded as artifacts.

`check_links.py` fetches no address (it counts external links), so it does not depend on the network. `cancho install` and the clones do, and a GitHub outage or a rate
limit fails the job at that step, with the step's name.

### 5.1 Three cases that run on macOS only

Running the conformance suite on Linux x86-64 found three cases whose answer is the platform's, not the library's: `path-symlink` and `path-symlink-dir` refuse a path through a
symbolic link with `"errno":40` in the detail on Linux (ELOOP) where the baseline, written on macOS, holds 62; `io-read-failed` opens `/dev/stdin` with descriptor 0 closed, which on Linux is a link
to a path that does not exist (`io.not-found`, exit 3) and on macOS a read that fails (`io.read-failed`, exit 1). The three fixtures carry `darwin_only` (`harness.runs_here`) and
`test_rules`, `test_cat_baseline` and `test_wire_baseline` skip them elsewhere. No Linux baseline for them is recorded: the Linux answers above were read from one run on one machine
[measured, once]. A refusal whose detail holds an `errno` is therefore not byte-equal across platforms, and no determinism claim is made for it.

## 6. The checks of the workflow itself

`scripts/check_workflow.py` reads `.github/workflows/ci.yml` and fails when: the top-level `permissions` is missing or is not exactly `contents: read`; a job other than
`release` declares a write permission; `release` declares anything but `contents: write`; a `uses:` is not pinned to a 40-hex SHA; the file mentions `pull_request_target`
or `secrets.` other than `secrets.GITHUB_TOKEN`; the compiler revision is a literal in the workflow instead of being read from `cancho.toml`; a script the workflow calls
(`scripts/...`, `ci/...`) does not exist; the triggers lack `push`, `pull_request` or `workflow_dispatch`; `full-gate` runs on push or pull request; `release` is not limited to tags. It
has mutants (`--mutants`), each of which must make it fail. `actionlint` is not installed on the machine this was written on and is not a dependency; it is not claimed to have run.

## 7. The full gate

`scripts/check.sh` takes 30 to 40 minutes on the Mac [open: measured in section 8] and needs numpy, scipy, scikit-learn, Schemathesis and the OpenAPI validator
(`ci/requirements-full.txt`), and the files of spike F, which are not in the repository: the job regenerates them with `scripts/spikes/gen.py` into the directory the tree
gates read (`../ml-data`). `CANCHO` points to the compiler built by the job. The job has a timeout of 120 minutes, runs on `workflow_dispatch` and on a nightly schedule only, and uploads
its log as an artifact.

## 8. Measured

(filled in by the commit that records the runs)

## 9. Release

On a tag `v*` the `release` job needs `build-test` on both platforms, downloads their binaries, writes `SHA256SUMS-<os>-<arch>` next to each platform's `ml`, `ml-predict` and
`ml-serve`, and creates the release with `gh release create` and uploads the files. The binaries are the ones the gates ran on in that same workflow run. They are unsigned;
the SHA-256 files are for a download check, not provenance. The tag is not checked against a version in the sources [open].

## 10. Not shown

* The first run of the workflow is the pull request's own: it cannot be run anywhere but on GitHub, and what was run before it is every step by hand on a Linux x86-64
  machine and on a Mac (section 8), not the YAML.
* Linux aarch64 is not in the matrix. A run on a Raspberry Pi 5 was done by hand, if at all, outside this workflow, and this document does not report it.
* Windows, other Linux distributions, musl, other macOS versions than the runner's, and other compiler revisions are untested.
* The runners' images change without a commit here; a red run after no commit is the first thing to read in the platform lines of the log.
* A compiler cache hit skips the compiler build: the cache key is the revision, the operating system and the architecture, so a compiler built by another Rust than the one
  recorded in the log could be restored only if the key were reused; the Rust channel is not part of the key because it is a function of the revision.
* A built binary is not bit-reproducible on Linux: three builds of `ml-predict` from one tree with one compiler gave three SHA-256 values [measured, Linux x86-64, 3 builds], and the bytes
  that differ are the file name `cancho-llvm-<pid>-0.ll` in the symbol table, which holds the compiler's process id. Two builds on the Mac gave one value [measured, 2 builds]. The checksums of a release name the
  files of one run and are not a build-reproducibility claim; the model bytes and the answers, which the gate fingerprints, do not depend on it.
