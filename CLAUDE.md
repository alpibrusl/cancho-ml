# CLAUDE.md -- cancho-ml

Design before code: each feature gets a section in `docs/design.md` before its implementation, and every claim in it is
either measured (with machine, threads, load and minimum of N) or marked as a choice or an estimate.

- **Agent-first, like cancho-table.** The CLI, errors and logs follow cancho `docs/agent-toolbox.md` through the `contract/`
  package of `cancho-tools`, pinned by commit and never copied. Errors are data with a stable `<area>.<name>` rule tag, an
  exit code and, where safe, a repair. No payload is ever logged.
- **Determinism.** The same input, seed and plan give the same bytes at any thread count. A feature that cannot keep that is
  refused or documented as the exception, with a test.
- **No file over 2,000 lines. Every refusal has a rule tag. No input reaches a panic.**
- **Gates are written before the code they judge and must be able to fail.** Mutants: all killed, or each survivor explained.
- **Measured claims only.** A claim found false is corrected in place, in the document that made it. Never state a speed
  against another library that was not measured here.
- **No GPU and no model runs without asking.**
- Commit trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; PRs stay draft until the maintainer says
  merge; no force-push to main.
