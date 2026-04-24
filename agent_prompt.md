# Task: Optimize `@pytest.mark.slow` Marker Placement in SciPy

## Objective

Increase code coverage of the "fast" test suite by reassigning `@pytest.mark.slow` markers — without increasing total test runtime across all submodules.

## Background

SciPy uses pytest markers to distinguish test tiers:
- **No marker** = "fast" test (always runs)
- **`@pytest.mark.slow`** = "slow" test (skipped in fast runs)
- **`@pytest.mark.xslow`** = "extra slow" test (never touch these)

"Fast" and "slow" are not strictly about wall-clock time — they indicate whether a test should run in a quick CI pass. The goal is to maximize what the fast suite covers.

## Your Task

Find `@pytest.mark.slow`-marked tests that cover source lines **not already covered** by the fast suite, and remove their `slow` marker. This increases fast-suite coverage without significantly increasing fast-suite runtime. Also, you can mark other tests as slow to gain more time.

## Critical: Work Backwards from Coverage Gaps

**Do not** search for "fast slow-marked tests" and promote them indiscriminately. Many slow-marked tests are short but redundant — they cover code already exercised by existing fast tests. Promoting them adds runtime without improving coverage.

**The correct direction is:**
1. Find source lines **not covered** by the current fast suite.
2. Find which slow test(s) cover those lines.
3. Promote only those tests.

A slow test that takes 500ms but covers zero new source lines is worthless. A slow test that takes 2s but covers 50 previously-uncovered lines is valuable.

## Commit message format

Decide whether your commit is increasing test coverage, or decreasing test time.

### Increases test coverage

Use the coverage report to measure how much coverage changed. Then, write a commit like this.

```
TST: remove slow marker in test.py

Increases test coverage by X lines in file foo.py
```

Ignore changes in test coverage inside of test files.

### Decreases test time

If your change makes the test suite faster, write a commit like this:

```
TST: add slow marker in test.py

Reduces time spent in test in submodule X by Y seconds.
```

### Other

If your change does not increase test coverage or make the test suite faster, you should undo the change.

## Constraints

- **Only** change `@pytest.mark.slow` markers (add or remove).
- **Never** add, remove, or modify `@pytest.mark.xslow` markers.
- **Do not** write any new tests.
- Do not increase total test runtime across all submodules.

## Key Commands

| Purpose | Command |
|---|---|
| Build SciPy | `spin build` |
| Run fast tests (all submodules) | `spin test` |
| Run fast tests (one submodule) | `spin test -s <submodule>` (e.g. `spin test -s io`) |
| Run with coverage | `spin test -c` |
| Generate JSON timing report | `spin test -- --json-report` |
| Coverage + JSON report (baseline) | `spin test -c -- --json-report` |
| View coverage report (no re-run) | `python3 -m coverage report` |
| View missing lines for a file | `python3 -m coverage report --include="**/foo.py" -m` |
| Summarize time by submodule | `python3 parse_test_times.py --by-submodule` |
| Other aggregation options | `python3 parse_test_times.py --help` |

Tests produce a lot of output. Do not consume test output directly — just check the exit code (0 = success).

Once tests have been run with `-c`, coverage data is saved to `.coverage`. You can re-inspect it at any time without re-running tests using `python3 -m coverage report` (summary) or `python3 -m coverage html` (browsable HTML in `htmlcov/`).

Tests take approximately 10 minutes per run. Batch your marker changes before each run.

## Targeted Test Execution (avoid running the full suite)

Running the full suite is expensive (~10 min). Prefer targeted runs:

| Goal | Command |
|---|---|
| Run one submodule with coverage | `spin test -c -s stats -- -q` |
| Run one file with coverage | `spin test -c scipy/stats/tests/test_resampling.py -- -q` |
| Run one class with coverage | `spin test -c -t scipy.stats.tests.test_resampling::TestBootstrap -- -q` |
| Run one method with coverage | `spin test -c -t scipy.stats.tests.test_resampling::TestBootstrap::test_bootstrap_against_theory -- -q` |
| Run by keyword in a file with coverage | `spin test -c scipy/stats/tests/test_resampling.py -- -k "bootstrap" -q` |
| Run only slow tests in a file with coverage | `spin test -c scipy/stats/tests/test_resampling.py -- -m slow -q` |

Note: `-t` takes a dotted Python path (`package.module::Class::method`); pytest flags go after `--`. The file-path form (`spin test path/to/test_foo.py`) also works. `-q` (quiet) suppresses per-test output and is recommended to reduce noise.

**Important:** coverage runs **replace** `.coverage` each time — they do not append. Use `check_slow_coverage.sh` to safely test whether a slow test adds new lines on top of the fast-suite baseline, then automatically restore the baseline:

```bash
# First establish a baseline (once per session)
spin test -c -- -q

# Then probe any slow test candidate:
./check_slow_coverage.sh <test-file> <keyword> <source-file>

# Example:
./check_slow_coverage.sh scipy/stats/tests/test_resampling.py \
    test_bootstrap_against_theory scipy/stats/_resampling.py
```

The script appends the slow test's coverage, prints the updated line-by-line report for the source file, then restores the baseline. If the output shows new covered lines, the test is worth promoting.

Use these to verify that a promoted test (a) passes when run without `-m slow`, and (b) actually covers the source lines you expect, before running the full submodule coverage pass.

## Reclaiming Time: Adding `slow` Markers to Fast Tests

Promoting slow tests adds runtime to the fast suite. If the total runtime grows too much, you can reclaim time by finding currently-fast tests that are expensive but redundant — they cover code already well-exercised by other fast tests — and marking them slow.

**How to find candidates:**

1. Get per-test timing from the JSON report:
   ```bash
   spin test -s <submodule> -- --json-report --json-report-file=/tmp/report.json -q
   python3 parse_test_times.py --by-submodule  # or inspect /tmp/report.json directly
   ```
2. Look for fast tests that take >1s and whose source coverage overlaps heavily with other fast tests (i.e., no unique lines they alone cover).
3. Add `@pytest.mark.slow` to those tests and verify the fast suite still passes and coverage hasn't dropped.

The net effect should be: runtime saved by demotion ≥ runtime added by promotion.

## Workflow

The baseline has already been established by running `establish_baseline.sh` before this session. Do not re-run the full test suite yourself — it takes ~10 minutes and the baseline is already saved.

1. Run `python3 -m coverage report -m` (from `build-install/usr/lib/python3/dist-packages/`) to find **source files with uncovered lines**. Focus on files with meaningful gaps (not 100%, not test files).
2. For a source file with gaps, examine the uncovered line numbers. Identify what functionality those lines implement.
3. Search for slow-marked tests that exercise that functionality. Use `check_slow_coverage.sh` to confirm they actually cover new lines without permanently changing the baseline.
4. If a slow test covers new lines, remove its `slow` marker and add its measured duration to `TEST_TIME_DEBT.md`. If the total exceeds 0, demote a fast test to bring it back under control before continuing.
5. Commit with a message stating exactly which source file gained how many lines.
6. Repeat from step 1.

## Tracking Time Debt

Keep a file `TEST_TIME_DEBT.md` with a running total in seconds. Add the measured duration each time you make a test fast; subtract it each time you make a test slow. The total must be **≤ 0** at the end of the session.

## Success Criteria

- Fast-suite **coverage increases** compared to baseline (measured in source lines, not test lines).
- `TEST_TIME_DEBT.md` total is **≤ 0s**.
- All fast tests **pass** (exit code 0).
- No `xslow` markers were modified.
