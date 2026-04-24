#!/usr/bin/env python3
"""Parse pytest-json-report output and summarize test times."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import PurePosixPath


#SCIPY_PREFIX = "build-install/usr/lib/python3/site-packages/"
SCIPY_PREFIX = "build-install/usr/lib/python3/dist-packages/"


def get_total_duration(test):
    total = 0.0
    for phase in ("setup", "call", "teardown"):
        if phase in test:
            total += test[phase].get("duration", 0.0)
    return total


def nodeid_to_scipy_path(nodeid):
    """Return the scipy-relative file path from a nodeid, or None."""
    file_part = nodeid.split("::")[0]
    if file_part.startswith(SCIPY_PREFIX):
        return file_part[len(SCIPY_PREFIX):]
    return file_part


def submodule(scipy_path):
    """Return the 2nd-level scipy submodule (e.g. 'scipy.ndimage')."""
    parts = PurePosixPath(scipy_path).parts
    # parts[0] == 'scipy', parts[1] == submodule name
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0] if parts else "<unknown>"


def test_name(nodeid):
    """Strip parameters from nodeid to get a stable test name."""
    return re.sub(r"\[.*\]$", "", nodeid)


def load_tests(path):
    with open(path) as f:
        data = json.load(f)
    return data["tests"]


def summarize(tests, key_fn, label):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for t in tests:
        k = key_fn(t)
        totals[k] += get_total_duration(t)
        counts[k] += 1

    rows = sorted(totals.items(), key=lambda x: -x[1])
    print(f"\n{'='*70}")
    print(f"  Time by {label}")
    print(f"{'='*70}")
    print(f"{'Time (s)':>12}  {'Count':>6}  Key")
    print(f"{'-'*12}  {'-'*6}  {'-'*48}")
    for k, t in rows:
        print(f"{t:12.4f}  {counts[k]:6d}  {k}")
    print(f"{'='*70}")
    print(f"  Total: {sum(totals.values()):.4f}s across {len(totals)} {label}s")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize test times from a pytest-json-report file."
    )
    parser.add_argument(
        "report",
        nargs="?",
        default="./build-install/usr/lib/python3/dist-packages/.report.json",
        help="Path to the .report.json file",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--by-test",
        action="store_true",
        default=False,
        help="Sum times per test name (default)",
    )
    group.add_argument(
        "--by-file",
        action="store_true",
        default=False,
        help="Sum times per test file",
    )
    group.add_argument(
        "--by-submodule",
        action="store_true",
        default=False,
        help="Sum times per 2nd-level submodule (e.g. scipy.ndimage)",
    )
    group.add_argument(
        "--all",
        dest="show_all",
        action="store_true",
        default=False,
        help="Show all three groupings",
    )
    parser.add_argument(
        "--outcome",
        choices=["passed", "failed", "error", "skipped"],
        help="Filter to a specific outcome",
    )
    args = parser.parse_args()

    tests = load_tests(args.report)

    if args.outcome:
        tests = [t for t in tests if t.get("outcome") == args.outcome]

    if args.show_all:
        summarize(tests, lambda t: test_name(t["nodeid"]), "test")
        summarize(tests, lambda t: nodeid_to_scipy_path(t["nodeid"]), "file")
        summarize(tests, lambda t: submodule(nodeid_to_scipy_path(t["nodeid"])), "submodule")
    elif args.by_file:
        summarize(tests, lambda t: nodeid_to_scipy_path(t["nodeid"]), "file")
    elif args.by_submodule:
        summarize(tests, lambda t: submodule(nodeid_to_scipy_path(t["nodeid"])), "submodule")
    else:
        # default: by test
        summarize(tests, lambda t: test_name(t["nodeid"]), "test")


if __name__ == "__main__":
    main()

