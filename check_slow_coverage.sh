#!/usr/bin/env bash
# Usage: ./check_slow_coverage.sh <test-file> <keyword> <source-file>
#
# Runs a specific slow test on top of the current fast-suite baseline,
# reports which new lines it covers in the given source file, then
# restores the baseline.
#
# Example:
#   ./check_slow_coverage.sh scipy/stats/tests/test_resampling.py \
#       test_bootstrap_against_theory scipy/stats/_resampling.py

set -e

TEST_FILE="$1"
KEYWORD="$2"
SOURCE_FILE="$3"

if [[ -z "$TEST_FILE" || -z "$KEYWORD" || -z "$SOURCE_FILE" ]]; then
    echo "Usage: $0 <test-file> <keyword> <source-file>"
    exit 1
fi

COVDIR="build-install/usr/lib/python3/dist-packages"
COVFILE="$COVDIR/.coverage"
BASELINE="$COVDIR/.coverage.baseline"

if [[ ! -f "$COVFILE" ]]; then
    echo "No .coverage baseline found. Run 'spin test -c -- -q' first."
    exit 1
fi

echo "==> Saving baseline coverage..."
cp "$COVFILE" "$BASELINE"

echo "==> Running slow test: $KEYWORD"
spin test -c "$TEST_FILE" -- -m slow -k "$KEYWORD" --cov-append -q

echo ""
echo "==> New lines covered in $SOURCE_FILE:"
cd "$COVDIR"
python3 -m coverage report -m --include="*/$SOURCE_FILE" 2>/dev/null
cd - > /dev/null

echo ""
echo "==> Restoring baseline..."
cp "$BASELINE" "$COVFILE"
echo "Done. Baseline restored. Edit the test file and re-run without restoring if you want to keep the change."
