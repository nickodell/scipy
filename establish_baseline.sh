#!/usr/bin/env bash
# Run this once before starting the optimization session to establish
# a coverage baseline. Takes ~10 minutes.
#
# Usage: ./establish_baseline.sh

set -e

COVDIR="build-install/usr/lib/python3/dist-packages"

echo "==> Building SciPy..."
spin build

echo "==> Running fast test suite with coverage (this takes ~10 minutes)..."
spin test -c -- -q

echo "==> Saving baseline..."
cp "$COVDIR/.coverage" "$COVDIR/.coverage.baseline"

echo ""
echo "==> Baseline coverage summary:"
cd "$COVDIR"
python3 -m coverage report | tail -3
cd - > /dev/null

echo ""
echo "Baseline saved to $COVDIR/.coverage.baseline"
echo "You can now run the agent."
