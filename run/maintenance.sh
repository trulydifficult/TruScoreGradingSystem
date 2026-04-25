#!/usr/bin/env bash
set -euo pipefail

cd /workspace/TruScoreGradingSystem

git fetch origin

git status
git branch -vv

echo "Maintenance complete."
