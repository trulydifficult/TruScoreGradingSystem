#!/usr/bin/env bash
set -euo pipefail

cd /workspace/TruScoreGradingSystem

# Ensure origin exists before any fetch/pull/push
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin https://github.com/trulydifficult/TruScoreGradingSystem.git
else
  git remote set-url origin https://github.com/trulydifficult/TruScoreGradingSystem.git
fi

# Ensure GitHub token auth is configured
if [ -n "${GH_TOKEN:-}" ]; then
  git config --global url."https://x-access-token:${GH_TOKEN}@github.com/".insteadOf "https://github.com/"
else
  echo "WARNING: GH_TOKEN is not set."
fi

git fetch origin

# Ensure work branch tracks origin/work if available
git checkout work 2>/dev/null || git checkout -b work

if git show-ref --verify --quiet refs/remotes/origin/work; then
  git branch --set-upstream-to=origin/work work 2>/dev/null || true
fi

git status
git branch -vv

echo "Maintenance complete."
