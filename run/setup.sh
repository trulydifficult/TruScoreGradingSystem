#!/usr/bin/env bash
set -euo pipefail

cd /workspace/TruScoreGradingSystem

# Ensure Git remote exists
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin https://github.com/trulydifficult/TruScoreGradingSystem.git
else
  git remote set-url origin https://github.com/trulydifficult/TruScoreGradingSystem.git
fi

# Use GH_TOKEN for GitHub HTTPS operations
if [ -n "${GH_TOKEN:-}" ]; then
  git config --global url."https://x-access-token:${GH_TOKEN}@github.com/".insteadOf "https://github.com/"
else
  echo "WARNING: GH_TOKEN is not set. GitHub push/pull may fail."
fi

# Fetch remote branches
git fetch origin || true

# Stay on work branch if it exists
if git show-ref --verify --quiet refs/heads/work; then
  git checkout work
elif git show-ref --verify --quiet refs/remotes/origin/work; then
  git checkout -b work origin/work
else
  git checkout -b work
fi

# Python deps
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

# Node deps, if applicable
if [ -f package-lock.json ]; then
  npm ci
elif [ -f package.json ]; then
  npm install
fi

echo "Setup complete."
