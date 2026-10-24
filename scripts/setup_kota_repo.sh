#!/usr/bin/env bash
# Restores the Kota repository at the audited commit into audit/repo from the bundled copy.
# Falls back to GitHub if the bundle is missing. No network is needed when the bundle is present.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=f76754b5bcff1a1d7a50c2f1073a4b6d44bd985c
if [ ! -d audit/repo/.git ]; then
  if [ -f third_party/kota_repo_f76754b.bundle ]; then
    git clone -q third_party/kota_repo_f76754b.bundle audit/repo
  else
    git clone -q https://github.com/Tkota10/Senior-Thesis-Multi-LLM-Resolution audit/repo
  fi
fi
git -C audit/repo checkout -q "$COMMIT"
test "$(git -C audit/repo rev-parse HEAD)" = "$COMMIT"
echo "audit/repo is at $COMMIT"
