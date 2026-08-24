#!/bin/bash
# Local -> workstation code sync (workstation-usage §3-2), with one hard-won extra rule.
#
# WHY THE *.out / *.err EXCLUDES ARE NOT OPTIONAL:
#   Job logs live inside workstation-records/<project>/ but are gitignored, so they exist
#   only on the remote.  A bare `rsync --delete` therefore DELETES THE LOG OF A RUNNING JOB.
#   The job survives (its fd stays open on the unlinked inode) but the log is unreachable
#   for the rest of the run.  Hit on 2026-08-24 during the S0 baseline sweep.
#   Belt and braces: launch scripts also write their real logs to /data (outside this tree).
set -euo pipefail
SRC="$(git rev-parse --show-toplevel)/"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
DEST="guoj0f@10.67.24.41:/home/guoj0f/repos/ProteinTTT/${BRANCH//\//-}/"
EX=(--exclude .git --exclude .claude/worktrees --exclude .synced_commit
    --exclude '*.out' --exclude '*.err' --exclude 'results/')

echo "src   : $SRC"
echo "branch: $BRANCH"
echo "dest  : $DEST"

rsync -a --delete "${EX[@]}" "$SRC" "$DEST"

git -C "$SRC" rev-parse HEAD > /tmp/.synced_commit
git -C "$SRC" status --porcelain >> /tmp/.synced_commit
rsync -a /tmp/.synced_commit "$DEST"
echo "stamped: $(head -1 /tmp/.synced_commit)"

echo "verification dry-run (must be empty):"
out=$(rsync -avn --delete "${EX[@]}" "$SRC" "$DEST" \
      | grep -vE '^(sending|sent |total size|building file list|\./$|$)' || true)
if [ -n "$out" ]; then echo "$out"; echo "NOT ALIGNED"; exit 1; fi
echo "ALIGNED"
