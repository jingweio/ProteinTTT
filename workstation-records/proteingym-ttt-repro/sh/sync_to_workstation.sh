#!/bin/bash
# Local -> workstation code sync.
#
# DESIGN RULE (set 2026-08-25): never `rsync --delete`.
#   The remote legitimately holds files the local tree does not -- job logs, caches,
#   anything a run writes inside the tree. A blanket --delete treats all of those as
#   "deleted locally" and removes them; on 2026-08-24 it removed the log of a running
#   job mid-sweep. Full mirroring is the wrong model: the two trees are *supposed* to
#   differ.
#
#   But every substantive local change to code -- add, modify, AND delete -- must still
#   reach the remote before anything runs there, or Python can import a stale module and
#   produce confidently wrong results. So deletions are propagated *explicitly*: the set
#   of git-tracked files is recorded on the remote as .synced_manifest, and each sync
#   removes exactly the tracked files that disappeared since the previous sync. Nothing
#   else is ever deleted, and every removal is printed.
set -euo pipefail
SRC="$(git rev-parse --show-toplevel)/"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
HOST=guoj0f@10.67.24.41
REMOTE="/home/guoj0f/repos/ProteinTTT/${BRANCH//\//-}"
# One multiplexed connection for the whole script. Without this each sync opens ~6
# separate sessions and sshd's rate limiting starts resetting them mid-run.
CM=/tmp/.ssh-cm-workstation-$$
SSH="ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=60"
trap '$SSH -O exit "$HOST" 2>/dev/null || true' EXIT
EX=(--exclude .git --exclude .claude/worktrees --exclude .synced_commit --exclude .synced_manifest)

echo "src   : $SRC"
echo "branch: $BRANCH"
echo "dest  : $HOST:$REMOTE"

# ---- 1. copy additions and modifications (no --delete) --------------------------
rsync -a -e "$SSH" "${EX[@]}" "$SRC" "$HOST:$REMOTE/"

# ---- 2. propagate deletions of tracked files, explicitly -----------------------
git -C "$SRC" ls-files | sort > /tmp/.synced_manifest.new
$SSH "$HOST" "cat $REMOTE/.synced_manifest 2>/dev/null || true" | sort > /tmp/.synced_manifest.old
removed=$(comm -23 /tmp/.synced_manifest.old /tmp/.synced_manifest.new || true)
if [ -n "$removed" ]; then
  echo "propagating deletions of tracked files:"
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    echo "  rm $f"
    $SSH "$HOST" "rm -f -- '$REMOTE/$f'"
  done <<< "$removed"
else
  echo "no tracked files deleted since last sync"
fi
rsync -a -e "$SSH" /tmp/.synced_manifest.new "$HOST:$REMOTE/.synced_manifest"

# ---- 3. provenance stamp -------------------------------------------------------
git -C "$SRC" rev-parse HEAD > /tmp/.synced_commit
git -C "$SRC" status --porcelain >> /tmp/.synced_commit
rsync -a -e "$SSH" /tmp/.synced_commit "$HOST:$REMOTE/.synced_commit"
echo "stamped: $(head -1 /tmp/.synced_commit)"

# ---- 4. verify every tracked file is present and identical ---------------------
# Transfer-check only, still no --delete: output must be empty.
echo "verification (every tracked file up to date; must be empty):"
# Directory entries are reported whenever a directory's mtime changed (e.g. because a
# file was deleted from it), which is not file drift -- drop anything ending in "/".
# rsync's own exit status must be checked separately: a dropped connection produces
# empty output, which would otherwise read as "no drift" and pass the gate.
if ! git -C "$SRC" ls-files -z \
     | rsync -avn -e "$SSH" --files-from=- --from0 "$SRC" "$HOST:$REMOTE/" > /tmp/.drift 2>&1; then
  echo "verification rsync FAILED -- cannot certify alignment:"; cat /tmp/.drift; exit 1
fi
drift=$(grep -vE '^(sending|sent |total size|building file list|.*/$|$)' /tmp/.drift || true)
if [ -n "$drift" ]; then echo "$drift"; echo "NOT ALIGNED"; exit 1; fi
echo "ALIGNED ($(wc -l < /tmp/.synced_manifest.new) tracked files)"
