#!/bin/bash
# Pull results back from the workstation.
#
# NEVER plain `rsync -a` this direction. The record .md is written LOCALLY and is normally
# newer than the copy the last code sync pushed out; rsync -a transfers on any difference,
# not on age, so a plain pull silently reverts the record to a stale version. That happened
# on 2026-09-13 and cost 247 lines of the experiment record (recovered from git).
#
# Two guards, both needed: exclude the records themselves, and refuse to overwrite anything
# newer than the remote copy.
set -euo pipefail
RD=/home/guoj0f/repos/ProteinTTT/bindingGYM-binding-sites-analysis
LOCAL="$(git rev-parse --show-toplevel)/workstation-records/mutation-landscape-TTT/"
rsync -a -u --exclude '*.md' \
  "guoj0f@10.67.24.41:$RD/workstation-records/mutation-landscape-TTT/" "$LOCAL"
echo "pulled (records excluded, --update on)"
