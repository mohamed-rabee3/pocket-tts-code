#!/usr/bin/env bash
# Download the Saudi TTS corpora from the Hetzner Storage Box with parallel rsync.
# Needs HBOX_HOST (user@host) and HBOX_PW in the environment. Resumable: rerun to continue.
set -euo pipefail

HOST=${HBOX_HOST:?set HBOX_HOST=user@host of the Storage Box}
SRC=saudi-tts/datasets
DST=/data/saudi-tts/raw
LOGS=/data/saudi-tts/logs
SHARDS=${SHARDS:-9}
CORPORA=(nafas maha shakhsiya sara swalif socrates youtube-001-part02-tashkeel)

ASKPASS=$(mktemp)
trap 'rm -f "$ASKPASS"' EXIT
printf '#!/bin/sh\necho "$HBOX_PW"\n' > "$ASKPASS"
chmod 700 "$ASKPASS"
export SSH_ASKPASS=$ASKPASS SSH_ASKPASS_REQUIRE=force DISPLAY=:0
SSH="ssh -p 23 -o PubkeyAuthentication=no -o ServerAliveInterval=30"

mkdir -p "$DST" "$LOGS/shards"

# 1. Per-corpus metadata (everything except the audio).
for c in "${CORPORA[@]}"; do
  rsync -a -e "$SSH" --exclude 'flac/' "$HOST:$SRC/$c/" "$DST/$c/"
done

# 2. Audio file list from each corpus's SHA256SUMS (skips the undiacritized `long` split).
LIST=$LOGS/shards/all_files.txt
: > "$LIST"
for c in "${CORPORA[@]}"; do
  awk -v c="$c" '{ sub(/^\*/, "", $2); print c "/" $2 }' "$DST/$c/SHA256SUMS" \
    | grep '/flac/' | grep -v '/flac/long/' >> "$LIST"
done
echo "$(wc -l < "$LIST") audio files listed"

# 3. Round-robin shards, one rsync per shard.
rm -f "$LOGS"/shards/shard_*
awk -v n="$SHARDS" -v dir="$LOGS/shards" '{ print > sprintf("%s/shard_%02d", dir, NR % n) }' "$LIST"
pids=()
for f in "$LOGS"/shards/shard_*; do
  rsync -a --files-from="$f" -e "$SSH" "$HOST:$SRC/" "$DST/" > "$f.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "rsync done (fail=$fail)"
exit $fail
