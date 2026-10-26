#!/usr/bin/env bash
# scripts/fetch_cd_chunks.sh — robustly download the CALVIN zip's central directory
# in 16 MB chunks with retries (the server drops connections frequently).
# Then scripts/fetch_zip_index.py replays the .part and finishes parsing.
set -u

URL="http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
CD_OFFSET=555080601096
CD_SIZE=229211511
CHUNK=16777216
PART="$(cd "$(dirname "$0")/.." && pwd)/data/calvin/task_ABC_D.index.part"
TMP=/tmp/cd_chunk.bin

mkdir -p "$(dirname "$PART")"
pos=$(stat -c%s "$PART" 2>/dev/null || echo 0)
echo "resume at $pos / $CD_SIZE"

while [ "$pos" -lt "$CD_SIZE" ]; do
  start=$((CD_OFFSET + pos))
  end=$((start + CHUNK - 1))
  [ "$end" -ge $((CD_OFFSET + CD_SIZE)) ] && end=$((CD_OFFSET + CD_SIZE - 1))
  want=$((end - start + 1))

  if curl -s --max-time 180 --retry 6 --retry-all-errors \
          -r "${start}-${end}" -o "$TMP" "$URL"; then
    got=$(stat -c%s "$TMP" 2>/dev/null || echo 0)
    if [ "$got" -eq "$want" ]; then
      cat "$TMP" >> "$PART"
      pos=$(stat -c%s "$PART")
      echo "$(date +%H:%M:%S) $pos / $CD_SIZE"
    else
      echo "$(date +%H:%M:%S) chunk incomplete ($got/$want), retrying"
    fi
  else
    echo "$(date +%H:%M:%S) curl failed, retrying"
  fi
done

echo "DONE: $PART complete"
