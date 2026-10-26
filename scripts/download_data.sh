#!/usr/bin/env bash
# scripts/download_data.sh — Phase 0 data acquisition (NO GPU REQUIRED).
#
# CALVIN full zips are huge (task_ABC_D = 517 GB); we never download them wholesale.
# Instead we parse the zip's central directory over HTTP range requests and selectively
# extract the post-training-scale subset (a few hundred env-A episodes + metadata):
#
#   python scripts/fetch_zip_index.py \
#       --url http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip \
#       --out data/calvin/task_ABC_D.index.json        # ~230 MB, resumable
#   python scripts/selective_extract.py --index data/calvin/task_ABC_D.index.json \
#       --pattern 'task_ABC_D/training/ep_start_end_ids.npy' \
#       --pattern 'task_ABC_D/training/ep_lens.npy' \
#       --pattern 'task_ABC_D/training/lang_annotations/auto_lang_ann.npy' \
#       --pattern 'task_ABC_D/training/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy' \
#       --pattern 'task_ABC_D/validation/ep_start_end_ids.npy' \
#       --pattern 'task_ABC_D/validation/lang_annotations/auto_lang_ann.npy' \
#       --out-dir data/calvin/meta
#   python scripts/plan_calvin_subset.py \
#       --training-meta data/calvin/meta/training \
#       --validation-meta data/calvin/meta/validation
#   python scripts/selective_extract.py --index data/calvin/task_ABC_D.index.json \
#       --pattern 'task_ABC_D/training/episode_*.npz' \
#       --episode-range <LO> <HI> --stride <N> --out-dir data/calvin/raw
#   python scripts/preprocess_data.py --dataset calvin \
#       --raw data/calvin/raw --lang-ann data/calvin/meta/training/lang_annotations/auto_lang_ann.npy
#
# see README §1 for the exact episode selection (env A = steps [1191339, 1795044]).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA/calvin" "$DATA/bridge" "$DATA/csi"

echo "==> CALVIN debug dataset (1.3 GB; pipeline smoke tests only — 1 episode per split)"
if [ ! -d "$DATA/calvin/calvin_debug_dataset" ]; then
  wget -q --show-progress -O "$DATA/calvin/calvin_debug_dataset.zip" \
    "http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip" \
    || curl -L -o "$DATA/calvin/calvin_debug_dataset.zip" \
    "http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip"
  unzip -q "$DATA/calvin/calvin_debug_dataset.zip" -d "$DATA/calvin/"
fi

echo "==> scene_info fix (525 B) — env-A/B/C step ranges for task_ABC_D training"
if [ ! -f "$DATA/calvin/scene_info.npy" ]; then
  curl -sL -o /tmp/task_ABC_D_scene_info.zip \
    "http://calvin.cs.uni-freiburg.de/scene_info_fix/task_ABC_D_scene_info.zip"
  unzip -q -o /tmp/task_ABC_D_scene_info.zip -d "$DATA/calvin/meta/"
fi

echo "==> BridgeData V2 (subsampled) — pulled by preprocess_data.py (HuggingFace)"
echo "    run: python scripts/preprocess_data.py --dataset bridge --max-trajectories 5000"

echo "Done. Full CALVIN subset extraction via fetch_zip_index + selective_extract (see header)."
