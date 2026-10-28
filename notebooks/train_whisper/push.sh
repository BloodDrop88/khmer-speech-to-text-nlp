#!/usr/bin/env bash
# Bundle src/normalize.py into the training script and push it to Kaggle.
# Usage: ./push.sh            full run
#        ./push.sh --smoke    short test run (few steps, 32 test clips)
#        ./push.sh --build    bundle only, don't push (for local testing)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
build="$here/build"
mkdir -p "$build"

{
  cat "$root/src/normalize.py"
  echo
  grep -v '^from normalize import' "$here/train_whisper.py"
} > "$build/train_whisper.py"
cp "$here/kernel-metadata.json" "$build/"

if [[ "${1:-}" == "--smoke" ]]; then
  sed -i '' 's/^SMOKE_TEST = False/SMOKE_TEST = True/' "$build/train_whisper.py"
fi
[[ "${1:-}" == "--build" ]] && { echo "built $build/train_whisper.py"; exit 0; }
grep -q '"KAGGLE_USERNAME/' "$build/kernel-metadata.json" && { echo "Set your Kaggle username in kernel-metadata.json first"; exit 1; }

kaggle kernels push -p "$build"
