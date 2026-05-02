#!/usr/bin/env bash
set -euo pipefail

readonly ENV_NAME="gnr_project_env"
readonly REPO_URL="${REPO_URL:-https://github.com/souparna21/gnr638-project-2.git}"

echo "==> Step 1/4: fetch repo content into current directory"
if [ -f "inference.py" ]; then
  echo "    repo content already present; skipping fetch"
else
  TMP="$(mktemp -d -t gnr638-clone-XXXXXX)"
  trap 'rm -rf "$TMP"' EXIT
  git clone --depth 1 "$REPO_URL" "$TMP"
  shopt -s dotglob
  for entry in "$TMP"/*; do
    base="$(basename "$entry")"
    if [ "$base" = "setup.bash" ]; then
      continue
    fi
    mv -f "$entry" "."
  done
  shopt -u dotglob
fi

echo "==> Step 2/4: ensure conda is on PATH"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first."
  exit 1
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "==> Step 3/4: create conda env '${ENV_NAME}' with Python 3.11"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "    env '${ENV_NAME}' already exists; skipping create"
else
  conda env create -f environment.yml
fi

echo "==> Step 4/4: download model weights (~5 GB; internet required)"
conda run --no-capture-output -n "${ENV_NAME}" python download_weights.py

echo "==> DONE. Run inference with:"
echo "      conda activate ${ENV_NAME}"
echo "      python inference.py --test_dir <absolute_path_to_test_dir>"
