#!/bin/bash
set -e

echo "[SETUP] Syncing repo and environment for my_project..."

# --- Config ---
REPO_DIR="$HOME/my_project"
PYTHON_MODULE="python/3.12.11-gcc-11.3.1-zbb5sqy"

# --- Load Python ---
module purge
module load $PYTHON_MODULE

# --- Pull latest changes ---
cd "$REPO_DIR"
echo "[GIT] Pulling latest changes..."
git fetch origin main
git reset --hard origin/main
echo "[GIT] Repo is up-to-date."

# --- Ensure Poetry is installed ---
if ! command -v poetry &> /dev/null; then
  echo "[SETUP] Installing poetry locally..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- Check for environment changes ---
ENV_CHANGED=false
if [[ ! -d "$REPO_DIR/.venv" ]]; then
  echo "[ENV] No .venv found — creating new environment..."
  ENV_CHANGED=true
else
  if [[ "$REPO_DIR/pyproject.toml" -nt "$REPO_DIR/.venv" ]] || [[ "$REPO_DIR/poetry.lock" -nt "$REPO_DIR/.venv" ]]; then
    echo "[ENV] Detected changes in pyproject.toml or poetry.lock."
    ENV_CHANGED=true
  fi
fi

# --- (Re)create environment if needed ---
if [ "$ENV_CHANGED" = true ]; then
  echo "[ENV] Updating environment..."
  poetry config virtualenvs.in-project true
  poetry install --no-root --sync
  echo "[ENV] Environment updated successfully."
else
  echo "[ENV] No changes detected — skipping reinstallation."
fi

echo "[SETUP] Environment ready."
