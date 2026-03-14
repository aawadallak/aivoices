#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <rclone-remote> <drive-path> [namespace ...]" >&2
  echo "example: $0 gdrive-aivoices aivoices/episodes dragonball beerschool" >&2
  exit 1
fi

REMOTE="$1"
DRIVE_PATH="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EPISODES_DIR="$REPO_ROOT/episodes"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found in PATH" >&2
  exit 1
fi

RCLONE_EXCLUDES=(
  --exclude '*.zip'
  --exclude '*.rar'
  --exclude '*.7z'
)

copy_namespace() {
  local namespace="$1"
  local src="$EPISODES_DIR/$namespace"
  local dst="$REMOTE:$DRIVE_PATH/$namespace"

  if [[ ! -d "$src" ]]; then
    echo "skip $namespace: local namespace directory not found at $src" >&2
    return 0
  fi

  echo "flush raw for $namespace"
  rclone copy "$src/raw" "$dst/raw" -P --create-empty-src-dirs "${RCLONE_EXCLUDES[@]}"

  if [[ -d "$src/extracted-audio" ]]; then
    echo "flush extracted-audio for $namespace"
    rclone copy "$src/extracted-audio" "$dst/extracted-audio" -P --create-empty-src-dirs "${RCLONE_EXCLUDES[@]}"
  fi
}

if [[ $# -eq 0 ]]; then
  while IFS= read -r namespace; do
    copy_namespace "$namespace"
  done < <(find "$EPISODES_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
else
  for namespace in "$@"; do
    copy_namespace "$namespace"
  done
fi
