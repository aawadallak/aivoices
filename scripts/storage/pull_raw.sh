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

pull_namespace() {
  local namespace="$1"
  local dst="$EPISODES_DIR/$namespace"
  local src="$REMOTE:$DRIVE_PATH/$namespace"

  mkdir -p "$dst"

  echo "pull raw for $namespace"
  rclone copy "$src/raw" "$dst/raw" -P --create-empty-src-dirs

  echo "pull extracted-audio for $namespace"
  rclone copy "$src/extracted-audio" "$dst/extracted-audio" -P --create-empty-src-dirs || true
}

if [[ $# -eq 0 ]]; then
  echo "at least one namespace is required for pull" >&2
  exit 1
fi

for namespace in "$@"; do
  pull_namespace "$namespace"
done
