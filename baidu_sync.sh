#!/usr/bin/env bash
# Move files to/from Baidu Pan (百度网盘).
#
# Baidu Pan has no rsync or SSH access -- it's a proprietary web/API
# storage service, not something rsync can talk to directly. rclone's
# "baidunetdisk" backend is the closest equivalent: `rclone sync` gives you
# the same delta-transfer behaviour rsync would (only changed files move,
# checksums, --dry-run), just over Baidu's API instead of a socket.
#
# Typical use on a rented GPU box (AutoDL etc.): pull input videos down
# from Baidu Pan before a job, push finished depth videos back up after.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="${BAIDU_REMOTE:-baidupan}"
REMOTE_DIR="${BAIDU_REMOTE_DIR:-depth-anything}"

usage() {
  cat <<EOF
Usage: $0 <command> [args]

Commands:
  install                      Install rclone if it's not already on PATH
  configure                    Set up the '$REMOTE' rclone remote
  push <local-path> [subdir]   Upload local-path to $REMOTE:$REMOTE_DIR/[subdir]
  pull <subdir> <local-path>   Download $REMOTE:$REMOTE_DIR/[subdir] to local-path
  ls   [subdir]                List files under $REMOTE:$REMOTE_DIR/[subdir]

Environment:
  BAIDU_REMOTE      rclone remote name (default: baidupan)
  BAIDU_REMOTE_DIR  base path on Baidu Pan (default: depth-anything)

Examples:
  $0 push storage/outputs results     # upload finished depth videos
  $0 pull inputs storage/uploads      # fetch source videos to process
EOF
}

require_rclone() {
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not found. Run '$0 install' first (or see https://rclone.org/install/)." >&2
    exit 1
  fi
}

require_remote() {
  if ! rclone listremotes | grep -qx "${REMOTE}:"; then
    echo "rclone remote '${REMOTE}:' is not configured. Run '$0 configure' first." >&2
    exit 1
  fi
}

cmd_install() {
  if command -v rclone >/dev/null 2>&1; then
    echo "rclone already installed: $(rclone version | head -n1)"
    return
  fi
  echo "Installing rclone..."
  curl -fsSL https://rclone.org/install.sh | sudo bash
}

cmd_configure() {
  require_rclone
  cat <<EOF
Setting up the '${REMOTE}' rclone remote for Baidu Pan.

Baidu's login is OAuth (needs a browser), but a rented GPU box is usually
headless. When 'rclone config' asks "Use auto config?", answer 'n' -- it
will print an 'rclone authorize "baidunetdisk"' command. Run THAT command
on a machine that has a browser, log into Baidu Pan when the page opens,
then paste the token it prints back into this prompt.

Starting 'rclone config': choose 'n' for a new remote, name it
'${REMOTE}', and pick 'baidunetdisk' as the storage type.
EOF
  rclone config
}

cmd_push() {
  local local_path="${1:?local path required}"
  local subdir="${2:-$(basename "$local_path")}"
  require_rclone
  require_remote
  echo "Uploading ${local_path} -> ${REMOTE}:${REMOTE_DIR}/${subdir}"
  rclone sync "$local_path" "${REMOTE}:${REMOTE_DIR}/${subdir}" --progress
}

cmd_pull() {
  local subdir="${1:?remote subdir required}"
  local local_path="${2:?local path required}"
  require_rclone
  require_remote
  mkdir -p "$local_path"
  echo "Downloading ${REMOTE}:${REMOTE_DIR}/${subdir} -> ${local_path}"
  rclone sync "${REMOTE}:${REMOTE_DIR}/${subdir}" "$local_path" --progress
}

cmd_ls() {
  local subdir="${1:-}"
  require_rclone
  require_remote
  rclone lsf "${REMOTE}:${REMOTE_DIR}/${subdir}"
}

case "${1:-}" in
  install)   cmd_install ;;
  configure) cmd_configure ;;
  push)      shift; cmd_push "$@" ;;
  pull)      shift; cmd_pull "$@" ;;
  ls)        shift; cmd_ls "$@" ;;
  *)         usage; exit 1 ;;
esac
