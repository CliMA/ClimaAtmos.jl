#!/usr/bin/env bash
#
set -euo pipefail

# List of whitelist write directories
# The paths must be both absolute and physical!
WRITE_DIRS=(
    $JULIA_DEPOT_PATH # Need to write julia files
    /home/mak60/.local/share/opencode
    # /run/user/$(id -u)
)

CWD="$(pwd -P)"

args=(
    bwrap
    # Shared kernel namespaces — NOTE: no --unshare-cgroup, NVIDIA driver
    # uses cgroups to enforce GPU access and blocks processes in a new
    # cgroup namespace ("GPU access blocked by the operating system").
    --unshare-user
    --unshare-pid
    --share-net
    # Host system
    --ro-bind /bin /bin
    --ro-bind /lib /lib
    --ro-bind /lib64 /lib64
    --ro-bind /usr /usr
    --ro-bind /etc /etc
    --ro-bind /home /home
    --ro-bind /opt /opt
    --ro-bind /var /var
    --ro-bind /clima /clima
    --ro-bind $HOME $HOME
    --ro-bind /net/sampo/data1/mak60 /net/sampo/data1/mak60
    # Pseudo-filesystems
    --proc /proc
    --dev /dev
    --tmpfs /tmp
    # NVIDIA driver needs /sys to verify cgroup device access
    --ro-bind /sys /sys
)

# Get access to the GPUs (dev-bind preserves device node type/permissions)
for dev in /dev/nvidia*; do
    args+=( --dev-bind "$dev" "$dev" )
done

# CWD should be writable
args+=( --bind "$CWD" "$CWD" )

# Add Whitelist
for d in "${WRITE_DIRS[@]}"; do
    [[ -d "$d" ]] || { echo "warn: '$d' not a directory, skipping" >&2; continue; }
    [[ "$d" = "$(readlink -f "$d")" ]] || { echo "warn: '$d' not a physical path, skipping" >&2; continue; }
    args+=( --bind "$d" "$d" )
done

# Launch the shell
exec "${args[@]}" -- $SHELL
