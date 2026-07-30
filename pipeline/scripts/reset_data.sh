#!/bin/bash
# Clear active generated data before rerunning the pipeline.
# Historical datav0/ and datav1/ snapshots are never modified.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ACTIVE_DATA="$ROOT/data"
PROTECTED_DIRS=("$ROOT/datav0" "$ROOT/datav1")
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash pipeline/scripts/reset_data.sh [--dry-run]

Only cleans the active data/ directory:
  - data/raw/* except clean.jsonl and seeds
  - data/final/*
  - data/processed/*

Protected and never deleted:
  - datav0/
  - datav1/
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

run_rm_file() {
    local f="$1"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would rm $f"
    else
        echo "  rm $f"
        rm "$f"
    fi
}

run_delete_files_under() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        find "$dir" -type f | sort | while read -r f; do echo "  would rm $f"; done
    else
        find "$dir" -type f -print -delete | sort | while read -r f; do echo "  rm $f"; done
    fi
}

echo "=== Active data directory ==="
echo "  $ACTIVE_DATA"
echo "=== Protected directories (never touched) ==="
for d in "${PROTECTED_DIRS[@]}"; do
    echo "  $d"
done
if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== DRY RUN: no files will be deleted ==="
fi

if [ ! -d "$ACTIVE_DATA" ]; then
    echo "Active data directory does not exist: $ACTIVE_DATA"
    exit 0
fi

echo ""
echo "=== Cleaning data/raw (preserving clean.jsonl) ==="
if [ -d "$ACTIVE_DATA/raw" ]; then
    for scene_dir in "$ACTIVE_DATA"/raw/*/; do
        [ -d "$scene_dir" ] || continue
        scene=$(basename "$scene_dir")
        for f in "$scene_dir"*.jsonl "$scene_dir"*.ckpt.json; do
            [ -f "$f" ] || continue
            fname=$(basename "$f")
            if [ "$fname" != "clean.jsonl" ]; then
                run_rm_file "$f"
            else
                echo "  keep $scene/$fname"
            fi
        done
    done
fi

echo ""
echo "=== Cleaning data/final ==="
run_delete_files_under "$ACTIVE_DATA/final"

echo ""
echo "=== Cleaning data/processed ==="
run_delete_files_under "$ACTIVE_DATA/processed"

echo ""
echo "=== Done; retained files ==="
echo "  seeds:"
if [ -d "$ACTIVE_DATA/seeds" ]; then
    find "$ACTIVE_DATA/seeds" -type f | sort | while read -r f; do echo "    $f"; done
fi
echo "  clean:"
if [ -d "$ACTIVE_DATA/raw" ]; then
    find "$ACTIVE_DATA/raw" -name "clean.jsonl" | sort | while read -r f; do echo "    $f"; done
fi
