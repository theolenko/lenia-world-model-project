#!/bin/bash
# Sync local repo (code only, no data) to the Clara cluster.

CLUSTER_HOME="jv72unec@login01.sc.uni-leipzig.de:~/lenia-world-model/"
DRY_RUN=""

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY-RUN MODE — no files will be transferred ==="
fi

echo "=== Starting rsync to $CLUSTER_HOME ==="

rsync -avz --progress $DRY_RUN \
    --exclude='.git/' \
    --exclude='data/' \
    --exclude='.venv/' \
    --exclude='lenia-wm/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='experiments/' \
    --exclude='*.h5' \
    --exclude='*.npy' \
    ./ "$CLUSTER_HOME"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Transfer complete ==="
    echo "Target:  $CLUSTER_HOME"
    if [[ -n "$DRY_RUN" ]]; then
        echo "Mode:    Dry-run (no files transferred)"
    else
        echo "Mode:    Live transfer"
    fi
else
    echo "=== ERROR: rsync exited with code $EXIT_CODE ==="
    exit $EXIT_CODE
fi
