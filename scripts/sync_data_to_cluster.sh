#!/bin/bash
# Transfer HDF5 data files to ~/lenia-world-model/data/ on the cluster.
# WARNING: Files are several GB — transfer may take several minutes.

CLUSTER_USER="jv72unec"
CLUSTER_HOST="login01.sc.uni-leipzig.de"
CLUSTER_DATA_DIR="~/lenia-world-model/data/"
LOCAL_DATA_DIR="data/"

FILES=(
    "lenia_train_chunked.h5"
    "lenia_val_chunked.h5"
    "v2_lenia_train_chunked.h5"
    "v2_lenia_val_chunked.h5"
)

echo "=== Data transfer to cluster ==="
echo "WARNING: Transfer of large files may take several minutes."
echo "rsync uses --partial, so you can interrupt with Ctrl+C and resume later."
echo ""

for FILE in "${FILES[@]}"; do
    LOCAL_PATH="$LOCAL_DATA_DIR/$FILE"
    REMOTE_PATH="$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DATA_DIR/$FILE"

    if [ ! -f "$LOCAL_PATH" ]; then
        echo "WARNING: $LOCAL_PATH not found — skipping."
        continue
    fi

    echo "--- Transferring: $FILE ---"
    rsync -avz --partial --progress "$LOCAL_PATH" "$REMOTE_PATH" || {
        echo "ERROR: Transfer of $FILE failed."
        exit 1
    }
done

echo ""
echo "=== Sanity check: comparing file sizes ==="
for FILE in "${FILES[@]}"; do
    LOCAL_PATH="$LOCAL_DATA_DIR/$FILE"
    if [ ! -f "$LOCAL_PATH" ]; then
        continue
    fi

    LOCAL_SIZE=$(du -sh "$LOCAL_PATH" | cut -f1)
    REMOTE_SIZE=$(ssh "$CLUSTER_USER@$CLUSTER_HOST" "du -sh $CLUSTER_DATA_DIR/$FILE 2>/dev/null | cut -f1" 2>/dev/null || echo "n/a")

    echo "$FILE"
    echo "  Local:    $LOCAL_SIZE"
    echo "  Cluster:  $REMOTE_SIZE"

    if [ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]; then
        echo "  WARNING: sizes differ — transfer may be incomplete."
    else
        echo "  OK"
    fi
done

echo ""
echo "=== Data transfer complete. ==="
