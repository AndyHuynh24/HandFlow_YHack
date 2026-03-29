#!/bin/bash
set -e

echo "=== Training ==="
python scripts/train.py --architecture tcn --epochs "${EPOCHS:-100}"

echo "=== Training complete ==="

if [ -z "$GH_TOKEN" ]; then
    echo "No GH_TOKEN set, skipping push"
    echo "Model at /workspace/project/models/hand_action.keras"
    tail -f /dev/null
fi

echo "=== Pushing to GitHub ==="

# Clone fresh repo (needed because Dockerfile only copies files, no .git)
cd /tmp
git clone "https://${GH_TOKEN}@github.com/AndyHuynh24/HandFlow.git" push_repo
cd push_repo

# Create new branch
BRANCH="trained-model-$(date '+%Y%m%d-%H%M%S')"
git checkout -b "$BRANCH"

# Copy trained model
cp /workspace/project/models/hand_action.keras models/hand_action.keras

# Copy logs if they exist
cp -r /workspace/project/logs/ logs/ 2>/dev/null || true

git config user.email "andy.hb.huynh@gmail.com"
git config user.name "HandFlow Training Bot"

git add models/hand_action.keras
git add logs/ 2>/dev/null || true
git commit -m "Trained model $(date '+%Y-%m-%d %H:%M UTC')"

git push origin "$BRANCH"

echo "=== Pushed to branch: $BRANCH ==="
echo "=== Create PR at: https://github.com/AndyHuynh24/HandFlow/compare/$BRANCH ==="

echo "=== Done. Container idle. ==="
tail -f /dev/null
