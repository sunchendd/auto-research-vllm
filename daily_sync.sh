#!/usr/bin/env bash
# daily_sync.sh — Auto-commit and push research results to GitHub every 24 hours.
# Usage:
#   ./daily_sync.sh              # run once immediately, then loop every 24h
#   ./daily_sync.sh --once       # commit and push once, then exit
#   ./daily_sync.sh --interval N # loop every N seconds (default: 86400)
#
# Run in background:
#   nohup ./daily_sync.sh > sync.log 2>&1 &

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INTERVAL=86400  # 24 hours in seconds
RUN_ONCE=false

for arg in "$@"; do
  case "$arg" in
    --once) RUN_ONCE=true ;;
    --interval) shift; INTERVAL="$1" ;;
  esac
done

cd "$REPO_DIR"

# Files/dirs to track in each commit
TRACKED=(
  "results.tsv"
  "speculative_runs/"
  "patent_tracker.md"
  "research_manifest.json"
  "speculative_candidates.json"
  "program.md"
  "speculative_benchmark.py"
  "speculative_loop.py"
  "speculative_registry.py"
  ".gitignore"
  "README.md"
)

do_sync() {
  local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  local date_tag
  date_tag="$(date '+%Y-%m-%d')"

  echo "[${timestamp}] Starting sync..."

  # Pull latest changes first (non-destructive)
  git fetch origin main 2>/dev/null || git fetch origin master 2>/dev/null || true
  git pull --rebase origin main 2>/dev/null || git pull --rebase origin master 2>/dev/null || true

  # Stage tracked files (skip missing ones)
  local staged=0
  for item in "${TRACKED[@]}"; do
    if [ -e "$item" ]; then
      git add "$item" 2>/dev/null && staged=1
    fi
  done

  if [ "$staged" -eq 0 ]; then
    echo "[${timestamp}] Nothing to stage. Skipping commit."
    return
  fi

  # Check if there's anything actually changed
  if git diff --cached --quiet; then
    echo "[${timestamp}] No changes detected. Skipping commit."
    return
  fi

  # Summarize what changed
  local changed_files
  changed_files="$(git diff --cached --name-only | tr '\n' ' ')"

  # Count results rows if results.tsv exists
  local results_summary=""
  if [ -f "results.tsv" ]; then
    local row_count
    row_count=$(wc -l < results.tsv)
    results_summary=" | results.tsv: ${row_count} rows"
  fi

  # Count patent ideas
  local patent_summary=""
  if [ -f "patent_tracker.md" ]; then
    local idea_count
    idea_count=$(grep -c "^### \[" patent_tracker.md 2>/dev/null || echo "0")
    patent_summary=" | patent ideas: ${idea_count}"
  fi

  local commit_msg="research: daily sync ${date_tag}${results_summary}${patent_summary}

Changed: ${changed_files}

Auto-committed by daily_sync.sh"

  git commit -m "$commit_msg"

  # Push to remote
  if git push origin HEAD:main 2>/dev/null || git push origin HEAD:master 2>/dev/null; then
    echo "[${timestamp}] Pushed to GitHub successfully."
  else
    # First push — set upstream
    git push -u origin HEAD:main 2>/dev/null || git push -u origin HEAD:master 2>/dev/null || {
      echo "[${timestamp}] Push failed. Will retry next cycle."
    }
  fi
}

# Main loop
echo "=== daily_sync.sh started at $(date) ==="
echo "=== Repo: ${REPO_DIR} ==="
echo "=== Interval: ${INTERVAL}s | Run-once: ${RUN_ONCE} ==="

do_sync

if [ "$RUN_ONCE" = true ]; then
  echo "=== Done (--once mode) ==="
  exit 0
fi

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sleeping ${INTERVAL}s until next sync..."
  sleep "$INTERVAL"
  do_sync
done
