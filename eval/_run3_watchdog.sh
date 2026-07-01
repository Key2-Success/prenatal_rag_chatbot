#!/usr/bin/env bash
# Run the RAGAS eval 3× sequentially, each guarded by a disk watchdog.
# If free space on / drops below FLOOR_MB, kill the eval immediately (the
# eval needs ~4.5GB scratch from reranker temp + RAGAS; a full disk OOM-
# SIGKILLs it silently, leaving a 0-byte log — the watchdog makes it loud).
set -u
cd "$(dirname "$0")/.."
FLOOR_MB=500

free_mb() { df -m / | awk 'NR==2{print $4}'; }

for i in 1 2 3; do
  echo "======== RUN $i/3 — free $(free_mb)MB at start ========"
  if [ "$(free_mb)" -lt "$FLOOR_MB" ]; then
    echo "ABORT: only $(free_mb)MB free (< ${FLOOR_MB}MB) before run $i — stopping." >&2
    exit 2
  fi

  env -u ANTHROPIC_API_KEY .venv/bin/python -m eval.ragas_eval \
    --runs 1 \
    -m "temp0-verify run $i/3: llm_temperature 0 (was 0.3). Measuring whether answer-side regeneration noise was the cause of faithfulness swing. Answer model gpt-4.1-mini, validator v2.4." &
  EVAL_PID=$!

  # Watchdog: poll disk; kill the eval process tree if we cross the floor.
  ( while kill -0 "$EVAL_PID" 2>/dev/null; do
      f=$(free_mb)
      if [ "$f" -lt "$FLOOR_MB" ]; then
        echo "WATCHDOG: ${f}MB free (< ${FLOOR_MB}MB) — killing eval PID $EVAL_PID" >&2
        kill -TERM "$EVAL_PID" 2>/dev/null
        sleep 2
        kill -KILL "$EVAL_PID" 2>/dev/null
        exit 0
      fi
      sleep 3
    done ) &
  WATCH_PID=$!

  wait "$EVAL_PID"
  EVAL_RC=$?
  kill "$WATCH_PID" 2>/dev/null
  wait "$WATCH_PID" 2>/dev/null

  echo "-------- RUN $i/3 exited rc=$EVAL_RC — free $(free_mb)MB after --------"
  if [ "$EVAL_RC" -ne 0 ]; then
    echo "Run $i failed (rc=$EVAL_RC) — stopping the sweep so we don't score on a bad run." >&2
    exit "$EVAL_RC"
  fi
done
echo "======== ALL 3 RUNS COMPLETE — free $(free_mb)MB ========"
