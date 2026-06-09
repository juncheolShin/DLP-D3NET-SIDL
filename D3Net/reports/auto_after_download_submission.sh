#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/user/Downloads/DLP-D3NET-SIDL}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/home/user/Downloads}"
ZIP_GLOB="${ZIP_GLOB:-test-*.zip}"
PART_GLOB="${PART_GLOB:-test-*.zip.part}"
EXTRACT_ROOT="${EXTRACT_ROOT:-$REPO_ROOT/D3Net/data/SIDL/leaderboard_input_from_zip}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/submissions/sidl_d3net_leaderboard_auto}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$REPO_ROOT/submissions/sidl_d3net_leaderboard_auto.zip}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/submissions}"
TRAIN_PATTERN="${TRAIN_PATTERN:-D3Net_SIDL}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/auto_after_download_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO_ROOT"
START_EPOCH="$(date +%s)"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

find_train_pgid() {
  ps -eo pid,pgid,cmd | awk -v pat="$TRAIN_PATTERN" '
    $0 ~ pat && $0 !~ /awk/ {
      print $2
      exit
    }
  '
}

wait_for_downloads() {
  log "Waiting for browser download to finish in $DOWNLOAD_DIR ($ZIP_GLOB)"
  local last_sig=""
  local stable_count=0

  while true; do
    shopt -s nullglob
    local parts=("$DOWNLOAD_DIR"/$PART_GLOB)
    local zips=("$DOWNLOAD_DIR"/$ZIP_GLOB)
    shopt -u nullglob

    if (( ${#parts[@]} > 0 )); then
      log "Download still active: ${parts[*]}"
      stable_count=0
      sleep "$POLL_SECONDS"
      continue
    fi

    if (( ${#zips[@]} == 0 )); then
      log "No matching ZIP files yet."
      stable_count=0
      sleep "$POLL_SECONDS"
      continue
    fi

    local bad=0
    local sig=""
    for zip_file in "${zips[@]}"; do
      if [[ ! -s "$zip_file" ]]; then
        bad=1
      fi
      sig+="$(stat -c '%n:%s:%Y' "$zip_file")"$'\n'
    done
    if (( bad )); then
      log "ZIP file exists but at least one is empty; waiting."
      stable_count=0
      sleep "$POLL_SECONDS"
      continue
    fi

    if [[ "$sig" == "$last_sig" ]]; then
      stable_count=$((stable_count + 1))
    else
      stable_count=0
      last_sig="$sig"
    fi

    if (( stable_count >= 2 )); then
      log "ZIP files look complete:"
      printf '%s\n' "${zips[@]}"
      return 0
    fi

    log "ZIP files present; waiting for size stability."
    sleep "$POLL_SECONDS"
  done
}

extract_downloads() {
  log "Extracting ZIP files to $EXTRACT_ROOT"
  mkdir -p "$EXTRACT_ROOT"
  shopt -s nullglob
  local zips=("$DOWNLOAD_DIR"/$ZIP_GLOB)
  shopt -u nullglob
  for zip_file in "${zips[@]}"; do
    log "Unzipping $zip_file"
    unzip -oq "$zip_file" -d "$EXTRACT_ROOT"
  done
}

detect_input_root() {
  local candidate
  for candidate in \
    "$EXTRACT_ROOT/test" \
    "$EXTRACT_ROOT" \
    "$EXTRACT_ROOT/SIDL/test" \
    "$EXTRACT_ROOT/sidl/test"
  do
    if [[ -d "$candidate/clean" || -d "$candidate/dust" || -d "$candidate/finger" || -d "$candidate/fingerprint" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  candidate="$(find "$EXTRACT_ROOT" -type d \( -name clean -o -name dust -o -name finger -o -name fingerprint \) -printf '%h\n' | sort -u | head -1)"
  if [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  return 1
}

wait_for_checkpoint() {
  log "Waiting for a checkpoint newer than script start."
  local checkpoint=""
  while true; do
    checkpoint="$(find "$REPO_ROOT/D3Net/ckpt" -type f \( -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' \) -newermt "@$START_EPOCH" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
    if [[ -n "$checkpoint" && -s "$checkpoint" ]]; then
      sleep 5
      if [[ -s "$checkpoint" ]]; then
        log "Detected checkpoint: $checkpoint"
        printf '%s\n' "$checkpoint"
        return 0
      fi
    fi
    sleep "$POLL_SECONDS"
  done
}

pause_training() {
  local pgid="$1"
  if [[ -n "$pgid" ]]; then
    log "Pausing training process group $pgid"
    kill -STOP "-$pgid"
  else
    log "No training process group found; continuing without pause."
  fi
}

resume_training() {
  local pgid="$1"
  if [[ -n "$pgid" ]]; then
    log "Resuming training process group $pgid"
    kill -CONT "-$pgid" || true
  fi
}

validate_inputs() {
  local input_root="$1"
  log "Validating input tree: $input_root"
  find "$input_root" -type f -name '*.png' | wc -l
  find "$input_root" -maxdepth 3 -type d | sort | sed -n '1,80p'
}

run_submission() {
  local input_root="$1"
  local checkpoint="$2"
  log "Running submission inference"
  log "Input root: $input_root"
  log "Checkpoint: $checkpoint"
  log "Output root: $OUTPUT_ROOT"
  log "Archive: $ARCHIVE_PATH"

  python "$REPO_ROOT/D3Net/reports/make_sidl_submission.py" \
    --input-root "$input_root" \
    --checkpoint "$checkpoint" \
    --output-root "$OUTPUT_ROOT" \
    --archive "$ARCHIVE_PATH" \
    --device cuda \
    --overwrite

  log "Submission archive created: $ARCHIVE_PATH"
  python - <<PY
from pathlib import Path
import zipfile
archive = Path("$ARCHIVE_PATH")
with zipfile.ZipFile(archive) as zf:
    names = zf.namelist()
    pngs = [n for n in names if n.lower().endswith(".png")]
    tops = sorted({n.split("/", 1)[0] for n in names if "/" in n})
print("archive", archive)
print("png_count", len(pngs))
print("top_level", ",".join(tops))
print("has_fingerprint", any(n.startswith("fingerprint/") for n in names))
print("has_finger", any(n.startswith("finger/") for n in names))
PY
}

main() {
  log "Auto workflow started. Log: $LOG_FILE"
  wait_for_downloads
  extract_downloads
  local input_root
  input_root="$(detect_input_root)"
  validate_inputs "$input_root"
  local checkpoint
  checkpoint="$(wait_for_checkpoint)"
  local pgid
  pgid="$(find_train_pgid || true)"
  trap 'resume_training "'"$pgid"'"' EXIT
  pause_training "$pgid"
  run_submission "$input_root" "$checkpoint"
  resume_training "$pgid"
  trap - EXIT
  log "Auto workflow completed."
}

main "$@"
