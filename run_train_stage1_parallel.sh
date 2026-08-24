#!/usr/bin/env bash

# Run this script from the ColorFM repository root.
# Edit the paths, GPU IDs, and non-overlapping [start, end) ranges as needed.

INPUT_DIR=""
OUTPUT_DIR=""
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

python main_train_stage1.py \
  --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" \
  --cuda 0 --start 0 --end 40 --no-progress \
  > "${LOG_DIR}/gpu0_0_40.log" 2>&1 &

python main_train_stage1.py \
  --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" \
  --cuda 0 --start 40 --end 80 --no-progress \
  > "${LOG_DIR}/gpu0_40_80.log" 2>&1 &

python main_train_stage1.py \
  --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" \
  --cuda 0 --start 80 --end 120 --no-progress \
  > "${LOG_DIR}/gpu0_80_120.log" 2>&1 &


wait
echo "All stage-1 generation tasks finished. Logs: ${LOG_DIR}"
