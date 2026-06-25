#!/bin/bash
HOST=$(hostname -s)
srun -Q --immediate=10 \
  -w "$HOST" \
  --partition=all_serial \
  --account=cvcs2026 \
  --gres=gpu:1 \
  --time=60:00 \
  --pty bash