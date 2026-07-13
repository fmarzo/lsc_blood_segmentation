#!/bin/bash

srun --partition=all_usr_prod --account=cvcs2026 --immediate=10 --gres=gpu:1 --time=30:00 --pty bash
