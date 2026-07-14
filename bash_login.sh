#!/bin/bash

srun -Q --partition=all_serial --mem=24G -w ailb-login-02 --account=cvcs2026 --immediate=10 --gres=gpu:1 --time=30:00 --pty bash
