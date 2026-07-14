#!/bin/bash

srun -Q --partition=all_usr_prod --mem=24G --account=cvcs2026 --immediate=10 --gres=gpu:1 --time=30:00 --pty bash
