#!/bin/bash
#SBATCH --job-name=lsc_train
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/work/cvcs2026/latent_space_cowboys/logs/train_%j.log
#SBATCH --error=/work/cvcs2026/latent_space_cowboys/logs/train_%j.log
#SBATCH --account=cvcs2026

source /homes/$USER/cvcs2026/venv/bin/activate

echo "Launching the Python script..."

cd ./lab || exit 1
python -m scripts.train_unet $1

echo "Python script completed."