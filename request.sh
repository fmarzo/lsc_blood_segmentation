#!/bin/bash
#SBATCH --job-name=u_rab_18_lsc
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/cvcs2026/latent_space_cowboys/logs/train_%j.log
#SBATCH --error=/work/cvcs2026/latent_space_cowboys/logs/train_%j.log
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_2080_11G|gpu_A40_45G|gpu_K80_12G|gpu_L40S_45G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"

source /homes/$USER/cvcs2026/venv/bin/activate

script_file="train_unet_resnet18_v1p0"

echo "${script_file} execution"

cd ./lab || exit 1

python -m scripts.rabbani.${script_file} "$1"

echo "Python script completed."