#!/bin/bash
#SBATCH --job-name=lsc_train
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00
#SBATCH --output=/work/cvcs2026/latent_space_cowboys/logs/train_%j.txt
#SBATCH --error=/work/cvcs2026/latent_space_cowboys/logs/train_%j.err
#SBATCH --account=cvcs2026


source /homes/$USER/cvcs2026/venv/bin/activate

echo "launching the python script.." 

# python ./lsc_blood_segmentation/lab/scripts/train.py --config config.yaml
cd ./lab
python -m scripts.train

echo "launching the python script.."