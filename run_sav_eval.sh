#!/bin/bash
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8
#SBATCH --job-name=videoseal_sav_155
#SBATCH --output=logs/eval_sav_155_%j.out
#SBATCH --error=logs/eval_sav_155_%j.err

echo "Starting Video Seal evaluation on SA-V (155 videos, first 5s each)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

source $HOME/.bashrc
conda activate videoseal

cd /home/fr/fr_fr/fr_sj200/videoseal

# Evaluate on all 155 SA-V videos (matching Video Seal paper protocol)
# First 5 seconds at original resolution
python -m videoseal.evals.full \
    --checkpoint /home/fr/fr_fr/fr_sj200/videoseal/ckpts/y_256b_img.pth \
    --lowres_attenuation True --scaling_w 0.2 \
    --dataset sav --is_video true \
    --num_samples 155 \
    --save_first 10

# Backup entire outputs folder
mkdir -p results/videoseal_baseline/sav_155videos
cp -r outputs/* results/videoseal_baseline/sav_155videos/

echo "End time: $(date)"
echo "Evaluation complete!"
echo "Results saved to: results/videoseal_baseline/sav_155videos/"
