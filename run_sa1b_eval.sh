#!/bin/bash
#SBATCH --partition=gpu_mi300
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=videoseal_sa1b_1k
#SBATCH --output=logs/eval_sa1b_1k_imgs_%j.out
#SBATCH --error=logs/eval_sa1b_1k_imgs_%j.err

echo "Starting Video Seal evaluation on COCO (1k images as per paper)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Activate conda environment
source $HOME/.bashrc
conda activate videoseal

cd /home/fr/fr_fr/fr_sj200/videoseal

# Run evaluation on only 1k images as stated in the paper
python -m videoseal.evals.full \
    --checkpoint /home/fr/fr_fr/fr_sj200/videoseal/ckpts/y_256b_img.pth \
    --lowres_attenuation True --scaling_w 0.2 \
    --dataset sa1b --is_video false \
    --num_samples 1000 \
    --save_first 10

# Backup entire outputs folder
mkdir -p results/videoseal_baseline/SA1b_1k
cp -r outputs/* results/videoseal_baseline/SA1b_1k/

echo "End time: $(date)"
echo "Evaluation complete!"
