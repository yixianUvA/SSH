#!/bin/bash
#SBATCH --job-name=Fourier
#SBATCH --time=12:30:00
#SBATCH -C A6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=y.shen@uva.nl


source /var/scratch/yshen/anaconda3/etc/profile.d/conda.sh
conda activate fourierft
module load gcc
module load cuDNN/cuda11.1/8.0.5
module load cuda11.1/toolkit/11.1.1


export HF_HOME=/var/scratch/yshen/.cache/huggingface
export PIP_CACHE_DIR=/var/scratch/yshen/.cache/pip



CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset qnli \
    --task qnli \
    --n_frequency 50 \
    --max_length 512 \
    --head_lr 0.001 \
    --fft_lr 0.1 \
    --num_epoch 40 \
    --bs 32  \
    --scale 29.0 \
    --seed 44444 \
    --share_entry

CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset qnli \
    --task qnli \
    --n_frequency 100 \
    --max_length 512 \
    --head_lr 0.001 \
    --fft_lr 0.1 \
    --num_epoch 40 \
    --bs 32  \
    --scale 29.0 \
    --seed 44444 \
    --share_entry

CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset qnli \
    --task qnli \
    --n_frequency 200 \
    --max_length 512 \
    --head_lr 0.001 \
    --fft_lr 0.1 \
    --num_epoch 40 \
    --bs 32  \
    --scale 29.0 \
    --seed 44444 \
    --share_entry

CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset qnli \
    --task qnli \
    --n_frequency 1000 \
    --max_length 512 \
    --head_lr 0.001 \
    --fft_lr 0.1 \
    --num_epoch 40 \
    --bs 32  \
    --scale 29.0 \
    --seed 44444 \
    --share_entry

CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset qnli \
    --task qnli \
    --n_frequency 6144 \
    --max_length 512 \
    --head_lr 0.001 \
    --fft_lr 0.1 \
    --num_epoch 40 \
    --bs 32  \
    --scale 29.0 \
    --seed 44444 \
    --share_entry

CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset qnli \
    --task qnli \
    --n_frequency 12288 \
    --max_length 512 \
    --head_lr 0.001 \
    --fft_lr 0.1 \
    --num_epoch 40 \
    --bs 32  \
    --scale 29.0 \
    --seed 44444 \
    --share_entry