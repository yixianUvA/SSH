
source /var/scratch/yshen/anaconda3/etc/profile.d/conda.sh
conda activate fourierft
module load gcc
module load cuDNN/cuda11.1/8.0.5
module load cuda11.1/toolkit/11.1.1


export HF_HOME=/var/scratch/yshen/.cache/huggingface
export PIP_CACHE_DIR=/var/scratch/yshen/.cache/pip

CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset rte \
    --task rte \
    --n_frequency 700 \
    --max_length 512 \
    --head_lr 0.011 \
    --fft_lr 0.09 \
    --num_epoch 100 \
    --bs 32  \
    --scale 110.0 \
    --seed 33333 \
    --share_entry


    # --share_entry

