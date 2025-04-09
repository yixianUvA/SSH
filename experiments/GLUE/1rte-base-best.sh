
source /var/scratch/yshen/anaconda3/etc/profile.d/conda.sh
conda activate fourierft
module load gcc
module load cuDNN/cuda11.1/8.0.5
module load cuda11.1/toolkit/11.1.1


# ssh -L 8083:127.0.0.1:8083 yshen@fs4.das6.science.uva.nl
# ssh -L 8083:127.0.0.1:8083 yshen@node412
# code-server --port 8083
    # --share_entry

