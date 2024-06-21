#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name l1evodiff
#SBATCH --output l1evodiff
#SBATCH -w gpu-3
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

module load cuda

source /stg3/data3/Jonathan/.bashrc
source /stg3/data3/Jonathan/.bash_profile

conda activate evodiff


# Remember to remove the --LoRA and --large_model lines if you don't want to use LoRA and/or the large model,
# and to update the other lines as appropriate.

python fine_tune.py --config_fpath /stg3/data3/Jonathan/jonathan2/pdl1_evodiff_tuning/evodiff/config/config38M.json \
	--out_fpath /stg3/data3/Jonathan/jonathan2/pdl1_evodiff_tuning/evodiff_results/ \
	--train_fpath /stg3/data3/Jonathan/jonathan2/pdl1_evodiff_tuning/train_pdl1.txt \
	--valid_fpath /stg3/data3/Jonathan/jonathan2/pdl1_evodiff_tuning/test_pdl1.txt \
        --checkpoint_freq 60 \
        --LoRA 16
#	--large_model
