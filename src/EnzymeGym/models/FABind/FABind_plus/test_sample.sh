#!/bin/bash -e
data_path=/root/projects/rl-enzyme-engineering/data/FABind/pdbbind2020
ckpt_path=/root/projects/rl-enzyme-engineering/ckpts/FABind/FABind_plus/confidence_model.bin
sample_size=10

python fabind/tools/generate_esm2_t33.py ${data_path}

python fabind/test_sampling_fabind.py \
    --batch_size 8 \
    --data-path ${data_path} \
    --resultFolder ./results \
    --exp-name test_exp \
    --ckpt ${ckpt_path} --use-clustering --infer-dropout \
    --sample-size ${sample_size} \
    --symmetric-rmsd ${data_path}/renumber_atom_index_same_as_smiles \
    --save-rmsd-dir ./rmsd_results