#!/bin/bash -e
index_csv=../inference_examples/example.csv
index_csv_mol=../inference_examples/example-mol.csv
pdb_file_dir=../inference_examples/pdb_files
num_threads=10
save_pt_dir=../inference_examples/temp_files
save_mols_dir=${save_pt_dir}/mol
ckpt_path=../../../../ckpts/FABind/FABind_plus/fabind_plus_best_ckpt.bin
output_dir=../inference_examples/inference_output

cd fabind

echo "======  preprocess molecules  ======"
#python inference_preprocess_mol_confs.py --index_csv ${index_csv_mol} --save_mols_dir ${save_mols_dir} --num_threads ${num_threads}

echo "======  preprocess proteins  ======"
#python inference_preprocess_protein.py --pdb_file_dir ${pdb_file_dir} --save_pt_dir ${save_pt_dir}

echo "======  inference begins  ======"
python inference_fabind.py \
    --ckpt ${ckpt_path} \
    --batch_size 4 \
    --write-mol-to-file \
    --sdf-output-path-post-optim ${output_dir} \
    --index-csv ${index_csv} \
    --preprocess-dir ${save_pt_dir} \
    --post-optim \
