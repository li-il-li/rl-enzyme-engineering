#!/bin/bash -e
python predict.py --mode esmfold --input_csv splits/atlas_val.csv --weights ../../ckpts/AlphaFlow/esmflow_md_distilled_202402.pt --samples 5 --outpdb ./pdbs/ --noisy_first --no_diffusion

