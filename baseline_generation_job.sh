#!/bin/bash

#SBATCH --job-name=baseline_gen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=baseline_gen_%j.out
#SBATCH --error=baseline_gen_%j.err

# Load environment and activate conda
source ~/.bashrc
cd /path/to/ml_peptide_self_assembly
conda activate ml_peptide_self_assembly

# Run the baseline peptide generation script
python SA_ML_generative/find_novel_peptides.py --ml-model AP_SP 