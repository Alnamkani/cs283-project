#!/bin/bash

#SBATCH --job-name=pepdiff_gen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=v100:1
#SBATCH --time=10:00:00
#SBATCH --output=pepdiff_gen_%j.out
#SBATCH --error=pepdiff_gen_%j.err

# Load environment and activate virtual environment
source ~/.bashrc
cd /path/to/PepDiffusion
source pepdiff_env/bin/activate

# Run the fine-tuned model generation
python main.py --work Generate \
    --Generate_VAE_model_path ../vae_fine_tuned.pth \
    --Generate_Diffusion_model_path ../best_model_diffusion.pth \
    --Generate_times 1 \
    --Generate_batch_num 512 \
    --Generate_batch_times 1 \
    --Generate_save_path ./generated_peptides.txt 