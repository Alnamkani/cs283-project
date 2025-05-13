# CS283 Project

## Getting Started

### Cloning the Repository

You can clone this repository using either HTTPS or SSH. Make sure to use the `--recursive` flag to get all submodules:

#### Using HTTPS
```bash
git clone --recursive https://github.com/Alnamkma/cs283-project.git
```

Note: If you forgot to use `--recursive` during cloning, you can initialize and update submodules later with:
```bash
git submodule update --init --recursive
```

## Environment Setup

### ML Peptide Self-Assembly Environment
```bash
cd ml_peptide_self_assembly
conda env create -f ml_peptide_self_assembly.yml
conda activate ml_peptide_self_assembly
```

### PepDiffusion Environment
```bash
# Create and activate a new virtual environment
python -m venv pepdiff_env
source pepdiff_env/bin/activate  # On Unix/macOS

# Install requirements
pip install -r pepdiff_requirements.txt
```

Note: PepDiffusion requires a GPU to run. Make sure you have a CUDA-compatible GPU and the appropriate drivers installed.

## Generating Peptides

### Using Baseline Genetic Algorithm
To generate peptides using the baseline genetic algorithm:

```bash
# Make sure you're in the ml_peptide_self_assembly directory and have activated the conda environment
cd ml_peptide_self_assembly
conda activate ml_peptide_self_assembly

# Run the peptide generation script
python SA_ML_generative/find_novel_peptides.py --ml-model AP_SP
```

The script will generate a CSV file named `suggested_SA_peptides_X_Y.csv` where X and Y are the minimum and maximum lengths of the generated peptides (default is 5-10 amino acids). The CSV file contains two columns:
- `Peptide`: The amino acid sequence of the generated peptide
- `Self-assembly probability [%]`: The predicted probability of self-assembly for the peptide, rounded to one decimal place

The peptides are sorted by their self-assembly probability in descending order.

### Using PepDiffusion
You can generate peptides using either the fine-tuned or non-fine-tuned PepDiffusion model. Note that PepDiffusion requires a GPU to run.

#### Fine-tuned Model
```bash
cd PepDiffusion
source pepdiff_env/bin/activate
python main.py --work Generate \
    --Generate_VAE_model_path ../vae_fine_tuned.pth \
    --Generate_Diffusion_model_path ../best_model_diffusion.pth \
    --Generate_times 1 \
    --Generate_batch_num 512 \
    --Generate_batch_times 1 \
    --Generate_save_path ./generated_peptides.txt
```

#### Non-fine-tuned Model
```bash
cd PepDiffusion
source pepdiff_env/bin/activate
python main.py --work Generate \
    --Generate_times 1 \
    --Generate_batch_num 512 \
    --Generate_batch_times 1 \
    --Generate_save_path ./generated_peptides.txt
```

The generated peptides will be saved in the specified output file (`generated_peptides.txt` by default).

### Running on IBEX Cluster

#### Baseline Generation
To run the baseline generation on the IBEX cluster:

1. First, modify the path in `baseline_generation_job.sh` to match your IBEX path to the project:
   ```bash
   # Change this line in the script
   cd /path/to/ml_peptide_self_assembly
   ```

2. Submit the job to the cluster:
   ```bash
   sbatch baseline_generation_job.sh
   ```

#### PepDiffusion Generation
To run PepDiffusion generation on the IBEX cluster:

1. First, modify the path in `PepDiffusion/pepdiff_generation_job.sh` to match your IBEX path to the project:
   ```bash
   # Change this line in the script
   cd /path/to/PepDiffusion
   ```

2. Submit the job to the cluster:
   ```bash
   sbatch PepDiffusion/pepdiff_generation_job.sh
   ```

The job will use 1 GPU, 4 CPUs, and 16GB of memory, with a runtime limit of 10 hours. Output and error logs will be saved in files named `pepdiff_gen_<job_id>.out` and `pepdiff_gen_<job_id>.err` respectively.

## Predicting Self-Assembly Probabilities

You can predict self-assembly probabilities for peptides using two methods:

### Single Sequence Prediction
To predict the self-assembly probability for a single peptide sequence:
```bash
cd ml_peptide_self_assembly
python SA_ML_predictive/code/predict.py --sequence <peptide_sequence> --ml-model AP_SP
```
Replace `<peptide_sequence>` with your peptide sequence (e.g., `ACDEFGHIKLMNPQRSTVWY`).

### Bulk Prediction
To predict self-assembly probabilities for multiple sequences at once:
```bash
python predict_bulk.py <input_file> <output_file> [model_name]
```
- `<input_file>`: A text file containing one peptide sequence per line
- `<output_file>`: (Optional) Name of the output file (default: predictions.txt)
- `[model_name]`: (Optional) Model to use (default: AP_SP)

Example input file format:
```
ACDEFGHIKLMNPQRSTVWY
ACDEFGHIKLMNPQRSTVW
ACDEFGHIKLMNPQRSTV
```

The output will be saved in a tab-separated file with two columns:
1. Peptide sequence
2. Self-assembly probability

# PepDiffusion Training Guide

This guide provides instructions for training and fine-tuning the PepDiffusion models on a single GPU setup.

## Data Preparation
Before starting the training pipeline, you need to prepare the training data:

1. Move the `VAE_Train.zip` from the base project directory to the PepDiffusion data folder:
   ```bash
   mv VAE_Train.zip PepDiffusion/data/
   ```

2. Unzip the file in the PepDiffusion data directory:
   ```bash
   cd PepDiffusion/data
   unzip VAE_Train.zip
   ```

**Important Note**: The `VAE_Train.zip` in the base directory and the one in PepDiffusion/data have the same name but contain different data. Make sure to use the one from the base directory as it contains the correct training data for fine-tuning.

## Training Pipeline

### 1. Fine-tune VAE Model
```bash
MASTER_ADDR=localhost MASTER_PORT=1234 RANK=0 WORLD_SIZE=1 python main.py \
    --work TransVAE \
    --vae_save_path ./fine_tuned_model \
    --vae_epoch 100 \
    --vae_batch_size 512 \
    --vae_lr 0.0001 \
    --vae_model_path ./data/model_299_0.10607143263973876_0.0909070645059858_.pth \
    --vae_train_path ./data/VAE_Train \
    --vae_val_path ./data/VAE_Val
```

### 2. Generate Memory Representations
```bash
MASTER_ADDR=localhost MASTER_PORT=1234 RANK=0 WORLD_SIZE=1 python main.py \
    --work GetMem_nc \
    --vae_model_path ./fine_tuned_model/model_99_*.pth \
    --mem_save_path_nc ./memory
```

### 3. Train Unconditional Diffusion Model
```bash
MASTER_ADDR=localhost MASTER_PORT=1234 RANK=0 WORLD_SIZE=1 python main.py \
    --work LatentDiffusion_nocondition \
    --LatentDiffusion_lr 0.0001 \
    --LatentDiffusion_save_path_nc ./model_mulgpu_nc \
    --LatentDiffusion_epoch 200 \
    --LatentDiffusion_batch_size 512 \
    --LatentDiffusion_num_steps 500 \
    --LatentDiffusion_shuffle False \
    --LatentDiffusion_num_workers 8 \
    --LatentDiffusion_pin_memory True \
    --LatentDiffusion_drop_last True
```


## Notes
- The `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE` environment variables are set for single GPU training
- Replace `model_99_*.pth` with the actual model checkpoint name after VAE training
- The training process follows a pipeline where each step builds on the previous one
- Best models are saved as "best_model.pth" in their respective directories
- Training logs are saved in the specified save paths