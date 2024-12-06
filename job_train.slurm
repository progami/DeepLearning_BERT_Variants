#!/bin/bash
#SBATCH --job-name=qa_training
#SBATCH --partition=killable.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Set to 1; torchrun will manage multiple processes
#SBATCH --gres=gpu:4                 # Request 4 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # Increased memory allocation
#SBATCH --time=24:00:00
#SBATCH --output=logs.out            # Redirect standard output to logs.out
#SBATCH --error=error.err            # Redirect standard error to error.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jarrar@ksu.edu

# Load necessary modules
module purge
module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0

# Move to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Activate your virtual environment
source ~/virtualenvs/automl_env/bin/activate

# Ensure that the environment variable for protobuf is set
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Set environment variables for NCCL and OpenMP
unset NCCL_DEBUG               # Disable NCCL logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional: Set a unique master port if needed
export MASTER_PORT=12355

# Run your script with desired options using torchrun
torchrun --nnodes=1 --nproc_per_node=4 training.py \
    --train_bert --train_albert --train_roberta \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 3e-5 \
    --max_length 384 \
    --doc_stride 128 \
    --gradient_accumulation_steps 2

