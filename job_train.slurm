#!/bin/bash
#SBATCH --partition=batch.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jarrar@ksu.edu

# No job-name here; provide --job-name=bert or --job-name=roberta or --job-name=albert at submission time.
# Example: sbatch --job-name=bert job_train.slurm

# The job name corresponds to the model (bert, roberta, albert).
# Store logs in a directory named after the model.
#SBATCH --output=%x/out_%x.out
#SBATCH --error=%x/err_%x.err

module purge
module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0

cd $SLURM_SUBMIT_DIR
source ~/virtualenvs/automl_env/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=12355

MODEL=${SLURM_JOB_NAME}  # Should be bert, roberta, or albert

# Create a folder named after the model if it doesn't exist
mkdir -p ${MODEL}

# Determine which flag to set based on job name and set the underlying Hugging Face model
if [ "$MODEL" == "bert" ]; then
    TRAIN_FLAG="--train_bert"
elif [ "$MODEL" == "roberta" ]; then
    TRAIN_FLAG="--train_roberta"
elif [ "$MODEL" == "albert" ]; then
    TRAIN_FLAG="--train_albert"
else
    echo "Unsupported model name: $MODEL. Use bert, roberta, or albert."
    exit 1
fi

# Run training script
torchrun --nnodes=1 --nproc_per_node=4 training.py \
    $TRAIN_FLAG \
    --epochs 20 \
    --batch_size 2 \
    --learning_rate 3e-5 \
    --max_length 384 \
    --doc_stride 128 \
    --gradient_accumulation_steps 2

