#!/bin/bash
#SBATCH --partition=batch.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jarrar@ksu.edu

module purge
module load Python/3.10.4-GCCcore-11.3.0

cd $SLURM_SUBMIT_DIR
source ~/virtualenvs/automl_env/bin/activate

# Ensure COVID-QA.json is present in the current directory.

python evaluate_all_models.py

