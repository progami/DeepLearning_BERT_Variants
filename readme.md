# Fine-Tuning Transformer Models on COVID-QA Dataset

This project involves fine-tuning transformer-based language models for question-answering tasks on the COVID-QA dataset.

## Directory Structure

- [`albert-base-v2-finetuned-covid-qa/`](#albert-base-v2-finetuned-covid-qa)
- [`bert-base-uncased-finetuned-covid-qa/`](#bert-base-uncased-finetuned-covid-qa)
- [`roberta-base-finetuned-covid-qa/`](#roberta-base-finetuned-covid-qa)
- [`results_albert-base-v2/`](#results_albert-base-v2)
- [`results_bert-base-uncased/`](#results_bert-base-uncased)
- [`results_roberta-base/`](#results_roberta-base)
- [`training.py`](#trainingpy)
- [`inference.py`](#inferencepy)
- [`COVID-QA.json`](#covid-qajson)
- [`job_train.slurm`](#job_trainslurm)
- [`logs.out`](#logsout)
- [`error.err`](#errorerr)
- [`results.out`](#resultsout)

---

### `albert-base-v2-finetuned-covid-qa/`

Contains the fine-tuned **ALBERT-base-v2** model on the COVID-QA dataset. [See README](./albert-base-v2-finetuned-covid-qa/README.md) for details.

### `bert-base-uncased-finetuned-covid-qa/`

Contains the fine-tuned **BERT-base-uncased** model on the COVID-QA dataset. [See README](./bert-base-uncased-finetuned-covid-qa/README.md) for details.

### `roberta-base-finetuned-covid-qa/`

Contains the fine-tuned **RoBERTa-base** model on the COVID-QA dataset. [See README](./roberta-base-finetuned-covid-qa/README.md) for details.

### `results_albert-base-v2/`

Contains training checkpoints and results for **ALBERT-base-v2**. [See README](./results_albert-base-v2/README.md) for details.

### `results_bert-base-uncased/`

Contains training checkpoints and results for **BERT-base-uncased**. [See README](./results_bert-base-uncased/README.md) for details.

### `results_roberta-base/`

Contains training checkpoints and results for **RoBERTa-base**. [See README](./results_roberta-base/README.md) for details.

### `training.py`

Python script used to fine-tune the transformer models on the COVID-QA dataset.

### `inference.py`

Python script used to perform inference using the fine-tuned models.

### `COVID-QA.json`

The COVID-QA dataset in JSON format, used for training and evaluation.

### `job_train.slurm`

SLURM job script for submitting training jobs to a computing cluster.

### `logs.out`

Standard output logs from training and evaluation runs.

### `error.err`

Error logs from training or evaluation runs.

### `results.out`

Output containing results from training or evaluation runs.

---

## Usage

1. **Training Models:**
   - Use `training.py` to fine-tune models. Adjust hyperparameters and model configurations as needed.
   - Submit the `job_train.slurm` script to your SLURM cluster for training jobs.

2. **Running Inference:**
   - Use `inference.py` to perform inference with the fine-tuned models.
   - Ensure that the appropriate model directory is specified in the script.

3. **Analyzing Results:**
   - Check `logs.out` and `results.out` for training progress and evaluation metrics.
   - Use the checkpoints in `results_*` directories if you wish to resume training or analyze intermediate models.

---
