# COVID-19 QA Chatbot Project

This repository contains code and experiments for developing a COVID-19 question-answering (QA) chatbot. We fine-tune large language models (LLMs) such as BERT, RoBERTa, and ALBERT on a COVID-19-specific QA dataset, as well as on the SQuAD dataset, to produce accurate, domain-specific responses.

## Project Overview

The goal is to adapt general-purpose transformers to the COVID-19 domain. We:
- Use the [COVID-QA.json](./COVID-QA.json) dataset containing context-question-answer triples.
- Compare multiple models (BERT, RoBERTa, ALBERT) after fine-tuning.
- Evaluate each model using QA metrics (EM, F1) and language similarity metrics (BLEU, ROUGE, BERTScore).
- Experiment with advanced fine-tuning techniques (e.g., QLoRA) and hyperparameter tuning.
- Potentially integrate sentiment analysis or Retrieval-Augmented Generation (RAG) in future work.

## Repository Structure

- `training.py`: Script for fine-tuning QA models. Takes arguments to choose which model to train (`--train_bert`, `--train_roberta`, `--train_albert`).
- `job_train.slurm`: SLURM submission script for HPC cluster training. Specify `--job-name=bert`, `roberta`, or `albert` when submitting.
- `evaluate_all_models.py`: Script to compute EM, F1, BLEU, ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore for all models that have `predictions.json`.
- `COVID-QA.json`: The COVID-specific QA dataset.
- `results_metrics.txt`: Generated after `evaluate_all_models.py` runs, containing all computed metrics.

## Requirements

- Python 3.10
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- NLTK, `rouge-score`, `bert-score`

Install requirements:
```bash
pip install transformers datasets nltk rouge-score bert-score

