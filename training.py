import argparse
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    BertTokenizerFast, BertForQuestionAnswering,
    RobertaTokenizerFast, RobertaForQuestionAnswering,
    AlbertTokenizerFast, AlbertForQuestionAnswering,
    Trainer, TrainingArguments,
    DataCollatorWithPadding
)
from transformers import logging
from evaluate import load as load_metric
from collections import OrderedDict
import warnings
from typing import Mapping
from tqdm import tqdm
import os

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.set_verbosity_error()

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Fine-tune QA models on COVID-19 dataset.')
parser.add_argument('--train_bert', action='store_true', help='Train BERT model')
parser.add_argument('--train_roberta', action='store_true', help='Train RoBERTa model')
parser.add_argument('--train_albert', action='store_true', help='Train ALBERT model')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size per device')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
args = parser.parse_args()

# Check if at least one model is selected
if not (args.train_bert or args.train_roberta or args.train_albert):
    parser.error('No model selected for training. Use --train_bert, --train_roberta, or --train_albert.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the dataset
file_path = "COVID-QA.json"  # Ensure this path is correct

with open(file_path, 'r') as f:
    covid_qa_data = json.load(f)

# Prepare data lists
contexts = []
questions = []
answers = []
ids = []

for item in covid_qa_data['data']:
    for paragraph in item['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            qa_id = qa['id']
            is_impossible = qa.get('is_impossible', False)
            if not is_impossible:
                if 'answers' in qa and qa['answers']:
                    # Use the first answer for training
                    answer = qa['answers'][0]
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    # Verify that answer_start is valid
                    if answer_start is not None and answer_text:
                        answers.append({'text': [answer_text], 'answer_start': [answer_start]})
                    else:
                        # If answer_start is None or answer_text is empty, set as unanswerable
                        answers.append({'text': [''], 'answer_start': [0]})
                else:
                    # If no answers provided, set as unanswerable
                    answers.append({'text': [''], 'answer_start': [0]})
            else:
                # For unanswerable questions
                answers.append({'text': [''], 'answer_start': [0]})
            contexts.append(context)
            questions.append(question)
            ids.append(str(qa_id))  # Ensure IDs are strings

# Assign unique IDs if necessary
data_dict = {
    'id': ids,
    'context': contexts,
    'question': questions,
    'answers': answers
}

# Create the dataset
dataset = Dataset.from_dict(data_dict)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

# Initialize tokenizer globally
tokenizer = None

# Function to prepare training features
def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length=256,  # Reduced max_length to reduce memory usage
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,  # We need offset mappings for training
        padding='max_length',
    )

    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    offset_mapping = tokenized_examples.pop('offset_mapping')  # Remove offset_mapping to save memory

    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples['answers'][sample_index]

        answer_starts = answers['answer_start']
        answer_texts = answers['text']

        # If no answers are given, set the cls_index as answer
        if len(answer_starts) == 0 or len(answer_texts) == 0 or answer_texts[0] == '':
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
        else:
            # Start/end character index of the answer in the text
            start_char = answer_starts[0]
            end_char = start_char + len(answer_texts[0])

            # Start token index of the current span in the text
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                # Move the token_start_index and token_end_index to the start and end of the answer
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_position = token_start_index - 1

                while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_position = token_end_index + 1

                tokenized_examples['start_positions'].append(start_position)
                tokenized_examples['end_positions'].append(end_position)

    return tokenized_examples

# Function to prepare evaluation features
def prepare_validation_features(examples):
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length=256,  # Reduced max_length to reduce memory usage
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )

    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')

    tokenized_examples['example_id'] = []

    for i in range(len(tokenized_examples['input_ids'])):
        # Map example index to example ID
        sample_index = sample_mapping[i]
        tokenized_examples['example_id'].append(examples['id'][sample_index])

        # Keep the offset mapping only for the context tokens
        sequence_ids = tokenized_examples.sequence_ids(i)
        tokenized_examples['offset_mapping'][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples['offset_mapping'][i])
        ]

    return tokenized_examples

# Define the postprocess_qa_predictions function
def postprocess_qa_predictions(
    examples,
    features,
    predictions,
    n_best_size=20,
    max_answer_length=30,
):
    all_start_logits, all_end_logits = predictions

    example_id_to_index = {k: i for i, k in enumerate(examples['id'])}
    features_per_example = {}
    for i, feature in enumerate(features):
        example_id = feature['example_id']
        if example_id not in features_per_example:
            features_per_example[example_id] = []
        features_per_example[example_id].append(i)

    # The dictionaries we have to fill
    final_predictions = OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples
    for example_id, feature_indices in tqdm(features_per_example.items()):
        # Those are the indices of the features associated to the current example.
        context = examples['context'][example_id_to_index[example_id]]
        answers = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]['offset_mapping']

            # Skip if offset_mapping is None
            if offset_mapping is None:
                continue

            # Update minimum null prediction
            cls_index = features[feature_index]['input_ids'].index(tokenizer.cls_token_id)
            score_null = start_logits[cls_index] + end_logits[cls_index]

            # Go through all possible positions
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid positions
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    # Get the answer text
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    answer_text = context[start_char:end_char]
                    total_score = start_logits[start_index] + end_logits[end_index]
                    answers.append({
                        'text': answer_text,
                        'score': total_score
                    })

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x['score'])
            final_predictions[example_id] = best_answer['text']
        else:
            final_predictions[example_id] = ''

    return final_predictions

# Custom Data Collator to exclude 'offset_mapping' and 'example_id'
from transformers.data.data_collator import DataCollatorWithPadding

class DataCollatorForQA(DataCollatorWithPadding):
    def __call__(self, features):
        # Remove 'offset_mapping' and 'example_id' from features
        features = [
            {k: v for k, v in f.items() if k not in ('offset_mapping', 'example_id')}
            for f in features
        ]
        return super().__call__(features)

# Function to train a model
def train_model(model_name, tokenizer_class, model_class):
    print(f"\n***** Training {model_name} *****")
    global tokenizer  # Ensure tokenizer is accessible in compute_metrics and feature functions
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)

    # Enable gradient checkpointing only if supported
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
        except ValueError:
            print(f"Gradient checkpointing is not supported for {model_name}. Continuing without it.")
    else:
        print(f"Gradient checkpointing method not found for {model_name}. Continuing without it.")

    # Tokenize the training dataset without multiprocessing
    tokenized_train_dataset = dataset['train'].map(
        prepare_train_features,
        batched=True,
        remove_columns=dataset['train'].column_names,
    )

    # Tokenize the evaluation dataset without multiprocessing
    tokenized_eval_dataset = dataset['test'].map(
        prepare_validation_features,
        batched=True,
        remove_columns=dataset['test'].column_names,
    )

    # Define compute_metrics inside train_model to access tokenized_datasets
    def compute_metrics(p):
        examples = dataset['test']
        features = tokenized_eval_dataset

        # Post-process the raw predictions to get final answers
        final_predictions = postprocess_qa_predictions(
            examples,
            features,
            p.predictions,
        )

        # Format the predictions and references as required by the metric
        formatted_predictions = [
            {'id': k, 'prediction_text': v} for k, v in final_predictions.items()
        ]
        references = [
            {'id': ex['id'], 'answers': ex['answers']} for ex in examples
        ]

        # Load the metric (use 'squad_v2' if unanswerable questions are present)
        has_impossible_answers = any(len(ans['text'][0]) == 0 for ans in examples['answers'])
        metric_name = 'squad_v2' if has_impossible_answers else 'squad'
        metric = load_metric(metric_name)

        # Compute the metric
        return metric.compute(predictions=formatted_predictions, references=references)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_{model_name}',
        evaluation_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        remove_unused_columns=False,  # Keep all columns
        fp16=False,
        logging_steps=50,
        report_to="none",
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
    )

    # Use the custom DataCollatorForQA
    data_collator = DataCollatorForQA(tokenizer, pad_to_multiple_of=8)

    # Initialize Trainer with compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results for {model_name}:")
    print(eval_results)

    # Save the model
    trainer.save_model(f'./{model_name}-finetuned-covid-qa')
    tokenizer.save_pretrained(f'./{model_name}-finetuned-covid-qa')

# Train selected models
if args.train_bert:
    train_model(
        model_name='bert-base-uncased',
        tokenizer_class=BertTokenizerFast,
        model_class=BertForQuestionAnswering
    )

if args.train_roberta:
    train_model(
        model_name='roberta-base',
        tokenizer_class=RobertaTokenizerFast,
        model_class=RobertaForQuestionAnswering
    )

if args.train_albert:
    train_model(
        model_name='albert-base-v2',
        tokenizer_class=AlbertTokenizerFast,
        model_class=AlbertForQuestionAnswering
    )

