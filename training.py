#!/usr/bin/env python
# training.py

import argparse
import json
import os
import warnings
import collections
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AlbertForQuestionAnswering,
    AlbertTokenizerFast,
    BertForQuestionAnswering,
    BertTokenizerFast,
    RobertaForQuestionAnswering,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    logging as transformers_logging,
)
from transformers.trainer_callback import TrainerCallback

warnings.simplefilter(action='ignore', category=FutureWarning)
transformers_logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Fine-tune QA models on COVID-19 dataset.')
parser.add_argument('--train_bert', action='store_true')
parser.add_argument('--train_roberta', action='store_true')
parser.add_argument('--train_albert', action='store_true')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=3e-5)
parser.add_argument('--max_length', type=int, default=384)
parser.add_argument('--doc_stride', type=int, default=128)
parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
args = parser.parse_args()

if not (args.train_bert or args.train_roberta or args.train_albert):
    parser.error('No model selected for training. Use --train_bert, --train_roberta, or --train_albert.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

file_path = "COVID-QA.json"
with open(file_path, 'r') as f:
    covid_qa_data = json.load(f)

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
                    answer = qa['answers'][0]
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    if answer_start is not None and answer_text:
                        answers.append({'text': [answer_text], 'answer_start': [answer_start]})
                    else:
                        answers.append({'text': [''], 'answer_start': [0]})
                else:
                    answers.append({'text': [''], 'answer_start': [0]})
            else:
                answers.append({'text': [''], 'answer_start': [0]})
            contexts.append(context)
            questions.append(question)
            ids.append(str(qa_id))

data_dict = {
    'id': ids,
    'context': contexts,
    'question': questions,
    'answers': answers
}

# 60% train, 20% val, 20% test
full_dataset = Dataset.from_dict(data_dict)
train_val_test = full_dataset.train_test_split(test_size=0.4, shuffle=True, seed=42)
val_test = train_val_test['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
train_dataset = train_val_test['train']
val_dataset = val_test['train']
test_dataset = val_test['test']

tokenizer = None

def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )

    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    offset_mapping = tokenized_examples.pop('offset_mapping')

    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        ans = examples['answers'][sample_index]

        answer_starts = ans['answer_start']
        answer_texts = ans['text']

        if len(answer_starts) == 0 or len(answer_texts) == 0 or answer_texts[0] == '':
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
        else:
            start_char = answer_starts[0]
            end_char = start_char + len(answer_texts[0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_position = token_start_index - 1

                while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_position = token_end_index + 1

                tokenized_examples['start_positions'].append(start_position)
                tokenized_examples['end_positions'].append(end_position)

    return tokenized_examples

def prepare_validation_features(examples):
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )

    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')

    tokenized_examples['example_id'] = []

    for i in range(len(tokenized_examples['input_ids'])):
        sample_index = sample_mapping[i]
        tokenized_examples['example_id'].append(examples['id'][sample_index])
        sequence_ids = tokenized_examples.sequence_ids(i)
        tokenized_examples['offset_mapping'][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples['offset_mapping'][i])
        ]

    return tokenized_examples

def postprocess_qa_predictions(
    examples,
    features,
    predictions,
    n_best_size=20,
    max_answer_length=30,
):
    all_start_logits, all_end_logits = predictions

    example_id_to_index = {k: i for i, k in enumerate(examples['id'])}
    features_per_example = collections.defaultdict(list)
    for i, f in enumerate(features):
        features_per_example[example_id_to_index[f['example_id']]].append(i)

    final_predictions = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example['context']
        answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]['offset_mapping']

            if offset_mapping is None:
                continue

            cls_index = features[feature_index]['input_ids'].index(tokenizer.cls_token_id)
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

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
            final_predictions[example['id']] = best_answer['text']
        else:
            final_predictions[example['id']] = ''

    return final_predictions

class DataCollatorForQA(DataCollatorWithPadding):
    def __call__(self, features):
        features = [
            {k: v for k, v in f.items() if k not in ('offset_mapping', 'example_id')}
            for f in features
        ]
        return super().__call__(features)

def train_model(model_name, tokenizer_class, model_class):
    global tokenizer

    print(f"\n***** Training {model_name} *****")
    # Map short model names to actual model identifiers on HF Hub
    if model_name == 'bert':
        hf_model_name = 'bert-base-uncased'
    elif model_name == 'roberta':
        hf_model_name = 'roberta-base'
    elif model_name == 'albert':
        hf_model_name = 'albert-base-v2'
    else:
        raise ValueError("Unsupported model_name. Use bert, roberta, or albert.")

    tokenizer = tokenizer_class.from_pretrained(hf_model_name)
    model = model_class.from_pretrained(hf_model_name)

    print(f"Gradient checkpointing is disabled for {model_name}")

    tokenized_train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_val_dataset = val_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    tokenized_test_dataset = test_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    # Use model_name as directory name
    os.makedirs(model_name, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_name,
        evaluation_strategy='no',
        logging_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True,
        report_to="none",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        disable_tqdm=True,
    )

    data_collator = DataCollatorForQA(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    class LogCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if trainer.is_world_process_zero():
                epoch = int(round(state.epoch))
                print(f"Epoch {epoch} completed for {model_name}")

    trainer.add_callback(LogCallback)
    trainer.train()

    if trainer.is_world_process_zero():
        print("Evaluating the model on test dataset...")
    raw_predictions = trainer.predict(tokenized_test_dataset)
    if trainer.is_world_process_zero():
        print("Prediction completed.")

    final_predictions = postprocess_qa_predictions(
        test_dataset,
        tokenized_test_dataset,
        raw_predictions.predictions,
    )

    if trainer.is_world_process_zero():
        formatted_predictions = [
            {"id": ex["id"], "prediction_text": final_predictions[ex["id"]]}
            for ex in test_dataset
        ]
        predictions_path = os.path.join(model_name, "predictions.json")
        with open(predictions_path, "w") as f:
            json.dump(formatted_predictions, f, indent=2)

        print(f"Predictions saved to {predictions_path}.")

    # Save the model with just the short name + "-finetuned"
    trainer.save_model(os.path.join(model_name, f"{model_name}-finetuned"))
    tokenizer.save_pretrained(os.path.join(model_name, f"{model_name}-finetuned"))

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

# Determine which model to train based on args
if args.train_bert:
    train_model('bert', BertTokenizerFast, BertForQuestionAnswering)
elif args.train_roberta:
    train_model('roberta', RobertaTokenizerFast, RobertaForQuestionAnswering)
elif args.train_albert:
    train_model('albert', AlbertTokenizerFast, AlbertForQuestionAnswering)

