#!/usr/bin/env python
import re
import math
import matplotlib.pyplot as plt

# Adjust the log file paths if necessary
albert_log = "albert/out_albert.out"
bert_log = "bert/out_bert.out"
roberta_log = "roberta/out_roberta.out"

# We'll just parse loss lines directly:
loss_pattern = re.compile(r"\{.*'loss':\s*([\d.]+).*'epoch':\s*([\d.]+).*}")

model_epochs = {'albert': [], 'bert': [], 'roberta': []}
model_losses = {'albert': [], 'bert': [], 'roberta': []}

# We'll deduce the model name from the file path
def parse_logs(log_file):
    if "albert" in log_file:
        current_model = "albert"
    elif "bert" in log_file:
        current_model = "bert"
    elif "roberta" in log_file:
        current_model = "roberta"
    else:
        # If the model name isn't in the file path, adjust this logic as needed
        return

    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        loss_match = loss_pattern.match(line)
        if loss_match:
            loss_str, epoch_str = loss_match.groups()
            loss_val = float(loss_str)
            epoch_val = float(epoch_str)
            # Round the epoch to the nearest integer to avoid multiple close values
            epoch_rounded = round(epoch_val)
            model_epochs[current_model].append(epoch_rounded)
            model_losses[current_model].append(loss_val)

# Parse each model's log file
parse_logs(albert_log)
parse_logs(bert_log)
parse_logs(roberta_log)

plt.figure(figsize=(10, 6))

for model in ['albert', 'bert', 'roberta']:
    if model_epochs[model]:
        # Sort by epoch to ensure lines are drawn correctly in ascending order
        combined = list(zip(model_epochs[model], model_losses[model]))
        combined.sort(key=lambda x: x[0])
        sorted_epochs, sorted_losses = zip(*combined)
        plt.plot(sorted_epochs, sorted_losses, marker='o', label=model.upper())

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curves for ALBERT, BERT, and RoBERTa')
plt.grid(True)
plt.legend()
plt.savefig('loss_curves.png', dpi=300)
plt.show()

