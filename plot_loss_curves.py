#!/usr/bin/env python
import re
import matplotlib.pyplot as plt

# Paths to each model's log file based on your directory structure
albert_log = "albert/out_albert.out"
bert_log = "bert/out_bert.out"
roberta_log = "roberta/out_roberta.out"

# Regex patterns to identify lines
epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+completed\s+for\s+(albert|bert|roberta)")
loss_pattern = re.compile(r"\{.*'loss':\s*([\d.]+),.*'epoch':\s*([\d.]+).*}")

model_epochs = {'albert': [], 'bert': [], 'roberta': []}
model_losses = {'albert': [], 'bert': [], 'roberta': []}

def parse_logs(log_file):
    # Parse a given log file and return epoch/loss for its model
    # We'll detect model name from the log lines themselves.
    current_model = None
    current_epoch = None

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # We'll find "Epoch X completed for MODEL" and then next line is the loss
    for i, line in enumerate(lines):
        line = line.strip()
        ep_match = epoch_pattern.match(line)
        if ep_match:
            current_epoch, current_model = ep_match.groups()
            current_epoch = float(current_epoch)
            # Next line should have the loss
            if i+1 < len(lines):
                next_line = lines[i+1].strip()
                loss_match = loss_pattern.match(next_line)
                if loss_match:
                    loss_str, epoch_str = loss_match.groups()
                    loss_val = float(loss_str)
                    model_epochs[current_model].append(current_epoch)
                    model_losses[current_model].append(loss_val)

# Parse each model's log file
parse_logs(albert_log)
parse_logs(bert_log)
parse_logs(roberta_log)

plt.figure(figsize=(10, 6))

for model in ['albert', 'bert', 'roberta']:
    if model_epochs[model]:
        plt.plot(model_epochs[model], model_losses[model], marker='o', label=model.upper())

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curves for ALBERT, BERT, and RoBERTa')
plt.grid(True)
plt.legend()
plt.savefig('loss_curves.png', dpi=300)
plt.show()

