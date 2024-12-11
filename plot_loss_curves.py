#!/usr/bin/env python
import re
import math
import matplotlib.pyplot as plt
from statistics import mean

# Adjust the log file paths as needed
albert_log = "albert/out_albert.out"
bert_log = "bert/out_bert.out"
roberta_log = "roberta/out_roberta.out"

# We'll just parse loss lines directly:
# Example line:
# {'loss': 0.7494, 'grad_norm': 94.9762, 'learning_rate': X, 'epoch': 2.0}
loss_pattern = re.compile(r"\{.*'loss':\s*([\d.]+).*'epoch':\s*([\d.]+).*}")

models = ['albert', 'bert', 'roberta']
model_epochs = {m: [] for m in models}
model_losses = {m: [] for m in models}

def parse_logs(model_name, log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # We'll store losses per integer epoch and average them later
    epoch_loss_map = {}

    for line in lines:
        line = line.strip()
        loss_match = loss_pattern.match(line)
        if loss_match:
            loss_str, epoch_str = loss_match.groups()
            loss_val = float(loss_str)
            epoch_val = float(epoch_str)
            # Round epoch to nearest integer
            epoch_int = round(epoch_val)
            if epoch_int not in epoch_loss_map:
                epoch_loss_map[epoch_int] = []
            epoch_loss_map[epoch_int].append(loss_val)

    # Now average the losses per epoch
    if epoch_loss_map:
        sorted_epochs = sorted(epoch_loss_map.keys())
        for e in sorted_epochs:
            avg_loss = mean(epoch_loss_map[e])
            model_epochs[model_name].append(e)
            model_losses[model_name].append(avg_loss)

# Parse each model's log file
parse_logs('albert', albert_log)
parse_logs('bert', bert_log)
parse_logs('roberta', roberta_log)

plt.figure(figsize=(10, 6))

# Plot each model line
for model in models:
    if model_epochs[model]:
        # Already averaged and sorted by epoch in parse_logs
        plt.plot(model_epochs[model], model_losses[model], marker='o', label=model.upper())

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curves for ALBERT, BERT, and RoBERTa')
plt.grid(True)
plt.legend()
plt.savefig('loss_curves.png', dpi=300)
plt.show()

