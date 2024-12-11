#!/usr/bin/env python
import re
import math
import matplotlib.pyplot as plt

# Log file paths
albert_log = "albert/out_albert.out"
bert_log = "bert/out_bert.out"
roberta_log = "roberta/out_roberta.out"

loss_pattern = re.compile(r"\{.*'loss':\s*([\d.]+).*'epoch':\s*([\d.]+).*}")

models = ['albert', 'bert', 'roberta']
model_epochs = {m: [] for m in models}
model_losses = {m: [] for m in models}

def parse_logs(model_name, log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Store only the last loss line per epoch
    epoch_loss_map = {}

    for line in lines:
        line = line.strip()
        loss_match = loss_pattern.match(line)
        if loss_match:
            loss_str, epoch_str = loss_match.groups()
            loss_val = float(loss_str)
            epoch_val = float(epoch_str)
            epoch_int = round(epoch_val)
            # Overwrite if multiple lines appear for the same epoch,
            # ensuring we end with the last reported loss for that epoch
            epoch_loss_map[epoch_int] = loss_val

    # Now we have one final loss per epoch
    if epoch_loss_map:
        sorted_epochs = sorted(epoch_loss_map.keys())
        for e in sorted_epochs:
            model_epochs[model_name].append(e)
            model_losses[model_name].append(epoch_loss_map[e])

# Parse each model's log file
parse_logs('albert', albert_log)
parse_logs('bert', bert_log)
parse_logs('roberta', roberta_log)

plt.figure(figsize=(10, 6))

for model in models:
    if model_epochs[model]:
        plt.plot(model_epochs[model], model_losses[model], marker='o', label=model.upper())

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curves for ALBERT, BERT, and RoBERTa (Last Loss per Epoch)')
plt.grid(True)
plt.legend()
plt.savefig('loss_curves.png', dpi=300)
plt.show()

