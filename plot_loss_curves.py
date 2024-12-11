#!/usr/bin/env python
import re
import math
import matplotlib.pyplot as plt

albert_log = "albert/out_albert.out"
bert_log = "bert/out_bert.out"
roberta_log = "roberta/out_roberta.out"

epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+completed\s+for\s+(albert|bert|roberta)")
loss_pattern = re.compile(r"\{.*'loss':\s*([\d.]+).*'epoch':\s*([\d.]+).*}")

model_epochs = {'albert': [], 'bert': [], 'roberta': []}
model_losses = {'albert': [], 'bert': [], 'roberta': []}

def parse_logs(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        ep_match = epoch_pattern.match(line)
        if ep_match:
            current_epoch, current_model = ep_match.groups()
            current_epoch = float(current_epoch)

            # Search forward for a line with a loss for approximately the same epoch
            j = i + 1
            found_loss = False
            while j < len(lines):
                loss_line = lines[j].strip()
                loss_match = loss_pattern.match(loss_line)
                if loss_match:
                    loss_str, epoch_str = loss_match.groups()
                    loss_val = float(loss_str)
                    loss_epoch = float(epoch_str)
                    # Use math.isclose to handle floating point differences
                    if math.isclose(loss_epoch, current_epoch, rel_tol=1e-3, abs_tol=0.01):
                        model_epochs[current_model].append(current_epoch)
                        model_losses[current_model].append(loss_val)
                        found_loss = True
                        break
                j += 1
            i = j if found_loss else j
        else:
            i += 1

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

