#!/usr/bin/env python
import re
import matplotlib.pyplot as plt

# Adjust the log file paths as needed
albert_log = "albert/out_albert.out"
bert_log = "bert/out_bert.out"
roberta_log = "roberta/out_roberta.out"

epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+completed\s+for\s+(albert|bert|roberta)")
# This pattern captures the 'loss' and 'epoch' values from a line like:
# {'loss': 0.7494, 'grad_norm': 94.9762, 'learning_rate': X, 'epoch': 2.0}
loss_pattern = re.compile(r"\{.*'loss':\s*([\d.]+).*'epoch':\s*([\d.]+).*}")

model_epochs = {'albert': [], 'bert': [], 'roberta': []}
model_losses = {'albert': [], 'bert': [], 'roberta': []}

def parse_logs(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # We'll keep track of whether we found an epoch line
    # and then search forward for a line with the corresponding loss.
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        ep_match = epoch_pattern.match(line)
        if ep_match:
            current_epoch, current_model = ep_match.groups()
            current_epoch = float(current_epoch)
            # Now search forward from the next line until we find a loss matching this epoch
            j = i + 1
            found_loss = False
            while j < len(lines):
                loss_line = lines[j].strip()
                loss_match = loss_pattern.match(loss_line)
                if loss_match:
                    loss_str, epoch_str = loss_match.groups()
                    loss_epoch = float(epoch_str)
                    # Check if the loss line corresponds to the same epoch we just found
                    if loss_epoch == current_epoch:
                        loss_val = float(loss_str)
                        model_epochs[current_model].append(current_epoch)
                        model_losses[current_model].append(loss_val)
                        found_loss = True
                        break
                j += 1
            # Move i to j, since we've scanned ahead
            i = j
            if not found_loss:
                # No loss found for this epoch, just continue
                i += 1
        else:
            i += 1

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

