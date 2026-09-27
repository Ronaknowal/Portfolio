"""Offline, CPU comparison of one effective batch and its microbatches."""
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(17)
records = np.loadtxt(Path(__file__).with_name("iris.csv"), delimiter=",", skiprows=1)
features = torch.tensor(records[:, 1:5])
targets = torch.tensor(records[:, 5], dtype=torch.long)
split_rng = np.random.default_rng(17)
training_rows, validation_rows = [], []
for class_id in range(3):
    rows = split_rng.permutation(np.flatnonzero(records[:, 5] == class_id))
    training_rows.extend(rows[:40])
    validation_rows.extend(rows[40:])
training_rows = np.array(training_rows)
validation_rows = np.array(validation_rows)
center = features[training_rows].mean(0)
scale = features[training_rows].std(0, correction=0)
features = (features - center) / scale

initial_model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
orders_rng = np.random.default_rng(29)
epoch_orders = [orders_rng.permutation(training_rows) for _ in range(20)]


def evaluate(model, rows):
    model.eval()
    with torch.no_grad():
        logits = model(features[rows])
        loss = F.cross_entropy(logits, targets[rows], reduction="sum") / len(rows)
        correct = (logits.argmax(1) == targets[rows]).sum()
    return float(loss), int(correct)


def train(microbatch_size):
    model = copy.deepcopy(initial_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    updates = microbatches = 0
    history = [(0, 0, *evaluate(model, training_rows), *evaluate(model, validation_rows))]
    for epoch, order in enumerate(epoch_orders, start=1):
        model.train()
        for start in range(0, len(order), 32):
            group = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(group), microbatch_size):
                rows = group[offset:offset + microbatch_size]
                logits = model(features[rows])
                loss_sum = F.cross_entropy(logits, targets[rows], reduction="sum")
                (loss_sum / len(group)).backward()
                microbatches += 1
            optimizer.step()
            updates += 1
        history.append((epoch, updates, *evaluate(model, training_rows), *evaluate(model, validation_rows)))
    return model, optimizer, history, microbatches


def main():
    full, full_optimizer, full_history, full_microbatches = train(32)
    accumulated, accumulated_optimizer, accumulated_history, microbatches = train(12)
    print("epoch updates train_loss train_correct validation_loss validation_correct")
    for epoch, updates, train_loss, train_correct, validation_loss, validation_correct in accumulated_history:
        if epoch in (0, 1, 5, 20):
            print(f"{epoch} {updates} {train_loss:.6f} {train_correct}/120 {validation_loss:.6f} {validation_correct}/30")
    parameter_gap = max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(full.parameters(), accumulated.parameters()))
    momentum_gap = max(float((full_optimizer.state[a]["momentum_buffer"] - accumulated_optimizer.state[b]["momentum_buffer"]).abs().max()) for a, b in zip(full.parameters(), accumulated.parameters()))
    print(f"full_forward_backward_calls {full_microbatches}")
    print(f"accumulated_forward_backward_calls {microbatches}")
    print(f"max_parameter_gap {parameter_gap:.3e}")
    print(f"max_momentum_gap {momentum_gap:.3e}")
    print("uniform_probability_loss", f"{np.log(3):.6f}")
    print("constant_class_validation_correct 10/30")
    return {"history_columns": ["epoch", "updates", "train_loss", "train_correct", "validation_loss", "validation_correct"], "history": accumulated_history, "parameter_gap": parameter_gap, "momentum_gap": momentum_gap, "full_calls": full_microbatches, "accumulated_calls": microbatches}


if __name__ == "__main__":
    main()
