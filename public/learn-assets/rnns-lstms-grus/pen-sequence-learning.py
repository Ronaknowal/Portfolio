"""RNN, LSTM and GRU on completed real pen trajectories; no online-input claim."""
from pathlib import Path
import csv
import json
import platform
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent


def load_data():
    with (ROOT / "pen-trajectories.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    coordinates = np.array([[float(row[f"{axis}{time + 1}"]) for time in range(8)
                             for axis in ("x", "y")] for row in records])
    identifiers = [row["source_id"] for row in records]
    assert len(set(identifiers)) == len(records)
    assert len(np.unique(coordinates, axis=0)) == len(records)
    assert np.isfinite(coordinates).all() and coordinates.min() >= 0 and coordinates.max() <= 100
    labels = np.array([int(row["digit"]) for row in records])
    train = np.array([row["partition"] == "train" for row in records])
    assert train.sum() == 600 and (~train).sum() == 300
    return coordinates.reshape(-1, 8, 2) / 50 - 1, labels, train, identifiers


class PenClassifier(nn.Module):
    def __init__(self, kind, hidden_size=32):
        super().__init__()
        self.kind = kind
        self.hidden_size = hidden_size
        self.recurrent = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind](
            2, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 10)
        with torch.no_grad():
            self.recurrent.bias_ih_l0.zero_()
            self.recurrent.bias_hh_l0.zero_()
            if kind == "lstm":
                self.recurrent.bias_ih_l0[hidden_size:2 * hidden_size].fill_(1)

    def forward(self, sequences, state=None):
        outputs, final_state = self.recurrent(sequences, state)
        return self.readout(outputs[:, -1]), outputs, final_state


@torch.no_grad()
def assess(model, sequences, labels, details=False):
    model.eval()
    logits, outputs, _ = model(sequences)
    predicted = logits.argmax(1)
    report = {"correct": int((predicted == labels).sum()),
              "cross_entropy": float(F.cross_entropy(logits, labels))}
    if details:
        report.update(predictions=predicted.tolist(), probabilities=logits.softmax(1).tolist())
    return report


def features(sequences, orderless=False):
    if orderless:
        return np.concatenate([sequences.mean(1), sequences.std(1),
                               sequences.min(1), sequences.max(1)], axis=1)
    return sequences.reshape(len(sequences), -1)


def main():
    torch.set_num_threads(1)
    sequences, labels, train, identifiers = load_data()
    x_train = torch.tensor(sequences[train], dtype=torch.float32)
    x_dev = torch.tensor(sequences[~train], dtype=torch.float32)
    y_train = torch.tensor(labels[train], dtype=torch.long)
    y_dev = torch.tensor(labels[~train], dtype=torch.long)
    report = {"versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "torch": torch.__version__, "sklearn": sklearn.__version__},
              "train_ids": np.array(identifiers)[train].tolist(),
              "development_ids": np.array(identifiers)[~train].tolist(),
              "development_labels": labels[~train].tolist(), "baselines": {}, "runs": [], "saved_models": {}}
    for orderless in (True, False):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
        model.fit(features(sequences[train], orderless), labels[train])
        original = model.predict_proba(features(sequences[~train], orderless))
        reversed_prob = model.predict_proba(features(sequences[~train, ::-1], orderless))
        name = "orderless_statistics" if orderless else "ordered_flattened"
        report["baselines"][name] = {"development_correct": int((original.argmax(1) == labels[~train]).sum()),
            "cross_entropy": float(log_loss(labels[~train], original)),
            "train_correct": int((model.predict(features(sequences[train], orderless)) == labels[train]).sum()),
            "reversed_correct": int((reversed_prob.argmax(1) == labels[~train]).sum()),
            "reverse_max_probability_change": float(np.max(np.abs(original - reversed_prob)))}
    for seed in (1, 2, 3):
        for kind in ("rnn", "lstm", "gru"):
            torch.manual_seed(seed)
            model = PenClassifier(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            generator = torch.Generator().manual_seed(100 + seed)
            run = {"seed": seed, "kind": kind, "parameters": sum(p.numel() for p in model.parameters()),
                   "trajectory": [], "clipped_updates": 0, "largest_preclip_gradient_norm": 0.0}
            for step in range(501):
                if step in (0, 1, 100, 300, 500):
                    run["trajectory"].append({"step": step, "train": assess(model, x_train, y_train),
                                               "development": assess(model, x_dev, y_dev)})
                if step == 500:
                    break
                model.train()
                batch = torch.randint(len(x_train), (64,), generator=generator)
                logits, _, _ = model(x_train[batch])
                loss = F.cross_entropy(logits, y_train[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                run["clipped_updates"] += int(norm > 1.0)
                run["largest_preclip_gradient_norm"] = max(run["largest_preclip_gradient_norm"], norm)
                optimizer.step()
            run["final"] = assess(model, x_dev, y_dev, details=True)
            run["reversed"] = assess(model, x_dev.flip(1), y_dev, details=True)
            swapped = x_dev.clone()
            swapped[:, [2, 3]] = swapped[:, [3, 2]]
            run["swapped_points_3_4"] = assess(model, swapped, y_dev, details=True)
            if seed == 1:
                report["saved_models"][kind] = {name: value.detach().tolist() for name, value in model.state_dict().items()}
                with torch.no_grad():
                    _, outputs, _ = model(x_dev[:2])
                run["first_two_examples"] = {"source_ids": np.array(identifiers)[~train][:2].tolist(),
                    "labels": y_dev[:2].tolist(), "coordinates": x_dev[:2].tolist(),
                    "hidden_states": outputs.tolist(),
                    "prefix_readout_probabilities": model.readout(outputs).softmax(-1).detach().tolist()}
            report["runs"].append(run)
            print(f"seed={seed} kind={kind} final={run['trajectory'][-1]}", flush=True)
    (ROOT / "calculated-inputs.json").write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(report["baselines"], indent=2))


if __name__ == "__main__":
    main()
