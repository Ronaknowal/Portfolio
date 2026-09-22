"""Small, fully unrolled vector-capsule learning experiment on offline digits."""
from pathlib import Path
import json
import platform
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent


def squash(vectors):
    radius = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return vectors * radius / (1 + radius.square())


def route(votes, iterations=3, capture=False, stop_gradient=False):
    """votes: batch, child, parent, parent-coordinate; softmax over parent."""
    if iterations < 1:
        raise ValueError("At least one routing iteration is required")
    logits = votes.new_zeros(votes.shape[:-1])
    trace = []
    for step in range(iterations):
        coupling = logits.softmax(dim=2)
        current_votes = votes.detach() if stop_gradient and step < iterations - 1 else votes
        sums = (coupling[..., None] * current_votes).sum(dim=1)
        output = squash(sums)
        agreement = (current_votes * output[:, None]).sum(dim=-1)
        if capture:
            trace.append({"coupling": coupling.detach().tolist(),
                          "sums": sums.detach().tolist(),
                          "outputs": output.detach().tolist(),
                          "agreement": agreement.detach().tolist()})
        if step < iterations - 1:
            logits = logits + agreement
    return output, trace


class TinyCapsules(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(1, 32, 3, padding=1)
        self.primary = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.transforms = nn.Parameter(torch.randn(64, 10, 8, 4) * 0.1)
        self.decoder = nn.Sequential(nn.Linear(80, 64), nn.ReLU(),
                                     nn.Linear(64, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.Sigmoid())

    def encode(self, images, iterations=3, capture=False):
        features = F.relu(self.features(images))
        primary = self.primary(features).reshape(-1, 4, 4, 4, 4)
        # First two axes after batch are capsule type and vector coordinate.
        primary = primary.permute(0, 3, 4, 1, 2).reshape(-1, 64, 4)
        primary = squash(primary)
        votes = torch.einsum("bid,ijod->bijo", primary, self.transforms)
        output, trace = route(votes, iterations, capture)
        return output, trace

    def reconstruct(self, capsules, labels):
        mask = F.one_hot(labels, 10)[..., None].to(capsules.dtype)
        return self.decoder((capsules * mask).flatten(1)).reshape(-1, 1, 8, 8)


def margin_loss(capsules, labels):
    lengths = torch.linalg.vector_norm(capsules, dim=-1)
    targets = F.one_hot(labels, 10).to(lengths.dtype)
    terms = targets * F.relu(0.9 - lengths).square()
    terms += 0.5 * (1 - targets) * F.relu(lengths - 0.1).square()
    return terms.sum(dim=1).mean()


def shifted(images, row_shift=0, column_shift=0):
    result = torch.zeros_like(images)
    height, width = images.shape[-2:]
    source_rows = slice(max(0, -row_shift), min(height, height - row_shift))
    source_cols = slice(max(0, -column_shift), min(width, width - column_shift))
    target_rows = slice(max(0, row_shift), min(height, height + row_shift))
    target_cols = slice(max(0, column_shift), min(width, width + column_shift))
    result[..., target_rows, target_cols] = images[..., source_rows, source_cols]
    return result


@torch.no_grad()
def assess(model, images, labels, iterations, include_outputs=False):
    model.eval()
    capsules, _ = model.encode(images, iterations)
    lengths = torch.linalg.vector_norm(capsules, dim=-1)
    predicted = lengths.argmax(dim=1)
    reconstruction = model.reconstruct(capsules, predicted)
    target_conditioned = model.reconstruct(capsules, labels)
    result = {"correct": int((predicted == labels).sum()),
              "margin_loss": float(margin_loss(capsules, labels)),
              "predicted_mask_mse": float(F.mse_loss(reconstruction, images)),
              "label_conditioned_mse": float(F.mse_loss(target_conditioned, images))}
    if include_outputs:
        result.update(predictions=predicted.tolist(), lengths=lengths.tolist())
    return result


def main():
    torch.set_num_threads(1)
    raw = np.genfromtxt(ROOT / "digits-400.csv", delimiter=",", skip_header=1)
    identifiers = raw[:, 0].astype(int)
    pixels = raw[:, 1:65]
    targets = raw[:, 65].astype(int)
    assert len(set(identifiers)) == 400
    assert len(np.unique(pixels, axis=0)) == 400
    train, development = train_test_split(np.arange(400), test_size=0.3,
                                         random_state=22, stratify=targets)
    images = torch.tensor(pixels / 16, dtype=torch.float32).reshape(-1, 1, 8, 8)
    labels = torch.tensor(targets, dtype=torch.long)
    x_train, y_train = images[train], labels[train]
    x_dev, y_dev = images[development], labels[development]
    report = {"versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "torch": torch.__version__, "sklearn": sklearn.__version__},
              "train_ids": identifiers[train].tolist(), "development_ids": identifiers[development].tolist(),
              "development_labels": y_dev.tolist(), "unique_ids": 400, "unique_images": 400,
              "training_mean_image_mse": float(F.mse_loss(x_train.mean(0).expand_as(x_dev), x_dev)),
              "runs": [], "saved_models": {}}
    for seed in (1, 2, 3):
        for training_iterations in (1, 3):
            torch.manual_seed(seed)
            model = TinyCapsules()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            batch_generator = torch.Generator().manual_seed(100 + seed)
            run = {"seed": seed, "training_iterations": training_iterations,
                   "parameters": sum(p.numel() for p in model.parameters()), "trajectory": []}
            for step in range(601):
                if step in (0, 1, 100, 300, 600):
                    run["trajectory"].append({"step": step,
                        "train": assess(model, x_train, y_train, training_iterations),
                        "development": assess(model, x_dev, y_dev, training_iterations)})
                if step == 600:
                    break
                model.train()
                batch = torch.randint(len(train), (64,), generator=batch_generator)
                batch_images, batch_labels = x_train[batch], y_train[batch]
                capsules, _ = model.encode(batch_images, training_iterations)
                reconstruction = model.reconstruct(capsules, batch_labels)
                reconstruction_sse = (reconstruction - batch_images).square().flatten(1).sum(1).mean()
                loss = margin_loss(capsules, batch_labels) + 0.0005 * reconstruction_sse
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            run["inference_iterations"] = {str(count): assess(model, x_dev, y_dev, count, True)
                                           for count in (1, 2, 3, 5)}
            run["shifted"] = {f"{dy},{dx}": assess(model, shifted(x_dev, dy, dx), y_dev, training_iterations)
                              for dy, dx in ((0, 1), (1, 0), (0, 0))}
            if seed == 1:
                key = str(training_iterations)
                report["saved_models"][key] = {name: value.detach().tolist()
                                               for name, value in model.state_dict().items()}
                capsules, trace = model.encode(x_dev[:2], training_iterations, capture=True)
                run["first_two_examples"] = {"source_ids": identifiers[development[:2]].tolist(),
                    "images": x_dev[:2, 0].tolist(), "labels": y_dev[:2].tolist(),
                    "capsules": capsules.detach().tolist(), "trace": trace}
            report["runs"].append(run)
            summary = run["trajectory"][-1]["development"]
            print(f"seed={seed} routing={training_iterations} development={summary}", flush=True)
    (ROOT / "calculated-inputs.json").write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
