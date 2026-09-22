"""CPU experiments for Loss Functions. Run beside digits-400.csv.

Python 3.12; numpy 2.3.5; torch 2.14.0; scikit-learn 1.9.1.
Outputs are a development experiment, not a benchmark or final test.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, log_loss,
)

torch.set_num_threads(1)


def binary_focal(logits, targets, gamma=2.0, alpha=None):
    """Unreduced stable binary focal loss; alpha=None means no class weighting."""
    if gamma < 0 or (alpha is not None and not 0 <= alpha <= 1):
        raise ValueError("gamma must be nonnegative; alpha must be in [0, 1]")
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    signed_logits = (2 * targets - 1) * logits
    miss_probability = torch.sigmoid(-signed_logits)
    weights = 1 if alpha is None else targets * alpha + (1 - targets) * (1 - alpha)
    return weights * miss_probability.pow(gamma) * cross_entropy


def pair_contrastive(first, second, same, margin=1.0):
    """same=1 for matching pairs; no one-half prefactor."""
    distance = torch.linalg.vector_norm(first - second, dim=-1)
    return same * distance.square() + (1 - same) * (margin - distance).clamp_min(0).square()


def squared_triplet(anchor, positive, negative, margin=1.0):
    positive_distance = (anchor - positive).square().sum(-1)
    negative_distance = (anchor - negative).square().sum(-1)
    return (positive_distance - negative_distance + margin).clamp_min(0)


def paired_info_nce(queries, keys, temperature=0.2):
    """One-way paired rows, all other key rows designated negative.

    This is not the full two-view SimCLR objective. Reject zero vectors:
    cosine geometry is undefined there.
    """
    if temperature <= 0 or queries.shape != keys.shape or queries.ndim != 2:
        raise ValueError("Use equal nonempty B by D arrays and positive temperature")
    if queries.shape[0] == 0:
        raise ValueError("At least one pair is required")
    if torch.any(queries.norm(dim=-1) == 0) or torch.any(keys.norm(dim=-1) == 0):
        raise ValueError("Zero vectors do not have a cosine direction")
    queries = queries / queries.norm(dim=-1, keepdim=True)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    logits = queries @ keys.T / temperature
    return F.cross_entropy(logits, torch.arange(len(queries)), reduction="none")


def choose_semi_hard(anchor, positive, negatives, margin=1.0):
    """Nearest strictly semi-hard negative; skip if none. Ties use first row."""
    positive_distance = float((anchor - positive).square().sum())
    negative_distances = (negatives - anchor).square().sum(-1)
    eligible = (negative_distances > positive_distance) & (
        negative_distances < positive_distance + margin
    )
    if not eligible.any():
        return None
    return int(torch.where(eligible, negative_distances, torch.inf).argmin())


def mechanism_fixtures():
    dtype = torch.float64
    focal = []
    for probability in (0.01, 0.1, 0.5, 0.9, 0.99):
        for gamma in (0.0, 2.0):
            logits = torch.tensor([np.log(probability / (1 - probability))],
                                  dtype=dtype, requires_grad=True)
            loss = binary_focal(logits, torch.ones_like(logits), gamma).sum()
            loss.backward()
            focal.append(dict(pt=probability, gamma=gamma, loss=loss.item(),
                              gradient=logits.grad.item()))
    candidates = torch.tensor([[0.5, 0], [1.2, 0], [2.0, 0]], dtype=dtype)
    anchor = torch.tensor([0., 0.], dtype=dtype)
    positive = torch.tensor([1., 0.], dtype=dtype)
    geometry = []
    for negative in candidates:
        geometry.append(dict(negative=negative.tolist(),
                             squared_distance=float(negative.square().sum()),
                             loss=float(squared_triplet(anchor, positive, negative))))
    info = []
    for sims in ([0.8, 0.2, -0.1], [0.2, 0.8, -0.1], [0.8, 0.8, -0.1], [0.4, 0.4, 0.4]):
        for temperature in (1., .2):
            logits = torch.tensor(sims, dtype=dtype) / temperature
            info.append(dict(similarities=sims, temperature=temperature,
                             probabilities=logits.softmax(0).tolist(),
                             loss=float(torch.logsumexp(logits, 0) - logits[0])))
    zeros = torch.zeros((1, 2), dtype=dtype, requires_grad=True)
    collapsed = squared_triplet(zeros, zeros, zeros).sum()
    collapsed.backward()
    return dict(focal=focal, triplets=geometry,
                semi_hard_index=choose_semi_hard(anchor, positive, candidates),
                info_nce=info, collapsed_triplet_loss=collapsed.item(),
                collapsed_triplet_gradient=zeros.grad.tolist(),
                regression=[dict(outlier=v, mse_minimum=v/7, mae_minimum=0.,
                                 huber_delta_one_minimum=1/6) for v in (10.,100.)])


def digit_experiment():
    rows = np.genfromtxt(Path(__file__).with_name("digits-400.csv"),
                        delimiter=",", names=True)
    features = np.column_stack([rows[f"pixel_{i}"] for i in range(64)]) / 16
    digits = rows["digit"].astype(int)
    train, validation = train_test_split(
        np.arange(len(rows)), test_size=.3, random_state=22, stratify=digits)
    inputs = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor((digits == 9).astype(float), dtype=torch.float32)
    records = []
    for seed in (1, 2, 3):
        for objective in ("bce", "positive_weight_9", "focal_gamma_2"):
            torch.manual_seed(seed)
            model = nn.Linear(64, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=.03)
            history = []
            for step in range(401):
                logits = model(inputs[train]).squeeze(-1)
                if objective == "focal_gamma_2":
                    loss = binary_focal(logits, targets[train]).mean()
                else:
                    weight = torch.tensor(9.) if objective == "positive_weight_9" else None
                    loss = F.binary_cross_entropy_with_logits(
                        logits, targets[train], pos_weight=weight)
                if step in (0, 1, 10, 100, 400):
                    history.append(dict(step=step, training_objective=loss.item()))
                if step == 400:
                    break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                probabilities = model(inputs[validation]).squeeze(-1).sigmoid().numpy()
            actual = targets[validation].numpy().astype(int)
            predictions = (probabilities >= .5).astype(int)
            records.append(dict(seed=seed, objective=objective, history=history,
                confusion_matrix=confusion_matrix(actual, predictions, labels=[0,1]).tolist(),
                balanced_accuracy=balanced_accuracy_score(actual, predictions),
                average_precision=average_precision_score(actual, probabilities),
                brier=brier_score_loss(actual, probabilities),
                unweighted_log_loss=log_loss(actual, probabilities, labels=[0,1]),
                validation_probabilities=probabilities.tolist()))
    return dict(train_source_ids=rows["source_id"][train].astype(int).tolist(),
                validation_source_ids=rows["source_id"][validation].astype(int).tolist(),
                records=records)


if __name__ == "__main__":
    result = dict(environment=dict(torch=torch.__version__, numpy=np.__version__),
                  mechanisms=mechanism_fixtures(), digits=digit_experiment())
    Path(__file__).with_name("calculated-inputs.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    for row in result["digits"]["records"]:
        print(row["seed"], row["objective"], row["confusion_matrix"],
              f'AP={row["average_precision"]:.6f} Brier={row["brier"]:.6f} '
              f'logloss={row["unweighted_log_loss"]:.6f}')
