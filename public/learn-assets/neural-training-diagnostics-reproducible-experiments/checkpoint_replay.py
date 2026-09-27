"""Compare one-variable checkpoint omissions on a fixed offline CPU run."""

import copy
import json
from pathlib import Path

import torch
from torch import nn

from wine_diagnostics import load_inputs

PACKET = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)


def fresh_training_state():
    torch.manual_seed(23)
    model = nn.Sequential(nn.Linear(13, 8), nn.Tanh(), nn.Dropout(0.25), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    order_generator = torch.Generator().manual_seed(88)
    return model, optimizer, scheduler, order_generator


def train_until(state, features, labels, order, cursor, completed, stop):
    model, optimizer, scheduler, order_generator = state
    model.train()
    trace = []
    for update in range(completed, stop):
        if cursor == len(features):
            order = torch.randperm(len(features), generator=order_generator)
            cursor = 0
        batch_ids = order[cursor:cursor + 12]
        cursor += len(batch_ids)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features[batch_ids]), labels[batch_ids])
        learning_rate = optimizer.param_groups[0]["lr"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        trace.append({"update": update + 1, "training_positions": batch_ids.tolist(),
                      "pre_update_loss": loss.item(), "learning_rate_used": learning_rate})
    return order, cursor, trace


def snapshot(state, order, cursor, completed):
    model, optimizer, scheduler, order_generator = state
    return copy.deepcopy({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "order_rng": order_generator.get_state(),
        "order": order,
        "cursor": cursor,
        "completed_updates": completed,
        "module_training": model.training,
    })


def restore(checkpoint, omission):
    state = fresh_training_state()
    model, optimizer, scheduler, order_generator = state
    model.load_state_dict(checkpoint["model"])
    optimizer_state = copy.deepcopy(checkpoint["optimizer"])
    if omission == "optimizer_buffers":
        optimizer_state["state"] = {}  # Preserve learning rate and other hyperparameters.
    optimizer.load_state_dict(optimizer_state)
    if omission != "scheduler":
        scheduler.load_state_dict(checkpoint["scheduler"])
    order_generator.set_state(checkpoint["order_rng"])
    if omission == "active_order_cursor":
        order = torch.randperm(len(checkpoint["order"]), generator=order_generator)
        cursor = 0
    else:
        order, cursor = checkpoint["order"].clone(), checkpoint["cursor"]
    model.train(checkpoint["module_training"])
    if omission != "torch_rng":
        torch.set_rng_state(checkpoint["torch_rng"])
    return state, order, cursor


def main():
    features, labels, split, _, _ = load_inputs()
    features, labels = features[split["train"]], labels[split["train"]]
    reference = fresh_training_state()
    order = torch.randperm(len(features), generator=reference[3])
    order, cursor, prefix = train_until(reference, features, labels, order, 0, 0, 5)
    checkpoint = snapshot(reference, order, cursor, 5)
    # Serialize through memory to test the tensor/plain-container checkpoint contract.
    import io
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    _, _, continuation = train_until(reference, features, labels, order, cursor, 5, 12)
    reference_parameters = [p.detach().clone() for p in reference[0].parameters()]
    comparisons = []
    for omission in ["none", "optimizer_buffers", "torch_rng", "active_order_cursor", "scheduler"]:
        state, resumed_order, resumed_cursor = restore(checkpoint, omission)
        _, _, trace = train_until(state, features, labels, resumed_order, resumed_cursor, 5, 12)
        differences = [(p.detach() - expected).abs().max().item()
                       for p, expected in zip(state[0].parameters(), reference_parameters)]
        exact_parameters = all(torch.equal(p.detach(), expected)
                               for p, expected in zip(state[0].parameters(), reference_parameters))
        comparisons.append({"omission": omission, "trace": trace,
                            "exact_trace_equal": trace == continuation,
                            "exact_parameters_equal": exact_parameters,
                            "maximum_parameter_difference": max(differences)})
    result = {"checkpoint_after_update": 5, "checkpoint_cursor": cursor,
              "optimizer_buffer_omission_preserves_parameter_groups": True,
              "training_row_ids": split["train"], "prefix": prefix,
              "uninterrupted_continuation": continuation, "comparisons": comparisons}
    (PACKET / "checkpoint-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for comparison in comparisons:
        print(comparison["omission"], comparison["exact_trace_equal"],
              comparison["exact_parameters_equal"], f"{comparison['maximum_parameter_difference']:.9f}")


if __name__ == "__main__":
    main()
