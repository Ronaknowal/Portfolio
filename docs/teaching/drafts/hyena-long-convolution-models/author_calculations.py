"""Bounded author evidence for the manuscript and later browser investigation."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from splice_models import ROOT, SpliceReader, load_data, causal_convolution, metrics, ALPHABET


def load_fit(kind, seed):
    saved = np.load(ROOT / "splice-fits.npz", allow_pickle=False)
    model = SpliceReader(kind)
    prefix = f"{kind}_{seed}__state__"
    model.load_state_dict({name.removeprefix(prefix): torch.tensor(saved[name])
                           for name in saved.files if name.startswith(prefix)})
    return model.eval(), saved


def main():
    torch.set_num_threads(2)
    tokens, labels, roles, data = load_data()
    report = {"role_counts": {key: len(ids) for key, ids in roles.items()},
              "role_group_counts": {}, "saved_logits_parity": {}, "numerics": {}}
    group_sets = {key: {data["groups"][i] for i in ids} for key, ids in roles.items()}
    for key, groups in group_sets.items():
        report["role_group_counts"][key] = len(groups)
    assert not any(group_sets[a] & group_sets[b] for a, b in [("fit", "validation"), ("fit", "assessment"), ("validation", "assessment")])
    validation_tokens = tokens[roles["validation"]]
    with torch.no_grad():
        for kind, seed in [("linear", 29), ("gated", 29), ("ungated", 29), ("gated", 71)]:
            model, saved = load_fit(kind, seed)
            key = f"{kind}_{seed}"
            actual = model(validation_tokens).numpy()
            report["saved_logits_parity"][key] = float(np.max(np.abs(actual - saved[f"{key}__validation_logits"])))
        model, _ = load_fit("gated", 29)
        before = model(validation_tokens)
        edited = validation_tokens.clone()
        edited[:, 30:32] = ALPHABET.index("A")
        after = model(edited)
        source_ids = data["role_source_ids"]["validation"]
        chosen = [i for i in range(len(edited)) if labels[roles["validation"][i]] == 0
                  and before[i].argmax() == 0 and after[i].argmax() != 0]
        fixtures = []
        for i in chosen[:2]:
            fixtures.append({"validation_index": i, "source_id": source_ids[i],
                             "true_class": 0, "sequence": "".join(ALPHABET[j] for j in validation_tokens[i]),
                             "edited_sequence": "".join(ALPHABET[j] for j in edited[i]),
                             "before_logits": before[i].tolist(), "after_logits": after[i].tolist(),
                             "before_probabilities": before[i].softmax(-1).tolist(),
                             "after_probabilities": after[i].softmax(-1).tolist()})
        report["sequence_fixtures"] = fixtures
        blank = torch.full((1, 60), ALPHABET.index("N"))
        report["all_ambiguity"] = model(blank).softmax(-1)[0].tolist()
        sample = validation_tokens[:4]
        full_hidden = model(sample, return_hidden=True)
        prefix_hidden = model(sample[:, :31], return_hidden=True)
        changed = sample.clone(); changed[:, 31:] = ALPHABET.index("N")
        changed_hidden = model(changed, return_hidden=True)
        report["numerics"]["prefix_length_error"] = (full_hidden[:, :31] - prefix_hidden).abs().max().item()
        report["numerics"]["future_edit_prefix_error"] = (full_hidden[:, :31] - changed_hidden[:, :31]).abs().max().item()
        report["learned_filters"] = [block.filter(60).numpy().tolist() for block in model.blocks]
    # A direct, separately indexed convolution and the FFT must have matching derivatives.
    generator = torch.Generator().manual_seed(301)
    values = torch.randn(2, 7, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    kernel = torch.randn(7, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    actual = causal_convolution(values, kernel)
    reference = torch.stack([sum(kernel[t-j] * values[:, j] for j in range(t+1)) for t in range(7)], dim=1)
    actual_gradient = torch.autograd.grad(actual.square().sum(), (values, kernel), retain_graph=True)
    reference_gradient = torch.autograd.grad(reference.square().sum(), (values, kernel))
    report["numerics"]["fft_direct_output_error"] = (actual - reference).abs().max().item()
    report["numerics"]["fft_direct_gradient_error"] = max((a-b).abs().max().item() for a,b in zip(actual_gradient, reference_gradient))
    assert max(report["saved_logits_parity"].values()) < 1e-6
    assert report["numerics"]["prefix_length_error"] < 1e-4
    assert report["numerics"]["future_edit_prefix_error"] < 1e-4
    assert report["numerics"]["fft_direct_output_error"] < 1e-12
    assert report["numerics"]["fft_direct_gradient_error"] < 1e-10
    assert len(fixtures) == 2
    (ROOT / "author-results.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in report.items() if key != "learned_filters"}, indent=2))


if __name__ == "__main__":
    main()
