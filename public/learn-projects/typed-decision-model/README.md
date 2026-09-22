# Build a typed decision model

This folder accompanies the eight-stage research project at `/learn/projects/typed-decision-model`.

## What runs

The original teaching implementation fits a small bidirectional transformer from random initialization. Its shared scorer reads option-marker positions, accepts 2–8 candidate descriptions, and emits probabilities over those options. The single supported question type is `choice`.

The program includes manual attention and an equivalent PyTorch scaled-dot-product-attention route, stable cross-entropy, shuffled candidate order, a lexical baseline, separate training/calibration/test fixtures, temperature fitting, model/configuration saving, and local JSON inference. It does not download anything or invoke external APIs.

It is not Jev, Laya, or a reproduction of either model. Pretrained-model fine-tuning, other question types, reinforcement-learning objectives and network serving are explained extensions, not executed deliverables.

## Setup and commands

Use Python 3.10+ and PyTorch 2.6+ (under 3). Create and activate a virtual environment using your shell's usual commands, then install PyTorch using the official platform instructions at https://pytorch.org/get-started/locally/ or the CPU package appropriate to your environment.

```shell
python -m pip install "torch>=2.6,<3"
python typed_decision.py verify
python typed_decision.py train --steps 600 --seed 7 --output decision-artifact
python typed_decision.py predict --artifact decision-artifact --input request.json --wrong-cost 10 --review-cost 1
```

`--library-attention` on `train` replaces the explicit attention arithmetic with `torch.nn.functional.scaled_dot_product_attention`. `predict` uses that library route. Verification compares them numerically with the same parameters and inputs. No runtime claims are made from this parity check.

Save Python and PyTorch versions with a run. The recorded author run used PyTorch 2.14.0+cpu and one CPU thread. Numerical results can vary across versions/devices.

## Artifacts and interpretation

Training writes these files to the requested output directory:

- `weights.pt`: state dictionary, read with `weights_only=True` by the inference program.
- `config.json`: vocabulary, temperature, architecture identity, supported question type and token budget.
- `fixture-data.json`: all constructed training and evaluation examples.
- `report.json`: environment, seed, training trace, losses and accuracies.

The checked-in `verified-report.json` is the measured 600-step author result. It shows training loss falling, 12/18 raw model test answers correct, all 18 lexical baseline answers correct, and worsened test log loss after an extreme temperature fit on easy calibration cases. Retain the failed result; it motivates better task data and calibration evaluation. A preliminary 120-step author run stayed near chance (6/18 test answers correct). Increasing the budget established that the model can learn the fixture while leaving unfavorable generalization evidence visible.

The 36/18/18 train/calibration/test cases use separate sentence template families but share an intentionally easy task vocabulary. Three stress cases use unseen wording. These are original constructed examples, not real support data and not evidence for language-model performance. The tiny model is not suitable for live routing based on these results.

The cost policy assumes correct actions cost zero, wrong actions share the supplied cost, and review resolves the case at its stated cost. It chooses review on equal expected costs using a tolerance of 1e-12 times the larger of 1 and both cost magnitudes. Real policies need observed error, review quality, capacity and delay evidence.

## Request format

`request.json` is the complete sample. Stable option IDs identify application outputs; descriptions are the model's text input. Each request needs nonempty `question`, `state`, and 2–8 unique-ID options. Text and token limits produce explicit errors. Inputs are rejected above 128 tokens instead of silently losing evidence.

The character limits and token limit are teaching-program bounds, not claims about pretrained encoders. The toy tokenizer lowercases text and extracts ASCII a-z word runs; digits, punctuation and non-Latin writing are not modeled. Question/state/description fields with no supported word tokens are rejected. Unknown supported words map to an unknown token because vocabulary is fit only on training data. Replace this tokenizer for multilingual or numeric tasks and rerun training and evaluation.

## Additional runnable tools

Keep these files beside `typed_decision.py`; they import its actual mechanisms instead of copying them:

- `research_tools.py verify` checks its trace/diagnostic arithmetic.
- `research_tools.py trace --output trace.json` follows the real tokenizer and tensors through a seeded random model and one illustrative SGD update. The public `trace-fixture.json` is this output, not a trained inference result.
- `research_tools.py evaluate --artifact decision-artifact --output diagnostics.json` reports examples, confusion matrices, reliability bins, review coverage/risk/cost and ID/order perturbations from your saved checkpoint. `verified-diagnostics.json` records the author's exact seed-7 checkpoint.
- `pretrained_decision.py verify` exercises a real Transformers adapter using a tiny random local BERT, without downloading pretrained weights. Install `transformers>=4.48,<5` in the same environment first; the author checked version 4.57.6 with PyTorch 2.14.0+cpu.
- `pretrained_decision.py train --checkpoint PATH_TO_LOCAL_ENCODER --output pretrained-artifact` and `pretrained_decision.py predict --artifact pretrained-artifact --input request.json` provide an implemented library training/inference route. Use a compatible bidirectional CLS/SEP/PAD encoder and inspect `--help` for freezing, revision and download controls. Actual fine-tuning of a pretrained checkpoint remains an unexecuted experiment.

The project pages explain each function, tensor transformation, numerical example and experiment interpretation. Their mechanism-specific links lead to the existing deeper lessons; the dedicated typed-decision companion remains a planned outline.

## Reproduction versus research

For real data, keep groups/conversations/entities disjoint, create a development split before model selection, reserve calibration data separately, and lock test access. Compare changed architectures or objectives across seeds and report uncertainty, label-order sensitivity, distractors, unseen tasks and measured inference costs. A schema-valid result can be wrong; a sharp distribution can be poorly calibrated.

Primary references are linked and annotated in the project stages: TypeSafe's API documentation for interface scope, public Laya source for its disclosed mechanism, and PyTorch/Transformers documentation for the ordinary library routes. Inspect upstream versions before extending the model. No pretrained-model training, GPU experiments, API deployment, or benchmark reproduction was performed for this guide.
