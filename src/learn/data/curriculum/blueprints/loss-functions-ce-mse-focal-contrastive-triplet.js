export default {
  "summary": "Understand losses as numerical teaching signals, inspect their actual derivatives, and distinguish changed objectives from changed decisions on recorded data.",
  "outcomes": [
    "Separate residual, penalty and parameter update",
    "Compare mean, median and Huber influence through editable observations",
    "Evaluate stable cross-entropy and the complete focal derivative",
    "Inspect recorded probabilities and threshold-induced confusion counts",
    "Choose and audit contrastive pairs, triplet distance conventions and mining",
    "Compute InfoNCE with explicit candidate masks, temperature and reduction denominators"
  ],
  "prerequisites": [
    "Backpropagation & Automatic Differentiation"
  ],
  "sequence": [
    "Residual → penalty → gradient → update",
    "Regression objectives and outlier influence",
    "Stable cross-entropy and focal gradient contributions",
    "Recorded imbalanced digit experiment and threshold decisions",
    "Pair and triplet geometry, mining and collapse",
    "InfoNCE competition, candidates, supervised contrast and angular margins",
    "Reduction, storage, practical objective selection and seven transfer exercises"
  ],
  "visual": {
    "type": "Spatial observation rows and fitted-loss curve; signed gradient contributions; recorded specimen probability plot and confusion cells; equal-scale triplet geometry with category shapes; probability shares and candidate masks; exact matrix area and denominator accounting.",
    "question": "Which observations and candidates determine the update, and which control changes decisions without changing fitted probabilities?",
    "interaction": "Edit actual observations, probabilities, counts, margins, coordinates, similarities and temperatures with immediate current calculations. Recorded nine-run inspection never implies browser training. No learner-prediction field or answer gate."
  },
  "practice": {
    "task": "Solve changed regression, focal, threshold, triplet, candidate-mask, reduction and collapse cases.",
    "success": "Use the stated objective, derivative, candidate set, distance units and denominator; explain a meaningful changed result and a genuine null."
  },
  "misconceptions": [
    "Penalty height equals derivative size",
    "Focal gradients are BCE gradients times the focal loss weight",
    "Threshold changes retrain or recalibrate a model",
    "Squared and unsquared margins are interchangeable",
    "Nonzero loss always produces a nonzero embedding gradient"
  ],
  "sources": [
    "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html",
    "https://arxiv.org/abs/1708.02002",
    "https://arxiv.org/abs/1503.03832",
    "https://arxiv.org/abs/2002.05709"
  ],
  "reviewFocus": "Full prepared prose/math/practice retained; independently verified focal derivatives, mining ties, saved specimen identity and null intervals; real pointer/numeric controls, all seven representations, readable phone charts and source-bound CPU evidence.",
  "depth": "core",
  "designRecord": "docs/teaching/drafts/loss-functions-ce-mse-focal-contrastive-triplet/design.md"
};
