export default {
  "summary": "Build normalization from a shared ruler, see exactly which values share statistics, and separate live batch calculations, learned affine values and running memory.",
  "outcomes": [
    "Compute centering, variance, scaling and affine output",
    "Map BN/LN/GN/IN onto explicit tensor coordinates",
    "Contrast LayerNorm with RMSNorm under offsets and scaling",
    "Separate training/evaluation modes, buffers and optimizer updates",
    "Trace every statistics gradient path",
    "Read twelve matched recorded runs without universal ranking claims",
    "Explain pre/post normalization through explicit residual forks"
  ],
  "prerequisites": [
    "Loss Functions (CE, MSE, Focal, Contrastive, Triplet)"
  ],
  "sequence": [
    "One ruler built from four values",
    "Interactive tensor membership and axis ownership",
    "Vector geometry: centering versus RMS scaling",
    "BatchNorm current output versus stored state",
    "Complete statistics gradients and affine update",
    "Recorded matched digit runs",
    "Residual placement, precision, resource accounting and eight exercises"
  ],
  "visual": {
    "type": "Staged ruler; editable 2×4×1×2 tensor groups; equal-scale vector geometry; state transition lanes; dependency graph and summed derivative table; recorded learning curves; explicit residual fork/merge paths.",
    "question": "Whose output changes when this cell, mode, affine value or residual placement changes—and why?",
    "interaction": "Expose every current result before action. Valid tensor/value/momentum/rate edits immediately recompute exact small mechanisms. Forward controls commit real buffer state. Recorded fit selectors inspect retained measurements only."
  },
  "practice": {
    "task": "Change normalization groups, offsets, state policies, epsilon, gradients and residual placements; critique recorded evidence.",
    "success": "Preserve group axes and denominators, separate population/corrected variances, include all gradient branches, and explain offset/null/mode contrasts."
  },
  "misconceptions": [
    "All normalization shares statistics over the batch",
    "BatchNorm always requires more than one image",
    "eval and no_grad mean the same thing",
    "RMSNorm subtracts the mean",
    "The normalization gradient treats statistics as constants",
    "An identity skip guarantees the total Jacobian is identity"
  ],
  "sources": [
    "https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html",
    "https://arxiv.org/abs/1502.03167",
    "https://arxiv.org/abs/1607.06450",
    "https://arxiv.org/abs/1803.08494",
    "https://arxiv.org/abs/1910.07467"
  ],
  "reviewFocus": "Exact group membership and cross-example nulls, corrected running variance, real state transitions, full input gradients, 12 reproduced fits, visible fork/merge paths and narrow-screen geometry without KaTeX SVG overrides.",
  "depth": "core",
  "designRecord": "docs/teaching/drafts/batch-layer-group-rms-normalization/design.md"
};
