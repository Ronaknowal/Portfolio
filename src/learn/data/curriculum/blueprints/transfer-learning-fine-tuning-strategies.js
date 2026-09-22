export default {
  "summary": "Reuse a learned representation, define a complete freeze policy, build low-rank and feature adapters, and choose a model from properly separated evidence.",
  "outcomes": [
    "Separate backbone ownership from head label meaning",
    "Control parameter gradients, optimizer membership and module state independently",
    "Run the complete fixed 400-row transfer comparison",
    "Calculate and edit both LoRA factors and their simultaneous gradients",
    "Count adapter/head/optimizer storage with explicit exclusions",
    "Choose from validation candidates under a budget without inventing new test evidence",
    "Preserve architecture, preprocessing, label order and weights in a replayable artifact"
  ],
  "prerequisites": [
    "Batch/Layer/Group/RMS Normalization"
  ],
  "sequence": [
    "Backbone/head reuse with real source and target specimens",
    "Freeze controls and gradient-through-frozen-weight mechanism",
    "Five evidence partitions and complete offline program",
    "LoRA factor paths and merged equivalence",
    "Feature adapter and honest resource accounting",
    "Derivatives, recorded validation selection and source retention",
    "Checkpoint semantics, advanced method boundaries, teaching schedule and six exercises"
  ],
  "visual": {
    "type": "Recorded pixels and architecture ownership; independent freeze/state/optimizer lanes; partition bins; editable LoRA matrices and two computation paths; parameter budget; recorded CE/retention evidence; checkpoint label semantics.",
    "question": "What is reused, what may change, and what evidence justifies adapting it?",
    "interaction": "Show outputs, gradients and next-step previews immediately. Edit factors, input, targets and rates; apply bounded real state steps. Budgets reinterpret saved validation records and retain the original unfavorable 77/100 test report."
  },
  "practice": {
    "task": "Diagnose head meanings and frozen behavior, select under a changed budget, calculate changed LoRA gradients, count resource units and design a controlled rank investigation.",
    "success": "Keep identical tensor shape distinct from identical semantics, evaluate gradients from one pre-update state, and preserve validation/test roles."
  },
  "misconceptions": [
    "Frozen base weights guarantee unchanged adapted behavior",
    "no_grad prevents running-buffer changes",
    "Low rank always means fewer parameters",
    "All same-shaped output heads have the same meaning",
    "Selecting another displayed method gives a fresh test result",
    "Trainable parameter count is measured peak memory or latency"
  ],
  "sources": [
    "https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html",
    "https://arxiv.org/abs/1411.1792",
    "https://arxiv.org/abs/2106.09685",
    "https://arxiv.org/abs/1902.00751",
    "https://aclanthology.org/P18-1031/"
  ],
  "reviewFocus": "Complete packet, real 18-fit replay, LoRA/BN gradient and state correctness, strict evidence selection and checkpoint semantics, readable plots and actual pointer/keyboard/phone controls.",
  "depth": "core",
  "designRecord": "docs/teaching/drafts/transfer-learning-fine-tuning-strategies/design.md"
};
