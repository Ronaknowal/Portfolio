export default {
  summary: 'Design, inspect and resume learning-rate policies using explicit update clocks, endpoint conventions, noise/stability reasoning and validation state.',
  outcomes: [
    'Derive the scalar error multiplier and explain a stable constant-rate counterexample',
    'Construct finite warmup/cosine, stepped, geometric and one-cycle schedules with declared endpoints',
    'Distinguish a consumed optimizer rate from a prepared next rate',
    'Trace microbatches, accumulation, skipped attempts and committed updates',
    'Derive the exact mean and noise contribution for a stated scalar model',
    'Explain plateau thresholds, patience, cooldown and rate floors using observed state',
    'Run actual PyTorch scheduling and restore a complete reproducible small training state',
    'Compare policies with appropriate data, validation, exposure and tuning controls',
  ],
  prerequisites: ['Gradient Descent Variants (SGD, Adam, AdaGrad, RMSProp, LAMB, LARS)'],
  sequence: ['A rate multiplies an update', 'Choose the clock', 'Build a finite schedule', 'Investigate noise and stability', 'Use OneCycleLR and restarts', 'Respond to validation', 'Accumulate and resume correctly', 'Practise and compare'],
  visual: {
    type: 'Discrete phase curves, exact scalar noise moments, event lanes and validation-state transitions',
    question: 'Which rate is actually consumed, why did it change, and what consequence follows under the stated model?',
    interaction: 'Inspect bounded schedule points, alter curvature/noise, step through accumulated/skipped updates and validate threshold-triggered changes.',
  },
  practice: {
    task: 'Derive a changed-budget schedule, repair a shifted update clock, calculate a noise contribution, trace plateau ties and restore an interrupted experiment.',
    success: 'Explicit rate/state values, justified endpoint and comparison contracts, independent native checks and an explanation of the policy limits.',
  },
  misconceptions: ['A rate is the distance a parameter travels', 'A smooth plotted curve defines discrete update timing', 'Every scheduler steps per batch or per epoch', 'Warmup makes an arbitrarily high rate safe', 'Cosine necessarily reaches its minimum on the last used update', 'A restart reinitializes model parameters', 'Equality always resets plateau patience', 'Restoring model weights alone resumes training', 'One tuned toy result establishes a universal ranking'],
  sources: ['https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.OneCycleLR.html', 'https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html', 'https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html', 'https://arxiv.org/pdf/1608.03983', 'https://arxiv.org/pdf/1706.02677', 'https://arxiv.org/pdf/1506.01186', 'https://arxiv.org/pdf/1708.07120', 'https://course.fast.ai/Lessons/lesson18.html'],
  depth: 'core',
  designRecord: 'docs/teaching/LEARNING-RATE-SCHEDULES-DESIGN.md',
  reviewFocus: 'Validate used versus prepared rates, finite endpoint/phase boundaries, exact scalar moments, native API version, accumulation and skip semantics, checkpoint state, mathematical and mobile labels, and appropriately qualified empirical claims.',
};
