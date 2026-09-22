export default {
  summary: 'Build and diagnose Xavier, Kaiming, orthogonal and width-aware initialization through exact mechanisms and matched recorded experiments.',
  outcomes: [
    'Derive the forward second-moment and fan-in/fan-out rules with their assumptions',
    'Distinguish average squared gain from every-direction sensitivity and a gated Jacobian',
    'Construct rectangular orthogonal draws with QR and compare library contracts',
    'Trace hidden symmetry and a zero head through simultaneous gradient updates',
    'Reproduce the fixed 400-digit initialization experiment without confusing validation with test evidence',
    'Implement the restricted μP forward/Adam recipe and compare two same-state updates with MuReadout/MuAdam',
    'Diagnose truncation, precision and normalization misconceptions with actual measured examples'
  ],
  prerequisites: ['Transfer Learning & Fine-Tuning Strategies'],
  sequence: ['Twenty-layer forward/backward signal observations', 'Four-value moment mechanism and activation-specific derivation', 'Fan conventions and initialization recipes', 'Directional geometry, orthogonal construction and symmetry', 'Complete real-input experiment', 'Width-aware initialization, forward paths and Adam updates', 'Failure checks, changed-case practice and residual connection'],
  visual: { type: 'Recorded log-scale signal traces, persistent value/distance views, equal-axis ellipse and gated Jacobian, hidden-unit gradient paths, actual digit specimens, shape/rate workbench and precision tables.', question: 'What survives an initialized forward pass, and what can change on the first update?', interaction: 'Immediate editable moments, directions, depth, rates and width; bounded real gradient steps; saved training selections remain distinct from newly computed arithmetic.' },
  practice: { task: 'Compute changed fan/gain recipes, diagnose mean/variance claims, construct a lost direction, trace a zero head, configure a wider μP model and design a controlled failure investigation.', success: 'Explain preserved quantities and assumptions, match ordinary library behavior, and retain fixed evidence roles.' },
  misconceptions: ['ReLU halves variance', 'Xavier exactly preserves unequal forward and backward fans', 'Orthogonal weights certify a ReLU network is an isometry', 'Any zero initialization prevents all learning', 'A good forward scale controls every training update', 'A μP rule for Adam can be copied into SGD', 'Validation rankings prove a universal initializer winner'],
  sources: ['https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf', 'https://arxiv.org/pdf/1502.01852', 'https://arxiv.org/pdf/1312.6120', 'https://arxiv.org/abs/2203.03466', 'https://github.com/microsoft/mup', 'https://docs.pytorch.org/docs/2.14/nn.init.html'],
  reviewFocus: 'Full packet conservation, native experiment identity, cross-language tiny mechanisms, μP same-state library parity, readable geometry and honest live/recorded boundaries.',
  depth: 'core',
  designRecord: 'docs/teaching/drafts/weight-initialization-xavier-kaiming-p/design.md'
};
