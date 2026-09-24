export default {
  summary: 'Build and examine ConvNeXt’s spatial/channel block, its V2 response normalization and a real masked-image learning experiment.',
  outcomes: ['Trace V1/V2 block shapes and distinguish spatial from channel mixing', 'Construct complete V1/V2 hierarchies and compare a matched-state Torchvision block', 'Identify normalization reduction axes and global response coupling', 'Calculate exact parameter and convolution/linear MAC budgets', 'Protect hidden targets from reconstruction input leakage', 'Interpret recorded paired fits, frozen probes and feature diagnostics separately', 'Fold compatible linear branches and diagnose the nonlinear boundary'],
  prerequisites: ['Depthwise Separable & Dilated Convolutions'],
  sequence: ['Architecture versus recipe evidence', 'Block topology and normalization axes', 'Hierarchy and parameter/work budget', 'GRN mechanism and initial gradients', 'Masked-input boundary and actual small training', 'Matched library route, branch folding and hybrid connections', 'Changed-design practice and Capsule Networks bridge'],
  visual: 'Feature lattices, channel maps, a live budget, editable trained-digit reconstruction and a kernel-folding stencil',
  practice: 'Diagnose wrong axes and target leakage, modify expansion, derive budgets, evaluate evidence and test branch-fusion limits',
  references: ['https://arxiv.org/pdf/2201.03545', 'https://arxiv.org/pdf/2301.00808', 'https://docs.pytorch.org/vision/0.29/models/convnext.html'],
  reviewFocus: 'Complete preserved manuscript; independent exact mechanisms, Torchvision gradients, hidden-edit invariance, live linked views and honest fit/benchmark boundaries.',
  depth: 'core',
  designRecord: 'docs/teaching/CONVNEXT-IMPLEMENTATION.md',
};
