// Authoring blueprint: NumPy: Arrays, Broadcasting & Vectorization.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Trace sensor-array calculations from coordinates and storage through broadcasting and reductions to a checked numerical report.",
  "outcomes": [
    "Map array shape, axes, indexing and masks to concrete observations",
    "Predict view/copy writes and identify every broadcast operand of an output cell",
    "Select reduction axes and verify an independent multi-sensor workflow"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules"
  ],
  "sequence": [
    "Introduce a time-by-sensor array and its dtype",
    "Select elements and observations by their coordinates",
    "Follow a view or a copy to the underlying values",
    "Match trailing dimensions and inspect broadcast operands cell by cell",
    "Trace contributions along a chosen reduction axis",
    "Separate elementwise operations, matrix products and numeric precision",
    "Build a calibrated report with explicit missing-value handling",
    "Solve a changed-sensor dataset before consulting deeper array tools"
  ],
  "visual": {
    "type": "Broadcast coordinate explorer",
    "question": "Which input and offset produce this output value?",
    "interaction": "Change the offset shape, inspect aligned dimensions and select an output cell; include incompatible shapes."
  },
  "visuals": [
    {
      "type": "Selection map",
      "question": "Which original coordinates survive this selection?",
      "interaction": "Compare scalars, rows, columns, slices and masks."
    },
    {
      "type": "Buffer mapping",
      "question": "Will writing the selection also change the original array?",
      "interaction": "Trace view/copy mappings and perform a controlled write."
    },
    {
      "type": "Reduction contributors",
      "question": "Which measurements contribute to this mean?",
      "interaction": "Switch axis and keepdims while tracing input groups to output cells."
    }
  ],
  "practice": {
    "task": "Calibrate and summarize a new three-sensor dataset while preserving source data and rejecting invalid observations.",
    "success": "Shape, coordinates, missing policy, broadcast alignment, axis choice and expected values all agree with an independent NumPy calculation."
  },
  "misconceptions": [
    "Axis numbers identify coordinates rather than a universal meaning of row or column",
    "Broadcasting does not imply materializing repeated copies",
    "Advanced indexing and basic slices differ in memory sharing",
    "Vectorized syntax alone does not prove better performance"
  ],
  "sources": [
    "https://numpy.org/doc/stable/user/basics.broadcasting.html",
    "https://numpy.org/doc/stable/user/basics.indexing.html",
    "https://numpy.org/doc/stable/user/basics.copies.html",
    "https://numpy.org/doc/stable/reference/generated/numpy.mean.html"
  ],
  "depth": "core",
  "reviewFocus": "Verify shapes, zero-length/scalar broadcasting, dtype and tolerance conventions, actual mutation behavior and the complete numerical workflow. No unmeasured speed claims.",
  "designRecord": "docs/teaching/numpy-foundations-reimplementation.md"
};
