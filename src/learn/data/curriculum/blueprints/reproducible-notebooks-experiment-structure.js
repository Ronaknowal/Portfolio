// Authoring blueprint: Reproducible Notebooks & Experiment Structure.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Build a repeatable computation by separating notebook source, live kernel state and saved outputs, then recording the exact inputs and executing in a fresh kernel.",
  "outcomes": [
    "Predict stale output after editing a cell and diagnose missing names after restart",
    "Separate pure transformation from IO and explicit path/environment inputs",
    "Explain random-state consumption and isolate experiment streams",
    "Identify changed run ingredients even when results happen to match",
    "Execute an actual notebook and create an independent replay package"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules",
    "Testing, Debugging & Dependency Management",
    "NumPy: Arrays, Broadcasting & Vectorization",
    "Scientific File Formats, Schemas & Reliable Data I/O"
  ],
  "sequence": [
    "Choose exact, numerical or conclusion-level agreement",
    "Inspect document/kernel/output state and replay dependency order",
    "Extract explicit functions and separate raw/derived artifacts",
    "Select the actual notebook interpreter and stable working directory",
    "Trace random-consumer cursors before using NumPy spawned streams",
    "Preserve train/evaluation boundaries; distinguish validity from repeatability",
    "Trace cache-key ingredients and serialize a run manifest",
    "Execute a complete notebook in a new kernel with failure-sensitive assertions",
    "Independently save source/data bytes and demonstrate identical-result/different-input evidence"
  ],
  "visual": {
    "type": "Document versus kernel versus saved output",
    "question": "What changes when a cell is edited but not run?",
    "interaction": "Edit source, execute individual cells, restart and run all while viewing separate states."
  },
  "visuals": [
    {
      "type": "Random-stream cursor ownership",
      "question": "Does an extra preprocessing draw change the model draw?",
      "interaction": "Switch shared/separate finite tapes and draw counts; real NumPy stream checks follow."
    },
    {
      "type": "Run ingredient identity",
      "question": "Does an identical filename or result establish the same run?",
      "interaction": "Change data, offset or code and compare filename-only versus complete fixed-fixture keys."
    }
  ],
  "practice": {
    "task": "Execute the offset notebook, deliberately expose a stale assertion, then build a source/data/config replay package and change data without changing the mean.",
    "success": "Real fresh kernels run baseline and corrected changed input; hidden variables and stale assertions fail. Independent source/data hashes distinguish equal-output runs."
  },
  "misconceptions": [
    "Editing source is not execution",
    "Restart does not undo files or erase historical output",
    "A seed does not identify the whole random execution",
    "A hash identifies bytes, not scientific truth",
    "Repeatable leakage remains invalid",
    "Recording a label is not archiving source"
  ],
  "sources": [
    "https://nbformat.readthedocs.io/en/latest/format_description.html",
    "https://nbconvert.readthedocs.io/en/latest/execute_api.html",
    "https://numpy.org/doc/stable/reference/random/parallel.html",
    "https://numpy.org/doc/stable/reference/random/compatibility.html"
  ],
  "depth": "core",
  "reviewFocus": "Check the installed kernel rather than assume terminal and notebook interpreters agree. Retain the real notebook artifacts; label random tapes as conceptual and archive identities as evidence rather than proof of trust. Current bridge goes to Code Documentation.",
  "designRecord": "docs/teaching/reliability-authoring-design.md"
};
