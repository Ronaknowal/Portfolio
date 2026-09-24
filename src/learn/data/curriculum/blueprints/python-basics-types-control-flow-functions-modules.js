// Authoring blueprint: Python Basics: Types, Control Flow, Functions & Modules.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Turn text readings into a validated report while explaining names, objects, control flow, function results and module boundaries.",
  "outcomes": [
    "Run and modify a complete Python program from a clean start",
    "Predict mutation versus rebinding and trace branches, loops and calls",
    "Build and diagnose a reusable two-file report with explicit input contracts"
  ],
  "prerequisites": [],
  "sequence": [
    "Run a tiny script and distinguish source from output",
    "Introduce text, numeric and missing values before operations",
    "Connect names to objects and distinguish copying from aliasing",
    "Trace a decision loop with the input cursor and accepted values visible",
    "Separate local call state, returned values and printed output",
    "Handle explicit failures and compose a two-file report",
    "Solve a changed-input investigation and an independent selection problem",
    "Revisit collection, scope, numeric and import details in optional branches"
  ],
  "visual": {
    "type": "Name-to-object explorer",
    "question": "Does this operation change a list or move a name?",
    "interaction": "Compare alias/copy behavior and step append versus rebinding with reference arrows and visible list elements."
  },
  "visuals": [
    {
      "type": "Control-flow gates",
      "question": "Which reading is retained and why?",
      "interaction": "Change the threshold and step missing-value checks, comparison and append."
    },
    {
      "type": "Call frame and output",
      "question": "What does the caller receive if the function prints instead of returns?",
      "interaction": "Trace the local frame, result path and console separately."
    }
  ],
  "practice": {
    "task": "Change the temperature report, diagnose empty/malformed/nonfinite input, then implement a highest-score selection with a stated tie policy.",
    "success": "Correct results, zero preservation, unchanged source data, explained exceptions and deliberately tested tie behavior."
  },
  "misconceptions": [
    "Assignment does not copy an object",
    "A printed value is not automatically a call result",
    "Zero and missing are different",
    "An import can execute top-level code"
  ],
  "sources": [
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/modules.html"
  ],
  "depth": "core",
  "reviewFocus": "Keep the beginner route self-contained, model identities distinct from memory addresses, finite-input limits explicit, and native outputs checked. User acceptance is pending.",
  "designRecord": "docs/teaching/python-foundations-reimplementation.md"
};
