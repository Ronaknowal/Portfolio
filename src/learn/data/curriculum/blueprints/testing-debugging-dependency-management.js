// Authoring blueprint: Testing, Debugging & Dependency Management.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Turn an independently justified expectation into a discriminating test, explain a failure through execution state, and control the interpreter and dependencies needed to repeat the check.",
  "outcomes": [
    "Trace consumed readings and return placement to diagnose a wrong mean",
    "Choose reference, boundary, invalid and property checks that distinguish known bugs",
    "Explain which boundaries mock and real-file tests exercise",
    "Resolve a small dependency conflict and select the interpreter consistently",
    "Implement and test an independent positive-reading report, including deliberate-bug rejection"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules",
    "Decorators & Context Managers"
  ],
  "sequence": [
    "Set up a standalone script and derive a known expected answer",
    "Trace the early return and construct a regression counterexample",
    "Discover and run a complete unittest suite",
    "Use complementary reference and property checks; investigate surviving mutants",
    "Inject a reader boundary and compare actual temporary-file integration",
    "Use traceback and pdb questions to isolate cause",
    "Choose a virtual environment; intersect transitive compatibility requirements",
    "Record direct requests versus installed snapshot and verify recreation",
    "Complete the independent filtered measurement contract and changed-input tests"
  ],
  "visual": {
    "type": "Execution with consumed inputs and accumulator",
    "question": "Which readings contribute before return exits?",
    "interaction": "Change return placement and input; step accumulation, unseen inputs and output."
  },
  "visuals": [
    {
      "type": "Test discrimination",
      "question": "Can a property let a wrong function survive?",
      "interaction": "Enable known references and a difference property; inspect each changed implementation."
    },
    {
      "type": "Dependency constraint intersection",
      "question": "Can one environment meet both requirements?",
      "interaction": "Change one package range; compare candidate accept/reject bands and common versions."
    }
  ],
  "practice": {
    "task": "Write positive_mean and independent tests before the solution; detect inclusion-of-zero and wrong-denominator mutants; prepare a reproducibility handoff.",
    "success": "All displayed programs run; the four-method suite rejects both introduced defects and correctly handles changed input, blanks, errors and no selected readings."
  },
  "misconceptions": [
    "A singleton can conceal an early return",
    "Coverage or zero discovered tests does not prove correctness",
    "A translation property cannot detect a constant offset",
    "Mocks bypass the very real boundary they replace",
    "A fresh environment cannot reconcile contradictory requirements",
    "pip check tests metadata, not every API contract"
  ],
  "sources": [
    "https://docs.python.org/3/library/unittest.html",
    "https://docs.python.org/3/library/pdb.html",
    "https://pip.pypa.io/en/stable/topics/dependency-resolution/",
    "https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/"
  ],
  "depth": "core",
  "reviewFocus": "Preserve the standard-library beginner route before NumPy/Linux/Git. Compare browser traces and test mutants with native Python. Annotate CS50P pytest differences and current standalone-test discovery. Three distinct labs are chosen for this topic, not a quota.",
  "designRecord": "docs/teaching/reliability-authoring-design.md"
};
