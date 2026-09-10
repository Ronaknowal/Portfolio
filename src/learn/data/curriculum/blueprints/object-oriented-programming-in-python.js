// Authoring blueprint: Object-Oriented Programming in Python.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Keep independent reading-log state with the operations that protect it, then compose interchangeable behavior without accidental sharing.",
  "outcomes": [
    "Explain classes, instances, attributes, receivers and bound methods",
    "Diagnose shared class state, rebinding and invalid updates",
    "Choose functions, composition or inheritance and implement a checked dataset-split object"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules"
  ],
  "sequence": [
    "Start with functions and dictionaries for two independent logs",
    "Introduce classes through a complete equivalent implementation",
    "Trace self and bound-method receiver identity",
    "Locate instance and class attributes before mutation",
    "Validate changes before committing object state",
    "Compose reports through explicit call/return boundaries",
    "Explore dataclasses, properties, identity/equality and inheritance in progressive depth",
    "Implement and test an independent dataset split with stated invariants"
  ],
  "visual": {
    "type": "Receiver and identity explorer",
    "question": "Which object does this method change?",
    "interaction": "Follow aliases and bound methods to independent log objects with reference arrows."
  },
  "visuals": [
    {
      "type": "Attribute lookup",
      "question": "Is this list owned by the instance or shared on its class?",
      "interaction": "Contrast lookup, mutation and instance shadowing."
    },
    {
      "type": "Validation gates",
      "question": "Can a rejected update leave the object partly changed?",
      "interaction": "Trace input checks and compare before/after state."
    },
    {
      "type": "Composition calls",
      "question": "How can a report use a different formatter?",
      "interaction": "Follow calls, returned values and object roles through interchangeable collaborators."
    }
  ],
  "practice": {
    "task": "Build a dataset-split object with explicit membership invariants, then diagnose copying, mutable defaults and identity/equality cases.",
    "success": "Independent instances, preserved invariants on rejection, deterministic results, justified interfaces and passing independent Python checks."
  },
  "misconceptions": [
    "Self is a supplied receiver parameter, not a global object",
    "Using self does not guarantee that a looked-up list is instance-owned",
    "A property follows descriptor rules beyond ordinary attribute lookup",
    "Frozen dataclasses are not recursively immutable",
    "Inheritance is not required for objects with a compatible interface"
  ],
  "sources": [
    "https://docs.python.org/3/tutorial/classes.html",
    "https://docs.python.org/3/reference/datamodel.html",
    "https://docs.python.org/3/library/dataclasses.html"
  ],
  "depth": "core",
  "reviewFocus": "Keep plain object reasoning before descriptors/MRO. Test all examples in Python, including failed updates and dataclass equality/default behavior. User acceptance pending.",
  "designRecord": "docs/teaching/oop-foundations-reimplementation.md"
};
