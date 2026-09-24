// Authoring blueprint: Code Documentation, Type Hints & API Design.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Make a reusable function a trustworthy boundary by deciding its units, values, ownership, errors and compatibility, then communicating and checking those promises.",
  "outcomes": [
    "Distinguish annotations, static checking, doctests and runtime validation",
    "Choose interfaces that match actual operations and handle absence separately from zero",
    "Trace shared defaults, aliasing and shallow copying",
    "Validate complete score input before filtering and preserve specific failures",
    "Predict semantic and argument-binding compatibility breaks",
    "Design a once-consuming duration adapter with complete documentation and tests"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules",
    "Iterators, Iterables & Generators",
    "Decorators & Context Managers",
    "Testing, Debugging & Dependency Management",
    "Scientific File Formats, Schemas & Reliable Data I/O"
  ],
  "sequence": [
    "State the caller task and distinguish documentation, hints and API decisions",
    "Read annotation syntax and observe unenforced runtime behaviour",
    "Write an executable docstring with units and failure conditions",
    "Handle optional values and preserve valid zero",
    "Trace object ownership, mutable defaults and shallow copies",
    "Use named results and structural protocols where the operation requires them",
    "Step complete schema validation before an inclusive filter; execute a real JSON loader",
    "Compare unchanged caller behaviour across keyword, threshold and unit changes",
    "Independently implement a duration adapter accepting one-shot input with explicit overflow and consumption policies"
  ],
  "visual": {
    "type": "API validation gates",
    "question": "Can filtering hide an invalid input?",
    "interaction": "Select zero, bool, NaN, equality and invalid low scores; trace the first failed gate and returned mapping."
  },
  "visuals": [
    {
      "type": "Name-to-object ownership",
      "question": "Which caller sees an append?",
      "interaction": "Step shared default, alias or shallow copy while names and object contents remain visible."
    },
    {
      "type": "Caller compatibility flow",
      "question": "Which revision fails binding and which silently changes meaning?",
      "interaction": "Keep the caller fixed and change keyword, comparison or units."
    }
  ],
  "practice": {
    "task": "Design durations_ms for list, tuple and one-shot inputs; specify units, nonmutation, generator consumption, errors and overflow; add executable docs and independent checks.",
    "success": "Runtime and doctests match; mypy accepts the typed duration declarations and rejects the deliberately bad call; changed inputs, invalid values and generator partial consumption behave as documented."
  },
  "misconceptions": [
    "A type hint does not convert or validate",
    "float does not encode physical units or finiteness",
    "None and zero have different meanings",
    "Copying the outer list does not detach nested mutables",
    "A frozen dataclass is not automatic deep validation",
    "Passing a checker does not establish semantic compatibility"
  ],
  "sources": [
    "https://docs.python.org/3/library/typing.html",
    "https://typing.python.org/en/latest/guides/libraries.html",
    "https://docs.python.org/3/library/doctest.html",
    "https://mypy.readthedocs.io/en/stable/getting_started.html"
  ],
  "depth": "core",
  "reviewFocus": "Preserve real-file schema checks and precise contract limitations; keep modern 3.12 syntax alongside the historical PyCon video annotation. Run native and static checks separately. Current bridge goes to Git.",
  "designRecord": "docs/teaching/reliability-authoring-design.md"
};
