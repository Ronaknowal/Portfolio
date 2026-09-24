// Authoring blueprint: Iterators, Iterables & Generators.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Separate sources from one-pass reading positions; trace delayed execution, demand and ownership while building a bounded streaming workflow.",
  "outcomes": [
    "Predict independent versus shared cursor consumption and permanent exhaustion",
    "Trace a generator through creation, suspension, resumption, completion and explicit close",
    "Explain upstream reads needed for filtered output and when delayed errors become visible",
    "Build one-pass summaries and batches with explicit empty/final-batch policies",
    "Independently stop a sensor scan without consuming later readings"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules",
    "Object-Oriented Programming in Python"
  ],
  "sequence": [
    "Give a concrete large-reading problem and distinguish iterable, iterator and generator",
    "Refresh identity versus assignment and trace shared/independent list positions",
    "Explain the for/iter/next/StopIteration contract with native exhaustion output",
    "Expose suspended local values and the need for an extra request after the last yield",
    "Compare eager/lazy work, construction-time expressions and retained inputs",
    "Trace demand upstream through blank-line filtering and number parsing",
    "Run a self-contained text pipeline while its resource remains open",
    "Batch a one-pass iterable and explain deferred validation and incomplete output",
    "Use bounded infinite streams, chain, pairwise, zip and buffering cautions",
    "Investigate tee replay and why a slow consumer retains source history",
    "Implement a custom reusable source versus one cursor",
    "Delegate with yield from and branch into send/return/close without confusing async protocols",
    "Independently find a first threshold crossing, preserve zero and unread suffix, then transfer to running summaries"
  ],
  "visual": {
    "type": "Source positions and cursor ownership",
    "question": "Does a second name have an independent next item?",
    "interaction": "Compare two iter calls with assignment, alternate requests, and try empty/exhausted input."
  },
  "visuals": [
    {
      "type": "Suspended generator frame and caller",
      "question": "Has cleanup run when the last yielded item arrives?",
      "interaction": "Step natural exhaustion, close after starting and close before first next."
    },
    {
      "type": "Upstream demand and downstream data",
      "question": "Why can one requested result read multiple lines?",
      "interaction": "Choose result count and a malformed line, then step read/filter/parse/deliver and inspect unread input."
    }
  ],
  "practice": {
    "task": "Write a first-crossing consumer that accepts one-pass finite readings, stops on None or first strict crossing, preserves zero and leaves all later values unread. Transfer to bounded batching and running means.",
    "success": "341 changed-input cases validate both answer and consumption; independent native islice, iterator state and tee behavior agree with the explanation."
  },
  "misconceptions": [
    "Iterable does not imply reusable",
    "Assignment does not clone a cursor",
    "yield suspends rather than finishing a function",
    "Getting the last item does not necessarily execute cleanup",
    "One result can require multiple upstream reads",
    "Generator syntax alone does not establish constant memory",
    "tee may retain growing history",
    "A break does not guarantee prompt generator closure"
  ],
  "sources": [
    "https://docs.python.org/3/library/stdtypes.html#iterator-types",
    "https://docs.python.org/3/reference/expressions.html#generator-expressions",
    "https://docs.python.org/3/reference/expressions.html#yield-expressions",
    "https://docs.python.org/3/library/itertools.html",
    "https://cs50.harvard.edu/python/notes/9/#generators-and-iterators",
    "https://dabeaz-course.github.io/practical-python/Notes/06_Generators/03_Producers_consumers.html"
  ],
  "depth": "core",
  "reviewFocus": "Three distinct investigations and 14 full programs. Check Python behavior independently, count read demand rather than claiming RAM benchmarks, keep file ownership and generator-expression timing qualifications. CS50 transcript/notes are alternate intuition; their informal async/iterator wording is not the precise language contract. Current module next is Decorators.",
  "designRecord": "docs/teaching/iteration-decorators-design.md"
};
