// Authoring blueprint: Decorators & Context Managers.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Choose function-call or block lifetime boundaries; preserve intended call contracts and release resources on successful and exceptional paths.",
  "outcomes": [
    "Translate @ syntax into callable application and retained function references",
    "Predict noncommuting wrapper order, forwarding, metadata and exception behavior",
    "Distinguish successful entry, body failure, suppressed errors and failed acquisition",
    "Explain exactly-once contextmanager yield and reverse ExitStack cleanup",
    "Independently restore a temporary binding including missing/None/nested/failure cases"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules",
    "Object-Oriented Programming in Python",
    "Iterators, Iterables & Generators"
  ],
  "sequence": [
    "Choose decorators for calls and context managers for blocks",
    "Show function objects, closure references and before/after name binding",
    "Implement a logger preserving args, kwargs, return and exceptions with wraps",
    "Separate factory configuration, decoration time and call time",
    "Trace result flow through cap/double in both orders and compare changed inputs",
    "Show definition-time plugin registration without a wrapper",
    "Time synchronous work with an explicit fake clock and inspect cache behavior/limits",
    "Trace with through entry, body and exit, contrasting normal/suppressed/propagated error paths",
    "Implement class and single-yield generator context managers with explicit ownership",
    "Explain failed partial setup and register successful acquisitions using ExitStack",
    "Trace reverse cleanup after any selected acquisition failure",
    "Independently implement temporary settings with nested restore, absent keys, original None and propagated exceptions",
    "Transfer to counted-call instrumentation and prepare executable checks for Testing"
  ],
  "visual": {
    "type": "Nested callable references and outward result flow",
    "question": "Does swapping two wrappers preserve the result?",
    "interaction": "Choose cap/double order and inputs 3, 8 or 12; step inward calls and outward transformations."
  },
  "visuals": [
    {
      "type": "Resource lifetime and exception route",
      "question": "Which paths close a resource, and where does suppression continue?",
      "interaction": "Choose success/body error/enter error and truthy/false exit result; inspect resource and exception state."
    },
    {
      "type": "Registered exit actions",
      "question": "Which resources must unwind after a failed acquisition?",
      "interaction": "Fail A, B, C or none; step successful registrations and reverse releases."
    }
  ],
  "practice": {
    "task": "Write temporary_value with explicit shallow single-threaded binding ownership, a unique missing sentinel, nested restoration and unchanged exception propagation. Extend counted decorators to verify results, keywords, metadata and failed attempts.",
    "success": "Native context protocols match all browser paths; 16 changed restoration cases preserve original identity/presence after deletion and failure; complete programs reproduce annotated output."
  },
  "misconceptions": [
    "A decorator need not wrap or replace its input object",
    "Decoration time is different from later call time",
    "One agreeing input does not prove wrapper order irrelevant",
    "wraps preserves metadata, not arbitrary behavior",
    "Timing generator creation does not time consumption",
    "Truthy exit suppresses a body exception rather than restarting its body",
    "A failed enter does not call that same manager exit",
    "Contextmanager requires exactly one yield",
    "Temporary rebinding is not a deep copy or concurrency-safe global configuration change"
  ],
  "sources": [
    "https://docs.python.org/3/library/functools.html",
    "https://docs.python.org/3/library/contextlib.html",
    "https://docs.python.org/3/reference/compound_stmts.html#the-with-statement",
    "https://pycon-archive.python.org/2020/schedule/presentation/75/",
    "https://dabeaz-course.github.io/practical-python/Notes/07_Advanced_Topics/03_Returning_functions.html"
  ],
  "depth": "core",
  "reviewFocus": "Three interactive mechanisms, a binding flow, 11 programs and explicit independent tasks. Synchronous wrappers only; async cancellation/lifetimes belong to asynchronous programming. Video page/link and intended coverage verified; no claim full video watched. Current module next is Testing.",
  "designRecord": "docs/teaching/iteration-decorators-design.md"
};
