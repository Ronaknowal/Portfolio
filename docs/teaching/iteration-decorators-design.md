# Iteration, decorators and managed lifetimes

Design and scoped evidence record, 10 September 2026. Current authorization: complete the remaining Programming & Scientific Computing lessons and repair module sequence. The root handoff owns queue/status; this is the design for two lesson implementations, not a competing rollout plan.

## Scope and ownership decisions

The exact topic-plan command was run for `iterators-iterables-generators` and `decorators-context-managers`. Neither had an individual brief or destination note; the unresolved inbox was empty. Both are published older lessons, not newly published topics. Read the teaching standard, programming domain strategy, topic-design brief, actual lesson/examples, frontend-design skill and site `.impeccable.md`.

Both existing titles are retained: the added mechanisms and applications fit their promises without adding a new field. IDs, module membership and progress remain unchanged. Parent integration owns sequence/prerequisite/catalogue changes. Required progression is Python → OOP → Iterators → Decorators → Testing before scientific-library work.

| Coverage or discovery | Earlier implementation | Best home and action | Finish-line evidence |
| --- | --- | --- | --- |
| Cursor identity, independent passes and permanent exhaustion | Good prose and output; little spatial help | Iterators core: expose two names and next positions over an unchanged source | Native next calls in independent/shared and empty cases |
| Suspended generator state; last value versus completion | One generic trace | Iterators core: explicit CREATED/SUSPENDED/CLOSED state and locals, including never-started close | inspect state and local value compared at every caller operation |
| Demand through a filter and delayed malformed input | Taught in prose, hidden multiple reads | Iterators core: three stages with demand/data directions, original line state, one/two requested results and malformed line | Native source read counts and partial received values |
| Slow tee consumer retains history | Mentioned in tool table | Optional local application with complete logged producer/consumer example; clearly separate replay semantics from measured byte allocation | Exact native output; no memory benchmark claim |
| Early threshold result without losing unread suffix | Generic generator practice | New independent consumer investigation, None/zero/equality/end boundaries and changed stream use | 341 cases compare result and leftover cursor to a separate position oracle |
| Function objects retained after wrapping | Explained, mostly abstract | Decorators core binding flow before/after decoration | Original output, metadata and forwarding tests |
| Noncommuting decorator order | Generic labeled logger trace | Concrete cap/double result flow for three inputs, yielding both agreement and disagreement | Native decorators and intermediate arithmetic agree |
| Decorator need not wrap | Passing definition | Optional registration application makes the original function identity visible | Complete registry example; duplicate contract explained |
| with error path and successful-entry obligation | Correct table/prose only | Core lifetime investigation: successful body, failed body, failed enter, truthy/false exit | Native __enter__/__exit__ event paths and resource closure |
| Dynamic acquisition rollback | One full example with fixed failure C | ExitStack investigation exposes registered actions and failures at A/B/C/none | Native ExitStack order and each event prefix |
| Restore temporary settings after nested/failing work | Absent | Independent context-manager task; local single-threaded mapping, absent key versus None, binding identity and deletion | 16 combinations plus native printed nested example |
| Asynchronous wrappers and cancellation | Existing brief caution | Retain local synchronous boundary; no new async lesson or broad audit. This increment adds no unresolved discovery requiring another owner's new content. | No async behavior claimed or tested |

All 18 earlier complete programs remain displayed and were re-executed. Seven added programs expose frame states, pull reads, tee replay, first crossing, cap/double order, registry identity and temporary-binding restoration. No useful old topic coverage removed. Long secondary material uses progressive disclosure; original iteration send/throw/close discussion, custom iterator, batching, timing, caching and cleanup utilities remain.

## Teaching contracts for the six investigations

All use explicit presets, prediction before manipulation, reversible stepping, reset, causal text, keyboard controls and text-based spatial equivalents. Color strengthens the active relationship; labels carry meaning independently. There is no autoplay, randomness or arbitrary-code claim.

| Investigation | Learner question, controls and boundary | Prediction and transfer |
| --- | --- | --- |
| Iterator ownership | Same list values shown through two cursor views. Choose two iter calls or aliasing, and empty/nonempty source. Highlight next item/end while returning alternating a/b values. Rows are views, not duplicated stored lists. | Predict b after a reads 18; then explain why empty input hides the distinction. Native list objects establish the contract, not model internals. |
| Generator frame | One countdown has initial count 2. Choose natural exhaustion, close after first item, or close before start. Expose local count and suspension status. State picture is abstract, not a physical memory layout. | Predict remaining after yielding 2 and whether cleanup ran after yielding 1. Transfer to initial count 3. |
| Pull pipeline | Source lines 18/blank/24/30; replace 24 with bad, request one or two results. Highlight read/filter/parse/consumer and unread lines. This is synchronous demand, not a producer running in the background. | Predict how many lines two outputs need and whether one output encounters a later error. Native source instrumentation checks the demand. |
| Decorator order | Keep body/input fixed; choose cap outside double or reverse and inputs 3/8/12. Highlight the called layer, then the outward value. Both change returned results, not arguments. | Predict 10 versus 16 for input 8; explain why agreeing at input 3 does not establish a universal identity. |
| Context lifetime | Choose success/body failure/entry failure and suppression result. Resource status and exception status remain distinct. Enter failure happens before acquisition; cleanup itself does not fail in the model. | Predict whether exit runs and where suppressed execution resumes. Transfer to return/break and partial acquisition; no abrupt-process cleanup guarantee. |
| ExitStack | Register successful entry actions in visible last-entered-first-exited order. Choose failure A/B/C/none. Failed acquisition registers no exit. | Predict whether a failed B is released and which previous resources are owed cleanup. Native stack protocol checks every event prefix. |

## Research ledger and learner alternatives

Verified 10 September 2026. Official Python pages currently resolve to Python 3.14.7 documentation; runnable fixtures were actually checked on CPython 3.12.14, and the lesson's shown features target Python 3.10+ with explicit notes for newer itertools.batched. References are technical authorities, not templates to reproduce.

| Source / inspected portion | Claim or learning use | Limit |
| --- | --- | --- |
| [Python iterator types](https://docs.python.org/3/library/stdtypes.html#iterator-types) | Iterable/iterator methods and permanent exhaustion | Fixed list cursor positions are a teaching view, not the layout of every iterator |
| [Generator expressions and yield](https://docs.python.org/3/reference/expressions.html#generator-expressions), generator-iterator methods | Leftmost iterable creation timing, retained suspension state, next/send/throw/close and returned StopIteration.value | close return-value changes in 3.13 do not affect shown code; never claim every expression is delayed |
| [itertools](https://docs.python.org/3/library/itertools.html) tool and buffering sections | islice demand, batching, groupby consecutive grouping, tee and cycle storage | No RAM byte benchmark or blanket lazy-is-faster claim |
| [functools](https://docs.python.org/3/library/functools.html) wraps and lru_cache | Metadata/inspection versus behavior; hashable cache arguments, retained references and concurrency boundary | Cache is illustrative conversion, not recommended optimization of this cheap function |
| [with language semantics](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement) and function definitions | Entry/exit, suppression, definition-time decoration and nested order | Simplified diagrams do not model failing exit callbacks |
| [contextlib](https://docs.python.org/3/library/contextlib.html), contextmanager and ExitStack | Exactly one yield, error injection at yield, cleanup order and suppression effects | This lesson implements synchronous contexts only |
| [CS50 Python Week 9](https://cs50.harvard.edu/python/weeks/9/) and [notes generator section](https://cs50.harvard.edu/python/notes/9/#generators-and-iterators) | Beginner alternate lecture/notes use a growing list to motivate incremental output. Official [2022 transcript generator portion](https://cdn.cs50.net/python/2022/x/lectures/9/lang/en/lecture9.txt) reviewed, including the state/suspension explanation. | Full video not watched. Transcript uses informal “asynchronous” and “yield returns an iterator” phrasing: our lesson instead states that calling the generator function creates the iterator and synchronous next drives execution; a yielded value goes to the caller. No claimed universal memory ratio. Official lecture page includes video and YouTube choices; no invented timestamp. |
| [David Beazley: Producers, Consumers and Pipelines](https://dabeaz-course.github.io/practical-python/Notes/06_Generators/03_Producers_consumers.html) | Substantive producer/transform/consumer notes and exercises inspected; intermediate practical follow-on | Exercises rely on earlier course files and live-feed setup; marked in learner annotation |
| [David Beazley: Returning Functions](https://dabeaz-course.github.io/practical-python/Notes/07_Advanced_Topics/03_Returning_functions.html) | Closure examples and retained-variable exercises inspected | Some exercises require earlier Stock/property scaffolding; conceptual examples still stand alone |
| [PyCon 2020 tutorial page](https://pycon-archive.python.org/2020/schedule/presentation/75/) and its [verified YouTube destination](https://www.youtube.com/watch?v=T8CQwGIsrx4) | Creator Geir Arne Hjelle, title and intended function/decorator coverage verified from the official event page; alternate extended tutorial | Full recording/transcript not reviewed. This is a provisionally selected alternate based on verified scope, not technical evidence for lesson claims. Event summary says compile time loosely; lesson uses precise definition-time execution. |

The substantive sources informed distinctions and choice of examples; local diagrams, finite inputs, examples and exercises are original to this implementation or retained repository work. No videos were represented as fully watched.

## Verification status

Implementation: complete, three investigations per lesson (six distinct jobs, not a lab quota), 14 iteration programs and 11 decorator/context programs. Current titles retained. User acceptance: pending.

- `node scripts/verify-iteration-decorators.mjs` passed: every displayed fixture is exported, every exported fixture is displayed; exports 25 programs and 27 lab configurations.
- `scratch/lesson-tools/Scripts/python.exe scripts/verify-iteration-decorators.py` passed with Python 3.12.14. All 25 exact outputs; real iterators/generator inspection/islice/context protocols; native wrapper result order; 341 first-crossing result-and-consumption cases, 78 batch cases, 16 temporary restoration cases, keyword/return/metadata/exception contracts.
- Native evidence: `scratch/iteration-decorators-review/native-results.json`. The script executes bounded in-memory standard-library fixtures; no network, host-process changes or third-party installation.
- `node scripts/review-iteration-decorators.cjs` passed on Edge/Chromium at 1440 × 1100 and 390 × 1100. All 27 configurations and their forward steps, back/reset, keyboard activation of reset/hints, valid section anchors, all 25 displayed programs and external-resource groups were checked. No page errors, horizontal page overflow or clipped lab text/controls. Evidence: `scratch/iteration-decorators-review/browser-results.json`.
- All six narrow-screen lab screenshots and four desktop views were visually inspected; representative states were then recaptured to show suspended generator locals, a shared next position, the malformed pipeline boundary, a transformed returned value, resource cleanup and registered B/A exit actions. Text labels, controls, nesting and mobile stacking remain readable. Screenshots: `scratch/iteration-decorators-review/<investigation>-390.png` and `-1440.png`.
- Parent integration owns final application build, catalogue conservation, route order, handoff and generated inventory checks. This record does not assert those have already run.
- Learner walkthrough is an author's heuristic review, not an observed beginner study. Memory, performance, async behavior, cancellation and thread-safe global state are not empirically verified by these models.

## Visual follow-up · 10 September 2026

The [Python Foundations representation review](PYTHON-VISUAL-REVIEW.md) replaces duplicate cursor rows with a single-source ownership map, replaces generic decorator-binding prose panels with actual callable/reference structure, and replaces the context phase strip with selected exception/cleanup routes. All existing models, controls, complete programs, practice and useful other representations are retained. Read that record for the three visual contracts, title/scope decisions and fresh native/browser evidence.
