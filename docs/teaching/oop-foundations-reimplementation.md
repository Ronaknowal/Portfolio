# Object-Oriented Programming in Python: implementation design and evidence

Written 9 September 2026 for the user-authorized reimplementation of the first five topics. Stable ID: `object-oriented-programming-in-python`. This record describes this lesson; [the teaching standard](../../LESSON-TEACHING-STANDARD.md) remains policy and [the handoff](../../LESSON-AUTHORING-HANDOFF.md) owns the overall queue. User acceptance of this replacement is pending; the Linux lesson remains the approved quality reference.

## Learning contract and retained scope

The learner knows Python names, functions, lists/dictionaries, loops and exceptions from the first topic. Short local bridges introduce classes, attributes, instance methods, decorators, protocols and dataclass field annotations before relying on them. SQL and file handling occur earlier on the current route, but this lesson does not require SQL syntax. It connects external text parsing to explicit object input contracts and connects forward to Pandas. Iteration protocols, decorators and maintained testing suites are later depth links, not mislabelled immediate next lessons.

Finish line: design independent stateful objects, explain the exact receiving object in a call, preserve a public invariant on failure, distinguish shared storage and aliases, replace a collaborator through a behavioural contract, and test a changed-domain design. The core anchor is independent morning/evening temperature logs. DatasetSplit provides independent transfer: string membership, duplicate rejection, caller isolation and interchangeable reports.

The old lesson already had useful precision about bound methods, shallow copying, bool-as-int, float conversion failure, properties, MRO, dataclasses and inheritance tradeoffs. Retain those ideas. Its single general trace and predominantly prose-based transitions did not expose class lookup, validation and composition as distinct mechanisms. The replacement uses four focused investigations, a record identity diagram, complete independent programs, local retrieval and a synthesis task. Four labs are a result of this topic's hurdles, not a quota copied from Linux.

Core sequence: persistent problem → functions/dictionaries → class/instance/self → alias and shared-state lookup → validation and public snapshots → composition and contracts → dataclass equality/identity → design choice and independent task. Saved bound-method rebinding, descriptor precedence, numerical/batch limits, inheritance/MRO and method kinds are expandable depth. The basics of inheritance/substitution and class/static methods remain introduced in the main text so the deeper branches do not hide what those choices mean.

## Outcome map

| Observable outcome / hurdle | Plain mechanism and representation | Example / assessment | Depth |
| --- | --- | --- | --- |
| Justify an object instead of a function | Begin with persistent independent logs; show dictionary+function first | `functions`, function-versus-class checkpoint | Core |
| Predict self, mutation and alias visibility | Name → log identity → owned list; bound function/receiver → local call parameters | `instances`, binding lab, evening/alias mean checkpoint | Core |
| Explain a saved bound method after reassignment | Stored method retains object/function, not the variable's future binding | `bound`, prediction before runnable result | Deeper |
| Diagnose class/instance/mutable-default sharing | Explicit own-attribute absence/found route into class list; distinguish append from assignment | `shared`, lookup lab, default-argument repair | Core |
| Maintain a contract and preserve state after error | Four ordered gates; only accepted value reaches append | `validated`, validation lab, rejected NaN checkpoint | Core |
| Explain snapshot access and encapsulation limits | Property getter produces separate immutable-membership tuple; hidden internal list is convention, not security | `validated`, old/current snapshots, read-only property checks | Core with descriptor/numeric depth |
| Compose and swap behaviour | Separate Report and formatter receivers; call and return arrows retain unit/meaning | `composition`, formatter lab, hierarchy diagnosis | Core |
| Test boundaries, not copied implementation | Recording formatter observes delegated input and gives controlled output | `composition` assertions; DatasetSplit tests | Core with interface depth |
| Distinguish identity, equality and mutable record defaults | Same-field dataclass objects have separate identity/list arrows | `dataclass`, alias checkpoint, static record diagram | Core |
| Recognise frozen/type-hint/hash limits | Complete frozen-list counterexample, explicit hashed-key contract | `frozen`, prediction before output | Deeper |
| Justify inheritance and method kind | Substitution preserves input/output meaning; table of implicit argument | `inheritance`, unit-contract diagnosis; `methods` | Core choice with implementation depth |
| Transfer to a new data model | Dataset membership contract, fresh owned collection, duplicate rejection and composable reporting | `mission`, hidden hint/solution, invalid and overlap transfer | Independent core synthesis |

## Visual contracts

### Receiver and bound-method investigation

- Question: which object receives an add call, and why do alias and morning observe the same mutation?
- Start: two empty ReadingLog instances A/B; names morning and alias reach A, evening reaches B. Each log reaches a distinct list bearing its stable letter. Area and location are layout, not memory measurements.
- Prediction: choose name and reading (18, 24 or 30), identify the changed list before stepping.
- Control: four causal stages, back, reset and explicitly keep result for another call. Selectors discard an unkept trace and start from the last kept state. This is stated beside the lab.
- Consequence: name lookup, function/receiver binding and local self appear as separate stages; no mutation before append. A/B values stay visible together.
- Transfer: morning 18, alias 24, evening 30 → A [18,24], B [30]. Saved-method rebinding is separate executable depth.
- Access: ordinary labelled controls; text names, identities and exact values accompany arrows; live causal feedback; 44px control targets; two-column name/object rows remain readable at 390px while the call flow stacks.
- Boundary: fixed simple instance method; no interpreter, custom descriptors, concurrent execution or memory size claims. Back changes the model view, not live Python history.

### Attribute lookup investigation

- Question: why can two independent objects still share one list, and why does assignment change that?
- Start: `Log.values=[]`; both instance attributes explicitly absent and a fallback path points toward class list C.
- Control: append 18 or assign [99] through A/B; toggle between class-list and per-instance initialisation; undo/reset.
- Consequence: read origin and exact observed lists are visible for both objects; class list stays visible after one or both objects shadow it. Assignment changes a reference; append changes the reached list.
- Transfer: shadow both lists, append independently and explain why the original class list still exists.
- Access: stacked labelled nodes and text lookup directions; the diagram does not rely only on colour or a changing table.
- Boundary: ordinary data attributes only. Property/descriptor precedence is explicitly excluded and revisited in the property explanation.

### Validation investigation

- Question: at which gate does an input fail, and can that failed call alter the log?
- Start: every attempt starts from [18.0], holding prior state fixed.
- Prediction/control: choose typed candidate and predict failure/state before Run validation. Six choices: int 24, bool True, str "24", NaN, infinity, huge int.
- Consequence: each gate shows passed/stopped/not reached in text as well as colour; a before/after pair preserves exact values. The failure path never reaches append.
- Transfer: explain why three failure mechanisms need different checks; executable negative-infinity, None/list and snapshot tests extend the visible cases.
- Boundary: six fixed Python cases; not an evaluator, a universal numeric-type policy, an accuracy test for sensors or a scientific range validator.

### Composition investigation

- Question: how can the same report work with a different formatter, and what does self refer to in each call?
- Start: report holds a reference to one formatter; input is a Celsius number. Controls change formatter or 0/20/30°C, then reset the trace.
- Stages: report receives input → delegates to formatter → formatter returns unit-bearing text → report returns labelled text to caller. Both receivers remain labelled, with separate call and return routes.
- Transfer: Kelvin formatter with two-decimal output, same report; explain why returning a bare number violates the stated interface.
- Access: flow stacks on narrow screens; call/return text remains explicit; each stage has live explanation and exact output; back/reset are meaningful.
- Boundary: synchronous fixed methods and modest finite inputs, no thread model or real hardware measurements.

### Dataclass record diagram

A static pair is sufficient: show two distinct same-type records, two list identities and equal initial fields. Explain `is` versus generated `==`, then predict the effect of appending to A. The runnable example validates it. No extra toggle is needed to teach this relationship.

## Claim ledger

Primary documentation retrieved 9 September 2026; `/3` pages identified themselves as Python 3.14.7 documentation. That is the documentation snapshot, not the executed runtime or a recommendation to upgrade. Programs were executed on Python 3.11.7 and use features available from 3.10. Wording/examples are original to this lesson; no videos were claimed reviewed.

| Claim / convention | Primary source and locator | Verification / qualification |
| --- | --- | --- |
| Names alias objects; ordinary method access binds receiver and function | [Python tutorial, methods and instance/class variables](https://docs.python.org/3/tutorial/classes.html#method-objects) | 81 two-call combinations independently executed; saved-method reassignment fixture |
| Class list versus per-instance list; append versus instance rebinding | [Python tutorial, class and instance variables](https://docs.python.org/3/tutorial/classes.html#class-and-instance-variables) | 128 three-action sequences compared with fresh Python classes |
| Default mutable argument evaluated once | [Tutorial, default argument values](https://docs.python.org/3/tutorial/controlflow.html#default-argument-values) | Explicit repair uses None and a fresh outer list |
| Assignment does not copy; shallow copy retains element references | [Copy documentation](https://docs.python.org/3/library/copy.html) | DatasetSplit uses immutable strings; no recursive isolation claim |
| Ordinary lookup is not the whole descriptor algorithm; __init__ initialises; special-method and hash contracts | [Python data model](https://docs.python.org/3/reference/datamodel.html#customizing-attribute-access) | Properties explicitly excluded from simple lookup diagram; read-only property independently tested; numerical and equality limits retained |
| Properties expose attribute-style getters and optional setters; method decorators bind differently | [Built-in functions](https://docs.python.org/3/library/functions.html#property) | Snapshot and property-assignment failure checked; all method-kind fixture outputs executed |
| NaN/infinity are not finite | [math.isfinite](https://docs.python.org/3/library/math.html#math.isfinite) | Six visible cases plus further rejected values tested against actual validated class |
| Dataclass generated methods, default_factory, frozen, hash and type-annotation behaviour | [Dataclasses reference](https://docs.python.org/3/library/dataclasses.html) | Ordinary and frozen fixtures executed; no claim of runtime type validation or recursive immutability; release-sensitive NaN equality excluded |
| Protocol structural interface and type-checking limitations | [typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol) | Optional conceptual depth; no claim a static checker proves semantic behaviour |
| Abstract methods prevent ordinary instantiation before implementation | [abc reference](https://docs.python.org/3/library/abc.html) | Optional explanatory contrast; no ABC implementation added |
| assert can be omitted by optimisation | [Language reference, assert](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement) | Assertions used for tests, explicit exceptions used for validation |

## Verification evidence

- `scripts/verify-oop-foundations.mjs`: all **11** displayed programs and expected outputs passed on Python 3.11.7. **221** model comparisons against independently executed Python: 81 method-call pairs, 128 lookup action sequences, six validation cases and six formatter/input combinations.
- Additional runtime checks: mutable-default repair, shallow nested sharing, independent lists, empty mean, negative/zero readings, rejected False/None/list/negative infinity, failure preservation, immutable snapshots, no property setter, constructor duplicate failure, blank/non-string IDs, saved snapshot, LinesFormatter newline/empty behaviour and explicit lack of cross-split disjointness enforcement.
- Invocation on this Windows host: set `PYTHON=C:/msys64/ucrt64/bin/python.exe`, then `node scripts/verify-oop-foundations.mjs`. Bare `python` from Node initially failed to spawn; using the resolved interpreter succeeded. This is a host command-resolution issue, not an example failure.
- `scripts/review-oop-foundations.cjs`: actual browser checks at **1440px** and **390px** passed for all four labs, changing inputs, step/back/reset/undo, retained state, validation failures, composition return, keyboard activation, lesson anchor targets, independent hint/solution reveals, Pandas continuation and page overflow. No page errors recorded.
- Screenshots and machine-readable records: `scratch/oop-foundations/`. Scoped lab captures temporarily hide fixed site navigation during the screenshot only, then restore it for interactions; this prevents a tall element screenshot from including a fixed bar across its middle.
- Visual review: original desktop/mobile screenshots inspected. Arrows, state identities, text failure labels and stacked flows remain readable. Clean scoped captures were regenerated after excluding the fixed-navigation screenshot artifact.
- `scripts/review-programming-batch-one.cjs` now retains iterator checks and delegates the current Python/OOP reviews; the combined public entry point passed at 1440 and 390 pixels. Legacy OOP fixtures in the old native batch script are regression evidence, not complete coverage of this replacement.
- [The first-five integration record](../../FIRST-FIVE-REIMPLEMENTATION.md) owns the final application build, curriculum conservation/inventory and cross-page integration checks. Those results are separate from this topic's model/runtime and browser evidence.

No external novice study or user acceptance has occurred. Runtime correctness and author walkthrough are evidence for this scoped implementation, not proof of mastery. Multiple inheritance, descriptor implementation, large-data performance and comprehensive type checker setup remain linked/deeper topics, while the core route reaches its small-design finish line.

## Visual follow-up · 10 September 2026

The [Python Foundations representation review](PYTHON-VISUAL-REVIEW.md) adds a reference map beside the saved-bound-method/rebinding experiment. Existing receiver, lookup, invariant and composition labs are retained. The linked review records the placement rationale, exact native example, accessible/narrow-screen treatment and fresh verification; the title and existing outcomes are retained.
