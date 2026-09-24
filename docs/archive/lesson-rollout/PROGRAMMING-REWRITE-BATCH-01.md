> HISTORICAL RECORD — archived 9 September 2026. All queue, authorization, approval and teaching instructions below describe an earlier increment. They are not current policy. Start at [the current handoff](../../../LESSON-AUTHORING-HANDOFF.md).

# Programming rewrite — batch 01

9 September 2026. Status: approved to continue via the next-three request. Batch 02 is now implemented; see [batch 02](PROGRAMMING-REWRITE-BATCH-02.md).

## Scope

Replaced the first three Programming & Scientific Computing articles in place. Existing URLs, track order and dark styling remain. No DSA or fourth-topic changes.

| Lesson | Coverage | Runnable examples |
| --- | --- | --- |
| Python Basics: Types, Control Flow, Functions & Modules | Running scripts; types and conversions; collections; mutation/aliasing; conditions and loops; functions, scope and defaults; exceptions; two-file report; solved practice | 9 |
| Object-Oriented Programming in Python | Instances and self; class versus instance state; validation and properties; special methods; method types; composition/polymorphism; inheritance; dataclasses; solved practice | 8 |
| Iterators, Iterables & Generators | Iteration protocol; yield/exhaustion; eager/lazy timing; streaming file-like pipeline; batches; itertools; custom iterables; delegation/cleanup; send/return; solved practice | 10 |

The examples follow readings through functions, stateful logs, then incremental processing. Each code example is independent and has the output actually produced by Python. The two-file project includes both files and a run command. Notes distinguish runnable Python from the illustrated browser traces.

## Teaching changes

- Replaced generic overview/reference wrappers with sequenced topic-specific modules.
- Added three step-through state visualisations: names referring to objects, independent log instances, and generator suspension/exhaustion.
- Explained outputs immediately after code, including why errors occur and what state remains after failure.
- Added first-reader routes, revision links and explicit prerequisite hand-offs.
- Added worked solutions, boundary cases and small checks rather than unsolved prompts alone.
- Linked official Python references for technical details. Python 3.10+ is the examples' baseline; newer conveniences are labelled with their version requirement.

This is a complete core learning path for each named topic, not a claim to document every Python built-in or advanced language mechanism. Advanced async execution, metaclasses, descriptors and larger test/environment workflows remain explicitly outside this batch or belong to later topics.

## Verification

- `scripts/verify-programming-batch-one.mjs` executes all 27 examples independently, compares stdout to the exact displayed output and checks additional edge cases. The project checks that importing report does not execute main.
- `scripts/review-programming-batch-one.cjs` checks all three pages at 1440 px and 390 px: displayed outputs, unique navigation targets, all answer reveals, forward/back/reset trace controls, keyboard activation, empty code fragments, browser errors and document overflow.
- `npm.cmd run build` passes. Existing warnings in the unrelated Bayesian-networks article and for the large shared content chunk remain.
- Screenshots are generated under `scratch/programming-batch-one`. The fixed site navigation is hidden only for element screenshots so it does not obscure captured content.

Run the numerical/example verifier with `LESSON_PYTHON` set to a Python executable if `python` is not available. The browser verifier needs the local development server on port 5173, Edge and Playwright; `PLAYWRIGHT_PACKAGE` can point to an existing Playwright installation.

## Next batch — not yet implemented

After review: Decorators & Context Managers; Testing, Debugging & Dependency Management; NumPy: Arrays, Broadcasting & Vectorization. Keep the three-at-a-time approval boundary.
