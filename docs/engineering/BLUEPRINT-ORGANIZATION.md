# Authored blueprint ownership

10 September 2026. Authored lesson designs now live in [`src/learn/data/curriculum/blueprints/`](../../src/learn/data/curriculum/blueprints/index.js), with one default-exported object per stable topic ID. The source filename identifies the topic rather than the date or implementation batch in which it was written.

This is a source organization change. All 19 authored objects, including their summaries, prerequisites, teaching sequences, visual contracts, practice, source links and review metadata, were preserved exactly. The resolved catalogue also stayed exactly equal: no title, level, module membership, prerequisite, topic order or progress ID changed. Individual design records remain human-authored Markdown evidence; no source code is generated from their prose.

## Edit a topic's plan

1. Run `node scripts/build-curriculum-inventory.mjs --topic "Exact topic title or stable ID"` and read its returned plan and destination notes.
2. Edit `src/learn/data/curriculum/blueprints/<stable-topic-id>.js` for an authored lesson. Its default export is that topic's blueprint; do not combine unrelated lesson designs into a batch bundle.
3. Register a newly authored topic once in [the semantic index](../../src/learn/data/curriculum/blueprints/index.js), using the exact existing title as the key and the stable file as the import. Preserve stable IDs when refining a title according to the teaching standard.
4. [track-definitions.js](../../src/learn/data/track-definitions.js) consumes the single `authoredBlueprints` map. Module/section/topic arrays there own teaching order; the alphabetical order of imports in the blueprint index is only source organization.
5. Run the normal curriculum verification and regenerate the inventory and compact browser metadata through their respective documented workflows. A newly written source plan is not a completed or user-approved lesson.

GPU, neural-engineering and cross-domain expansion plans remain in their semantic domain expansion sources. Those are planned coverage, not implementation batches or a claim that all 193 published lessons already have a reviewed authored design. The 19 files extracted here are precisely the existing authored-blueprint set: 17 programming/scientific-computing lessons and two DSA lessons.

The authored index is an authoring/build-time aggregate. It deliberately contains complete teaching plans for inventory, prerequisite and design tooling. It is not the reader's initial browser payload; reader loading and generated compact navigation data have a separate runtime boundary.

## Migration mapping

All former source bundles below were removed. There are no compatibility re-export files at their old paths.

| Former bundle | Current topic-owned source files under `blueprints/` |
| --- | --- |
| `reference-blueprints.js` | [linux-basics-filesystems-processes.js](../../src/learn/data/curriculum/blueprints/linux-basics-filesystems-processes.js) |
| `first-five-blueprints.js` | [python-basics-types-control-flow-functions-modules.js](../../src/learn/data/curriculum/blueprints/python-basics-types-control-flow-functions-modules.js), [numpy-arrays-broadcasting-vectorization.js](../../src/learn/data/curriculum/blueprints/numpy-arrays-broadcasting-vectorization.js), [scientific-file-formats-schemas-reliable-data-i-o.js](../../src/learn/data/curriculum/blueprints/scientific-file-formats-schemas-reliable-data-i-o.js), [sql-relational-data-transactions-for-ml.js](../../src/learn/data/curriculum/blueprints/sql-relational-data-transactions-for-ml.js), [object-oriented-programming-in-python.js](../../src/learn/data/curriculum/blueprints/object-oriented-programming-in-python.js) |
| `next-three-blueprints.js` | [pandas-data-wrangling-joins-grouping.js](../../src/learn/data/curriculum/blueprints/pandas-data-wrangling-joins-grouping.js), [matplotlib-scientific-plotting.js](../../src/learn/data/curriculum/blueprints/matplotlib-scientific-plotting.js), [git-github-collaborative-version-control.js](../../src/learn/data/curriculum/blueprints/git-github-collaborative-version-control.js) |
| `systems-structures-blueprints.js` | [os-processes-virtual-memory-isolation.js](../../src/learn/data/curriculum/blueprints/os-processes-virtual-memory-isolation.js), [arrays-strings-hash-maps.js](../../src/learn/data/curriculum/blueprints/arrays-strings-hash-maps.js), [linked-lists-stacks-queues.js](../../src/learn/data/curriculum/blueprints/linked-lists-stacks-queues.js) |
| `iteration-decorator-blueprints.js` | [iterators-iterables-generators.js](../../src/learn/data/curriculum/blueprints/iterators-iterables-generators.js), [decorators-context-managers.js](../../src/learn/data/curriculum/blueprints/decorators-context-managers.js) |
| `reliability-blueprints.js` | [testing-debugging-dependency-management.js](../../src/learn/data/curriculum/blueprints/testing-debugging-dependency-management.js), [reproducible-notebooks-experiment-structure.js](../../src/learn/data/curriculum/blueprints/reproducible-notebooks-experiment-structure.js), [code-documentation-type-hints-api-design.js](../../src/learn/data/curriculum/blueprints/code-documentation-type-hints-api-design.js) |
| `thread-completion-blueprints.js` | [threads-concurrency-locks-deadlocks.js](../../src/learn/data/curriculum/blueprints/threads-concurrency-locks-deadlocks.js) |
| `bash-completion-blueprint.js` | [bash-scripting-command-line-automation.js](../../src/learn/data/curriculum/blueprints/bash-scripting-command-line-automation.js) |

## Precedence and conservation

The pre-migration resolver checked Bash, Threads, reliability, iteration/decorators, systems/structures, next-three and first-five authored bundles before inline topic plans. The Linux reference was a fallback alongside domain-expansion blueprints. An explicit snapshot evaluated all eight source bundles and each resolved catalogue occurrence before changing any source.

There were **no overlapping authored titles** and **no mismatches between an authored object and its active resolved catalogue object**, including Linux. Consequently extraction required no content merge or silent choice between conflicting versions. The new rule is explicit: the topic's authored design takes precedence, then an inline plan, then a domain fallback. This gives every current authored lesson the same result as before, without a growing batch-precedence expression.

The pre-migration snapshot is [`scratch/blueprint-organization/before.json`](../../scratch/blueprint-organization/before.json). The one-time extraction scripts remain in that scratch evidence directory, not the authoring source workflow. They refuse to overwrite the captured baseline or existing semantic topic files; they must not be used to regenerate over future edits.

Verification performed:

- [`node scripts/verify-blueprint-organization.mjs`](../../scripts/verify-blueprint-organization.mjs): passed exact deep equality for all 19 per-topic default exports and the aggregate; unique stable ownership; absence of all eight legacy bundles; and deep equality of the entire resolved 28-module catalogue with all **1,222 topic memberships**. This is a one-time migration proof against a saved baseline, not a test to run after intentional future curriculum/content changes.
- The serialized resolved catalogue's SHA-256 before and after was identical: `c47c3290cbb59806d40f8db19ebb4396df2f1233d33b307f2eb4752b02dd10b3`. [Machine-readable result](../../scratch/blueprint-organization/conservation-results.json).
- `node scripts/verify-curriculum.mjs`: passed 28 modules, 1,218 unique stable topics, 289 individual briefs and seven paths, with prerequisite inclusion/acyclicity, shared memberships, module-order reading and preservation of all 1,002 pre-expansion topics.
- `node scripts/build-curriculum-inventory.mjs`: succeeded; 193 published, 289 briefs, 339 prerequisite reviews, seven paths. These count different states; a brief is not a published or accepted lesson.
- Direct old-source import/reference search under `src/` and `scripts/` returned no remaining consumers. Only `track-definitions.js` imported the old bundles; the inventory and conservation tools already consume the resolved catalogue and required no old-import compatibility layer. Current documentation links were migrated, with pointers from historical first-five/next-three records.

The root integration task owns the final application build and browser loading measurements after runtime metadata and lazy content changes land. This source migration alone makes no measured browser-performance claim.
