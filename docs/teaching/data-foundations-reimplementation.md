# File I/O and SQL: implementation design

Current visual follow-up, 10 September 2026: [scientific representation review](SCIENTIFIC-VISUAL-REVIEW.md) records the CSV-boundary diagram, file publication artifacts, SQL reference diagram and fresh native/browser evidence. This is a historical increment record; current reading order comes from the catalogue and [handoff](../../LESSON-AUTHORING-HANDOFF.md).

9 September 2026. Authorized topics 3 and 4 in the first-five reimplementation. Read the current handoff and standard. Publication and user acceptance remain separate. The full sequence is Python → NumPy → Scientific File Formats → SQL → OOP → Pandas; this increment does not reorder it.

## Learning contracts

File I/O starts with Python strings/lists/dictionaries, functions, exceptions and NumPy shapes/dtypes. A reader must be able to move an invented measurement dataset from bytes to validated records, explain preserved identifiers/nulls/units, choose an appropriate storage representation, check a semantic round trip and distinguish atomic replacement from durable multi-file publication. It is not a comprehensive implementation guide to every scientific format or filesystem.

SQL restates the needed measurement-contract ideas and requires only Python Basics as a formal prerequisite. Local bridges explain multiline SQL strings, connection methods, setup statements, parameter tuples and finally before the complete workflow relies on them. A reader must identify one row's meaning and keys, trace filtering and joining into aggregation, preserve entities with no observations, construct one feature row per sensor at an explicit time cutoff, parameterize values and explain/execute a failed transaction with explicit rollback. Administration and distributed transaction protocols are subsequent study.

File I/O explicitly introduces its compact Python syntax and library boundaries: any over field checks, a conditional expression, enumerate record numbers, Path joining, text reads/writes and JSON encoding/decoding. Its missing-value policy differs deliberately from the first Python report: only an empty CSV temperature field becomes None; whitespace-only numeric fields are rejected. Sites must be nonblank. An empty batch is rejected; an accepted batch whose measurements are all missing has no mean and reports it as unavailable.

## Concept/outcome map

| Hurdle | Plain mechanism and support | Evidence / practice |
| --- | --- | --- |
| Bytes are not typed records | Decode → parse → convert → validate; CSV fixture with quoted comma, leading-zero ID, empty reading and observed zero | Predict each cell's type and whether malformed units/numbers/duplicate IDs pass |
| A parsed number may be invalid | Separate structure, type and domain gates; keep the batch unpublished on error | Schema explorer shows the first failing gate and exact rejected field |
| Saving is not proof of preservation | CSV/JSON/NPY/Parquet/HDF5 comparison plus executable JSON and NPY round trips | Inspect IDs, nulls, dtype, shape, units and counts; distinguish equivalent data from identical bytes |
| Publishing a partial file | Old destination beside staged replacement with explicit failure point | Write strategy/failure explorer; predict what a fresh reader sees; native replacement check |
| A result row needs a defined grain | Tiny sensor/reading tables with explicit primary/foreign-key correspondence | Count rows before running SQL; distinguish sensor ID from observation ID |
| Joins multiply and drop records | Linked source/result rows with selectable left/inner join, duplicate key and predicate placement | Predict output count; inspect lineage and nullable measurements separately |
| Filtering and aggregation differ | Logical SQL flow, grouped result with COUNT(*) vs COUNT(id) vs COUNT(value) | Trace a cutoff query; produce one feature row per sensor without treating absent readings as zero |
| Transaction changes are provisional | Writer working state beside independently committed state | Step explicit/individual-commit updates; inject failure; explain why the scenario ends in rollback or commit |
| ML feature extraction can leak time | Parameterized cutoff in the join and deterministic ORDER BY | Independent changed-sensor/cutoff exercise; uniqueness and exclusion tests |

## Visual contracts and boundaries

FileSchemaLab uses fixed valid CSV parsing fixtures, not an arbitrary CSV parser. Select a file variation, predict publication, step/back/reset through raw text, parsed strings, conversion and domain validation. Exact values and types remain readable in HTML. Error messages name the failed rule; no file writes occur in the browser.

FilePublicationLab uses a named destination, staging buffer and observed reader value. Strategy, a fixed failure after the first record, and stepping expose truncation versus complete replacement. It models ordinary successful same-filesystem replace semantics, not operating-system crash recovery, hardware caches, directory fsync or object-store transactions. Native Python tests exercise real temporary files separately.

SqlJoinLab traces every output pair to its source IDs. Join type, a duplicate reference key, and ON/WHERE cutoff placement change output and counts. The duplicate case is explicitly an unconstrained staging table; the real primary-key table rejects it. Compact aggregate headings (Rows, Readings, Measured, Mean) have an explicit legend mapping them to COUNT(*), COUNT(reading_id), COUNT(value) and AVG(value). This readability change preserves the exact meanings and computations. This is a small relational model, not a SQL engine or physical query plan.

SqlTransactionLab shows a two-update invariant with writer-local state and committed state. Explicit transaction / separate commits and success / failure scenarios have next/back/reset. Readers are represented by a snapshot of committed state; the model does not promise all isolation levels share SQLite's exact locking or visibility behavior. Native SQLite checks verify commits/rollbacks and constraint failure.

## Claim ledger

Reviewed primary documentation on 2026-09-09; moving documentation pages are research sources, not a claim that local runtimes match their latest release. Original datasets, diagrams, wording, teaching sequence and exercises are project work.

| Claim / convention | Primary source and locator | Verification / boundary |
| --- | --- | --- |
| CSV yields strings by default, quoting handles commas, newline handling, None write loses distinction | https://docs.python.org/3/library/csv.html — reader, writer, DictReader | Python fixture compares exact fields; explicit application null convention |
| Text encoding differs from binary reads | https://docs.python.org/3/library/functions.html#open | UTF-8 source and explicit newline handling in local examples |
| JSON mappings and strict nonfinite serialization | https://docs.python.org/3/library/json.html — conversion table, allow_nan | null/None and finite number envelope checked; JSON doesn't preserve arbitrary Python types |
| NPY retains array representation; load pickle risk | https://numpy.org/doc/stable/reference/generated/numpy.save.html and https://numpy.org/doc/stable/reference/generated/numpy.load.html | Numeric dtype/shape/value round trip with allow_pickle=False; units stored explicitly |
| Typed column selection and row groups | https://arrow.apache.org/docs/python/parquet.html — reading subsets / row groups | Format-selection discussion; no unmeasured performance claim |
| Chunked scientific arrays | https://docs.h5py.org/en/stable/high/dataset.html — chunked storage | Access-pattern diagram and scope note, not an executed HDF5 benchmark |
| Replace and filesystem boundary | https://docs.python.org/3/library/os.html#os.replace | Temporary same-directory staging; distinguish atomic name replacement from durability |
| Logical selection, joins and NULL extension | https://www.sqlite.org/lang_select.html — simple SELECT processing / join clauses | Browser model compared to SQLite queries; logical explanation isn't promised execution order |
| COUNT and AVG null behavior | https://www.sqlite.org/lang_aggfunc.html — count, avg | Exact group counts/averages checked with empty and missing observations |
| Foreign-key enforcement | https://www.sqlite.org/foreignkeys.html — enabling support | Explicit PRAGMA before transaction and orphan-insert check |
| Python parameter binding and transaction configuration | https://docs.python.org/3/library/sqlite3.html — placeholders / transaction control | isolation_level=None plus explicit BEGIN/COMMIT/ROLLBACK avoids relying on changing defaults |
| Transaction boundaries and failures | https://www.sqlite.org/lang_transaction.html and https://www.postgresql.org/docs/current/tutorial-transactions.html | Explicit rollback on exception; statement failure is not described as universal automatic full rollback |

## Runtime and review evidence

- Implementation: both stable lessons are published and contain their complete examples, diagrams, independent practice, optional hints and explained solutions. File I/O has separate schema and publication investigations; SQL has separate join and transaction investigations. User acceptance remains pending.
- Native verification: [the saved result](../../scratch/first-five-review/data-native-results.json) records Python **3.12.14**, NumPy **2.3.5** and SQLite **3.53.1**. All **10 displayed programs** passed: six File I/O examples (including the all-missing boundary) and four SQL examples. The verifier also passed **15 CSV cases**, **16 join variants**, **four real-file publication traces** and **four transaction traces** against actual Python/NumPy/filesystem/SQLite behavior. All writes use disposable fixtures; no database server is required.
- Browser verification: [the saved integration result](../../scratch/first-five-review/browser-results.json) passed at **1440** and **390** CSS pixels for five published sequential lessons and **15 total labs**. It checks named adjacent-topic links, section/related links, full schema/publication/join/transaction controls against their models, keyboard behavior, hints, reset/back and absence of document overflow. The recorded page-error list is empty. Python, NumPy and OOP also retain their dedicated checks.
- Visual evidence: scoped schema, publication, join and transaction captures at both widths are under `scratch/first-five-review/`. The final captures exclude fixed site navigation during the screenshot only; the original UI and normal interaction checks retain it. The SQL aggregate-label cleanup changes presentation only, with the COUNT/AVG correspondence explicit beside the table.
- Batch integration: [the first-five integration record](../../FIRST-FIVE-REIMPLEMENTATION.md) owns final application build, cross-page review, curriculum conservation and shared handoff status. Source research, native checks, browser inspection and user acceptance remain distinct evidence categories.
- Learner evidence: author review only. No observed novice study has been conducted, and reading-completion controls do not establish mastery. User review is pending.
