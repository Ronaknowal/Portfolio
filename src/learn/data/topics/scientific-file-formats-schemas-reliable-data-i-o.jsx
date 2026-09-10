import { Code, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from '../../components/lesson-labs/PythonExample';
import { FileSchemaLab, FilePublicationLab } from "../../components/lesson-labs/ScientificFileLabs.jsx";

import { CsvBoundaryDiagram } from "../../components/lesson-labs/ScientificFileFigures.jsx";
import { fileExamples } from "../scientific-file-examples.js";
import '../../components/lesson-labs/data-foundations.css';

export default {
  title: 'Scientific File Formats, Schemas & Reliable Data I/O',
  readTime: '~40 min read + 65 min practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot data-lesson file-io-lesson">
    <LessonIntro prerequisites="Python strings, lists, dictionaries, functions and exceptions; NumPy arrays, shape and dtype. No database or file-format experience is assumed."
      sections={[["1-a-file-must-preserve-meaning", "The problem"], ["2-follow-bytes-into-records", "Bytes to records"], ["3-write-the-contract-before-converting", "Schema gates"], ["5-choose-a-format-for-the-data-and-the-work", "Formats"], ["7-publish-a-complete-update", "Reliable writes"], ["9-practise-an-independent-import", "Practise"]]}>
      A sensor export contains an ID called 001, a missing reading, a real zero and a comma inside a room name. An import can finish without an error and still change the meaning of all four. Learn to carry data from a file to a trustworthy result, prove what survived, and avoid publishing a half-written update.
    </LessonIntro>
    <Prose><strong>Your finish line:</strong> import a small measurement batch with an explicit contract, preserve missing values and units, verify a round trip, and explain what readers see if an update fails. The first pass covers sections 1–7 and the investigation in 9. The scaling branch helps you choose the next technique when files stop fitting comfortably in memory.</Prose>

    <H2>1. A file must preserve meaning</H2>
    <Prose>A file stores bytes: small numbered units of information. An application interprets those bytes as text, arrays, images or some other structure. <strong>Input/output, or I/O,</strong> means moving data into or out of a program. Successfully opening a file proves only that the program could access it. It does not prove that the values have the right types, units or interpretation.</Prose>
    <Prose>Imagine an experiment recorded three samples. The identifier <Code>001</Code> is a label, not a quantity to add. A blank temperature means the instrument did not produce a reading. A temperature of zero is an actual observation. The room name <Code>room,north</Code> contains a comma that belongs to the name. Our report should use the two measured values, 18.5 and 0, and report their mean as 9.25 °C.</Prose>
    <LessonTable caption="What must survive the trip?" headers={['Source value', 'Intended meaning', 'A plausible but wrong import']} rows={[
      ['001', 'Sample identifier, stored as text', 'Convert to integer 1 and lose the leading zeros'],
      ['blank temperature', 'Missing observation', 'Replace with zero and invent a measurement'],
      ['0', 'Measured zero degrees Celsius', 'Drop it because zero is false in a condition'],
      ['"room,north"', 'One field containing a comma', 'Split at every comma and create two fields'],
      ['C', 'Temperature unit', 'Combine with Fahrenheit because both values look numeric'],
    ]} />
    <Prose>A <strong>round trip</strong> means write data, read it again, and compare the restored meaning with the original contract. We may want equal records and units without demanding identical whitespace or byte order. Decide what “the same” means before testing it.</Prose>

    <H2>2. Follow bytes into records</H2>
    <ol className="data-flow" aria-label="From bytes to trusted measurements">
      <li><strong>Bytes → text</strong><small>Decode with a stated encoding, such as UTF-8.</small></li>
      <li><strong>Text → fields</strong><small>Use CSV quoting and delimiter rules to find boundaries.</small></li>
      <li><strong>Fields → values</strong><small>Apply explicit types and missing-value conventions.</small></li>
      <li><strong>Values → accepted data</strong><small>Check identity, units, range and cross-record rules.</small></li>
    </ol>
    <Prose>An <strong>encoding</strong> maps between characters and bytes. It is separate from the file format: UTF-8 tells us how to read characters, while CSV tells us where fields and records begin. A filename ending in <Code>.csv</Code> is a convention, not a guarantee that the contents obey CSV rules.</Prose>
    <Prose><strong>Parsing</strong> finds structure. Python's default CSV reader returns strings, so parsing <Code>18.5</Code> does not yet produce a floating-point number. A CSV writer quotes fields when necessary; a CSV reader recognises those quotes. Calling <Code>line.split(",")</Code> has no such understanding and breaks our room name. A record can even contain a quoted newline, so one physical line need not equal one record.</Prose>
    <CsvBoundaryDiagram />
    <Prose>Run the first example with Python 3.10 or newer. <Code>io.StringIO</Code> presents a string as a readable text stream so we can inspect parsing without creating a file. <Code>DictReader</Code> uses the header as dictionary keys; <Code>next</Code> asks for one data record. The escape <Code>\n</Code> inside the Python string represents a newline.</Prose>
    <PythonExample example={fileExamples.parse}><Prose>The comma is still inside one room string, and both the ID and temperature are initially strings. We deliberately convert only the temperature before arithmetic. Keeping the identifier as text preserves its spelling.</Prose></PythonExample>
    <Checkpoint prompt="Why would float(row['temperature']) fail for a blank field? Would int(row['sample_id']) be a good repair for the ID?">
      <Prose>A blank string is not a number. The importer needs a stated missing-value rule instead of a numeric guess. Converting the ID to an integer would discard its spelling; it fixes no problem in a label that was already valid text.</Prose>
    </Checkpoint>

    <H2>3. Write the contract before converting</H2>
    <Prose>A <strong>schema</strong> is an explicit agreement about structure and meaning. It can state column names, types, required fields and allowed missing values. Application rules add conditions such as “each sample ID appears once” or “all readings in this batch are Celsius.” A format may carry some of that information, but the application still needs to check its own rules.</Prose>
    <LessonTable caption="The contract for this particular measurement batch" headers={['Field / rule', 'Contract', 'Reason']} rows={[
      ['Header', 'sample_id, temperature, unit, site in this order', 'Catch accidental exports with a different shape'],
      ['sample_id', 'Exactly three ASCII digits, kept as str; unique in the batch', 'Preserve identity and detect repeated samples'],
      ['temperature', 'Empty field → None; whitespace-only text is invalid; otherwise a finite float in [−80, 80]', 'Instrument range for this invented experiment'],
      ['unit', 'Exactly C', 'One common scale before aggregation'],
      ['site', 'Nonblank text', 'Retain observation context'],
      ['Batch', 'At least one record; reject the whole batch on any invalid record', 'No silent partial import'],
    ]} />
    <Prose>The range is a rule for this example, not a universal range for scientific measurements. A real instrument needs its own contract. We also reject unexpected fields and headers rather than guessing how to use them. Other applications may allow extra columns, but that is a decision to record explicitly.</Prose>
    <Prose>The earlier Python report chose to skip whitespace-only strings. This file contract is stricter: only an empty temperature field means missing; a field containing spaces alone is invalid. Missing-value conventions belong to the dataset contract, so do not silently transfer one parser's rule to another format.</Prose>
    <Prose><Code>None</Code> is a Python value meaning no value here. It differs from numeric zero and the empty string. <Code>NaN</Code> is a special floating-point value, often used for missing or invalid numeric results; it is not an ordinary finite reading. Python can convert the text <Code>"NaN"</Code> into a float, which is why a successful conversion still needs a finiteness check.</Prose>
    <FileSchemaLab />
    <Checkpoint prompt="A row says 77,F. It passes the numeric range check. Can you safely change its unit label to C and keep 77?">
      <Prose>No. Units are part of the value's meaning. A declared Fahrenheit conversion would compute (77 − 32) × 5 / 9 = 25 °C. Relabelling it as 77 °C changes the measurement. This importer rejects mixed units so that any conversion must be explicit and traceable.</Prose>
    </Checkpoint>

    <H2>4. Run a complete import and check the round trip</H2>
    <Prose>Save the next two files together. The first defines the contract in one reusable function. The second creates disposable input, imports it, saves JSON, reloads it and checks the result. <Code>TemporaryDirectory</Code> creates a practice folder and removes it when the block finishes; this code does not need your existing files. <Code>Path</Code> builds paths without manually joining separators.</Prose>
    <Prose>Three less familiar tools are worth naming before the code. <Code>re.fullmatch</Code> checks that the whole ID fits a pattern; <Code>[0-9]&#123;3&#125;</Code> means three digits. A <Code>set</Code> called <Code>seen</Code> remembers IDs already encountered. <Code>math.isfinite</Code> rejects infinity and NaN. These are checks on different properties, not interchangeable ways to detect a bad row.</Prose>
    <Prose>The <Code>with open(...)</Code> block closes the file even if validation raises an exception. We request UTF-8 explicitly. For the CSV module, <Code>newline=""</Code> lets the CSV parser handle newline conventions. A wrong field count is checked separately from an empty field: missing columns and present-but-blank values mean different things.</Prose>
    <Prose><strong>Read the compact syntax before running it:</strong> <Code>any(value is None for value in row.values())</Code> examines the field values one at a time and is true if at least one is None. It detects an absent column. <Code>None if raw == "" else float(raw)</Code> is a conditional expression: choose None for a blank, otherwise convert the text. <Code>enumerate(..., start=1)</Code> attaches a record number to each row so an error can point to its source.</Prose>
    <Prose>For a Path object, <Code>root / "measurements.csv"</Code> builds a child path; here the slash joins a path, rather than dividing numbers. <Code>write_text</Code> writes a string and <Code>read_text</Code> reads one. <Code>json.dumps</Code> turns Python values into JSON text; <Code>json.loads</Code> turns that text back into Python values. Neither operation knows our measurement rules without the checks around it.</Prose>
    <PythonExample example={fileExamples.roundtrip}><Prose>The restored IDs remain strings. JSON turns Python <Code>None</Code> into <Code>null</Code> and reads it back as <Code>None</Code>. The average includes zero and excludes only missing values. Equality checks the complete controlled envelope, including units and schema version; it is stronger than merely asking whether JSON parsing succeeded.</Prose></PythonExample>
    <Prose><Code>allow_nan=False</Code> stops the writer from emitting nonstandard NaN or infinity values. JSON supports objects, arrays, strings, numbers, booleans and null; it does not preserve arbitrary Python classes, a NumPy dtype, tuples as tuples, or physical units by itself. We put our unit and schema version into an explicit <strong>envelope</strong>: a container around the records.</Prose>
    <Prose>For an externally supplied JSON file, validate its schema and values too. Loading JSON is not an approval step. In this controlled round trip, comparison with the original validated envelope checks that our own serialization preserved the intended data. An empty batch is rejected. A batch containing records whose temperatures are all missing is allowed, but it has no numeric mean: the conditional calculation returns None instead of dividing by zero.</Prose>
    <details className="data-deeper"><summary>Try the all-missing boundary</summary><PythonExample example={fileExamples.missing}><Prose>Two records are present, so the batch is structurally valid. No numeric reading contributes to the average, so the report says unavailable. That is different from a batch that measured a mean of zero.</Prose></PythonExample></details>
    <Checkpoint prompt="What changes if the mean code uses 'if row[temperature]' rather than 'if row[temperature] is not None'?">
      <Prose>Zero is false in a Python condition, so that filter would drop a real observation. Our two values would incorrectly become only 18.5, making the reported mean 18.5 instead of 9.25. Check the condition against the intended missing-value convention.</Prose>
    </Checkpoint>

    <H2>5. Choose a format for the data and the work</H2>
    <Prose>There is no single best scientific file format. Ask what structure you need to preserve, what readers must understand, and which pieces will be accessed together. A simple text exchange and a large multi-dimensional recording pose different problems.</Prose>
    <LessonTable caption="Choose by representation and access pattern" headers={['Format', 'Useful for', 'What still needs an explicit contract']} rows={[
      ['CSV / TSV', 'Small rectangular exchanges that many tools can inspect', 'Types, delimiter, encoding, units and null convention; nested data is awkward'],
      ['JSON / JSON Lines', 'Metadata and nested records; JSON Lines stores one JSON value per line', 'Schema, units, numeric limits and versioning; repeated field names add size'],
      ['NPY / NPZ', 'NumPy arrays with dtype and shape; NPZ bundles named NPY arrays', 'Axis meaning, units, provenance; object arrays can require pickle'],
      ['Parquet', 'Typed columnar tables; reading selected columns and suitable row groups', 'Business rules, keys, units and compatible schema evolution'],
      ['HDF5 / chunked array stores', 'Large multi-dimensional recordings, arrays and hierarchical metadata', 'Dataset layout, chunk shape, library compatibility and scientific conventions'],
    ]} />
    <Prose><strong>Row-oriented</strong> access brings the fields of a record together. <strong>Columnar</strong> storage groups values of the same field, which can help when an analysis needs a few columns across many records. Compression and skipping depend on data layout, metadata and the reader; a format choice alone does not promise a speedup.</Prose>
    <Prose>CSV has no universal missing-value marker. Python's CSV writer writes <Code>None</Code> as an empty field, losing the distinction between None and an originally empty string unless you define a separate convention. JSON's null makes that particular distinction explicit, but neither format knows that a number is a temperature.</Prose>

    <H2>6. Preserve an array and describe its axes</H2>
    <Prose>NumPy taught you that an array's shape and dtype affect its meaning. Saving only formatted numbers can lose both. NPY stores the array representation; we still add metadata saying what each axis and value mean. Here rows are the observations at minutes 0 and 1, columns are sensors 001 and 002, and values are degrees Celsius.</Prose>
    <Prose>This example requires NumPy in the same Python environment, as in the preceding lesson. The examples were checked with Python 3.12 and NumPy 2.3.5; they do not depend on a newly introduced API. The other examples on this page use Python's standard library.</Prose>
    <PythonExample example={fileExamples.array}><Prose>The checks cover shape, dtype, exact values for these representable numbers, and metadata. A separate test may need a justified tolerance when the workflow deliberately changes precision. Arbitrarily rounding both sides can conceal an actual error.</Prose></PythonExample>
    <Prose>Keep <Code>allow_pickle=False</Code> when loading ordinary numeric arrays. Pickle can reconstruct Python objects by executing code; it is not an appropriate interchange mechanism for an untrusted file. Numeric arrays do not need it. This small example writes two related files for clarity; publishing both as one consistent dataset requires the manifest strategy described next.</Prose>

    <H2>7. Publish a complete update</H2>
    <Prose>Opening an existing file with mode <Code>"w"</Code> truncates it before the replacement is complete. If the process fails after one record, there is no automatic return to the old contents. A safer single-file pattern is to build a separate file in the destination directory, close and validate it, then switch the published name to that complete file.</Prose>
    <FilePublicationLab />
    <PythonExample example={fileExamples.publish}><Prose>The old version stays at the public name until the replacement step. <Code>os.replace</Code> removes the staging name when the operation succeeds. The temporary directory makes this fixed staging filename suitable for an isolated exercise; concurrent production writers need distinct staging names and a coordination policy.</Prose></PythonExample>
    <Prose><strong>Atomic visibility and durability answer different questions.</strong> Successful same-filesystem replacement is about readers seeing the old name binding or the new one, rather than an in-between copy. Durability asks what survives a power loss. Flushing user-space buffers, filesystem synchronization, directory metadata and storage guarantees need a platform-specific protocol; this example does not claim to supply that protocol.</Prose>
    <Prose>If arrays and metadata live in several files, replacing one at a time can mix versions. Write a new versioned dataset folder, validate all files, then publish a manifest or pointer naming that version using the storage system's supported atomic operation. A <strong>manifest</strong> records file names, shapes, schema version, units and provenance. A checksum can detect changed bytes relative to a trusted expected checksum; it cannot prove that the measurements themselves are scientifically correct.</Prose>

    <details className="data-deeper"><summary id="8-scale-without-losing-the-contract">8. Deeper: read only the pieces the task needs</summary>
      <Prose>A file can be larger than the program's working memory. Iterating a CSV reader processes one record at a time, but our importer stores accepted records and the set of seen IDs, so its memory use still grows with the batch. To keep memory bounded, process or write validated records incrementally and place whole-dataset checks such as uniqueness in an external store or an appropriate partitioned workflow. Streaming alone does not solve every global rule.</Prose>
      <ol className="data-flow" aria-label="Chunked reading workflow"><li><strong>Choose a slice</strong><small>For example, all sensors during one second.</small></li><li><strong>Load overlapping chunks</strong><small>The chunk layout determines how much extra data is read.</small></li><li><strong>Validate and compute</strong><small>Keep only the state needed for the next result.</small></li><li><strong>Save checked output</strong><small>Record source version and transformation.</small></li></ol>
      <Prose>A <strong>chunk</strong> is a stored block of an array. Reading one cell of a compressed chunk may require reading and decompressing the whole block. Long time-wise chunks can fit time-window queries but make all-time single-sensor queries expensive, or vice versa depending on layout. Choose chunks for measured access patterns. HDF5 and other array stores expose such choices; their APIs and concurrency rules differ.</Prose>
      <Prose>Parquet similarly lets a reader select columns and use row-group metadata where it can skip irrelevant data. It is not a substitute for a database transaction. For NPY arrays, <Code>np.load(path, mmap_mode="r", allow_pickle=False)</Code> can expose file-backed numeric data without eagerly reading the entire array, but operations can still allocate large result arrays. Measure peak memory as well as load time.</Prose>
      <Prose>Schema evolution also needs a policy. Adding an optional field may be compatible; changing Celsius to Fahrenheit or seconds to milliseconds under the same field name is a semantic change. Record versions and test old-reader/new-file behavior deliberately. Keep source provenance and the transformation version with the result so a later investigator can reproduce it.</Prose>
    </details>

    <H2>9. Practise an independent import</H2>
    <Prose>You receive a new batch with IDs 004, 005 and 006. Temperatures are 0, blank and 24, all in Celsius at the lab. Before running code, write down the number of records, measured and missing counts, mean, and which IDs should survive unchanged. Then build this fixture and use the saved importer. Finally, replace ID 006 with another 004: the whole import should fail.</Prose>
    <details className="data-deeper"><summary>Hint: separate counting from arithmetic</summary><Prose>Keep all accepted records. Select numeric values with an explicit None check. The denominator is the count of measured values, not the number of records. Repeated IDs violate a batch rule even if every temperature is plausible.</Prose></details>
    <details className="data-deeper"><summary>Explained solution and runnable check</summary><PythonExample example={fileExamples.practice}><Prose>There are three records, one missing value, two measured values and a mean of (0 + 24) / 2 = 12. Reusing 004 loses the guarantee that each row is a different sample, so the second file is rejected before publication.</Prose></PythonExample></details>
    <Checkpoint prompt="Transfer: a file loads successfully but its axes are swapped, its unit metadata says F, and its values match the original numbers. Has the round trip succeeded? Design three checks that would catch it.">
      <Prose>No. Check shape and axis labels together, check the declared unit and any conversion, and check selected coordinate-to-sensor/time correspondences. Equal flattened values do not prove equal meaning. Compare the complete metadata contract and a few known observations, then test the full array appropriately.</Prose>
    </Checkpoint>
    <H3>Readiness and the next connection</H3>
    <Prose>You are ready to continue when you can explain each import gate, distinguish missing from zero, justify a format, detect a lossy round trip, and predict the state after a failed write. In the opening curriculum sequence, <a href="/learn/topic/sql-relational-data-transactions-for-ml">SQL, Relational Data &amp; Transactions for ML</a> takes the same idea of explicit contracts into related tables: how do we join records without duplicating examples, and update several values as one change? On another guided path, follow the reader's named Next link for that route's sequence.</Prose>
    <Sources>
      <li><a href="https://docs.python.org/3/library/csv.html">Python CSV documentation</a> — quoting, parsed strings, newlines and null conventions.</li>
      <li><a href="https://docs.python.org/3/library/json.html">Python JSON documentation</a> — supported values and nonfinite-number handling.</li>
      <li><a href="https://numpy.org/doc/stable/reference/generated/numpy.load.html">NumPy load</a> and <a href="https://numpy.org/doc/stable/reference/generated/numpy.save.html">save</a> — array persistence, memory mapping and object-array boundaries.</li>
      <li><a href="https://docs.python.org/3/library/os.html#os.replace">Python os.replace</a> — replacement semantics and filesystem constraints.</li>
      <li><a href="https://arrow.apache.org/docs/python/parquet.html">Apache Arrow's Parquet guide</a> and <a href="https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage">h5py chunked datasets</a> — deeper format and access-pattern study.</li>
    </Sources>
  </div>,
};
