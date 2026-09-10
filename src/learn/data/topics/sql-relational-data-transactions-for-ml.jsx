import { Code, CodeBlock, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from '../../components/lesson-labs/PythonExample';
import { SqlJoinLab, SqlTransactionLab } from "../../components/lesson-labs/SqlLabs.jsx";

import { SqlRelationshipDiagram } from "../../components/lesson-labs/SqlFigures.jsx";
import { sqlExamples, featureQuery } from "../sql-examples.js";

import '../../components/lesson-labs/data-foundations.css';

export default {
  title: 'SQL, Relational Data & Transactions for ML',
  readTime: '~45 min read + 75 min practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot data-lesson sql-lesson">
    <LessonIntro prerequisites="Python values, lists, functions and exceptions. The preceding File I/O lesson introduces schemas and missing values; the key ideas are restated here. No database experience or server installation is needed."
      sections={[["1-decide-what-one-row-means", "Rows & keys"], ["2-ask-a-table-a-question", "Read SQL"], ["4-trace-a-join-before-trusting-it", "Join explorer"], ["6-build-one-feature-row-per-sensor", "Feature table"], ["7-make-related-updates-one-change", "Transactions"], ["9-practise-a-changed-feature-request", "Practise"]]}>
      Your experiment has a table of sensors and a table of readings. Build one trustworthy feature row per sensor, including sensors with no measurements. Then make a two-step update survive a failure. SQL is a way to describe the records you want; keys, null rules and transaction boundaries make that description dependable.
    </LessonIntro>
    <Prose><strong>Your finish line:</strong> explain every output row's origin, preserve the intended unit of analysis, write a parameterized cutoff query, and show that a failed transaction leaves no partial update. Follow sections 1–7 and the investigation in 9 first; the deeper branch introduces performance and time-aware extensions without making them hidden prerequisites.</Prose>

    <H2>1. Decide what one row means</H2>
    <Prose>A <strong>relational database</strong> stores data in related tables and provides rules for querying and changing it. Each table has named columns and rows. Before writing a query, state its <strong>grain</strong>: what one row represents. Our sensors table has one row per sensor; our readings table has one row per observation. These are different grains even though both tables mention a sensor ID.</Prose>
    <LessonTable caption="sensors — one row per sensor" headers={['sensor_id', 'room']} rows={[["A", "north"], ["B", "south"], ["C", "spare"]]} />
    <LessonTable caption="readings — one row per recorded observation" headers={['reading_id', 'sensor_id', 'minute', 'value · °C']} rows={[[1, 'A', 0, 18], [2, 'A', 10, 22], [3, 'B', 0, 'NULL']]} />
    <Prose>A <strong>primary key</strong> identifies a row uniquely. <Code>sensors.sensor_id</Code> identifies a sensor, and <Code>readings.reading_id</Code> identifies an observation. <Code>readings.sensor_id</Code> is allowed to repeat: sensor A was observed twice. A <strong>foreign key</strong> states that this sensor reference must correspond to a sensor in the other table.</Prose>
    <Prose>C exists as a sensor but has no observation. B has an observation record, but its value is <Code>NULL</Code>, SQL's marker for missing or unknown information. Neither should silently become a measured zero. Keeping a separate sensor table lets us represent a device before its first reading and change its room without editing every historical observation.</Prose>
    <SqlRelationshipDiagram />
    <Checkpoint prompt="Could sensor_id alone be the primary key of readings? What information would that forbid?">
      <Prose>It would forbid a second reading from the same sensor, because a primary key cannot repeat. An observation needs its own ID or an appropriate composite key. A composite key uses several columns together; sensor plus time works only if your actual measurement contract guarantees one observation for that combination.</Prose>
    </Checkpoint>
    <Prose>Separating facts into tables to reduce contradictory duplication is part of <strong>normalization</strong>. It does not mean splitting every field into its own table. Define the entities and dependencies first. When “room” means the room at measurement time, storing only a sensor's current room would lose history; that requires a timestamped history or a recorded observation attribute.</Prose>

    <H2>2. Ask a table a question</H2>
    <Prose><strong>SQL</strong> is a language for describing operations on tables. <Code>SELECT</Code> asks for a result. You say which columns, rows, combinations and summaries you need; the database chooses an execution strategy. It is not a Python loop written with different punctuation.</Prose>
    <CodeBlock language="sql">{`SELECT reading_id, sensor_id, value
FROM readings
WHERE value >= 20
ORDER BY reading_id;`}</CodeBlock>
    <Prose>Read this as “from readings, keep rows with a measured value at least 20, show these three columns, and sort the result by observation ID.” The result contains only reading 2: sensor A, value 22. SQL keywords are conventionally capitalized, while table and column names describe the data. The semicolon ends a statement in a script.</Prose>
    <ol className="data-flow" aria-label="A conceptual reading order for a SELECT query">
      <li><strong>FROM / JOIN</strong><small>Form candidate rows from the named tables.</small></li>
      <li><strong>WHERE</strong><small>Keep rows whose condition is true.</small></li>
      <li><strong>GROUP BY / HAVING</strong><small>Form summaries; optionally filter groups.</small></li>
      <li><strong>SELECT / ORDER BY</strong><small>Choose result expressions and their order.</small></li>
    </ol>
    <Prose>This is a model for reasoning about a query's meaning, not a promise about physical execution order. The optimizer may push a filter earlier or use an index. Without <Code>ORDER BY</Code>, do not rely on a stable output order. For ties, add enough ordering columns to make the order you need explicit.</Prose>
    <Prose>We will use <strong>SQLite</strong>, a database engine included with Python's standard-library module <Code>sqlite3</Code>. The next setup makes a fresh in-memory database, so it does not contact a server or modify an existing database. Each later example imports this saved setup file and creates its own fixture.</Prose>
    <Prose><Code>CREATE TABLE</Code> defines columns and constraints. <Code>NOT NULL</Code> requires a value; <Code>CHECK</Code> checks a condition; <Code>REFERENCES</Code> defines a foreign key. In SQLite we explicitly enable foreign-key enforcement for this connection before using the tables. A room cannot be NULL, but a reading value can: that missing-value policy is deliberate.</Prose>
    <Prose><strong>Read the setup:</strong> Python's triple quotes hold a string across several lines. <Code>executescript</Code> runs the fixed SQL statements in that string, while <Code>INSERT INTO ... VALUES</Code> adds our demonstration rows. <Code>PRAGMA</Code> sets an SQLite-specific option; it is not a portable SQL keyword. The <Code>try ... finally</Code> block always reaches its cleanup code, unlike an <Code>except</Code> block that handles a particular failure. Query values will be passed separately through placeholders, as the first caller shows.</Prose>
    <PythonExample example={sqlExamples.select}><Prose><Code>connection.execute</Code> submits one statement. <Code>fetchall</Code> collects its result rows as a list of tuples. <Code>(20,)</Code> is a one-item parameter tuple; its comma matters in Python. The second query asks for the missing reading explicitly. The database connection is closed in <Code>finally</Code>, even if a query fails.</Prose></PythonExample>
    <Prose>The setup uses <Code>isolation_level=None</Code> so the Python wrapper does not implicitly open transactions for us. Later examples issue <Code>BEGIN</Code>, <Code>COMMIT</Code> and <Code>ROLLBACK</Code> explicitly. This keeps transaction boundaries visible and avoids depending on changing wrapper defaults. The examples were verified with Python 3.12 and SQLite 3.53.1; the core statements also use long-established SQL features.</Prose>

    <H2>3. Treat NULL as unknown, not as zero</H2>
    <Prose>In SQL, comparing a missing value with a number normally produces <strong>unknown</strong>, not true or false. <Code>WHERE</Code> keeps rows only when its condition is true. That is why B's missing reading does not satisfy <Code>value &gt;= 20</Code>, and why asking <Code>value = NULL</Code> does not find it. Use <Code>IS NULL</Code> or <Code>IS NOT NULL</Code>.</Prose>
    <LessonTable caption="Predicates and missing values" headers={['Expression', 'Result for value = NULL', 'Does WHERE retain the row?']} rows={[
      ['value = NULL', 'Unknown', 'No'], ['value >= 20', 'Unknown', 'No'], ['value IS NULL', 'True', 'Yes'], ['value IS NOT NULL', 'False', 'No'],
    ]} />
    <Prose><Code>AVG(value)</Code> ignores NULL inputs and returns NULL when no non-NULL values contribute. <Code>COUNT(value)</Code> counts non-NULL values; <Code>COUNT(*)</Code> counts rows. When Python receives a SQL NULL through sqlite3, it represents it as <Code>None</Code>. The two names belong to different languages but play the corresponding role in these examples.</Prose>
    <Checkpoint prompt="Does replacing every NULL with zero make a mean more complete? What does it claim about the instrument?">
      <Prose>It invents measurements. A missing reading and a measured zero may have very different causes. If you later choose imputation for a model, justify it, keep missingness information where appropriate, and fit learned imputation rules using the training data only. A database convenience function cannot make that scientific decision for you.</Prose>
    </Checkpoint>

    <H2>4. Trace a join before trusting it</H2>
    <Prose>A <strong>join</strong> forms rows from matching records in two tables. For each sensor, look for all readings whose sensor ID matches. A has two matches, so it contributes two output rows. B contributes one match with a NULL value. A <Code>LEFT JOIN</Code> preserves C by adding one row with NULL reading fields; an <Code>INNER JOIN</Code> keeps matches only and drops C.</Prose>
    <Prose><Code>AS s</Code> and <Code>AS r</Code> give short local names to the tables in this query. They do not copy or rename stored data. <Code>s.sensor_id</Code> specifies which table's column we mean. <Code>ON</Code> supplies the matching condition.</Prose>
    <CodeBlock language="sql">{`SELECT s.sensor_id, s.room, r.reading_id, r.minute, r.value
FROM sensors AS s
LEFT JOIN readings AS r ON r.sensor_id = s.sensor_id
ORDER BY s.sensor_id, r.reading_id;`}</CodeBlock>
    <Prose>The join has four rows, even though there are only three sensors. That is not automatically an error: its grain is now a sensor paired with a matching observation, plus a preserved unmatched sensor. If your model expects one row per sensor, aggregation or a deliberate selection rule must restore that grain.</Prose>
    <SqlJoinLab />
    <Prose>More generally, if a key appears <Code>m</Code> times on one side and <Code>n</Code> times on the other, an equality join produces <Code>m × n</Code> matching pairs for that key. A duplicate sensor A with two readings therefore creates four A pairs. The laboratory's duplicate case is an unconstrained staging table; the primary key in our real setup rejects that duplicate before the join.</Prose>
    <Checkpoint prompt="Why is SELECT DISTINCT not a reliable repair for an unexpectedly large join?">
      <Prose>It removes duplicate selected rows, not the cause of the multiplication. Distinct observations can be lost if their selected values happen to match, while nonidentical duplicates can remain. Check key uniqueness and source cardinality, then choose the intended grain. An unchanged average is not proof that no duplication happened.</Prose>
    </Checkpoint>

    <H2>5. Put each filter at the stage where it belongs</H2>
    <Prose>Suppose features may use only readings observed through minute 0. We still want every sensor represented. Putting <Code>r.minute &lt;= 0</Code> in <Code>ON</Code> controls which readings are eligible matches; the left join can then preserve unmatched sensors. Putting it in <Code>WHERE</Code> filters the already-joined rows, including C's NULL-extended row. NULL compared with zero is unknown, so C disappears.</Prose>
    <CodeBlock language="sql">{`-- Preserve every sensor; restrict matching observations.
FROM sensors AS s
LEFT JOIN readings AS r
  ON r.sensor_id = s.sensor_id AND r.minute <= 0

-- Different meaning: drop result rows that fail the time condition.
FROM sensors AS s
LEFT JOIN readings AS r ON r.sensor_id = s.sensor_id
WHERE r.minute <= 0`}</CodeBlock>
    <Prose>These are query fragments showing the placement difference, not complete statements to execute alone. Explore both in the lab with the same cutoff. A condition about the sensor itself, such as <Code>s.room = 'north'</Code>, may appropriately go in WHERE when you really want only those sensors.</Prose>
    <Prose><Code>GROUP BY</Code> collects rows with the same grouping keys and computes aggregates. <Code>HAVING</Code> filters those groups after aggregation. For example, a requirement for at least two measured values belongs in <Code>HAVING COUNT(r.value) &gt;= 2</Code>. Filtering individual temperatures first answers a different question and can change the average.</Prose>

    <H2>6. Build one feature row per sensor</H2>
    <Prose>A <strong>feature</strong> is an input value used by a model. Our small feature table has one row per sensor, containing the number of observation records, the number with measured values and the mean of those measured values. The counts distinguish “no observation” from “observation attempted but value missing.”</Prose>
    <CodeBlock language="sql">{featureQuery}</CodeBlock>
    <Prose>The question mark is a <strong>parameter placeholder</strong>. Python supplies the cutoff separately as data. The aliases after <Code>AS</Code> name the output columns. Every selected item is either the grouping key or an aggregate; do not select an arbitrary ungrouped room or time and assume every database chooses the same row.</Prose>
    <PythonExample example={sqlExamples.features}><Prose>At minute 0, A has one allowed reading with mean 18. At minute 10 it has two, with mean (18 + 22) / 2 = 20. B has one observation record but no measured value. C has neither. Both means are None because neither sensor has a number to average.</Prose></PythonExample>
    <Prose>C illustrates the count distinction precisely. The left join produced one placeholder row, so <Code>COUNT(*)</Code> would be 1. But <Code>r.reading_id</Code> and <Code>r.value</Code> are NULL, so both corresponding counts are 0. For B, the reading ID is present while its value is missing, producing counts 1 and 0.</Prose>
    <Prose><strong>Time is part of the feature contract.</strong> If the prediction is made at minute 0, using A's minute-10 reading leaks future information into the input. The example's cutoff is a simple shared time boundary; a real training table may need a separate prediction time per row and an “as of” join. Also define when data became available, not only when an event says it happened.</Prose>
    <Prose>Parameter binding keeps a value separate from SQL structure and handles quoting correctly. Do not build SQL by inserting user text into the query string. Placeholders represent values, not table names, column names or keywords; dynamic identifiers need a deliberate set of allowed choices. After extracting features, check uniqueness, expected entity coverage, missingness and time boundaries before training.</Prose>

    <H2>7. Make related updates one change</H2>
    <Prose>Reading a feature table is only half the story. Databases also change. Imagine two research jobs with a shared compute-credit allocation: A has 6 credits and B has 4. Moving 2 from A to B requires two updates. If the debit commits but the credit fails, the total falls from 10 to 8 even though neither balance is negative.</Prose>
    <Prose>A <strong>transaction</strong> groups operations into one unit. <Code>BEGIN</Code> starts it; <Code>COMMIT</Code> accepts the unit; <Code>ROLLBACK</Code> discards its pending changes. The transaction can protect the two-update operation, while constraints protect individual rules such as a nonnegative balance. A correct transaction must still perform the right operations.</Prose>
    <SqlTransactionLab />
    <PythonExample example={sqlExamples.transaction}><Prose>The first attempt deliberately sets B's balance to −1, causing a constraint error. The application catches that specific error and explicitly rolls back, restoring A to 6 and B to 4. The second attempt completes both changes and commits A=4, B=6. A failed statement is not a universal promise that every earlier statement was automatically undone.</Prose></PythonExample>
    <LessonTable caption="Four transaction properties, with practical limits" headers={['Property', 'Question it answers', 'Boundary to remember']} rows={[
      ['Atomicity', 'Do the grouped changes take effect as a unit?', 'It does not repair earlier changes already committed outside the transaction'],
      ['Consistency', 'Are declared rules preserved by the transaction?', 'The database cannot infer every scientific or business rule you failed to express'],
      ['Isolation', 'How do concurrent transactions interact?', 'Visibility, locking and possible anomalies depend on the engine and isolation level'],
      ['Durability', 'What happens to acknowledged commits after failure?', 'This depends on the storage system and configured durability guarantees'],
    ]} />
    <Prose>The feature extractor may also need a consistent snapshot when several queries read changing data. A transaction is not a historical dataset version: record query text, parameters, source version or snapshot identity, schema and extraction time to reproduce a training table. SQLite's local file architecture and concurrency differ from a server database such as PostgreSQL; do not transfer every locking assumption between them.</Prose>

    <details className="data-deeper"><summary id="8-deeper-query-design-and-production-boundaries">8. Deeper: query design and production boundaries</summary>
      <H3>Indexes support access patterns</H3>
      <Prose>An index is an extra structure that helps find rows without scanning every possible row. An index on readings' sensor and time columns may help repeated sensor/time lookups; it also consumes storage and work on updates. Inspect the database's query plan and measure the actual workload. This three-row example cannot establish a meaningful speedup.</Prose>
      <H3>Choose a row deliberately when you do not want a mean</H3>
      <Prose>“Latest observation per sensor” is a different feature definition. A window function can assign row numbers within each sensor, ordered by time and an explicit tie-breaker such as reading ID, then select the first. A window function retains row-level data rather than collapsing each group. Learn this after you can explain the join and aggregation here; do not use an ungrouped time column as a shortcut.</Prose>
      <H3>State what the database actually enforces</H3>
      <Prose>SQL dialects differ in types and constraints. SQLite normally uses type affinity rather than the rigid typing many newcomers expect; its STRICT tables are a separate option. Our Python fixture inserts numeric readings and supplies range constraints, but a production contract should validate incoming values and test the actual engine's conversions. Primary/foreign keys and NOT NULL solve distinct problems.</Prose>
      <Prose>A transfer service must also check that both accounts exist, the intended rows were updated, authorization is correct and retries cannot duplicate an already committed transfer. Real failures include deadlocks, busy databases and uncertain client acknowledgements. Use the chosen engine's documented transaction/retry behavior; do not put an external email or file write inside a transaction and assume database rollback can undo it.</Prose>
      <H3>Connect to the rest of the data workflow</H3>
      <Prose>Files preserve data for exchange; tables provide relationships, queries and coordinated changes. A reproducible system may use both: ingest validated files, store constraints and lineage in a database, and export versioned feature files. Later lessons on Pandas, data contracts, time-series validation and MLOps extend these mechanisms. None of those tools removes the need to define a row's meaning.</Prose>
    </details>

    <H2>9. Practise a changed feature request</H2>
    <Prose>Start from the original fixture. Add reading 4 for sensor B at minute 5 with a real value of zero. Add sensor D in room west without any readings. Extract features using only observations through minute 5. Before running the query, predict the four output rows, both counts and each mean. Keep C and D, exclude A's future reading, and prove each sensor appears exactly once.</Prose>
    <details className="data-deeper"><summary>Hint: work sensor by sensor</summary><Prose>A contributes only reading 1. B contributes a missing-valued observation and a measured zero. C and D each need a preserved sensor row. Keep the cutoff in ON, then group by the sensor key and count observation IDs separately from measured values.</Prose></details>
    <details className="data-deeper"><summary>Explained solution and runnable verification</summary><PythonExample example={sqlExamples.practice}><Prose>B has two observation records but only one measured value, zero, so its mean is 0. C and D have no observation IDs and no numeric mean. A's mean stays 18 because the minute-10 value is excluded. The uniqueness check catches duplicate entity rows even if their averages happen to look plausible.</Prose></PythonExample></details>
    <Checkpoint prompt="Repair challenge: a teammate moves the cutoff to WHERE, uses COUNT(*) for n_readings, and replaces every missing mean with zero. Name one wrong result caused by each change.">
      <Prose>WHERE drops C and D because their NULL-extended time is not at most 5. COUNT(*) reports one observation for an unmatched sensor if it is preserved, even though no reading exists. Replacing missing means with zero makes C and D look like B, which actually measured zero. These are three different errors; fixing only one leaves the others.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Transaction transfer: what if A is debited outside BEGIN, then the application starts a transaction for B's credit and rolls it back?">
      <Prose>The already committed debit is outside the rollback boundary. The credits can still total 8. Put the complete logical change inside one transaction and verify the intended rows and rules, rather than adding BEGIN around only the statement that might fail.</Prose>
    </Checkpoint>
    <H3>Readiness and the next connection</H3>
    <Prose>Before continuing, explain the four original joined rows, the difference between B and C, the ON/WHERE cutoff behavior, and the two transaction outcomes. Next in this module, <a href="/learn/topic/pandas-data-wrangling-joins-grouping">Pandas data wrangling, joins and grouping</a> carries keyed relationships and missing-value decisions into in-memory tables. Compare the two systems' actual join and null rules instead of assuming that similar vocabulary means identical behavior. On a focused path, use the reader's named Next link for its selected module topics.</Prose>
    <Sources>
      <li><a href="https://www.sqlite.org/lang_select.html">SQLite SELECT</a> — joins, NULL extension and query semantics.</li>
      <li><a href="https://www.sqlite.org/lang_aggfunc.html">SQLite aggregate functions</a> — COUNT and AVG with missing values.</li>
      <li><a href="https://docs.python.org/3/library/sqlite3.html">Python sqlite3</a> — parameter binding and explicit transaction control.</li>
      <li><a href="https://www.sqlite.org/foreignkeys.html">SQLite foreign keys</a> and <a href="https://www.sqlite.org/datatype3.html">type affinity</a> — what this engine enforces.</li>
      <li><a href="https://www.sqlite.org/lang_transaction.html">SQLite transactions</a> and <a href="https://www.postgresql.org/docs/current/tutorial-transactions.html">PostgreSQL's transaction tutorial</a> — atomic changes and engine-specific behavior.</li>
      <li><a href="https://www.sqlite.org/queryplanner.html">SQLite query planning</a> — deeper study of indexes and access paths.</li>
    </Sources>
  </div>,
};
