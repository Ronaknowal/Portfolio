import { Code, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from "../../components/lesson-labs/PythonExample";
import PandasJoinLab from "../../components/lesson-labs/PandasJoinLab";
import { pandasExamples } from "../pandas-examples";
import { pandasPracticeExamples } from "../pandas-practice-examples.js";
import { PandasAlignmentLab, PandasCleaningLab, PandasGroupingLab } from "../../components/lesson-labs/PandasFoundationsLabs";
import { PandasPivotLab } from "../../components/lesson-labs/PandasPivotLab.jsx";
import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";

export default {
  title: "Pandas: Data Wrangling, Joins & Grouping",
  readTime: "~45 min read + 90 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot">
    <LessonIntro prerequisites="Python collections/functions and NumPy's shapes, masks and missing values. No SQL knowledge is required."
      sections={[["1-start-with-the-question-and-the-row", "Table model"], ["3-labels-align-even-when-you-do-not-ask", "Alignment"], ["4-clean-with-an-explicit-missing-data-policy", "Clean"], ["7-joins-preserve-or-change-the-unit-of-analysis", "Explore joins"], ["8-group-with-the-right-denominator", "Group"], ["11-build-an-auditable-report", "Project"], ["12-practise-and-check", "Practise"]]}>
      Turn messy orders and a customer lookup into an auditable regional report. Understand every row selected, matched, rejected and aggregated—not just how to produce a table that looks plausible.
    </LessonIntro>
    <Prose><strong>Your finish line:</strong> follow a record from arrival to a checked report, explain why each value was matched or excluded, and independently summarize a different dataset. Read the main route first; the calibration branch shows how the same matching idea extends to time.</Prose>
    <H2>1. Start with the question and the row</H2>
    <Prose>Suppose orders arrive as a CSV file and customer regions live in another table. You want total paid-order revenue by region. Before writing a chain of methods, define what one row represents: one order, not one customer or one product line. This unit is the table's <strong>grain</strong>. If one order can contain several lines, order counts and line counts are not interchangeable.</Prose>
    <Prose>A DataFrame is a two-dimensional labelled table. Each column is a Series with its own dtype, and the row index provides labels used for selection and alignment. Unlike a single ordinary NumPy array, different columns may have different types. The row index need not be your business identifier and is not required to be unique unless you enforce that rule.</Prose>
    <Prose>The examples run independently and include their data; no download is required. They were checked with Python 3.12.14 and Pandas 3.0.1. To recreate that Pandas version, use a Python 3.11+ virtual environment and its interpreter's <Code>-m pip install pandas==3.0.1</Code>. This records the tested version, not a claim that it is the newest. Copy-on-write behaviour below is specifically explained for Pandas 3.</Prose>
    <PythonExample example={pandasExamples.inspect}><Prose>The table has three rows and three columns. Selecting amount with one pair of brackets returns a Series; selecting a list of column names returns a DataFrame even when there is only one column. Int64 with a capital I is a nullable integer dtype, allowing an integer column to represent an unknown value without inventing zero.</Prose></PythonExample>
    <LessonTable caption="A useful first inspection" headers={["Question", "Tool", "What to watch for"]} rows={[
      ["How much data arrived?", "shape, len(df)", "Unexpected empty input or a large row-count change."],
      ["What columns and types arrived?", "columns, dtypes, info()", "IDs parsed as numbers, amounts left as text, surprising missing-value types."],
      ["What do actual rows look like?", "head, tail, a controlled sample", "A small preview is not a full validation."],
      ["Where is information absent?", "isna().sum()", "Count by column; do not infer missingness from visual blanks alone."],
      ["Are values plausible?", "describe, value_counts(dropna=False)", "Units, ranges and unexpected categories."],
      ["Is a key actually unique?", "duplicated(key), index.is_unique", "Business-key uniqueness and index uniqueness are different claims."],
    ]} />
    <Prose>Info prints a structural summary and returns None; printing its return value adds a misleading extra None. For large data, <Code>memory_usage(deep=True)</Code> helps inspect memory, although it is still not a full process-memory measurement. Inspect the schema before fitting a model or plotting a summary.</Prose>

    <H2>2. Select rows and columns deliberately</H2>
    <div className="nt-flow" aria-label="What changes in a table workflow"><span>Select<small>Keep particular rows or columns</small></span><span>Align / join<small>Connect values using labels or keys</small></span><span>Group / reduce<small>Turn members into a summary</small></span><span>Check<small>Account for records and meaning</small></span></div>
    <Prose>Read a chain one operation at a time: what identifies a row now, what changed, and what should remain the same? A new index of 0, 1, 2 is only a set of row labels; it does not establish that the underlying orders are unique. The investigations keep source identities visible so these distinctions are concrete.</Prose>
    <PythonExample example={pandasExamples.select}><Prose>Loc uses labels: the slice from a through b includes both endpoints here. Iloc uses integer positions: [0:1] includes position 0 and excludes position 1. A boolean mask keeps only order 102. The single loc assignment updates that row's amount in the original table.</Prose></PythonExample>
    <LessonTable caption="Selection syntax without guessing" headers={["Expression", "Meaning", "Typical result"]} rows={[
      ['df["amount"]', "One named column", "Series"],
      ['df[["order_id", "amount"]]', "A list of columns", "DataFrame"],
      ["df.loc[row_labels, column_labels]", "Label-based selection", "Shape depends on scalar versus list/slice selectors."],
      ["df.iloc[row_positions, column_positions]", "Position-based selection", "Python-style half-open slices."],
      ["df.at[label, column] / df.iat[i, j]", "One scalar by label / position", "A scalar when labels are unique."],
    ]} />
    <Prose>Use parenthesised comparisons joined with &amp; or | for elementwise boolean conditions, not Python's and/or. Loc aligns a boolean Series by its labels; an unalignable mask can fail. Iloc is positional and can use a positional boolean array instead. Integer-looking index labels remain labels for loc: label 10 does not mean the eleventh row.</Prose>
    <Prose>Label slicing is easiest to reason about with a sorted, unique index. Duplicates and missing slice bounds have additional rules; do not assume every unordered index behaves like a Python list. Query can make some filters readable, but is unnecessary here and should not be fed untrusted expression strings.</Prose>

    <H2>3. Labels align even when you do not ask</H2>
    <PandasAlignmentLab />
    <PythonExample example={pandasExamples.alignment}><Prose>The first order has index b, so it receives fee 2 even though that fee is second in its Series. The second order has index a and receives fee 1. Converting fees to a NumPy array removes labels and assigns the sequence [1, 2] by position instead. Both assignments run, but they mean different things.</Prose></PythonExample>
    <Prose>Arithmetic and Series assignment commonly align by index labels, not visual row order. Alignment can introduce missing values when labels do not match. Reindex requests an explicit label order and inserts missing entries for absent labels; it is not merely a positional shuffle. Reset_index moves the old index into a column unless drop=True is used. Set_index chooses columns as an index; check the resulting index.is_unique when uniqueness is required.</Prose>
    <H3>Copy-on-write changes the old “view or copy?” habit</H3>
    <Prose>In Pandas 3, an object derived by selection behaves independently when written through the Pandas API. The example changes part but leaves orders unchanged. Storage may be shared until a write requires a copy; behavioural independence does not promise eager physical copying. A plain alias such as <Code>other = orders</Code> is still the same object, not a derived one.</Prose>
    <Prose>Do not use chained assignment such as <Code>orders["amount"][mask] = 25</Code> to modify the original. Use <Code>orders.loc[mask, "amount"] = 25</Code> in one operation. Older tutorials based on Pandas 2's warning/view rules may describe different behaviour. An explicit copy is useful to express ownership, but it does not recursively duplicate arbitrary mutable Python objects stored inside object-dtype cells.</Prose>
    <Checkpoint prompt="A prediction Series has the right length but a shuffled index. Will df['prediction'] = predictions assign by row position?">
      <Prose>No: Series assignment aligns by labels. Verify that its index identifies the intended rows. Use a positional array only when positional matching is explicitly intended and its length/order have been checked.</Prose>
    </Checkpoint>

    <H2>4. Clean with an explicit missing-data policy</H2>
    <PandasCleaningLab />
    <PythonExample example={pandasPracticeExamples.nullable}><Prose>The capital-F Float64 dtype supports pd.NA. An unknown comparison stays unknown in the mask; selection with loc excludes it. The invalid-source mask singles out row 2, while preserving row 1 as an originally empty input. Conversion success and valid source data are different claims.</Prose></PythonExample>
    <PythonExample example={pandasExamples.cleaning}><Prose>We preserve amount_text while creating a parsed amount. Coercion turns both the malformed string and the missing input into missing numbers, but the invalid mask distinguishes the malformed source value "bad" from an originally absent value. Keeping that distinction makes a rejected-row report possible.</Prose></PythonExample>
    <Prose>String accessors operate on a column: str.strip removes surrounding whitespace, str.upper normalises case and str.replace performs replacements. Normalisation must respect meaning—case-folding case-sensitive IDs would corrupt them. Map is useful for a known value-to-value lookup; unmapped values need a policy. Avoid astype(str) as a universal cleaner because it can turn missing markers into ordinary text in workflows using Python string conversion.</Prose>
    <Prose>Missingness depends on dtype: you may encounter pd.NA, NaN, NaT or None. Use isna/notna rather than equality comparisons to a sentinel. Nullable booleans can represent unknown as well as True/False; in a filter, explicitly decide whether unknown should be retained, rejected or separated. Zero remains valid data and must not be confused with missing.</Prose>
    <LessonTable caption="A cleaning operation is also a decision" headers={["Operation", "What it does", "What it does not justify"]} rows={[
      ["to_numeric(errors='coerce')", "Convert bad parses to missing", "Silently accepting or discarding the bad source rows."],
      ["dropna(subset=[...])", "Remove rows missing selected fields", "Dropping rows for missing information irrelevant to the question."],
      ["fillna(value)", "Substitute a chosen value", "Claiming that an unknown measurement was really zero."],
      ["replace / map", "Change values by a rule or mapping", "Assuming every category was recognised."],
      ["astype / convert_dtypes", "Convert or infer storage types", "Validating units, business ranges or key uniqueness."],
    ]} />
    <Prose>Many reductions skip missing measurements. An all-missing sum can otherwise appear as zero; min_count=1 requires at least one known value. Report known-value counts alongside means. In ML, learn imputation statistics from training data only, then reuse them for validation/test data; computing them from the full dataset leaks information.</Prose>

    <H2>5. Parse dates before grouping by time</H2>
    <PythonExample example={pandasExamples.dates}><Prose>The invalid timestamp is counted and excluded explicitly. Resampling on the UTC index assigns 10 to January 1 and 20 to January 2. Converting both actual timestamps to Asia/Kolkata puts them on January 2 locally. If your question is daily local sales, convert to the intended timezone before daily aggregation.</Prose></PythonExample>
    <Prose>To_datetime converts text into datetime values; use an explicit known format where possible. Errors='coerce' needs the same rejected-value audit as numeric parsing. Utc=True normalises offset-aware timestamps, and treats naive inputs as UTC; it does not discover the source's local timezone. Localising naive local times requires a declared timezone and a daylight-saving ambiguity/nonexistent-time policy.</Prose>
    <Prose>Datetime accessors such as dt.year, dt.month and dt.strftime derive date features or display text. Sort time before rolling or previous-row calculations. Resampling groups into time bins; the bin frequency, timezone, boundary closure and labels are part of the analytical definition, not cosmetic formatting.</Prose>

    <H2>6. Resolve duplicates before counting or joining</H2>
    <PythonExample example={pandasExamples.duplicates}><Prose>One duplicate row repeats exactly. Removing it still leaves two different amounts for order 102. Dropping duplicates on order_id alone would silently choose one conflicting value. Instead, the example exposes the conflict and the integrity check rejects the non-unique key.</Prose></PythonExample>
    <Prose>Keep='first' or 'last' is meaningful only after defining an ordering and a conflict policy. For event updates, “latest” needs a trustworthy timestamp and tie handling. Do not use deduplication to hide a row multiplication caused by a bad join.</Prose>
    <Prose>Concat appends compatible tables along an axis; it does not look up matching business keys. Ignore_index=True supplies a new positional index, not new order identifiers. Collect frames and concatenate once rather than repeatedly growing a frame inside a loop. For differently named columns, rename them deliberately and validate the expected schema rather than letting unexpected columns appear unnoticed.</Prose>

    <H2>7. Joins preserve or change the unit of analysis</H2>
    <Prose>A merge connects rows by key values. Here each order should acquire at most one region, so many orders may reference one customer row. That is a many-to-one relationship. If the lookup contains two C1 rows, each C1 order finds two matches and appears twice.</Prose>
    <PandasJoinLab />
    <PythonExample example={pandasExamples.merge}><Prose>Left merge preserves the unmatched C9 order with an absent region and a left_only indicator. Inner merge drops it. Outer merge also includes C3 from the lookup, even though no order exists for it. Validation rejects duplicate lookup keys instead of producing a plausible but duplicated revenue table.</Prose></PythonExample>
    <LessonTable caption="Choose join type and cardinality separately" headers={["Choice", "Meaning", "Check"]} rows={[
      ["inner", "Keep matching keys", "Which source rows disappeared?"],
      ["left / right", "Preserve rows from that side", "Matches can still multiply a preserved row."],
      ["outer", "Keep keys from both sides", "Which rows have no counterpart?"],
      ["validate='one_to_one'", "Keys unique on both sides", "One record per entity on each side."],
      ["validate='many_to_one'", "Right keys unique", "Order table to customer lookup."],
      ["validate='one_to_many'", "Left keys unique", "Reverse relationship."],
      ["validate='many_to_many'", "Permit duplicates on both sides", "Does not protect against a Cartesian multiplication of matches."],
    ]} />
    <Prose>For one key, m left matches and n right matches can produce m × n output rows. Always declare join columns with on or left_on/right_on. Same-named unrelated columns can otherwise become accidental keys, and overlapping non-key columns need meaningful suffixes. Join commonly works with indexes; merge is clearer here because customer is an explicit column.</Prose>
    <PythonExample example={pandasExamples.nullJoin}><Prose>Pandas can match missing join keys to one another, unlike the usual SQL NULL join behaviour. If missing customer IDs must never match, reject or quarantine them before merging. A many-to-one check alone does not express that missing-key policy.</Prose></PythonExample>
    <Prose>Indicator distinguishes both, left_only and right_only, which is more reliable than testing a payload column for missingness: a legitimately matched customer's region could itself be unknown. Validate that payload separately. Comparing row counts alone can also miss an error if some rows were duplicated while others disappeared.</Prose>

    <H2>8. Group with the right denominator</H2>
    <Prose>Groupby splits rows by keys, performs an operation per group, then combines the results. It changes the grain when it aggregates: one row per order becomes one row per region. Decide which rows and missing values are eligible before choosing the statistic.</Prose>
    <PandasGroupingLab />
    <PythonExample example={pandasExamples.group}><Prose>North has two rows but only one known amount, so size is 2, count is 1 and the mean is 10—not 5. Dropna=False retains the missing-region group; the default grouping would omit that key and account for only three source rows. Transform broadcasts each group mean back onto its member rows instead of shrinking the table.</Prose></PythonExample>
    <LessonTable caption="Pick the grouping operation by its output shape" headers={["Operation", "Result grain", "Typical job"]} rows={[
      ["agg with named aggregations", "One row per group", "Report totals, means and counts with clear column names."],
      ["transform", "One result per original row for these column transformations", "Attach a group mean or normalise within a group."],
      ["filter", "Keep or discard complete groups", "Retain groups meeting a minimum size criterion."],
      ["apply", "Flexible, depends on the function", "Use only when the more explicit operations cannot express the task."],
    ]} />
    <Prose>As_index=False keeps grouping keys as ordinary columns. Observed controls whether unused categorical combinations appear; it matters for categorical groupers, not ordinary text keys. The examples set observed explicitly rather than relying on a version-dependent default. Sort controls key ordering, not the chronological order inside each group.</Prose>
    <Prose>Averaging already-averaged groups weights each group equally, not each observation. To recover an observation-level mean, combine the relevant totals and known-value counts. For sum, consider min_count when groups may be entirely missing. Report coverage alongside the number so a mean based on one known value is not mistaken for a mean based on hundreds.</Prose>

    <H2>9. Reshape a table without silently aggregating it</H2>
    <Prose>A long table stores one region/month amount per record. A wide table uses region as the row label and month as the column label. Together, those two labels act as a destination address for the amount. A <strong>pivot</strong> changes that layout; an aggregation also decides how multiple records become one value.</Prose>
    <PandasPivotLab />
    <PythonExample example={pandasExamples.reshape}><Prose>Pivot turns month values into columns when each region/month cell is unique. Melt reverses the layout into identifier, variable and value columns. Adding a duplicate North/Jan row makes pivot fail; pivot_table can resolve multiple cell values using an explicitly chosen aggregation. Here sum doubles that cell from 10 to 20.</Prose></PythonExample>
    <Prose>Pivot_table defaults to a mean if you do not specify aggfunc, which may be wrong for totals. A zero-filled missing cell asserts no activity, while an unknown cell asserts absent information; choose fill behaviour deliberately. Multiple index/column keys can create a MultiIndex, which is a hierarchy of labels rather than hidden nested tables.</Prose>
    <Prose>Stack/unstack move index levels between axes. Explode expands list-like cells into rows and therefore changes the grain and row count. Crosstab counts category combinations, and get_dummies builds indicator columns; neither removes the need to define the underlying unit of observation. Learn the shape contract before chaining these operations.</Prose>

    <H2>10. Previous values and running summaries need order</H2>
    <PythonExample example={pandasExamples.windows}><Prose>Sorting by customer and day makes a previous order well-defined. Grouped shift leaves the first row of each customer without a predecessor; grouped cumsum restarts at each customer boundary. The standalone rolling example averages up to two rows, yielding 10, 20 and 40.</Prose></PythonExample>
    <Prose>A two-row window is not a two-day window. Time-based rolling requires an appropriate time index, and min_periods specifies how many observations are required before reporting a value. For predictive features, decide whether the current row may contribute; you may need to shift before rolling to avoid using information unavailable at prediction time. A stable sort preserves input order for ties but does not invent a scientifically meaningful tie-breaker.</Prose>

    <details className="nt-deeper"><summary>A different use of matching: which calibration was active?</summary>
      <Prose>A reading and its calibration need not have identical timestamps. An equality merge would miss a reading at time 2 when the latest calibration was recorded at time 0. In this invented instrument example, an offset becomes applicable at its recorded time and expires after 3 ms. Subtract it from a reading only while it is recent enough.</Prose>
      <LessonTable caption="Choose a prior calibration, then check its age" headers={['Reading time','Latest prior calibration','Age','Decision']} rows={[[2,'time 0: offset 1',2,'10 − 1 = 9'],[5,'time 4: offset 3',1,'12 − 3 = 9'],[9,'time 4: offset 3',5,'Too old: result unknown']]}/>
      <Prose>Merge_asof searches an ordered time key. Backward means the right key must be less than or equal to the reading time; tolerance limits the distance. This is a matching policy, not interpolation. We chose 3 ms for the exercise, not as a physical recommendation. Sort the time key; for multiple instruments, use an appropriate by key and keep merge times globally sorted.</Prose>
      <PythonExample example={pandasPracticeExamples.temporal}><Prose>Itertuples supplies one row object at a time, so row.time_ms reads a field. Retaining the right timestamp lets us inspect age. An unmatched offset produces an unknown corrected reading; it does not imply an offset of zero.</Prose></PythonExample>
      <Checkpoint prompt="What changes if tolerance becomes 5? What if the second calibration is only delivered at time 8?">
        <Prose>With tolerance 5, time 9 uses offset 3 and becomes 12. But a timestamp alone does not prove availability: if the time-4 calibration arrives at 8, a real-time decision at 5 cannot use it. Event time and availability time need a separate contract. This lesson teaches matching; <a href="/learn/topic/ml-problem-formulation-baselines-data-leakage">data leakage and prediction-time information</a> is the stronger home for that evaluation question.</Prose>
      </Checkpoint>
    </details>
    <H2>11. Build an auditable report</H2>
    <Prose>The complete example introduces six raw records: one repeated row, one malformed amount, one cancelled order, one paid order with an unknown customer and two accepted orders. Amounts are deliberately tiny integer cents. The goal is a report whose exclusions are visible, not a silent reduction from six rows to two.</Prose>
    <PythonExample example={pandasExamples.project}><Prose>The audit reconciles all six source records: 1 exact duplicate + 1 invalid amount + 1 not-paid record + 1 unmatched record + 2 accepted records. North and South each have one accepted order, totalling 10 and 20 cents respectively. Order 105 is explicitly quarantined, not treated as revenue belonging to an invented region.</Prose></PythonExample>
    <LessonTable caption="Pipeline checkpoints and what each proves" headers={["Stage", "Rows remaining", "Evidence"]} rows={[
      ["Read raw CSV", "6", "Identifiers loaded as strings."],
      ["Remove exact repeated rows", "5", "One exact duplicate counted; conflicting IDs rejected."],
      ["Reject invalid amounts", "4", "Original invalid row retained separately."],
      ["Select paid orders", "3", "One valid but cancelled row excluded."],
      ["Attach regions", "3", "Many-to-one validation prevents row multiplication."],
      ["Separate unmatched customer", "2 accepted + 1 unmatched", "The indicator preserves the reason for exclusion."],
      ["Aggregate", "2 region rows", "Accepted amount total reconciles with reported revenue."],
    ]} />
    <Prose>Read_csv supports dtype, usecols, parse_dates and chunksize for controlling ingestion. Text identifiers can have meaningful leading zeros. Default CSV missing-value rules also interpret some tokens such as "NA" as missing; configure keep_default_na and na_values when those tokens are legitimate IDs. The inline StringIO supplies a file-like input without requiring a download.</Prose>
    <Prose>This example assumes small whole-cent amounts; fractional or out-of-range values need an explicit schema/rejection policy before conversion to Int64. Status handling is also deliberate: only exact paid values qualify here. Normalise or reject unexpected statuses according to the real data contract. Assertions demonstrate development checks; replace critical runtime validations with explicit exceptions because Python optimisation may remove asserts.</Prose>
    <Prose>Assert_frame_equal checks the expected table's values, labels and dtypes, not just its printed appearance. CSV stores text rather than Pandas dtype metadata, so the round-trip read specifies types again. Index=False avoids exporting an accidental positional-index column. Parquet can preserve richer schema efficiently but requires an appropriate engine such as PyArrow; Excel likewise requires a suitable engine. Do not use pickle for untrusted data.</Prose>
    <H3>Keep the workflow inspectable as it grows</H3>
    <Prose>Use vectorised arithmetic and built-in string/group operations when they express the task. Assign and pipe can organise readable transformations; row-wise apply is not an automatic speed optimisation. If iteration is genuinely needed, itertuples is often preferable to iterrows, which can coerce row types. Avoid turning a clear process into one enormous chain that is difficult to audit.</Prose>
    <Prose>Pandas is primarily in-memory. Select needed columns/dtypes early and measure memory. Chunked CSV reading helps only when the downstream algorithm can combine partial results correctly: partial sums and counts can combine, while a global exact median or unrestricted join needs more planning. Deduplication across chunks must also track keys across boundaries.</Prose>

    <H2>12. Practise and check</H2>
    <Checkpoint prompt="In the group example, why are North's row count, known count and mean 2, 1 and 10? What would replacing the missing amount with zero change?">
      <Prose>One of North's two amounts is missing. Size counts both rows; count and mean use the one known measurement. Filling with zero would change the known count to 2 and mean to 5, asserting something different about the underlying data.</Prose>
    </Checkpoint>
    <Checkpoint prompt="In the join explorer, add a second C1 lookup row. Predict the left-join row count with validation on, then off.">
      <Prose>With validation on the merge fails because the right key is non-unique. With validation off it produces four rows: order 101 matches twice, order 102 once, and unmatched order 103 is preserved once. The extra row is not an extra order.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Change the final raw CSV's C9 customer to C2. Predict the complete audit and regional totals before rerunning.">
      <Prose>The duplicate, invalid and not-paid counts stay 1 each. Unmatched becomes 0 and accepted becomes 3. North remains one order / 10 cents; South becomes two orders / 60 cents. Update the expected-table and unmatched assertions deliberately to this new contract.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Add another row with order_id 102 but amount 25 to the project. Should drop_duplicates make the conflict disappear?">
      <Prose>No: the new row is not an exact duplicate. After exact-row deduplication, order_id is still duplicated and the explicit validation raises ValueError. Investigate which record is authoritative before choosing a conflict-resolution policy.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Two region averages are 10 from one known order and 30 from three known orders. What is the combined order-level mean?">
      <Prose>(10 × 1 + 30 × 3) / (1 + 3) = 25. The simple average of the two region means is 20, which weights regions equally instead. Always connect the denominator to the question.</Prose>
    </Checkpoint>
    <H3>Independent task: off, observed or unknown?</H3>
    <Prose>Use sample IDs 1–6, devices A,A,A,B,B,C and watts 0,4,missing,0,0,missing. Each row is one scheduled observation. Report scheduled rows, observed readings and the fraction of observed readings with watts greater than zero, for each device. Zero means observed but inactive; missing means no observation. Keep an entirely unobserved device visible with an unknown fraction. This is an invented monitoring example; a missing reading cannot tell you whether a device was physically off.</Prose>
    <details><summary>Hint: create a nullable indicator before grouping</summary><Prose>Preserve missing values when comparing watts with zero. Count all sample IDs for scheduled observations, count known watts for the denominator, and sum the active indicator. Turn a zero denominator into missing before division.</Prose></details>
    <details><summary>Worked solution and checks</summary><PythonExample example={pandasPracticeExamples.uptime}><Prose>A is active in one of two observed readings, not one of three scheduled readings. B is observed and inactive. C has no measured denominator, so its fraction stays unknown. The active sum can be zero even for C; the observed count prevents that zero being misread as evidence of inactivity.</Prose></PythonExample></details>
    <Checkpoint prompt="Add sample 7 for C at 6 watts. Predict C's report. Then introduce a duplicate sample ID and explain the response.">
      <Prose>C has two scheduled rows, one observed value and active fraction 1.00. A duplicate sample ID must fail the explicit identity check. None of this proves a time-weighted uptime percentage: irregular sampling requires a different denominator and a stated interpolation policy.</Prose>
    </Checkpoint>
    <H3>Connect the checked report to a chart</H3>
    <Prose>You should be able to explain the grain, key rules, missingness policy and row-count changes at every stage. Once those checks hold, <a href="/learn/topic/matplotlib-scientific-plotting">Matplotlib &amp; Scientific Plotting</a> can communicate the result. Plotting cannot repair an incorrectly joined or aggregated table.</Prose>
    <Sources alternatives={<LearningResources>
        <li><a href="https://www.youtube.com/playlist?list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y">Data School / Kevin Markham — pandas video playlist</a> · Short question-led videos. Use the index, groupby, missing-values and merging entries in the <a href="https://github.com/justmarkham/pandas-videos">creator's notebook and chapter guide</a>. Older recordings predate Pandas 3; follow current copy-on-write rules here rather than old chained-assignment advice.</li>
        <li><a href="https://pandas.pydata.org/docs/getting_started/intro_tutorials/">Pandas getting-started articles</a> · A written second route through selection, derived columns, summary statistics and reshaping. Work one tutorial after its corresponding investigation.</li>
      </LearningResources>}>
      <li><a href="https://pandas.pydata.org/docs/user_guide/indexing.html">Pandas: selection and label alignment</a></li>
      <li><a href="https://pandas.pydata.org/docs/user_guide/copy_on_write.html">Pandas: copy-on-write and Pandas 3 behaviour</a></li>
      <li><a href="https://pandas.pydata.org/docs/user_guide/missing_data.html">Pandas: missing values and nullable dtypes</a></li>
      <li><a href="https://pandas.pydata.org/docs/user_guide/merging.html">Pandas: merges, validation, indicators and concatenation</a></li>
      <li><a href="https://pandas.pydata.org/docs/user_guide/groupby.html">Pandas: aggregation and transformation</a></li>
      <li><a href="https://pandas.pydata.org/docs/user_guide/reshaping.html">Pandas: pivoting, melting and reshaping</a></li>
      <li><a href="https://pandas.pydata.org/docs/user_guide/timeseries.html">Pandas: time zones and time-series operations</a></li>
      <li><a href="https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html">Pandas merge_asof: sorted keys, direction, tolerance and exact-match policy</a></li>
      <li><a href="https://pandas.pydata.org/docs/user_guide/io.html">Pandas: import/export options</a></li>
      <li><a href="https://pypi.org/project/pandas/3.0.1/">Pandas 3.0.1: recorded example version and Python requirements</a></li>
    </Sources>
  </div>,
};
