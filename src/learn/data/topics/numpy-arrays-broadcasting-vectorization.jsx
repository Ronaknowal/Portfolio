import { Code, CodeBlock, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from '../../components/lesson-labs/PythonExample';
import { NumpyDataDiagram, NumpySelectionLab, NumpyMemoryLab, NumpyBroadcastLab, NumpyReductionLab } from "../../components/lesson-labs/numpy-foundations-labs";
import { NumpyShapeComparison } from "../../components/lesson-labs/NumpyFigures.jsx";
import NumpyFoundationsDeeper from '../../components/lesson-labs/numpy-foundations-deeper';
import { numpyFoundationsExamples } from "../numpy-foundations-examples";
import { numpyReferenceExamples } from "../numpy-reference-examples.js";

export default {
  title: 'NumPy: Arrays, Broadcasting & Vectorization',
  readTime: '~45 min core read + 75 min practice; optional deeper branches',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot numpy-lesson">
    <LessonIntro prerequisites="Python names, lists, zero-based indexing, comparisons, functions and imports. No matrix algebra required. The previous Python lesson teaches these; we refresh the notation where it changes."
      sections={[
        ['1-start-with-readings-and-a-question', 'Start with data'], ['2-select-values-without-losing-their-meaning', 'Coordinates'],
        ['3-follow-the-data-buffer-through-a-view', 'Views & copies'], ['4-broadcast-by-matching-coordinates', 'Broadcasting'],
        ['5-reduce-an-axis-keep-the-question', 'Reductions'], ['9-build-a-small-sensor-report', 'Complete workflow'],
        ['10-investigate-a-new-dataset', 'Independent practice'], ['11-deeper-tools-when-your-task-needs-them', 'Deeper reference'],
      ]}>
      Two sensors measured the same room at several times. How do you correct each sensor's offset, find the warm observations and summarise the readings without mixing up times, sensors or raw data? Learn to trace every array result back to its inputs before relying on compact code.
    </LessonIntro>
    <Prose><strong>Your finish line:</strong> read shapes and coordinates, select whole observations, predict memory sharing, explain which value broadcasts into each output cell, choose the correct reduction axis, and build a checked numerical workflow. Four focused explorers make those hidden mechanisms visible. Follow sections 1–10 first; section 11 preserves more advanced tools for later revision.</Prose>

    <H2>1. Start with readings and a question</H2>
    <Prose>NumPy is a Python library for working with organised collections of numbers. Its central object, an <strong>array</strong>, keeps values together with a description of their shape and storage type. Instead of telling Python to visit each reading in a loop, you can describe an operation on the collection: subtract these calibration offsets, keep these observations, or average over these times.</Prose>
    <Prose>That compact expression is useful only when the organisation is correct. In this lesson, X has three rows for times 0, 1 and 2, and two columns for sensors A and B. Every value is a temperature in degrees Celsius. The numbers are invented so you can check them by hand; NumPy itself stores neither the sensor names nor the units.</Prose>
    <NumpyDataDiagram />
    <Prose>An <strong>axis</strong> is a coordinate direction in the array. Axis 0 chooses the time; axis 1 chooses the sensor. The <strong>shape</strong> <Code>(3, 2)</Code> lists the lengths in that order. There are two axes (<Code>ndim == 2</Code>) and six stored values (<Code>size == 6</Code>). A single reading needs both coordinates: <Code>X[1, 0]</Code> is 24, because indices begin at zero.</Prose>
    <Prose>A <strong>dtype</strong> describes how each element is stored. We explicitly use <Code>float64</Code>, a 64-bit floating-point numeric type: eight bytes per reading. Unlike Python's list, this ordinary numeric array uses one dtype for all of its elements. Floating-point numbers can represent fractions, with finite precision; section 8 explains why that affects answers.</Prose>
    <Prose><strong>Run the examples:</strong> use the Python environment from the previous lesson. In that environment, run the command below once to install NumPy if it is absent. Save each complete code block into its own <Code>.py</Code> file and run <Code>python filename.py</Code>, or run the whole block in a notebook cell. Every example includes its imports and data; there is no hidden prior cell to execute.</Prose>
    <CodeBlock language="bash">python -m pip install numpy</CodeBlock>
    <Prose>The examples were checked with Python 3.12.14 and NumPy 2.3.5. Those are tested versions, not a claim about the newest release. The browser explorers simulate their stated small examples; they do not run arbitrary Python. Printed lists use <Code>.tolist()</Code> only to make the expected output easier to inspect.</Prose>
    <PythonExample example={numpyFoundationsExamples.create}><Prose><Code>import numpy as np</Code> gives the imported library the short name np. The nested lists supply rows to <Code>np.array</Code>. Python list multiplication repeats references to its elements; NumPy numeric multiplication applies arithmetic to corresponding values. The same <Code>*</Code> symbol therefore has a different contract depending on the object.</Prose></PythonExample>
    <Checkpoint prompt="If a fourth time is recorded by the same two sensors, what changes: the shape, the number of axes, or both?">
      <Prose>The shape becomes (4, 2), with eight values. There are still two axes: time and sensor. More readings along an existing axis do not create an additional axis.</Prose>
    </Checkpoint>

    <H2>2. Select values without losing their meaning</H2>
    <Prose>Inside <Code>X[rows, columns]</Code>, the comma separates instructions for the two axes. A colon <Code>:</Code> means take the whole axis. A slice such as <Code>:1</Code> means start at zero and stop before 1. As in Python, the stop is excluded. An integer fixes one coordinate and removes that axis; a slice keeps an axis even when only one position survives.</Prose>
    <LessonTable caption="Read the coordinates before reading the result" headers={['Selection', 'Read it aloud', 'Result meaning']} rows={[
      ['X[1, 0]', 'Time 1, sensor 0', 'One numeric scalar: 24 °C.'],
      ['X[1, :]', 'Time 1, every sensor', 'Two readings, shape (2,).'],
      ['X[:, 0]', 'Every time, sensor 0', 'Three readings, shape (3,).'],
      ['X[:, :1]', 'Every time, a one-column slice', 'A three-row, one-column array, shape (3, 1).'],
      ['X[-1, :]', 'The last time, every sensor', 'The final row, [30, 32].'],
    ]} />
    <Prose>The comma in <Code>(3,)</Code> is Python's notation for a tuple with one item. It means one axis of length three. It is neither inherently a row nor a column. Shape <Code>(3, 1)</Code> explicitly has two axes; that difference will decide whether an operation broadcasts correctly.</Prose>
    <H3>Ask for rows or ask for cells</H3>
    <Prose>A <strong>boolean mask</strong> is a collection of True/False choices. <Code>{'X[:, 0] >= 24'}</Code> checks sensor A at each time and produces [False, True, True]: one choice per row. Using that mask as <Code>X[mask]</Code> keeps whole observations. By contrast, <Code>{'X >= 24'}</Code> tests every cell, and using that full-array mask collects individual values into a one-dimensional array.</Prose>
    <NumpySelectionLab />
    <PythonExample example={numpyFoundationsExamples.select}><Prose>Whole-row selection keeps the two-sensor relationship. Cell selection removes that grid, even though this particular dataset gives the same four values. The last expression combines two conditions on each row; parentheses keep each comparison together before elementwise <Code>&amp;</Code> combines them.</Prose></PythonExample>
    <Prose>Use <Code>&amp;</Code> for elementwise “and”, <Code>|</Code> for “or” and <Code>~</Code> for boolean negation. Python's <Code>and</Code>/<Code>or</Code> ask for the truth of an entire object, which is ambiguous for a multi-element array. To ask whether at least one value is True use <Code>mask.any()</Code>; for every value use <Code>mask.all()</Code>.</Prose>
    <Prose><strong>A condition's shape is part of its meaning.</strong> A row mask must match the row axis it selects. Boolean indexing does not expand a (3, 1) mask across a (3, 2) table using the arithmetic broadcasting rules. We will use <Code>all(axis=1)</Code> to turn one validity decision per cell into one decision per row after learning reductions.</Prose>
    <Checkpoint prompt="Keep the complete observation where A is at least 24 °C and B is below 30 °C. Predict the mask and output shape before checking the code above.">
      <Prose>The mask is [False, True, False], so the result is [[24, 26]] with shape (1, 2). A single selected row still has its two-column structure; a row mask does not remove that axis.</Prose>
    </Checkpoint>

    <H2>3. Follow the data buffer through a view</H2>
    <Prose>The previous Python lesson separated a name from the object it refers to. NumPy adds another distinction: an array object includes a coordinate-to-storage description, while its values live in a <strong>data buffer</strong>. A <strong>view</strong> is another array object describing positions in the same buffer. A <strong>copy</strong> stores the selected numeric values independently.</Prose>
    <Prose>For a numeric ndarray, a basic slice such as <Code>X[:, 0]</Code> returns a view. It does not copy the selected column out of each row. Writing through that view changes the same values visible through X. Calling <Code>.copy()</Code> makes independent numeric storage. Reading an integer-array or boolean selection is <strong>advanced indexing</strong>, which returns copied data.</Prose>
    <NumpyMemoryLab />
    <PythonExample example={numpyFoundationsExamples.memory}><Prose>The copy and gathered selection were created before the view write, so their middle reading stays 24 until explicitly changed. The view's step between consecutive values is 16 bytes: it passes over the other sensor's 8-byte reading each time. Those byte steps are called <strong>strides</strong>. A view can therefore follow a column without moving the underlying values together.</Prose></PythonExample>
    <Prose><strong>Reading a selection and assigning into the original are different operations.</strong> <Code>{'picked = X[[0, 2], 0]'}</Code> returns copied data. But <Code>{'X[[0, 2], 0] = 7'}</Code> tells NumPy to write directly to those positions of X. Chaining <Code>{'X[[0, 2]][:, 0] = 7'}</Code> instead writes to a temporary gathered array. It leaves X unchanged.</Prose>
    <Prose>When independence matters, request it before modifying the selection. <Code>np.shares_memory(a, b)</Code> checks whether two arrays overlap in memory; equality of values does not answer that question. Also, <Code>other = X</Code> merely adds another name for the same array object—it is not a new view or a copy.</Prose>
    <details className="numpy-deeper"><summary>Deeper: non-contiguous arrays and what copying cannot promise</summary>
      <Prose>The lab shows one contiguous float64 buffer with strides (16, 8). Transposes and slices can expose the same buffer with other strides, including negative strides for a reversed view. Reshape returns a view when the requested organisation permits it, but may need a copy; do not promise that every reshape shares storage. <Code>ravel()</Code> returns a flattened view when possible and otherwise copies; <Code>flatten()</Code> copies.</Prose>
      <Prose>A tiny view may keep a large base allocation alive. Copying that small selection can release the dependency when no other reference needs the original. Numeric storage independence is the promise used here: copying an object-dtype array copies its object references, not recursively every referenced Python object.</Prose>
    </details>

    <H2>4. Broadcast by matching coordinates</H2>
    <Prose>Suppose sensor A reads 2 °C too high and sensor B reads 4 °C too high. Every time needs the same two corrections. A Python solution would loop over times, then subtract the appropriate sensor offset. NumPy can express the whole question as <Code>X - offsets</Code>, where offsets is [2, 4].</Prose>
    <Prose><strong>Broadcasting</strong> specifies how differently shaped operands supply values to an array operation. Picture one output cell, such as time 2, sensor B: take X[2, 1] = 32 and offsets[1] = 4, then compute 28. The offset has no time coordinate, so the same sensor offset is reused at every time. That coordinate rule matters more than picturing a physically copied table.</Prose>
    <ol className="numpy-workflow">
      <li><strong>Align both shapes on the right.</strong> X is (3, 2); offsets is (2,). Its missing leading axis behaves like length 1, so compare (3, 2) with (1, 2).</li>
      <li><strong>Compare corresponding axis lengths.</strong> They must be equal, or one must be 1. Here 2 matches 2, and the singleton time axis supplies an offset for each of three times.</li>
      <li><strong>Determine the output shape.</strong> Matching axes keep their length; a length-one axis uses the other's length. This result is (3, 2).</li>
      <li><strong>Trace one output coordinate.</strong> On a length-one input axis use coordinate 0. On an equal-length axis use the output coordinate. That selects the two values to combine.</li>
    </ol>
    <NumpyBroadcastLab />
    <PythonExample example={numpyFoundationsExamples.broadcast}><Prose>One offset per time is a different question. Shape (3,) aligns against the sensor axis of length 2, so it fails. <Code>time_offsets[:, None]</Code> inserts a new axis: (3, 1) explicitly means three times with one value to reuse across sensors. <Code>None</Code> here adds an axis; it does not represent a missing reading.</Prose></PythonExample>
    <Prose>Successful broadcasting does not establish the intended meaning. Combining a (3, 1) column with a (3,) vector produces (3, 3): all pairs. That is useful for pairwise differences and wrong for three elementwise self-differences. Keep axis labels in mind and assert the expected output shape, especially when both dimensions happen to have the same length.</Prose>
    <Prose>Broadcasting does not require tiling the offsets into repeated input storage. Calculated output and intermediate arrays can still be large: a million rows by a thousand columns contains a billion values. At float64, the values alone need about eight billion bytes. Avoid a giant pairwise operation when your task needs only a small subset.</Prose>
    <Checkpoint prompt="For images shaped (batch, height, width, 3), what shape can three colour-channel offsets have? What about one offset per image?">
      <Prose>Channel offsets can have shape (3,), aligning with the final channel axis. Per-image offsets should have shape (batch, 1, 1, 1). Singleton spatial and channel axes reuse each image's offset at every pixel. A plain (batch,) array aligns with channels, even when batch happens to equal three and the mistake runs without an error.</Prose>
    </Checkpoint>

    <H2>5. Reduce an axis, keep the question</H2>
    <Prose>A <strong>reduction</strong> combines several values into fewer values: a sum, mean, maximum, or an all/any decision. To find each sensor's average temperature, vary the time coordinate while holding the sensor fixed. For A, average 18, 24 and 30 to get 24 °C. For B, average 20, 26 and 32 to get 26 °C.</Prose>
    <Prose>That operation reduces <strong>axis 0</strong>, because axis 0 is the coordinate you vary and combine. The sensor axis survives. Reducing axis 1 combines sensors at each time and leaves three results. Avoid memorising “axis 0 means rows” without asking whether rows are being combined or retained.</Prose>
    <NumpyReductionLab />
    <PythonExample example={numpyFoundationsExamples.reduce}><Prose><Code>keepdims=True</Code> keeps the reduced axis with length one. Per-time means therefore have shape (3, 1) and subtract cleanly from (3, 2). Without keepdims they would be (3,), aligning with the wrong axis. No axis argument combines every value. <Code>argmax</Code> returns a position; <Code>max</Code> returns the maximum value.</Prose></PythonExample>
    <Prose>Because both sensors measure the same quantity in the same unit, the per-time mean is meaningful here. A table containing temperature in one column and humidity in another would still accept <Code>mean(axis=1)</Code>, but the result would not be a meaningful physical average. Correct shapes and sensible quantities are both necessary.</Prose>
    <Prose>The same grouping rule works for booleans. <Code>np.isfinite(X)</Code> gives one True/False value per cell. <Code>.all(axis=1)</Code> combines the sensor checks at each time, leaving one row decision. A row is kept only when every sensor reading in that row is finite. This connects cell validity to complete observations in the final workflow.</Prose>
    <H3>Write the coordinates before compressing the code</H3>
    <Prose>Use ordinary Python loops to implement this small calibration and column-mean contract. The subtraction holds the sensor coordinate fixed when choosing its offset. The mean holds that same sensor fixed while visiting every time. The two NumPy expressions below express those same choices; their results are compared on the same inputs.</Prose>
    <PythonExample example={numpyFoundationsExamples.loopReference}><Prose>The loop reference accepts a nonempty rectangular table with one finite offset per sensor. It explains this operation, rather than implementing NumPy's general broadcasting or storage engine. The reusable array function checks that contract, creates a separate calibrated result and performs the numeric loops inside NumPy. For T times and S sensors, both routes do O(TS) arithmetic and store O(TS) corrected values plus O(S) means. Returning every corrected cell already requires visiting TS values. The library route removes Python's per-cell loop overhead; it does not change that growth rate or promise a speedup for every tiny input. Very large finite magnitudes can still overflow, and different summation orders can round differently; use the numerical checks in section 8 for those questions.</Prose></PythonExample>
    <details className="numpy-deeper"><summary>Deeper: variability, empty groups and more axes</summary>
      <Prose><Code>var</Code> averages squared deviations and <Code>std</Code> takes their square root. Their divisor is N − ddof, with N contributing observations; NumPy defaults to ddof=0. Under the usual independent, identically distributed sampling assumptions, ddof=1 gives an unbiased estimator of population variance, but not a universally unbiased standard-deviation estimator. The divisor must be positive.</Prose>
      <Prose>An empty mean has no ordinary finite answer; check that accepted observations exist. Boolean all on an empty collection is True and any is False, reflecting their logical definitions; neither demonstrates that data was observed. For shape (run, time, sensor), reducing axis 1 keeps run and sensor. A tuple axis=(0, 1) combines both runs and times.</Prose>
    </details>

    <H2>6. Change the shape deliberately</H2>
    <Prose>Reshape and transpose can produce the same shape while giving cells different meanings. Default C-order <strong>reshape</strong> reads the logical row-by-row sequence and regroups it; the total number of elements must stay equal. <strong>Transpose</strong> reorders axes. Here X.T turns one row per time into one row per sensor. Compare both maps below; the earlier selection explorer's final two choices let you trace additional cells.</Prose>
    <NumpyShapeComparison />
    <PythonExample example={numpyFoundationsExamples.shapes}><Prose><Code>v.T</Code> leaves a one-dimensional array one-dimensional: there are no two axes to swap. <Code>v[:, None]</Code> adds the second axis. <Code>concatenate</Code> extends the time axis with an observation. <Code>stack</Code> creates a new run axis, so two (3, 2) runs become (2, 3, 2). Repeating X illustrates shape; it does not create independent scientific observations.</Prose></PythonExample>
    <LessonTable caption="Create the array your question needs" headers={['Need', 'Operation', 'Contract to check']} rows={[
      ['Convert known numbers', 'array / asarray', 'array normally copies; asarray may reuse suitable input. Use copy() for explicit independence.'],
      ['Known initial values', 'zeros / ones / full', 'Specify shape and dtype deliberately.'],
      ['Integer positions', 'arange(start, stop, step)', 'Stop is excluded; floating-point steps can give surprising endpoints.'],
      ['A fixed count of samples', 'linspace(start, stop, num)', 'Endpoint included by default; num is a count.'],
      ['A buffer to fill completely', 'empty(shape, dtype=...)', 'Values are unspecified until written, not measurements, zeros or random samples.'],
      ['Remove a known singleton', 'squeeze(axis=...)', 'Naming the axis catches unexpected length rather than removing every singleton.'],
    ]} />
    <Prose>Shape (0, 2) is an empty two-axis table; shape () has zero axes and one scalar value. With more than two axes, <Code>.T</Code> reverses their order. For a particular change, name it with <Code>transpose</Code> or <Code>moveaxis</Code> rather than assuming a batch axis stays fixed.</Prose>

    <H2>7. Separate elementwise arithmetic from weighted sums</H2>
    <Prose>Suppose sensor B should contribute three times as much as sensor A to a combined reading. Choose weights [0.25, 0.75], which sum to one. First multiply readings by their weights; then add the contributions at each time. At time 0: 18 × 0.25 + 20 × 0.75 = 4.5 + 15 = 19.5 °C.</Prose>
    <LessonTable caption="Two steps for the first weighted reading" headers={['Sensor', 'Reading × weight', 'Contribution']} rows={[
      ['A', '18 °C × 0.25', '4.5 °C'], ['B', '20 °C × 0.75', '15.0 °C'], ['Combine', '4.5 + 15.0', '19.5 °C at time 0'],
    ]} />
    <PythonExample example={numpyFoundationsExamples.weighted}><Prose><Code>X * w</Code> multiplies elementwise with broadcasting, preserving a contribution for every cell. For two-dimensional X and one-dimensional w, <Code>X @ w</Code> multiplies matching sensor positions and sums them, leaving one result per row. The @ operator is matrix multiplication; its inner dimensions must agree. You can verify this case without prior matrix algebra by checking each weighted sum.</Prose></PythonExample>
    <Prose>The weights illustrate computation only; real weights need a justified measurement or decision model. The assertion checks shape separately before checking the agreement between two formulations. NumPy does not know whether sensor B deserves greater influence.</Prose>
    <Prose><strong>Vectorisation</strong> expresses work as array operations, such as calibration or weighted sums. For ordinary numeric operations, NumPy executes repeated work in compiled code. Loops still exist underneath. Fewer Python lines do not prove lower memory use or faster execution. <Code>np.vectorize</Code> wraps repeated Python calls for convenience; it does not automatically compile your function into a fast numerical kernel.</Prose>

    <H2>8. Check numbers as carefully as shapes</H2>
    <Prose>Unlike ordinary Python integers, fixed-width NumPy integers have a bounded range. Signed int8 holds −128 through 127. Adding 120 to 120 in an int8 array overflows. Converting afterwards cannot recover the intended answer; convert to a sufficiently wide type before the arithmetic.</Prose>
    <PythonExample example={numpyFoundationsExamples.numerical}><Prose><Code>/=</Code> tries to write back into the original array. Integer storage cannot accept the floating division result under the default casting rule, so it raises instead of silently changing dtype. Ordinary <Code>/</Code> produces a new floating result. Even dtype conversion can lose information: converting fractional values to integers discards their fractional parts.</Prose></PythonExample>
    <Prose>Floating-point stores a finite binary approximation to many decimal values. Tiny discrepancies can therefore be expected. Near zero, an absolute tolerance states an acceptable error. Away from zero, relative tolerance allows error proportional to a reference value. A tolerance for a tiny float64 calculation need not suit a long float32 computation or noisy physical measurements.</Prose>
    <Prose><Code>array_equal</Code> checks exact shape and value equality. <Code>allclose</Code> checks approximate values but may broadcast its arguments; check shapes explicitly when a wrong shape must fail. <Code>testing.assert_allclose</Code> provides a failing assertion for tests. Agreement with a reference does not establish that units, calibrations or scientific assumptions were valid.</Prose>
    <Prose><Code>NaN</Code> represents a missing or invalid floating value here; infinity is also non-finite. Use <Code>isnan</Code> to identify NaN and <Code>isfinite</Code> to reject both NaN and infinity. NaN does not equal itself. Ordinary means propagate NaN; <Code>nanmean</Code> ignores NaNs but does not choose a missing-data policy, discard infinity, or solve an all-missing group.</Prose>
    <details className="numpy-deeper"><summary>Deeper: division masks and domain errors</summary>
      <PythonExample example={numpyReferenceExamples.numpyMissing}><Prose>The initialised output supplies NaN where division is excluded. In <Code>np.divide(..., out=out, where=valid)</Code>, where controls the operation's selected positions. <Code>np.where(valid, numerator / denominator, np.nan)</Code> does not prevent invalid division: Python calculates numerator / denominator before where selects results.</Prose></PythonExample>
      <Prose>Give excluded positions a defined output; an uninitialised buffer would leave their values unspecified. <Code>errstate</Code> can turn selected warnings into exceptions. Log needs positive real inputs for finite real results, and real square root needs nonnegative inputs. Clipping or suppressing warnings changes what you see; it does not justify the mathematical or scientific choice.</Prose>
    </details>

    <H2>9. Build a small sensor report</H2>
    <Prose>A fourth observation arrives, and one earlier reading is missing. Preserve the raw table, reject incomplete rows under an explicit policy, subtract independently known sensor offsets, and identify times whose corrected mean exceeds 23 °C. Keep original time positions so filtering does not relabel time 2 as time 1.</Prose>
    <ol className="numpy-workflow">
      <li><strong>Validate shape:</strong> rows are times; columns match the two offsets.</li>
      <li><strong>Build a row decision:</strong> finite checks give (4, 2) booleans; all over axis 1 gives [True, False, True, True]. Rejecting incomplete rows is this teaching example's policy, not a universal recommendation.</li>
      <li><strong>Preserve raw data and labels:</strong> keep times [0, 2, 3] and make a separate clean array.</li>
      <li><strong>Calibrate:</strong> broadcast offsets of shape (2,) across the (3, 2) clean table.</li>
      <li><strong>Summarise:</strong> reduce axis 0 for sensor means and axis 1 for time means; apply the threshold to time means.</li>
    </ol>
    <Checkpoint prompt="Predict the shapes of valid, clean, per_sensor and per_time. Which original time should be reported as warm?">
      <Prose>The shapes are (4,), (3, 2), (2,) and (3,). The corrected rows are [16, 16], [28, 28], [20, 20]. Only the middle retained row exceeds 23 °C, and its original time is 2.</Prose>
    </Checkpoint>
    <PythonExample example={numpyFoundationsExamples.workflow}><Prose>Each sensor's mean is (16 + 28 + 20) / 3 = 64/3 ≈ 21.333 °C. Only printing is rounded. The copy after a boolean selection is redundant for these numeric data because advanced indexing already copied, but makes independence explicit if the selection changes later.</Prose></PythonExample>
    <Prose>The checks address distinct errors: shape catches an unintended broadcast, equal_nan allows unchanged missing entries, expected means check arithmetic, and the original index checks label alignment. The report describes accepted observations; it does not prove that rejected observations were unimportant or calibrations physically correct.</Prose>
    <Checkpoint prompt="A valid row [26, 28] arrives at time 4. Predict the new per-sensor means and warm times without running code.">
      <Prose>Its corrected values are [24, 24]. Each sensor mean becomes (16 + 28 + 20 + 24) / 4 = 22 °C. Warm times are [2, 4]. Accepted rows retain their original labels.</Prose>
    </Checkpoint>

    <H2>10. Investigate a new dataset</H2>
    <Prose><strong>Your task:</strong> three electricity meters report daily usage in kWh. Calibration uses multiplication rather than subtraction. Start with these complete inputs and write your report before opening the solution.</Prose>
    <Prose>First adapt the coordinate loops from section 5 to multiply by a per-meter gain, after selecting complete finite days. Then implement the NumPy route and compare corrected cells, daily totals and per-meter means on the same retained rows. A successful comparison must also preserve the original day identities; agreeing numbers in the wrong order are not the same report.</Prose>
    <CodeBlock language="python">{`import numpy as np

raw = np.array([[10, 20, 30], [20, 10, 20],
                [30, np.nan, 10], [40, 20, 10]], dtype=np.float64)
gains = np.array([1.0, 0.5, 2.0])
# Rows: days 0–3. Columns: meters A, B, C. Gains are dimensionless.`}</CodeBlock>
    <Prose>Preserve raw. Keep complete finite days and their original indices. Apply each meter's gain. Calculate total corrected usage per day and report days above 65 kWh. Then calculate each meter's mean with a retained singleton axis and subtract it from that meter's accepted readings.</Prose>
    <Prose><strong>Success criteria:</strong> explain each intermediate shape; produce totals [80, 65, 70] for accepted days; identify original days [0, 3] above the strict threshold; preserve raw; and verify that each meter's deviations average to approximately zero. Explain why summing across meters and averaging across days use different axes.</Prose>
    <details className="numpy-deeper"><summary>Hint: keep one label for each row</summary><Prose>Reduce a cellwise finite mask across the meter axis. Apply the same row mask to the table and <Code>np.arange(raw.shape[0])</Code>. Gain shape (3,) aligns with meters. Daily totals reduce axis 1; meter means reduce axis 0. Do not round before checking deviations.</Prose></details>
    <details className="numpy-deeper"><summary>Full runnable solution and reasoning</summary>
      <PythonExample example={numpyFoundationsExamples.transfer}><Prose>After rejecting day 2, the accepted table happens to be square, (3, 3). This is a trap: the wrong axis can still yield a broadcast-compatible shape. Labels tell you that meter means must combine days. Keeping axis 0 at length one gives (1, 3), with values for A, B and C that broadcast across days.</Prose></PythonExample>
      <Prose>The strict comparison excludes 65 exactly. Subtracting each meter's mean makes its accepted deviations sum to zero up to roundoff, a property check as well as an expected-number check. A (3, 1) mean would instead subtract one daily mean across meters and answer another question.</Prose>
    </details>
    <Checkpoint prompt="Someone proposes replacing every NaN with zero before reporting totals. Is that a neutral operation?">
      <Prose>No. It asserts that the missing meter used zero energy. Rejecting the day, imputing from a justified model, or reporting an incomplete total are different policies. State the policy and its limits; an array operation cannot choose it for you.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Diagnose: v = X[:, 0]; v -= 2. The author says raw X is unchanged because v is a different name.">
      <Prose>The slice is a view and in-place subtraction writes through shared storage. Use <Code>v = X[:, 0].copy()</Code> before modifying v if X must remain unchanged. A new name does not mean a new buffer.</Prose>
    </Checkpoint>

    <H2>11. Deeper tools when your task needs them</H2>
    <Prose>The core route is complete. These branches retain useful earlier material and name its additional prerequisites. Open one when you have its question; memorising an API list is not the beginner finish line. The earlier examples use their own complete fixtures, distinct from the continuing sensor dataset.</Prose>
    <details className="numpy-deeper"><summary>Advanced indexing: pairs, cross-products and repeated writes</summary>
      <Prose><Code>{'X[[0, 2], [0, 1]]'}</Code> selects pairs (0, 0) and (2, 1), giving [18, 32] in the original X. For every combination of those rows and columns, use <Code>{'X[np.ix_([0, 2], [0, 1])]'}</Code>, producing [[18, 20], [30, 32]].</Prose>
      <Prose>Repeated advanced indices with <Code>+=</Code> are not a general accumulation operation because a gathered intermediate is updated before writing back. For every repeated contribution to accumulate, use an operation such as <Code>np.add.at</Code> with your intended indices.</Prose>
    </details>
    <NumpyFoundationsDeeper />
    <details className="numpy-deeper"><summary>Performance: measure correct work, then diagnose the cost</summary>
      <Prose>Validate units, shapes, dtype, selected inputs and outputs first. Then time representative data sizes repeatedly, separating setup and I/O from the operation. A view can avoid copying yet have less convenient memory access; a compact broadcast expression can allocate a large temporary. Neither line count nor NumPy usage proves a speedup.</Prose>
      <Prose><Code>nbytes</Code> reports element storage, excluding full Python-object overhead. Inspect strides and flags when layout matters. Array operations such as add and exp are universal functions, or ufuncs. They may accept <Code>out</Code> to reuse storage, but overwrite only values that no later step still needs. Chunk data that will not fit comfortably in memory.</Prose>
      <Prose>Float32 uses less storage than float64, with different range and precision. Accumulation order and numerical libraries can affect roundoff. Document a justified tolerance and hardware/software context when comparing implementations. The browser models make no performance speedup claim.</Prose>
    </details>

    <H2>12. Check your readiness and carry the structure forward</H2>
    <Prose>Without looking back, explain each report axis, trace an output to its source cells, distinguish a view from a copy, repair a per-row broadcast, and predict a reduction's shape. Explain why a successful calculation can still answer the wrong scientific question. If one is difficult, revisit its explorer and change the case.</Prose>
    <Prose>The next topic in the opening curriculum sequence is <a href="/learn/topic/scientific-file-formats-schemas-reliable-data-i-o">Scientific File Formats, Schemas &amp; Reliable Data I/O</a>. You can now organise, transform and check arrays in memory. Next, learn what is preserved or lost when values, types, labels and units cross a file boundary. Follow the reader's named Next link if you are studying a different guided route.</Prose>
    <Sources>
      <li><a href="https://numpy.org/doc/stable/user/absolute_beginners.html">NumPy beginner guide: arrays, axes and creation</a></li>
      <li><a href="https://numpy.org/doc/stable/user/basics.indexing.html">Indexing: basic selections, masks and advanced-index assignment</a></li>
      <li><a href="https://numpy.org/doc/stable/user/basics.copies.html">Copies and views: buffers, strides and reshape limits</a></li>
      <li><a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">Broadcasting: compatibility and memory implications</a></li>
      <li><a href="https://numpy.org/doc/stable/reference/generated/numpy.mean.html">Mean: axes, keepdims and accumulator precision</a></li>
      <li><a href="https://numpy.org/doc/stable/reference/generated/numpy.matmul.html">Matrix multiplication: the contract behind @</a></li>
      <li><a href="https://numpy.org/doc/stable/user/basics.types.html">Data types: storage limits and overflow</a></li>
      <li><a href="https://numpy.org/doc/stable/reference/random/compatibility.html">Random-generator compatibility and reproducibility limits</a></li>
      <li><a href="https://numpy.org/doc/stable/reference/routines.html">Routines by topic: optional deeper API reference</a></li>
    </Sources>
  </div>,
};
