import { Code, CodeBlock, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from "../../components/lesson-labs/PythonExample";
import PlotOutput from "../../components/lesson-labs/PlotOutput";
import { plottingExamples } from "../plotting-examples.js";
import { plottingPracticeExamples } from "../plotting-practice-examples.js";
import { PlotCoordinateLab, PlotHistogramLab, PlotIntervalLab } from "../../components/lesson-labs/PlottingFoundationsLabs";
import { PlotOwnershipDiagram } from "../../components/lesson-labs/PlotOwnershipFigure.jsx";
import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";

export default {
  title: "Matplotlib & Scientific Plotting",
  readTime: "~40 min read + 90 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot">
    <LessonIntro prerequisites="Python functions, NumPy arrays and reductions; the Pandas report is useful but not required."
      sections={[["1-ask-a-question-before-drawing", "Choose a chart"], ["2-build-and-read-a-training-curve", "First figure"], ["5-show-a-distribution-not-just-an-average", "Distributions"], ["6-say-what-your-error-bars-mean", "Uncertainty"], ["9-export-and-test-the-figure", "Export"], ["10-practise-a-complete-report", "Practise"]]}>
      Turn a verified table into evidence someone can read. Build real figures, explore three visual investigations, and diagnose a proposed model with a residual plot. Connect each encoding choice to the question it answers.
    </LessonIntro>
    <H2>1. Ask a question before drawing</H2>
    <Prose>The first example uses a model-learning experiment, but you do not need an ML course to read it. An epoch is one pass through the training data. Loss is a numerical error score: for this score, lower is better. Training loss measures fit to the data used to adjust the model; validation loss uses separate data to help choose a version. Generalisation means performing well on new data. Here these are simply two measured series indexed by training pass.</Prose>
    <Prose>Imagine reviewing an experiment: training loss keeps falling, validation loss stops improving, and latency differs between configurations. A spreadsheet contains every number, but a chart can reveal the trend or variation much faster. First decide what the reader should compare. A plotting library cannot decide whether your join, measurement unit or denominator was correct.</Prose>
    <LessonTable caption="Match the visual encoding to the question" headers={["Question", "Starting method", "Why / limitation"]} rows={[
      ["How does loss change with epoch?", "ax.plot", "Connect ordered observations; a line implies an order."],
      ["Which region has more revenue?", "ax.bar or ax.barh", "Compare lengths from a meaningful zero baseline."],
      ["Do payload size and latency vary together?", "ax.scatter", "Each point is one paired observation; no causal claim follows."],
      ["How spread out are latencies?", "ax.hist, ax.ecdf, ax.boxplot", "Show distribution shape or quantiles rather than only a mean."],
      ["How variable are repeated runs?", "ax.errorbar / ax.fill_between", "Supply and label what the interval represents."],
      ["Where are errors in a two-dimensional table?", "ax.imshow", "Map values to colour with a stated scale and orientation."],
    ]} />
    <Prose>Each example contains its data, runs independently and saves the real SVG shown below it. Examples were checked with Python 3.12.14, NumPy 2.3.5 and Matplotlib 3.11.1. In an environment using that Python version, install the recorded plotting packages with this terminal command. These pins identify a tested snapshot, not every compatible combination.</Prose>
    <CodeBlock language="bash">{`python -m pip install matplotlib==3.11.1 numpy==2.3.5`}</CodeBlock>
    <Prose>Run examples in a scratch folder: savefig overwrites an existing file of the same name. The examples save and close figures so they also work without an interactive window. To experiment locally, put <Code>plt.show()</Code> before close; in a notebook, use an appropriate display backend. A backend is the renderer/display connection, not a different dataset.</Prose>
    <Prose>Charts fit the screen initially. Use Enlarge chart and scroll horizontally to inspect small labels, or open the full-size image. The caption explains the numerical result so the image is not the only way to understand it.</Prose>

    <H2>2. Build and read a training curve</H2>
    <Prose>Figure is the whole canvas; an Axes is one plotting panel; each Axis controls an x or y scale, ticks and labels. Lines, text and legends are Artists in that hierarchy. <Code>fig, ax = plt.subplots()</Code> creates an explicit place to draw. Ax.plot adds lines to that panel; ax.set changes labels or limits. This avoids drawing on whichever pyplot panel happens to be current.</Prose>
    <PlotOwnershipDiagram />
    <Prose>Start with three distinct things: the measurements, the rule mapping them to positions, and the visible marks. A plotting library can change the last two while the first stays identical. Trace that distinction before judging a chart by its shape.</Prose>
    <PlotCoordinateLab />
    <PythonExample example={plottingExamples.curves}><Prose>Argmin returns position 2, so indexing epoch gives epoch 3. Validation loss then rises from 0.61 to 0.66 while training loss falls from 0.48 to 0.39. This is a reason to investigate generalisation, not proof of one cause from four observations.</Prose></PythonExample>
    <PlotOutput example={plottingExamples.curves} alt="Training loss decreases from 1.10 to 0.39. Validation reaches 0.61 at epoch 3, then rises to 0.66.">Squares and a dashed line distinguish validation even without colour. The arrow marks the selected point.</PlotOutput>
    <H3>Read the code as a hierarchy</H3>
    <Prose>Figsize is in inches, not pixels. Constrained layout allocates room for labels at render time. The format strings o- and s-- combine marker and line style. Label supplies legend text; legend() creates the legend. Set_xticks makes the four epochs explicit. Annotate uses data coordinates for both the arrow target and text position here.</Prose>
    <Prose>The shape check makes the one-loss-per-epoch assumption visible. Plot connects points in supplied order; it does not sort x for you. Sort paired x/y values together when chronological order is needed, never each column independently. Missing values can create gaps; filling a missing measurement with zero invents an event that may never have happened.</Prose>
    <Checkpoint prompt="Why not choose epoch 4 because its training loss is smallest?">
      <Prose>Training loss measures fit to optimisation data. Checkpoint selection here uses validation loss, lowest at epoch 3. Repeated validation-based decisions also adapt to validation data; reserve independent test data for final evaluation rather than selection.</Prose>
    </Checkpoint>

    <H2>3. Compare categories without distorting magnitude</H2>
    <PythonExample example={plottingExamples.bars}><Prose>This is the accepted Pandas report: North contributes 10 cents and South 20. These small numbers are teaching data, not a real business estimate. Values are already aggregated; bar does not group or sum transactions.</Prose></PythonExample>
    <PlotOutput example={plottingExamples.bars} alt="Zero-baseline bars show North at 10 cents and South at 20 cents; total 30 cents.">Direct labels make the comparison exact; the y-axis includes zero.</PlotOutput>
    <Prose>Readers compare bar lengths. Starting at 9 would make lengths of 1 and 11 look elevenfold different, although the values differ twofold. A zoomed line or dot plot can reveal small changes if its scale is conspicuous; “every chart must start at zero” is too broad. For long category labels, use barh. For grouped bars, define positions and offsets and label the groups.</Prose>
    <Prose>Check the denominator and exclusions before writing the title. These are accepted paid orders, not every raw row. Revenue does not imply twice as many customers. Explain that in a caption instead of expecting readers to reconstruct the cleaning pipeline.</Prose>

    <H2>4. Preserve pairs in relationship plots</H2>
    <PythonExample example={plottingExamples.scatter}><Prose>The third payload is larger than the second but slightly faster. Scatter preserves that exception. Connecting points adds a path with no measurement meaning unless the observations really are a sequence.</Prose></PythonExample>
    <PlotOutput example={plottingExamples.scatter} alt="Payload-latency pairs: (1,9), (2,13), (3,12), (4,20), (5,25), in MB and ms.">The association is visible; other variables could influence both size and time.</PlotOutput>
    <Prose>Scatter's s specifies marker area in points squared, not radius or data units. Alpha controls transparency. Overlap can hide sample count; smaller markers, transparency, hexbin or a two-dimensional histogram may help large datasets. Colour can encode a measurement with a colourbar, but should not carry every distinction alone.</Prose>

    <H2>5. Show a distribution, not just an average</H2>
    <PlotHistogramLab />
    <PythonExample example={plottingExamples.histogram}><Prose>The bins are [0,3), [3,6) and [6,10], with the last right endpoint included. Therefore 3 belongs to the second bin. Counts are 3, 1 and 2. Unequal widths make counts versus densities especially important.</Prose></PythonExample>
    <PlotOutput example={plottingExamples.histogram} alt="Histogram counts 3,1,2 across widths 3,3,4. Densities approximately 0.1667,0.0556,0.0833 integrate to one.">The first density is 3 / (6 × 3) = 1/6 per millisecond. Its area is one half: three of six observations.</PlotOutput>
    <Prose>Hist returns heights, edges and patches; underscores ignore results we do not need. Density=True normalises area, not the sum of heights. Density can exceed 1 for narrow intervals without being an invalid probability. Counts and density answer different questions, so change the y-axis label too.</Prose>
    <Prose>Too few bins hide structure; too many turn a small sample into noise. Compare plausible widths and retain raw observations where practical. Values outside supplied bin edges are not counted: compare counts.sum with the intended population. Ecdf answers “what fraction is at or below x?” without bins. Boxplot summarises quartiles and whiskers; whiskers are not automatically minimum/maximum or confidence intervals. Violin plots add a density estimate whose shape depends on smoothing.</Prose>
    <Checkpoint prompt="Does adding the density heights give the probability of falling in any bin?">
      <Prose>No. Multiply by each width: (1/6 × 3) + (1/18 × 3) + (1/12 × 4) = 1. Heights have units 1/ms; multiplying by ms gives dimensionless probability mass.</Prose>
    </Checkpoint>

    <H2>6. Say what your error bars mean</H2>
    <PlotIntervalLab />
    <PythonExample example={plottingExamples.uncertainty}><Prose>A's runs are 10, 12, 14: mean 12, sample SD 2. B's are 13, 14, 15: mean 14, SD 1. Ddof=1 divides squared-deviation sums by n−1 for sample variance. Individual points reveal how little data supports the summaries.</Prose></PythonExample>
    <PlotOutput example={plottingExamples.uncertainty} alt="A: run values 10,12,14 and mean 12 ± SD 2. B: 13,14,15 and mean 14 ± SD 1.">The bars show run-to-run spread, not a 95% confidence interval or significance test.</PlotOutput>
    <Prose>Yerr supplies nonnegative distances from each centre. Asymmetric intervals use separate lower and upper distances; absolute endpoints are not error magnitudes. Errorbar draws what you supply, without estimating its statistical meaning. Fill_between(x, lower, upper) shows a band over ordered x values; smooth rendering does not validate interval assumptions.</Prose>
    <LessonTable caption="Intervals support different statements" headers={["Quantity", "Interpretation", "Needed context"]} rows={[
      ["Standard deviation", "Spread of measurements or run summaries", "Observation unit, sample size, ddof."],
      ["Standard error", "Estimated variability of a mean under a sampling model", "Dependence and estimator."],
      ["Confidence interval", "Interval from a repeated-sampling procedure", "Method, level and assumptions."],
      ["Quantile range", "Specified portion of a distribution", "Which quantiles and population."],
    ]} />
    <Prose>If runs are paired by seed or dataset, analyse paired differences when appropriate. Overlap alone is not a general hypothesis test. For construction, see the <a href="/learn/topic/hypothesis-testing-confidence-intervals">confidence-interval lesson</a>; here we communicate an already-defined quantity.</Prose>

    <H2>7. Make matrix orientation and colour explicit</H2>
    <PythonExample example={plottingExamples.heatmap}><Prose>Rows are true class; columns are predicted class. Eight true negatives and nine true positives give 17 correct classifications out of 20. Swapping the axis labels would swap the interpretation of the off-diagonal errors.</Prose></PythonExample>
    <PlotOutput example={plottingExamples.heatmap} alt="Confusion table with true classes in rows and predictions in columns: [[8,2],[1,9]]. Colour range zero to ten.">The counts can be read independently of colour.</PlotOutput>
    <Prose>Imshow maps a two-dimensional scalar array to colour. Origin='upper' puts row zero on top; interpolation='nearest' avoids smoothing discrete counts. Fixed vmin/vmax support comparison across figures. Auto-scaling each table can make different magnitudes look identical. The colourbar ties colours to values and units.</Prose>
    <Prose>Use sequential colours for ordered magnitudes and a diverging map with a meaningful centre for signed departures. Category IDs should not appear continuous. Default cell coordinates are column/row indices; extent can supply physical bounds when those match the geometry. Aspect determines the displayed x/y unit ratio, not the values. Counts and row-normalised proportions answer different questions: label which is shown.</Prose>

    <H2>8. Understand what a scale changes</H2>
    <PythonExample example={plottingExamples.scales}><Prose>Each error is one tenth of the previous value. A log axis spaces equal ratios equally, producing a straight declining line. Only the position mapping changes, not the source values.</Prose></PythonExample>
    <PlotOutput example={plottingExamples.scales} alt="Errors 1,0.1,0.01,0.001 appear curved on a linear axis and equally spaced vertically on a log axis.">Log scaling separates the small values. Read the ticks, not just the slope.</PlotOutput>
    <Prose>Ordinary log scales cannot represent zero or negative values. Do not silently replace them with tiny positives; decide whether they are invalid, censored, or need another scale. Symlog supports signed values with a specified linear region near zero. Limits crop the view rather than filter source data; disclose zooms that exclude observations.</Prose>
    <Prose>Subplots(1, 2) returns an array of two Axes. Squeeze=False keeps a two-dimensional array even for one panel; axes.flat is convenient for iterating a grid. Share axes only for compatible quantities. Separate aligned panels are often clearer than unrelated dual y-axes whose scaling can manufacture visual agreement.</Prose>

    <H2>9. Export and test the figure</H2>
    <Prose>The examples deliberately set dark styling and save the Figure object. Library functions should use local rc_context settings or accept a caller's Axes instead of unexpectedly changing global preferences. Return Figure/Axes so callers choose whether to show, save or close.</Prose>
    <LessonTable caption="Export choices affect the deliverable" headers={["Choice", "Effect", "Check"]} rows={[
      ["SVG / PDF", "Usually vector text and paths; some artists can be rasterised", "Fonts in the actual viewer."],
      ["PNG + dpi", "Inches × dpi gives nominal pixel dimensions", "Text at final display size."],
      ["bbox_inches='tight'", "Crops to content bounds", "Final pixel dimensions can change."],
      ["facecolor / transparent", "Exported background", "Contrast on its destination page."],
      ["plt.close(fig)", "Releases pyplot's figure registration", "Close saved figures in loops."],
    ]} />
    <Prose>Test numbers before pixels: pairing, series count, finite inputs, histogram totals, units, interval definition and limits. Then inspect the saved file for clipping, unreadable labels and legend overlap. Pixel snapshots can change with fonts or renderers; they complement numerical checks. These images were generated from the displayed code, not drawn separately to resemble its output.</Prose>

    <H2>10. Practise a complete report</H2>
    <H3>A chart can help find what the model misses</H3>
    <Prose>A residual is an observed value minus its predicted value. Plotting only observations and predictions can hide small systematic departures on a wide axis. In this invented experiment, a proposed model predicts response = 1 + input. A second panel gives those departures their own scale while retaining the same x positions.</Prose>
    <PythonExample example={plottingPracticeExamples.residual}><Prose>Positive residuals mean underprediction; negative ones mean overprediction. The last observation is two units above the proposed model. A mean residual of 0.4 alone hides where this happens. The pattern suggests checking the model and measurement process; five invented points do not prove a scientific cause or validate a replacement model.</Prose></PythonExample>
    <PlotOutput example={plottingPracticeExamples.residual} alt="Observed values versus a straight-line prediction, with residuals 0, -0.5, 0, 0.5 and 2 in an aligned lower panel.">The zero reference line means exact agreement. Both panels preserve observation pairing.</PlotOutput>
    <Checkpoint prompt="Change revenue to North 10 and South 60 cents. What else must change?">
      <Prose>Total becomes 70; South is six times North. Update the title and y limit, for example to 72. Leaving ylim at 24 clips the bar. Keep zero and the accepted-order policy.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Add latency 10 without changing the histogram edges. Predict the counts and density area.">
      <Prose>The last bin includes 10, giving [3, 1, 3]. Divide each count by 7 times its width; areas still sum to 1. A value of 11 would be excluded by the supplied edges.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Build a two-panel report: training/validation curves and regional revenue. What makes it ready?">
      <Prose>Use separate Axes without sharing y: loss and cents are different units. Include line labels, integer epochs, zero-baseline bars and exclusions. Save before closing, inspect the export and check that plotted arrays match the source examples.</Prose>
    </Checkpoint>
    <H3>Independent investigation: localize a timing discrepancy</H3>
    <Prose>At distances 1, 2, 3 and 4 m, measured times are 2, 4, 7 and 8 ms. A proposed prediction is twice the distance. Before writing code, find the residuals and the distance with the largest absolute discrepancy. Build aligned measured/predicted and residual panels, label units, include a zero residual line, and export a readable figure. Keep distance/time pairs together.</Prose>
    <details><summary>Hint: share x, not the meaning of y</summary><Prose>The top panel contains actual times; the lower contains their differences from prediction. Both use milliseconds but answer different questions. Compute measured − predicted once, then plot that array; absolute values are only needed to find the largest discrepancy.</Prose></details>
    <details><summary>Worked solution and expected result</summary><PythonExample example={plottingPracticeExamples.transfer}/><PlotOutput example={plottingPracticeExamples.transfer} alt="A distance/time report with a single positive residual of 1 ms at distance 3 m.">Investigate the discrepancy at 3 m instead of silently deleting that observation.</PlotOutput></details>
    <Checkpoint prompt="Change the final measured time from 8 to 10 ms. What should change in the chart and conclusion?">
      <Prose>Residuals become 0, 0, 1, 2; the largest is now at 4 m. Recompute the array and annotation, inspect the limits, and explain the new pattern. Changing only the caption disconnects the story from the marks.</Prose>
    </Checkpoint>
    <Prose>This covers a core plotting workflow; animation, maps, 3D and statistical model checking have deeper owners. Next in this module, <a href="/learn/topic/reproducible-notebooks-experiment-structure">Reproducible Notebooks &amp; Experiment Structure</a> makes the inputs, execution state and environment behind a figure inspectable and rerunnable. Documentation and Git follow with public contracts and change history. The reader's named Next link follows the selected topics on your route.</Prose>
    <Sources alternatives={<LearningResources>
      <li><a href="https://www.youtube.com/watch?v=6gdNUDs6QPc">Benjamin Root &amp; Hannah Aizenman — Anatomy of Matplotlib, SciPy 2018</a> · A beginner video workshop on plot types, vocabulary and the object hierarchy. Follow the <a href="https://github.com/matplotlib/AnatomyOfMatplotlib">companion notebooks</a> after the coordinate investigation. The recording is older; check current APIs and layout options against the reference links.</li>
      <li><a href="https://matplotlib.org/stable/users/explain/quick_start.html">Matplotlib quick-start article</a> · A written guide to Figure/Axes/Artist ownership and common plots; useful when translating an intended picture into code.</li>
    </LearningResources>}>
      <li><a href="https://matplotlib.org/stable/users/explain/quick_start.html">Figure, Axes and explicit plotting</a></li>
      <li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html">Histogram bins and density</a></li>
      <li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html">Error-bar distances and shapes</a></li>
      <li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html">Images and coordinates</a></li>
      <li><a href="https://matplotlib.org/stable/users/explain/colors/colormaps.html">Choosing colour scales</a></li>
      <li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html">Saving a Figure</a></li>
      <li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html">Scatter marker sizes and colour arguments</a></li>
      <li><a href="https://matplotlib.org/stable/users/explain/axes/axes_scales.html">Linear, logarithmic and symmetric-log scales</a></li>
      <li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html">Subplot grids and squeeze behaviour</a></li>
    </Sources>
  </div>,
};
