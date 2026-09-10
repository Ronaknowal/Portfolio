import { CodeBlock, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from "../../components/lesson-labs/PythonExample";
import { NotebookKernelLab, RandomStreamLab, ProvenanceLab } from "../../components/lesson-labs/NotebookLabs.jsx";


import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { notebookExamples } from "../notebook-examples.js";
import { notebookPracticeExamples } from "../notebook-practice-examples.js";
import { NotebookStatePicture } from "../../components/lesson-labs/NotebookFigures.jsx";

export default {
  title: "Reproducible Notebooks & Experiment Structure",
  readTime: "~38 min read + 90 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot">
    <LessonIntro prerequisites="Python functions, files and environments; NumPy for the randomness and preprocessing examples."
      sections={[["1-define-what-must-be-repeatable", "The goal"], ["2-expose-hidden-notebook-state", "Kernel state"], ["4-separate-inputs-code-and-results", "Project layout"], ["6-randomness-is-state-too", "Randomness"], ["8-record-a-run-you-can-replay", "Run record"], ["9-execute-a-real-notebook-from-a-clean-kernel", "Notebook project"], ["11-practise-a-handoff", "Practise"]]}>
      Make an experiment another person can rerun without inheriting your open notebook. Follow the path from hidden state to explicit inputs, checked transformations, recorded results and a fresh-kernel execution.
    </LessonIntro>
    <H2>1. Define what must be repeatable</H2>
    <Prose>You send a colleague a chart showing an adjusted mean of 18 ms. They rerun the notebook and get 28 ms—or a NameError. The chart is not enough to tell which data, offset, cell order or package environment produced 18. Reproducibility means making those dependencies visible and checking the outcome against a stated target.</Prose>
    <LessonTable caption="Choose the agreement you need" headers={["Target", "Example", "Evidence"]} rows={[
      ["Exact values or bytes", "An integer count; a content checksum", "Exact comparison under a specified contract."],
      ["Numerical agreement", "Floating-point model predictions", "A justified absolute/relative tolerance; shapes and labels checked too."],
      ["Comparable conclusions", "Variation across repeated training runs", "A defined protocol and uncertainty, not one lucky seed."],
    ]} />
    <Prose>These targets are not interchangeable. Two CSV files can contain equivalent tables but different line endings and therefore different hashes. Two GPU runs can differ numerically while supporting the same conclusion. State which result should agree and how you will decide, instead of promising universal bit-for-bit identity.</Prose>

    <H2>2. Expose hidden notebook state</H2>
    <Prose>A <strong>cell</strong> is one editable block: code executes, while Markdown explains the question, choices and result. The frontend is the editor you see. It sends code to a <strong>kernel</strong>, a running Python process, and displays the returned output. Saving the document stores the source and selected outputs; it does not save every live object in that process. Those three places can disagree.</Prose>
    <Prose>A notebook file stores ordered cells and possibly saved outputs. The kernel is the running process holding Python objects. Editing a cell changes the document; it does not run the code. Running one cell updates only the expressions executed there. Previously computed variables do not automatically become formulas that recalculate when an input changes.</Prose>
    <NotebookStatePicture />
    <NotebookKernelLab />
    <Prose>Follow the dependency direction: input readings and offset → adjusted values and mean → displayed result. To update the final answer after changing an input, execute the affected downstream calculations. A cell's position on the screen does not make that dependency execute automatically. A short notebook with this chain is easier to inspect than a long notebook whose dependencies are spread across hidden global variables.</Prose>
    <Prose>Now verify the same mechanism in Python with a second small calculation: rate × 10 gives cost. Here rate plays the role of the changed input, and cost plays the role of the previously calculated mean. The arithmetic changes; the source → stored value → display dependency is the same.</Prose>
    <PythonExample example={notebookExamples.state}><Prose>The second 20 is stale: rate changed to 3 but cost still stores 20. A fresh namespace cannot display cost at all. Executing the complete dependency chain with rate 3 produces 30. Exec is used only to model these fixed cells; it must not be used to execute untrusted user strings.</Prose></PythonExample>
    <Prose>Execution counts are useful clues about order, not proof that saved results match current source. “Run all” in a dirty kernel can still see variables from deleted cells. Restart and run all clears that hidden memory first. It does not undo written files, restore a changed database, refresh a cached download or recreate your package environment.</Prose>
    <Checkpoint prompt="You edit offset from 2 to 5 but rerun only a cell that prints a previously computed mean. What should you expect?">
      <Prose>The old value remains until the calculation is rerun with the new input. Restart and execute the full visible chain; for [10,20,30], the adjusted mean should become 15 rather than 18.</Prose>
    </Checkpoint>

    <H2>3. Move calculations into explicit functions</H2>
    <PythonExample example={notebookExamples.pure}><Prose>The offset is an explicit keyword-only argument. The function builds a new list rather than changing raw. Running the same call twice returns the same small result dictionary. Repeating a plotting cell or file write has different side effects; this example isolates computation from those effects.</Prose></PythonExample>
    <Prose>A pure calculation depends on its arguments and returns a value without changing outside state. This makes it easy to test, reuse in a script, and call from a notebook. Purity is a useful boundary, not a requirement that the whole application avoid IO. Keep reading, computing and writing separate so an accidental rerun does not append duplicate results or silently overwrite evidence.</Prose>
    <Prose>Document the input contract. This tiny function assumes finite numeric measurements in a common unit; it only demonstrates an empty-input check. Real input needs schema validation, including missing values and units. The <a href="/learn/topic/code-documentation-type-hints-api-design">next lesson</a> develops that contract and its validation explicitly.</Prose>

    <H2>4. Separate inputs, code and results</H2>
    <CodeBlock language="text">{`experiment/
  README.md                 # exact setup and rerun command
  requirements.txt          # recorded package environment
  data/raw/                 # original inputs or retrieval manifest
  data/processed/           # derived tables, reproducible from raw
  analysis.py               # reusable transformations
  run.py                    # explicit command-line entry point
  notebooks/01_analysis.ipynb
  tests/                    # contracts and small known examples
  runs/run-001/
    config.json
    manifest.json
    metrics.json
    figures/`}</CodeBlock>
    <Prose>This is a proposed layout, not files the next snippet requires you to create. Raw data is treated as read-only; processed data is derived. A notebook explains a run and imports reusable functions instead of becoming the only copy of the algorithm. Use an explicit run identifier and refuse collisions, or document an intentional replace policy. “latest.csv” alone hides which experiment a chart belongs to.</Prose>
    <H3>Paths are inputs, not guesses</H3>
    <Prose>A relative path is resolved against the process working directory, not necessarily the notebook's directory. Path.cwd() helps inspect that context. A normal script can anchor resources with Path(__file__).resolve().parent; notebook cells do not normally define __file__. For automation, choose a project root or data path explicitly and check it exists. Avoid silently changing directories in a late cell just to make a read succeed.</Prose>
    <Prose>Do not overwrite original data to “clean it once.” Save the cleaning logic and a derived output. For private or very large inputs, record access instructions, schema, immutable version and checksum rather than checking the data into source control. A checksum identifies bytes; it neither grants access nor proves the data is truthful.</Prose>

    <H2>5. Identify the actual execution environment</H2>
    <Prose>The Python selected in a terminal can differ from the kernel selected in a notebook. Check sys.executable and sys.version inside the notebook, and read package versions there. Install into the intended environment with that interpreter's -m pip rather than assuming a bare pip command targets it.</Prose>
    <CodeBlock language="bash">{`python -m pip install numpy==2.3.5 nbconvert==7.17.1 nbformat==5.11.1 ipykernel==7.3.0
python -m ipykernel install --sys-prefix --name python3 --display-name "Experiment Python"`}</CodeBlock>
    <Prose>These commands are for a dedicated Python 3.12 environment; registration sets that environment's python3 kernel entry. Select the matching kernel in your notebook frontend. Examples were verified on Python 3.12.14 with these package versions. The notebook frontend itself can be installed separately; the final project below needs an execution kernel, not a browser UI.</Prose>
    <Prose>A pip freeze snapshot can record installed packages, but it is not a portable, fully specified environment by itself. Also record Python, platform and relevant native libraries; distinguish direct dependencies from transitive resolutions. GPU work adds drivers, accelerator libraries and hardware. A lockfile or container narrows uncertainty but does not capture an external dataset or every hardware-dependent operation.</Prose>

    <H2>6. Randomness is state too</H2>
    <Prose>A pseudorandom generator stores a state and advances it as you request values. A <strong>seed</strong> initializes that state; it is not attached separately to every future result. If two parts of an experiment share a generator, a new request in one part can change which values the other receives. First inspect that consumption mechanism without any probability calculation.</Prose>
    <RandomStreamLab />
    <PythonExample example={notebookExamples.random}><Prose>Constructing two generators from seed 7 repeats the same draws in this checked environment. Reusing one generator advances its state, so successive arrays differ in this example. Child streams keep an extra data-splitting draw from consuming the model's random sequence.</Prose></PythonExample>
    <Prose>The equality of random samples in general is not logically impossible: chance can produce repeats. Here we verified the particular five-integer draws. The important rule is state advancement, not a promise that every pair of draws must be different.</Prose>
    <Prose>Put generator construction at a deliberate run boundary. Reseeding every batch can repeat samples accidentally; using one global generator makes unrelated cells affect future draws. Record how streams are allocated as well as the root seed. Standard-library random, NumPy and model frameworks have distinct random-state systems.</Prose>
    <Prose>A seed alone does not guarantee identical results across library versions, algorithms, hardware, thread scheduling or nondeterministic kernels. NumPy's generator compatibility has environment and call-sequence qualifications. Record relevant versions and use exact saved split IDs when membership is important. If an algorithm is stochastic, evaluate multiple runs under a fixed protocol rather than selecting a favourable seed.</Prose>

    <H2>7. Preserve the experimental boundary</H2>
    <PythonExample example={notebookExamples.leakage}><Prose>Training measurements have mean 20. The validation value 100 becomes 80 when transformed using that training mean. Including validation in the fit changes the mean to 40 and the transformed value to 60. The second pipeline can be perfectly repeatable and still be an invalid evaluation.</Prose></PythonExample>
    <Prose>Split before learning preprocessing statistics. Fit imputation, scaling and feature selection on training data, then reuse the fitted transformation. Record row identifiers and split rules. Time forecasting needs chronological boundaries; repeated users or patients may need group separation. A random row split is not universally appropriate.</Prose>
    <Prose>Keep a record of hypotheses and evaluation choices before inspecting the final test set. Track failed runs as failures, not missing rows in a summary that only includes successes. Reproducibility lets someone repeat your procedure; validity asks whether that procedure answers the scientific question. You need both.</Prose>

    <H2>8. Record a run you can replay</H2>
    <Prose>A <strong>manifest</strong> is a small record connecting an output to its inputs and decisions. <strong>Provenance</strong> means the account of where that output came from. When a later run asks to reuse a cached result, compare the inputs that actually determine the calculation rather than assuming a reused filename means unchanged data.</Prose>
    <ProvenanceLab />
    <PythonExample example={notebookExamples.manifest}><Prose>The manifest connects exact input bytes, configuration, environment and result. Reloading its config reconstructs the calculation and returns count 3 / mean 18. The temporary directory is deleted after the example; a real run should save to a persistent, uniquely named run folder.</Prose></PythonExample>
    <Prose>The seed is recorded for consistency with a broader experiment schema, but this particular mean calculation uses no randomness. A frozen dataclass prevents field reassignment, not recursive mutation of nested lists. Here the configuration contains only scalar values. Its field names make choices explicit; they do not validate arbitrary input JSON automatically.</Prose>
    <LessonTable caption="A run record should answer these questions" headers={["Question", "Record", "Limit"]} rows={[
      ["Which data?", "Version, checksum, schema, split IDs", "A local path alone may later point to different bytes."],
      ["Which code?", "Commit plus any uncommitted patch, or archived source", "The example's lesson-example-v1 label is illustrative, not a Git hash."],
      ["Which settings?", "Serialised configuration and seed/stream policy", "Defaults can change between releases."],
      ["Which environment?", "Interpreter, dependencies, platform, relevant hardware", "A single package version is not the whole environment."],
      ["What happened?", "Metrics, output paths, warnings, status and failure details", "Do not label a partially completed run successful."],
    ]} />
    <Prose>The helper reads trusted inline numeric JSON, not arbitrary uploaded data; a production entry point needs validation. A data hash detects byte changes, not whether changed data gives the same mean. Cache keys must include all inputs that affect a result—data, config, code and relevant environment—not just the filename.</Prose>

    <H2>9. Execute a real notebook from a clean kernel</H2>
    <Prose>This project creates an actual notebook and executes its three code cells through nbconvert. Run it in a new scratch directory with the python3 kernel configured above. It refuses to replace either named notebook if it already exists. The source notebook is saved before execution; a separate file records executed outputs.</Prose>
    <PythonExample example={notebookExamples.execute}><Prose>The fresh kernel defines inputs, calculates adjusted values, then checks and prints 18.0. Execution counts [1,2,3] accompany this verified run. The meaningful acceptance criteria are the assertions and successful execution, not the appearance of sequential counts alone.</Prose></PythonExample>
    <Prose>Download the generated <a href="/learn-assets/notebooks/offset-analysis.ipynb" download>source notebook</a> or <a href="/learn-assets/notebooks/offset-analysis.executed.ipynb" download>executed notebook</a>. Inspect code before running any downloaded notebook: notebook execution has the same access to files and services as ordinary Python.</Prose>
    <Prose>Timeout is a per-cell bound in seconds. The execution path is explicit. Allow_errors=False stops on an unexpected failing cell instead of creating an apparently successful analysis containing errors. Deliberate teaching failures can be caught and explained inside a cell; do not suppress every error just to reach the final plot.</Prose>
    <CodeBlock language="bash">{`jupyter nbconvert --to notebook --execute offset-analysis.ipynb --output offset-analysis.checked.ipynb --ExecutePreprocessor.timeout=60`}</CodeBlock>
    <Prose>This is the command-line equivalent for an existing trusted source notebook. Use a fresh output name if preserving prior evidence. Headless execution is especially valuable in continuous integration because it tests a clean process rather than your interactive session. It does not automatically check network inputs, scientific assumptions or outputs without assertions.</Prose>

    <H2>10. Review the handoff, not just the final cell</H2>
    <LessonTable caption="Ready-to-share checks" headers={["Check", "Concrete action", "Failure to catch"]} rows={[
      ["Fresh execution", "Restart and run all; run automation from the declared directory", "Hidden variables or wrong paths."],
      ["Inputs and environment", "Follow the README in a new environment", "Undocumented packages, unavailable data or wrong kernel."],
      ["Assertions", "Check counts, schema, totals and selected known outputs", "A pipeline that runs but silently changes meaning."],
      ["Side effects", "Separate raw/derived files and protect output collisions", "Duplicate appends or overwritten evidence."],
      ["Privacy", "Inspect source, outputs, tracebacks and metadata", "Credentials, private paths or sensitive sample records."],
      ["Narrative", "Explain question, choices, evidence and limitations", "A sequence of unexplained code cells."],
    ]} />
    <Prose>Large outputs can slow notebooks and obscure meaningful changes. Keep compact representative outputs with useful captions and link large artifacts. Clearing an output may reduce noise but removes visible evidence; preserve an executed artifact separately where appropriate. If a secret was exposed, deleting the visible cell is not enough—rotate it and address stored history.</Prose>

    <H2>11. Practise a handoff</H2>
    <H3>Independent investigation: identical answer, different evidence</H3>
    <Prose>Create a replay package for the trusted [10,20,30] millisecond fixture with offset 5. It must preserve the raw bytes, the actual source bytes and a JSON manifest containing their hashes, configuration, Python version, result and completion status. Recompute the identity from the saved ingredients, then rerun the calculation. Expected result: count 3 and mean_ms 15.0.</Prose>
    <Prose>Now replace the input by [9,20,31]. Show that the answer remains 15 but the data identity changes. A correct solution must not claim those are the same run merely because their result dictionaries compare equal. Explain what you would add if the calculation depended on a package, operating-system library or external service.</Prose>
    <details><summary>Hint: preserve evidence separately from its digest</summary><Prose>Hash exact bytes with sha256; keep the bytes too, because a digest cannot reconstruct them. Encode configuration with a stable key order. Read your script through __file__ in this standalone program. A manifest should describe a completed result only after calculation succeeds.</Prose></details>
    <details><summary>One complete replay package and changed-data check</summary><PythonExample example={notebookPracticeExamples.notebookTransfer}/><Prose>The temporary folder makes this exercise rerunnable; use a uniquely named persistent run directory when retaining evidence. The code stores actual source bytes, replacing the earlier illustrative revision label. It recalculates with the current function; it does not execute arbitrary archived source or prove that an untrusted manifest is truthful. Only Python is recorded because this calculation uses the standard library; broader dependencies need a broader environment record.</Prose></details>
    <Checkpoint prompt="Modify the generated notebook to use offset 5. Which values and checks should change?">
      <Prose>The adjusted values become [5,15,25], and the mean is 15.0. Update the assertion from 18 to 15. Use a new folder or explicit new filenames, then execute the whole notebook in a new kernel. Do not merely edit the saved output text.</Prose>
    </Checkpoint>
    <Checkpoint prompt="The raw data changes from [10,20,30] to [9,20,31]. What happens to the manifest comparison?">
      <Prose>The exact bytes and checksum change but the mean remains 20 before the offset, hence 18 after it. Equal results do not prove equal inputs. Record the new data version rather than reusing the old manifest as if nothing changed.</Prose>
    </Checkpoint>
    <Checkpoint prompt="A colleague can run every cell but gets a different random split. What should you investigate first?">
      <Prose>Check the actual kernel/interpreter, package versions, seed and generator, the call order, and the ordering of input rows before splitting. For a fixed benchmark, compare saved split IDs rather than only regenerating from a seed. Then inspect hardware or nondeterministic framework behaviour if relevant.</Prose>
    </Checkpoint>
    <Prose>You now have a small reproducible analysis, not a full experiment-tracking platform. Distributed orchestration, remote storage, dataset governance and GPU determinism need additional design. Next, <a href="/learn/topic/code-documentation-type-hints-api-design">document and type the boundaries</a> so another person can use the reusable parts correctly.</Prose>
    <Sources alternatives={<LearningResources>
      <li><a href="https://swcarpentry.github.io/python-novice-gapminder/01-run-quit.html">Software Carpentry: running Python in a notebook</a> — beginner walkthrough of cells, execution and the interface; use it if the kernel/document distinction is new.</li>
      <li><a href="https://nbconvert.readthedocs.io/en/latest/execute_api.html">nbconvert: execute notebooks programmatically</a> — hands-on follow-on for clean execution, explicit directories, errors and saved artifacts. Requires the local environment introduced in this lesson.</li>
      <li><a href="https://pyvideo.org/jupytercon-2023/five-guiding-principles-to-make-jupyter-notebooks-educational-and-reusable.html">Julia Wagemann, JupyterCon 2023: educational and reusable notebooks</a> — conference video page connecting modular code and computational narrative through Earth-observation teaching. Useful after completing your first notebook; its suggested text/code ratio is context-specific, not a requirement for your analysis.</li>
    </LearningResources>}>
      <li><a href="https://nbformat.readthedocs.io/en/latest/format_description.html">Notebook format: cells, source, execution counts and stored outputs</a></li>
      <li><a href="https://jupyterlab.readthedocs.io/en/stable/user/commands.html">JupyterLab kernel and execution commands</a></li>
      <li><a href="https://nbconvert.readthedocs.io/en/latest/execute_api.html">nbconvert: execution, working directory, timeouts and errors</a></li>
      <li><a href="https://numpy.org/doc/stable/reference/random/compatibility.html">NumPy random-stream compatibility qualifications</a></li>
      <li><a href="https://numpy.org/doc/stable/reference/random/parallel.html">NumPy: spawned random streams</a></li>
      <li><a href="https://ipython.readthedocs.io/en/stable/install/kernel_install.html">Registering a kernel for the intended environment</a></li>
      <li><a href="https://docs.python.org/3.12/library/pathlib.html">Python: explicit filesystem paths</a></li>
      <li><a href="https://docs.python.org/3.12/library/dataclasses.html">Python: configuration dataclasses and frozen fields</a></li>
    </Sources>
  </div>,
};
