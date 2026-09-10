import { Code, CodeBlock, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from "../../components/lesson-labs/PythonExample";
import { apiExamples } from "../api-design-examples.js";
import { apiPracticeExamples } from "../api-practice-examples.js";
import { ApiBoundaryLab, ApiOwnershipLab, ApiCompatibilityLab } from "../../components/lesson-labs/ApiDesignLabs.jsx";


import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { ApiCheckingPicture } from "../../components/lesson-labs/ApiDesignFigures.jsx";

export default {
  title: "Code Documentation, Type Hints & API Design",
  readTime: "~38 min read + 90 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot">
    <LessonIntro prerequisites="Python functions, exceptions, files and basic classes; the testing lesson helps with verification."
      sections={[["1-design-for-the-person-calling-your-code", "API contract"], ["2-separate-hints-from-runtime-checks", "Type hints"], ["3-write-a-contract-you-can-execute", "Docstrings"], ["5-make-ownership-and-defaults-predictable", "Ownership"], ["7-build-a-truthful-file-loading-api", "Real loader"], ["10-practise-and-review", "Practise"]]}>
      Turn a calculation into a trustworthy interface. Explain what callers can pass, what they receive, what can fail, and what remains unchanged—then check those promises with actual files, executable documentation and a type checker.
    </LessonIntro>
    <H2>1. Design for the person calling your code</H2>
    <Prose>You have a working calculation in a notebook. A teammate wants to use it in a report without copying your whole notebook. The next job is to make the boundary clear: what to pass, what comes back, and which mistakes will be reported. <strong>Documentation</strong> explains those promises; <strong>type hints</strong> express part of them in a form tools can inspect; <strong>API design</strong> chooses the promises themselves.</Prose>
    <Prose>An API is the boundary through which another piece of code uses yours. A Python function is already an API even if there is no web server. A caller should not need to read its implementation to discover that it mutates a list, measures seconds instead of milliseconds, or returns None on a missing file.</Prose>
    <Prose>Start with a real use case: a notebook needs an offset-adjusted mean, and a report needs to load named evaluation scores from JSON. We will make both contracts explicit. Names should describe actual behaviour: a function called load_scores must read the supplied source, not ignore its path and return sample constants.</Prose>
    <LessonTable caption="The public contract has several parts" headers={["Part", "Question to answer", "Example decision"]} rows={[
      ["Input", "Which types, units, shapes and ranges?", "Nonempty finite measurements in ms."],
      ["Output", "Which meaning and representation?", "A float mean in ms, after subtracting the offset."],
      ["Failure", "Which invalid inputs and exceptions?", "Empty sequence raises ValueError."],
      ["Ownership", "What is read, changed or consumed?", "Do not mutate caller measurements."],
      ["Side effects", "Which files or external state change?", "The loader reads; it does not rewrite the source."],
      ["Compatibility", "What can callers rely on across changes?", "Threshold comparison remains inclusive."],
    ]} />
    <Prose>Keep the public surface small. Read data at the boundary, validate it, compute with clear internal assumptions, then format or save at another boundary. Avoid a single function that loads, cleans, trains, plots and emails with many interacting flags. Small does not mean every line needs a helper; split where responsibilities or contracts genuinely differ.</Prose>

    <H2>2. Separate hints from runtime checks</H2>
    <Prose>Read <Code>def double(value: float) -&gt; float</Code> as “double expects a number called value and promises a numeric result.” The colon annotates the parameter; the arrow annotates the returned value. Unlike <Code>float(value)</Code>, neither notation converts data. Save each full example as lesson.py and run it with the intended environment's Python, as in the testing lesson. The examples use Python 3.12; the optional checker is separate software.</Prose>
    <ApiCheckingPicture />
    <PythonExample example={apiExamples.hints}><Prose>The second call contradicts the annotation, yet Python multiplies a string and prints haha. An annotation communicates intent to readers and analysis tools; ordinary Python does not automatically enforce it. The displayed call is deliberately wrong for a static type checker.</Prose></PythonExample>
    <Prose>Static checking examines source without running the program. Runtime validation checks actual values during execution. Tests exercise selected behaviours. These are complementary: a type checker can flag the string argument, but float alone does not express “finite”, “nonnegative”, “milliseconds” or an array shape.</Prose>
    <LessonTable caption="Choose hints according to the operations you need" headers={["Hint", "Meaning", "Important limit"]} rows={[
      ["list[float]", "A mutable list of numeric values under the type contract", "Not automatic per-element runtime validation."],
      ["Sequence[float]", "An ordered, indexable input such as a list or tuple", "Does not promise the underlying object is immutable."],
      ["Iterable[float]", "Something you can iterate", "Could be a one-shot generator; may have no len or indexing."],
      ["Mapping[str, float]", "A key/value lookup interface", "Does not promise a concrete dict or enforce value ranges."],
      ["float | None", "Either a number or absence", "Callers must handle both cases."],
      ["object / Any", "Unknown Python object / opt out of many checks", "object needs narrowing; Any can hide mistakes."],
    ]} />
    <Prose>For example, a generator is not a Sequence merely because it yields numbers. Decide whether to accept streaming input or require repeatable indexing. Narrow types with checks such as is None or isinstance before using type-specific operations. Cast and type-ignore comments do not convert or validate data; use them only when you can explain evidence the checker cannot see.</Prose>
    <CodeBlock language="bash">{`python -m pip install mypy==2.3.1
python -m mypy --strict example.py`}</CodeBlock>
    <Prose>Save the first example as example.py and run this command from its folder. The checker reports an incompatible argument to double; exact diagnostic wording and line numbers depend on the source and tool version. We verified rejection with mypy 2.3.1. The executable examples use Python 3.12.14; this is the recorded environment, not a requirement to adopt that checker for every project.</Prose>

    <H2>3. Write a contract you can execute</H2>
    <Prose>A <strong>docstring</strong> is the string immediately inside a function, class or module. Editors and <Code>help(mean_ms)</Code> can show it to a caller. The triple quotes allow several lines. The <Code>*</Code> in the signature means arguments after it must be named: <Code>offset_ms=2</Code> makes the unit and purpose visible where the function is called.</Prose>
    <PythonExample example={apiExamples.contract}><Prose>The answer is (10 + 20 + 30) / 3 − 2 = 18 ms. The keyword-only offset_ms makes the adjustment explicit at the call site. Empty input, NaN and bool each fail according to the documented rules. Doctest runs the two examples inside the docstring and checks their displayed answers.</Prose></PythonExample>
    <Prose>The first docstring sentence tells a caller what the function does. The next lines specify units, input conditions, mutation and errors. That is more useful than comments narrating “loop over values” or “return the result.” Explain why a surprising choice exists: bool is rejected even though Python treats it as an int subclass because a truth value is not a measurement here.</Prose>
    <Prose>This deliberately narrow runtime contract accepts built-in int and float values, not arbitrary NumPy scalars, Decimal objects or numeric subclasses. The hint describes ordinary intended usage; the docstring adds restrictions the type system does not capture. For huge integers or extreme floats, conversion or summation can overflow; this basic numerical implementation is intended for ordinary measurement ranges, not arbitrary-precision arithmetic.</Prose>
    <Prose>Doctests make small documentation examples testable; they do not replace unit tests for missing values, overflow policy, mutation or unusual inputs. Stable examples should avoid timestamps, memory addresses and random unseeded outputs. For larger results, assert properties and show a meaningful summary rather than depending on incidental formatting.</Prose>
    <H3>Different documentation serves different jobs</H3>
    <LessonTable caption="Put explanations where readers need them" headers={["Document", "Main purpose", "Useful content"]} rows={[
      ["README", "Get a user to a first successful run", "Purpose, setup, exact command and small expected result."],
      ["Tutorial", "Teach a task in sequence", "Motivation, data, code, output and explanation."],
      ["Docstring / API reference", "Answer a caller's precise question", "Arguments, defaults, units, return value, exceptions and example."],
      ["Inline comment", "Explain a local non-obvious decision", "A constraint or reason, not a paraphrase of the next line."],
      ["Change notes", "Help users migrate", "Changed behaviour, old/new calls and compatibility window."],
    ]} />

    <H2>4. Represent absence without losing valid values</H2>
    <PythonExample example={apiExamples.optional}><Prose>Baseline has a valid score of 0.0; missing has no entry. Testing if score would treat both as false. An explicit is None check preserves the distinction and narrows the remaining branch to a number.</Prose></PythonExample>
    <Prose>Choose absence and failure deliberately. A lookup may return None for a normal “not found” result; a required configuration entry might raise KeyError. Returning an empty dictionary for every IO or parse failure makes “valid empty file” indistinguishable from “could not load data.” Catch only exceptions you can handle meaningfully, preserve context when translating them, and let unexpected failures remain visible.</Prose>
    <Prose>A return hint documents every path, not only the successful one. If a function sometimes returns a list and sometimes a single element based on a flag, consider separate functions or a consistent result shape. Avoid making callers guess the type after every call.</Prose>

    <H2>5. Make ownership and defaults predictable</H2>
    <ApiOwnershipLab />
    <details><summary>Run the identity counterexample, including a shallow copy</summary><PythonExample example={apiPracticeExamples.ownership}/><Prose>The first and second names observe one shared object. Copying creates a new outer object, but the last experiment deliberately mutates an inner list that both outer lists still reference. “Returns a new list” and “nothing inside is shared” are different contracts.</Prose></details>
    <PythonExample example={apiExamples.defaults}><Prose>The function copies the supplied list before appending. Original stays ['raw']; calls without a list receive independent results. None is a sentinel meaning “no starting tags supplied”, not a shared container.</Prose></PythonExample>
    <Prose>Default argument objects are created when the function definition runs. A default tags=[] can therefore accumulate state across calls if mutated. Replacing the default with None fixes the shared-default problem; copying a supplied list additionally establishes non-mutation of the caller's list. Those are two distinct decisions.</Prose>
    <Prose>List copying is shallow: nested mutable objects remain shared. If a function intentionally mutates an input, name and document that behaviour. If it consumes a generator or closes a file it receives, that is also ownership information. Do not silently close a caller-owned resource just because your function has finished reading it.</Prose>

    <H2>6. Return named results and accept useful interfaces</H2>
    <Prose>An interface describes the operations a caller needs, such as “can read text.” A concrete class is one implementation. Accepting the interface lets a real file and an in-memory text buffer serve the same role in tests. A <strong>Protocol</strong> lets a static checker express that relationship without requiring both implementations to inherit from your own base class.</Prose>
    <PythonExample example={apiExamples.result}><Prose>The dataclass gives count and mean_ms names, so callers do not need to remember tuple positions or units. Frozen prevents field reassignment, shown by the caught exception. Reader is a Protocol: StringIO works because it provides read() returning str, without inheriting from Reader.</Prose></PythonExample>
    <Prose>The ellipsis in the Protocol method is an intentional interface declaration, not an unfinished implementation. Static structural typing checks the required method signature. The example does not use a runtime protocol check, and protocols do not validate the contents of returned strings. Prefer ordinary parameters when a protocol would add complexity without a real substitutability need.</Prose>
    <Prose>Dataclasses reduce record boilerplate; they do not automatically validate field types or deeply freeze nested objects. A TypedDict can describe the keys and value types of a dictionary for static checking, but is also not a JSON validator. Choose a result record when names improve clarity; a simple scalar is still best for a simple mean.</Prose>
    <Prose>The attempted frozen-field assignment is deliberately caught at runtime. A type checker also rejects assigning that field; when checking the reusable declarations, omit this demonstration driver. The declarations themselves are checked separately in the lesson's verification suite.</Prose>

    <H2>7. Build a truthful file-loading API</H2>
    <Prose>Now implement load_scores as promised: read the given UTF-8 JSON file, validate a name-to-score object, and keep values at or above minimum. Scores and minimum must be finite numbers in [0,1]. Bool is rejected; a valid score of zero is preserved. Validation happens before filtering so an invalid low-scoring entry cannot quietly disappear.</Prose>
    <ApiBoundaryLab />
    <Checkpoint prompt="At minimum 1, a valid file with a score 0 produces an empty dictionary. Is that an error, or evidence of a broken loader?"><Prose>Neither. The file passed validation but no score met the filter. By contrast, a file containing -0.1 must fail even though that entry also would not pass. Separate valid no-matches results from invalid inputs; callers need that distinction to diagnose their data.</Prose></Checkpoint>
    <PythonExample example={apiExamples.loader}><Prose>The fixture creates a real file in a temporary folder. Only larger survives minimum=0.85. Replacing the file with a boolean value triggers the schema error, and a nonexistent path raises FileNotFoundError. The temporary directory is removed when the demonstration finishes; no existing user file is overwritten.</Prose></PythonExample>
    <LessonTable caption="Trace the boundary from bytes to return value" headers={["Stage", "Successful value", "Possible failure"]} rows={[
      ["Read UTF-8 text", "The contents of the named file", "FileNotFoundError, PermissionError or another OSError; UnicodeDecodeError."],
      ["Parse JSON", "A Python object", "JSONDecodeError for invalid syntax."],
      ["Validate schema", "Nonblank names and finite scores in [0,1]", "ValueError; the complete object must pass."],
      ["Filter inclusively", "A new dict of float values meeting minimum", "An empty dict is a legitimate no-matches result."],
    ]} />
    <Prose>Path accepts either a string or Path object. The loader does not change the file or mutate a caller dictionary. It does not silently strip or case-normalise names: 'Baseline' and 'baseline' remain different keys. Reading the full text is appropriate for this small fixture, not a streaming strategy for unbounded uploads.</Prose>
    <Prose>Python's default JSON decoder can accept non-finite numeric tokens; the range/finite validation rejects their resulting values. Duplicate JSON member names are normally collapsed by the decoder, so this implementation cannot detect them after parsing. If duplicates must be an error, use an object_pairs_hook-based decoder before building the dictionary. Likewise, define file-size limits and path-access policy at a service boundary. A type hint is not a security boundary.</Prose>
    <Prose>This API intentionally exposes useful IO and parsing exceptions rather than replacing all of them with “failed.” JSONDecodeError is a ValueError subclass; callers needing separate parse and schema handling must catch the more specific type first. A future wrapper can add domain context with exception chaining while keeping the original cause available.</Prose>

    <H2>8. Test promises, not only the happy path</H2>
    <Prose>A useful test matrix follows the contract: threshold equality, zero and one, empty object, absent file, invalid syntax, wrong root type, booleans, null, out-of-range numbers and NaN. Verify non-mutation where promised. Tests using actual temporary files exercise IO and parsing that a mocked dictionary would skip.</Prose>
    <Prose>Static checking was run on the reusable API declarations, and the intentional bad double call was separately checked for rejection. Runtime examples and doctests were also executed. Each catches different errors: changing the comparison to strict greater-than can still type-check but breaks inclusive-threshold behaviour.</Prose>
    <CodeBlock language="bash">{`python -m doctest -v mean_api.py
python -m mypy --strict mean_api.py`}</CodeBlock>
    <Prose>For these commands, save the contract example's imports and mean_ms definition as mean_api.py, leaving out its demonstration driver. The docstring examples remain part of the function. A module intended for import should not read files, print demonstrations or run experiments merely because it is imported; keep such actions behind a main guard or in separate examples.</Prose>

    <H2>9. Evolve the interface without surprising callers</H2>
    <ApiCompatibilityLab />
    <Prose>Changing a default, a return shape, units, mutation behaviour, accepted range or exception can break callers even when the function name stays the same. Add new optional settings as keyword-only where practical. Introduce a deprecation path for public behaviour changes and document how to migrate; do not assume a type-checking pass proves compatibility.</Prose>
    <Prose>For example, renaming minimum to min_score breaks callers using minimum=. Returning percentages from a score loader that previously returned fractions is a semantic break even if both are floats. Preserve the old contract or give the new quantity an explicit name and migration. Tests should include representative external calls, not just internal helpers.</Prose>

    <H2>10. Practise and review</H2>
    <H3>Independent API: a duration adapter for two instruments</H3>
    <Prose>One device reports seconds; another reports milliseconds. Design durations_ms(values, *, input_unit="ms") to return a new list in milliseconds. It must accept a list, tuple or one-shot iterable, consume it once, preserve valid zeroes and return [] for empty input. Accept only built-in finite, nonnegative int/float measurements, rejecting bool. Reject unknown units and conversion overflow. Document error types and what happens if a generator contains a bad value halfway through.</Prose>
    <Prose>Before implementing, write the call a colleague should make and its expected answer: [0,0.25,2] seconds becomes [0.0,250.0,2000.0] milliseconds. Add tests for unchanged list input, a consumed generator, empty input, invalid units, negative/bool/non-finite values and a very large conversion. A single float hint cannot guarantee any of these value rules.</Prose>
    <details><summary>Hint: define one traversal and one ownership promise</summary><Prose>Validate the unit before consuming anything. Traverse values once and append validated conversions to a new list. A failed generator cannot generally be rewound; document partial consumption. For a value whose conversion overflows, raise the promised error rather than returning infinity.</Prose></details>
    <details><summary>One complete typed implementation, docstring and run</summary><PythonExample example={apiPracticeExamples.apiTransfer}/><Prose>Iterable is justified by the single traversal; a sequence requirement would unnecessarily exclude generators. The new output list does not mutate a supplied list. The explicit factor prevents a silent unit change. Transfer: [0.001,0.002] seconds becomes [1.0,2.0]; the same input with default ms stays [0.001,0.002].</Prose></details>
    <Checkpoint prompt="A score equals the threshold, 0.85. Should the loader include it? What test would catch an accidental change?">
      <Prose>Yes: the documented comparison is inclusive. Write a real JSON fixture with that value and assert it remains in load_scores(path, minimum=0.85). Replacing <Code>{">="}</Code> with <Code>{">"}</Code> should make that test fail.</Prose>
    </Checkpoint>
    <Checkpoint prompt="The input contains {'ok': 0.9, 'bad': -0.1}, and minimum is 0.8. Should filtering hide bad?">
      <Prose>No. The whole source object must meet the contract before filtering. Raise ValueError for the negative score even though it would not be returned. Otherwise the validity of a file would depend on the caller's threshold.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Can you accept a generator in mean_ms just by changing Sequence to Iterable?">
      <Prose>No. The implementation uses len and iterates the values more than once. Either materialise and validate once with a documented memory cost, or implement a streaming sum/count with an explicit empty-input and numerical policy. Change the implementation and tests along with the hint.</Prose>
    </Checkpoint>
    <Checkpoint prompt="A teammate asks for strict rejection of duplicate JSON score names. What needs changing besides the docstring?">
      <Prose>Detect duplicates during decoding, before a dict discards the earlier occurrence. Add a duplicate-name fixture, verify rejection, and document the new behaviour. Updating prose alone would create a promise the current implementation cannot keep.</Prose>
    </Checkpoint>
    <Prose>You can now build a small interface whose examples are truthful and whose contracts are testable. Advanced generic typing, overload design, async protocols and packaged API documentation are deeper extensions, not prerequisites for this loader. Next, <a href="/learn/topic/git-github-collaborative-version-control">use Git to review and preserve changes</a> to these contracts.</Prose>
    <Sources alternatives={<LearningResources>
      <li><a href="https://www.youtube.com/watch?v=ST33zDM9vOE">Dustin Ingram, PyCon US 2020: Static Typing in Python</a> and the author's <a href="https://dustingram.com/talks/2020/03/19/static-typing-in-python/">written talk</a> — beginner-to-intermediate explanation of gradual typing and checker purpose. The recording is historical; use this lesson's modern built-in generics and union syntax rather than its older typing spellings.</li>
      <li><a href="https://mypy.readthedocs.io/en/stable/getting_started.html">mypy: getting started</a> — practical checker feedback to try after the annotation experiment; compare a reported source error with what Python does at runtime.</li>
      <li><a href="https://diataxis.fr/how-to-use-diataxis/">Diátaxis: choosing the documentation a reader needs</a> — use when a reference page is technically complete but still fails to teach a task. This is guidance on documentation purpose, not a mandatory lesson template.</li>
    </LearningResources>}>
      <li><a href="https://typing.python.org/en/latest/guides/libraries.html">Python typing guide: useful public interfaces and library contracts</a></li>
      <li><a href="https://docs.python.org/3.12/library/typing.html">Python: annotations, unions, protocols and static typing</a></li>
      <li><a href="https://mypy.readthedocs.io/en/stable/getting_started.html">mypy: running a type checker</a></li>
      <li><a href="https://docs.python.org/3/library/doctest.html">Doctest: execute documentation examples</a></li>
      <li><a href="https://docs.python.org/3.12/tutorial/controlflow.html#documentation-strings">Python: parameter conventions, defaults and documentation strings</a></li>
      <li><a href="https://docs.python.org/3.12/library/dataclasses.html">Dataclasses and frozen-instance limitations</a></li>
      <li><a href="https://docs.python.org/3.12/library/json.html">JSON decoding and interoperability details</a></li>
    </Sources>
  </div>,
};
