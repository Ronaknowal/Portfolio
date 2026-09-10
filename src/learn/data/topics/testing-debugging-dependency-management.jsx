import { Code, CodeBlock, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from "../../components/lesson-labs/PythonExample";
import { DebugExecutionLab, TestDiscriminationLab, DependencyConstraintsLab } from "../../components/lesson-labs/TestingLabs.jsx";


import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { testingPracticeExamples } from "../testing-practice-examples.js";
import { testingExamples } from "../testing-examples.js";
import { TestingBoundaryFigure } from "../../components/lesson-labs/TestingBoundaryFigure.jsx";

export default {
  title: "Testing, Debugging & Dependency Management",
  readTime: "~35 min read + 80 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot">
    <LessonIntro prerequisites="Functions, exceptions, imports and simple context managers. The executable tests use only Python's standard library."
      sections={[["1-turn-an-expectation-into-evidence", "Find a bug"], ["2-build-a-test-suite-that-actually-runs", "Run tests"], ["4-debug-with-a-specific-question", "Debug"], ["6-choose-the-interpreter-before-the-packages", "Environments"], ["8-practise-and-check", "Practise"]]}>
      Reproduce a failure, fix its cause, run regression and edge-case tests, and record enough environment information for another person to repeat the result.
    </LessonIntro>
    <H2>1. Turn an expectation into evidence</H2>
    <Prose>You change a function, get an answer, and wonder whether the answer is right. Start with a case whose answer you can work out without the program. A <strong>test</strong> records that expectation as an executable check. A <strong>bug</strong> is a mismatch between the promised behaviour and what the program actually does; a crash is only one possible symptom.</Prose>
    <Prose>A script that ran once proves only that one execution completed. A test makes a particular claim about behaviour: given these inputs, the program should return this result or raise this error. Debugging investigates a failed claim. Dependency management controls part of the environment in which that claim is evaluated.</Prose>
    <Prose>To run a standalone example, create a new folder in your editor, save the complete code as lesson.py, and open the editor's terminal in that folder. A terminal accepts commands that launch programs; type <Code>python lesson.py</Code> and press Enter. Use <Code>py lesson.py</Code> on Windows or <Code>python3 lesson.py</Code> on macOS/Linux if that is your installed Python command. Code blocks contain Python unless labelled as terminal commands. The examples need Python 3.12 and its included standard library; no testing package is required.</Prose>
    <Prose>Start with a mean function for a nonempty sequence of ordinary finite numbers. The mean of 18 and 24 must be 21. You can calculate that independently before writing the test. Here is a deliberately incorrect implementation:</Prose>
    <PythonExample example={testingExamples.testingRed}><Prose>Return is inside the loop, so only 18 contributes before the function exits. Dividing 18 by the original length 2 gives 9. Changing the expected value to 9 would make the assertion pass while preserving the bug.</Prose></PythonExample>
    <DebugExecutionLab />
    <Checkpoint prompt="Why would a test with the single input [18] pass even with the early-return bug? Choose a better regression case."><Prose>The first value is also the last, so both versions return 18/1. Two nonzero readings expose missing accumulation: [18,24] should produce 21. A regression test protects a previously observed failure; it is valuable because of the behaviour it distinguishes, not the number of lines it executes.</Prose></Checkpoint>
    <Prose>A useful cycle is red → green → refactor: demonstrate a failure, make the smallest correct change, then improve structure while keeping the test passing. “Red” here means the assertion really failed; the teaching snippet catches it only so its message is stable and readable. Normal test runners should record the failure rather than hide it.</Prose>

    <H2>2. Build a test suite that actually runs</H2>
    <Prose>A <strong>test case</strong> is one checked situation. A <strong>suite</strong> collects cases, and a <strong>runner</strong> executes them and reports failure. Unittest supplies a TestCase base class: each method named test_ becomes a check the runner can discover. A fixture is the starting data or resource arrangement for a test. These names describe roles; they are not additional algorithms to learn.</Prose>
    <Prose>Save the following two files in the same folder. The implementation uses fsum for more accurate floating-point summation and gives empty and non-finite inputs explicit error behaviour. The contract assumes a sized collection of numeric readings, not arbitrary text, generators or numbers outside the practical range of float calculations.</Prose>
    <PythonExample example={testingExamples.testingSuite}><Prose>This script discovers four methods on MeanTests and executes them. subTest lets a method try several labelled inputs without hiding later cases after one assertion failure. The output says four test methods ran, not merely that four functions were defined. The script exits unsuccessfully if the suite fails.</Prose></PythonExample>
    <Prose>For normal development, run <Code>python -m unittest discover -s . -p "test_*.py" -v</Code> from that folder. Unittest imports matching modules, discovers test cases, and reports individual names and tracebacks. Its normal timing and formatting vary; our teaching runner captures that report and prints only the stable summary above. Imported tests still run through discovery; the main guard prevents them running merely because another module imports the file.</Prose>
    <LessonTable caption="Arrange, act, assert" headers={["Step", "In the ordinary test", "Why it matters"]} rows={[
      ["Arrange", "Choose [18, 24] and independently calculate 21", "Control the starting conditions."],
      ["Act", "Call mean once", "Exercise the behaviour under test."],
      ["Assert", "Compare with the expected result", "Produce a visible failure if the claim is false."],
    ]} />
    <Prose>Use equality for exact discrete results, assertAlmostEqual or an explicit tolerance for appropriate floating-point results, and assertRaises for an error contract. Unittest assertion methods remain active under optimised Python; plain assert statements can be removed with optimisation. Neither kind of assertion should replace runtime input validation in production code.</Prose>
    <Prose>If discovery reports zero tests, inspect the directory, filename pattern, TestCase inheritance and test_ method names. A successful process exit with zero discovered tests is not evidence that your implementation works.</Prose>

    <H2>3. Test boundaries, properties and interactions</H2>
    <Prose>Some answers are known directly: water's standard temperature reference points give 0°C → 32°F and 100°C → 212°F under the conversion formula. Other checks express a relationship: adding 10°C must add 18°F. Such a relationship can check many inputs without independently knowing every complete answer, but it may fail to distinguish particular bugs.</Prose>
    <TestDiscriminationLab />
    <PythonExample example={testingExamples.testingTolerance}><Prose>The three reference points check known temperatures. The translation property checks that adding 10°C changes Fahrenheit by 18°F at several starting values. These complement each other: a function with a constant offset bug could satisfy the difference property while failing the reference cases.</Prose></PythonExample>
    <Prose>An absolute tolerance is in the output's units; a relative tolerance scales with magnitude. Near zero, an appropriate absolute tolerance matters. Choose values based on the task's precision needs instead of increasing tolerances until a test stops failing. Tests that reimplement the same algorithm often repeat the same mistake; use known examples and independent properties.</Prose>
    <LessonTable caption="Build a small but deliberate case set" headers={["Case", "Question", "Example for a mean"]} rows={[
      ["Ordinary", "Does the usual path work?", "[18, 24] → 21"],
      ["Boundary", "What is the smallest valid input?", "[0] → 0"],
      ["Invalid", "Is failure explicit?", "[] raises ValueError"],
      ["Numerical", "Are special values accepted or rejected deliberately?", "NaN / infinity rejected"],
      ["Regression", "Can the old bug return unnoticed?", "Two unequal readings catch the early return"],
      ["Property", "What relationship should survive many inputs?", "Shifting all finite inputs shifts their mean, within numerical limits"],
    ]} />
    <H3>Mock a boundary, then test the real boundary too</H3>
    <Prose>Suppose the mean comes from a reader that normally fetches text. To ask “does my calculation handle this text?”, supply a small substitute reader returning a fixed string. This is <strong>dependency injection</strong>: the caller supplies the collaborator. A <strong>mock</strong> is a controllable substitute that can also record how it was used. It changes the question the test can answer, so keep a separate check of the real file boundary.</Prose>
    <TestingBoundaryFigure />
    <PythonExample example={testingExamples.testingIsolation}><Prose>The injected reader returns known text without touching a disk or network. Mock records that it was called once without arguments. Setting side_effect exercises an unavailable-reader path. This verifies the calculation's interaction contract; it does not prove that opening a real file works.</Prose></PythonExample>
    <PythonExample example={testingExamples.testingIntegration}><Prose>The integration example writes an actual UTF-8 file inside a dedicated temporary directory and reads it through the real file API. Blank lines are skipped. The directory is removed when its context exits; nothing is written over a user's existing dataset.</Prose></PythonExample>
    <Prose>Unit tests focus on a small behaviour. Integration tests connect real components. End-to-end tests exercise a user workflow. Mock only dependencies you need to control, not every internal method; otherwise tests can pass because the implementation has effectively been replaced. With patch, patch the name where the code looks it up, and undo changes after the test. Prefer dependency injection when it makes that boundary clearer.</Prose>
    <Prose>Unittest fixtures such as setUp and tearDown prepare and release per-test state; addCleanup can register cleanup even when later setup fails. Tests should not depend on execution order. Keep clocks, randomness, files and external services controlled where possible. A mock should not become your only evidence about a real integration.</Prose>

    <H2>4. Debug with a specific question</H2>
    <Prose>Read a traceback from the final exception message back to the relevant line in your code. Reduce the input until the failure is easy to reproduce. Then ask one question: was this branch taken, is this value missing, did this array have the expected shape, or did this loop stop early?</Prose>
    <Prose>For the early-return bug, put <Code>breakpoint()</Code> just before return and run the file in a terminal, or use <Code>python -m pdb lesson.py</Code>. The debugger pauses execution so you can inspect state. Remove or disable diagnostic breakpoints before unattended test/production runs.</Prose>
    <LessonTable caption="A focused pdb session" headers={["Command", "What it does", "Question it answers"]} rows={[
      ["l", "List nearby source", "Is return actually inside the loop?"],
      ["p values; p total", "Evaluate and print an expression (run each p command separately)", "What is in memory at this point?"],
      ["n", "Execute the next line without stepping into called functions", "Does execution reach another iteration?"],
      ["s", "Step into a function call", "Where does this value come from?"],
      ["w", "Show the call stack", "Who called this function?"],
      ["c / q", "Continue / quit the debugger", "Resume or end the session."],
    ]} />
    <Prose>Debuggers can execute expressions, so avoid evaluating state-changing functions just to inspect them. When a notebook behaves differently from a fresh script, restart and run cells in order. For intermittent issues, record context and timestamps with logging, but avoid secrets and sensitive payloads. A print statement is evidence only for the exact state and run where it was observed.</Prose>

    <H2>5. What passing tests do not prove</H2>
    <Prose>Passing examples do not establish correctness for every input, numerical scale, operating system or dependency version. Code coverage measures which code executed, not whether the assertions were meaningful. A test that calls a function without checking anything may increase coverage while proving little.</Prose>
    <Prose>Use slow or external tests separately when they make rapid feedback impractical. Keep regression cases after fixing a bug. Run a clean, noninteractive test command in continuous integration and honour its exit status. Property-based testing can generate many boundary cases; it still needs a correct property and a realistic input domain.</Prose>
    <Prose>Pytest is a popular alternative with fixtures, parametrisation and concise assertions. This lesson uses unittest so every executable example runs without installing a test framework. The core design principles transfer; fixtures and assertion APIs differ. More tests are useful when they cover new behaviour, not merely more lines.</Prose>

    <H2>6. Choose the interpreter before the packages</H2>
    <Prose>A <strong>dependency</strong> is another package your code needs. A direct dependency is one you request; a transitive dependency is requested by another package. The interpreter is the Python program executing your code. Installing a package adds it to a particular environment; it does not make every Python on the machine see it.</Prose>
    <Prose>A virtual environment isolates Python package installations for a project. It is not a security sandbox and does not automatically reproduce operating-system libraries, drivers or data. Use <Code>python -m pip</Code> with the intended interpreter so installation and execution refer to the same Python.</Prose>
    <Prose>The following are terminal recipes, not Python source or fixed-output examples. Create the environment once in a project folder. For the next lesson's recorded NumPy 2.3.5 setup, use Python 3.11 or newer; its examples were checked here with Python 3.12.14.</Prose>
    <H3>Windows PowerShell: activation is optional</H3>
    <CodeBlock language="powershell">{String.raw`python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.3.5
.\.venv\Scripts\python.exe -c "import sys, numpy; print(sys.executable); print(numpy.__version__)"
.\.venv\Scripts\python.exe -m pip check`}</CodeBlock>
    <Prose>Calling the environment's interpreter directly avoids changing PowerShell's execution policy to activate it. If the system command is py rather than python, use that to create the environment. Do not copy another machine's executable path from an example.</Prose>
    <H3>macOS/Linux: the same idea, a different path</H3>
    <CodeBlock language="shell">{String.raw`python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5
.venv/bin/python -c "import sys, numpy; print(sys.executable); print(numpy.__version__)"
.venv/bin/python -m pip check`}</CodeBlock>
    <Prose>These installation commands need package-index access and a compatible interpreter/platform. The exact path and pip messages vary; verify that sys.executable points inside your environment. The pinned version is the tested example environment, not a claim that it is the newest or appropriate forever. No installation is needed for the earlier standard-library examples.</Prose>
    <LessonTable caption="Diagnose environment errors by checking the boundary" headers={["Symptom", "Check", "Likely action"]} rows={[
      ["Import works in terminal, fails in notebook", "Notebook kernel's sys.executable", "Select a kernel using the intended environment."],
      ["Package installed, still ModuleNotFoundError", "The Python used by pip versus the Python used to run code", "Install through that interpreter's -m pip."],
      ["No matching distribution", "Python version, platform and requested package version", "Choose a supported combination; do not randomly edit source imports."],
      ["Unexpected module behaviour", "Local filenames and module __file__", "Avoid files shadowing packages, such as numpy.py."],
    ]} />

    <H2>7. Record and recreate an environment</H2>
    <DependencyConstraintsLab />
    <Prose>The resolver searches for a set of releases whose requirements agree. Trying another candidate after a conflict is called backtracking. If no set works, inspect the reported requirements and supported Python versions; select compatible releases or separate genuinely independent applications into environments. Forcing an install without dependencies suppresses part of the check and can leave imports or APIs broken.</Prose>
    <Prose>A dependency declaration records what your project asks for. A resolved snapshot records what is installed, including transitive packages brought in by those requests. A lock workflow additionally resolves and records exact artifacts/versions according to its tool. These are related but not interchangeable.</Prose>
    <CodeBlock language="text">{`# requirements.txt — minimal declaration for this numerical example
numpy==2.3.5`}</CodeBlock>
    <Prose>Save that file, then use the environment's Python to run <Code>-m pip install -r requirements.txt</Code>. For a snapshot of the installed environment, run <Code>-m pip freeze</Code> and save its output to a separate requirements.snapshot.txt. Freeze reports installed distributions; it is not a solver-generated, cross-platform lockfile. Keep direct requirements distinct from a snapshot so accidental local tools do not become permanent project dependencies.</Prose>
    <Prose>For a reusable package, pyproject.toml can declare project metadata, supported Python versions and dependencies. Exact pins suit a reproducible application snapshot; libraries often declare tested compatibility ranges. Neither approach justifies ignoring upgrades: update deliberately, rerun tests and review results.</Prose>
    <Prose>To check reproducibility, create a fresh environment, install from the recorded file, run pip check, then run the test command. Pip check detects dependency-metadata inconsistencies, not every API incompatibility. Record the Python version, operating system, required system libraries, data version and relevant random seeds too. Do not commit the .venv directory, credentials or machine-specific absolute paths. Only install packages from sources you trust.</Prose>

    <H2>8. Practise and check</H2>
    <H3>Independent investigation: test a filtered measurement report</H3>
    <Prose>Write positive_mean(text) before opening the solution. Text contains one reading per line. Skip blank lines, reject invalid or non-finite numbers, and average only strictly positive readings. Reject when no positive readings remain. For -3, 0, 6, 12 on separate lines, the answer is 9, because only 6 and 12 count in the denominator. Use ordinary finite magnitudes; arbitrary-precision and overflow policies are outside this exercise.</Prose>
    <Prose>Write tests for mixed values, one positive value, whitespace, no positive values, and a malformed/non-finite value appearing after a good value. Introduce two deliberate bugs in scratch copies: divide by the original line count, then change the selection to ≥0. Demonstrate a failing test for each, restore the correction, and run the suite again.</Prose>
    <details><summary>Hint: separate input validity from selection</summary><Prose>Parse and validate each nonblank field before deciding whether it enters the positive list. Use the selected list's length as the denominator. The [0,6,12] case catches accidental inclusion of zero: 6 would be wrong, 9 is correct.</Prose></details>
    <details><summary>One complete solution and executable suite</summary><PythonExample example={testingPracticeExamples.testingRepair}/><Prose>The four methods include multiple subcases and return a nonzero process status on failure. Change the input to 2, -4, 8: only 2 and 8 contribute, giving 5. A suite that merely checks “some number returned” would miss both introduced bugs.</Prose></details>
    <Checkpoint prompt="Add a regression test for the old early return, and demonstrate that it fails against the old code before passing against the correction.">
      <Prose>The existing ordinary test with [18, 24] already does this: expected 21 versus old result 9. Temporarily point the test at wrong_mean in a scratch copy and run it. A test that does not fail on the known bug is not protecting that regression.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Design a test that catches a Fahrenheit function which multiplies by 1.8 but forgets the +32. Would the translation property alone catch it?">
      <Prose>Check 0°C → 32°F, as in the runnable reference cases. The incorrect function still changes by 18°F for every 10°C increment, so the translation property alone would pass. Combine reference cases with properties.</Prose>
    </Checkpoint>
    <Checkpoint prompt="The mock-reader tests pass, but the program cannot read a UTF-8 file. What evidence is missing?">
      <Prose>A real file integration test, like the temporary-directory example. The mock bypassed encoding, path selection and file I/O. Test the actual failing boundary without abandoning the fast isolated calculation tests.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Write a short reproducibility hand-off for another learner.">
      <Prose>Include the required Python version, the dependency file, how to create/select the environment, the exact test command, expected test count, data assumptions and any platform limitations. Ask them to run it from a fresh environment, not your existing notebook session.</Prose>
    </Checkpoint>
    <Prose>Continue to <a href="/learn/topic/numpy-arrays-broadcasting-vectorization">NumPy</a>, where shape assertions, numerical tolerances and environment checks become especially useful.</Prose>
    <Sources alternatives={<LearningResources>
      <li><a href="https://www.youtube.com/watch?v=tIrcxwLqzjQ">CS50P, David Malan: Unit Tests</a> — beginner video with <a href="https://cs50.harvard.edu/python/notes/5/">written examples</a> for assertions, deliberately broken code and boundary cases. It uses pytest; this lesson's executable baseline is unittest. The notes' folder advice is simplified: modern pytest also collects tests outside packages; consult its current import documentation.</li>
      <li><a href="https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/">PyPA: environments and installation walkthrough</a> — follow after the interpreter explanation to practise choosing the environment on your own platform.</li>
    </LearningResources>}>
      <li><a href="https://pip.pypa.io/en/stable/topics/dependency-resolution/">pip: constraints, backtracking and resolution failures</a></li>
      <li><a href="https://docs.pytest.org/en/stable/explanation/pythonpath.html">pytest: package and standalone test import behaviour</a></li>
      <li><a href="https://docs.python.org/3/library/unittest.html">Python: unittest discovery, assertions and fixtures</a></li>
      <li><a href="https://docs.python.org/3/library/unittest.mock.html">Python: mocks, patching and interaction checks</a></li>
      <li><a href="https://docs.python.org/3/library/pdb.html">Python: debugger commands</a></li>
      <li><a href="https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/">Python Packaging Guide: pip and virtual environments</a></li>
      <li><a href="https://pip.pypa.io/en/stable/cli/pip_freeze/">pip: what freeze does and does not record</a></li>
      <li><a href="https://pypi.org/project/numpy/2.3.5/">NumPy 2.3.5: recorded example version and supported Python versions</a></li>
    </Sources>
  </div>,
};
