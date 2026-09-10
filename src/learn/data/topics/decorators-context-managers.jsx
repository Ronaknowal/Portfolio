import { Code, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from "../../components/lesson-labs/PythonExample";
import { DecoratorOrderLab, ContextLifetimeLab, ExitStackLab } from "../../components/lesson-labs/DecoratorContextLabs.jsx";


import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { decoratorExamples } from "../decorator-examples.js";
import { DecoratorBindingFigure } from "../../components/lesson-labs/DecoratorContextFigures.jsx";

export default {
  title: "Decorators & Context Managers",
  readTime: "~32 min read + 60 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot">
    <LessonIntro prerequisites="Functions, return, exceptions, positional/keyword arguments and simple classes from Python and OOP; suspended yield and cleanup from Iterators, Iterables & Generators. Argument forwarding and retained functions are refreshed below."
      sections={[["1-two-kinds-of-surrounding-work", "Choose the tool"], ["2-wrap-a-function-without-losing-its-result", "Decorators"], ["3-configuration-and-stacking-order", "Trace a call"], ["5-give-a-resource-a-clear-lifetime", "Context managers"], ["8-practise-and-check", "Practise"]]}>
      Add logging and timing without duplicating a function, understand decorator order, and manage resources on both the success and failure paths.
    </LessonIntro>
    <H2>1. Two kinds of surrounding work</H2>
    <Prose>Our reading pipeline now needs to report when a calculation starts and finishes, and to close an input file even when parsing fails. Neither requirement changes the mean formula. Both surround useful work, but they have different boundaries.</Prose>
    <LessonTable caption="Choose the boundary you want to control" headers={["Need", "Tool", "Lifetime"]} rows={[
      ["Run behaviour around calls to one function", "Decorator", "Each call through the decorated name"],
      ["Set up and clean up around a block", "Context manager", "One with block, possibly containing many calls"],
      ["A one-off explicit decision", "Ordinary code or try/finally", "No abstraction required"],
    ]} />
    <Prose>A decorator is a callable applied to another object at definition time. Function wrappers are the common case we will build; decorators can also return the original object or a different kind of callable. A context manager implements the enter/exit protocol used by with. These are language mechanisms, not guarantees that arbitrary wrappers preserve behaviour or arbitrary cleanup code cannot fail.</Prose>
    <Prose>Save each complete example separately as <Code>lesson.py</Code> and run <Code>python lesson.py</Code> in a terminal. They use only the standard library and were checked with Python 3.12.14; the demonstrated features work on Python 3.10+. The StringIO examples use in-memory text instead of requiring a file you do not yet have.</Prose>

    <H2>2. Wrap a function without losing its result</H2>
    <Prose>A function is an object: a variable can refer to it, another function can receive it, and a function can return it. A wrapper is a new function that calls the original one. Keep the original reference inside the wrapper; simply reassigning its public name must not make the wrapper accidentally call itself forever.</Prose>
    <DecoratorBindingFigure />
    <PythonExample example={decoratorExamples.decoratorBasic}><Prose>The decorated mean first enters the wrapper. The wrapper calls the original function, returns its result, and executes finally before the caller receives that result. On empty input the original raises ValueError; finally still prints leave, and the exception reaches the caller's except block.</Prose></PythonExample>
    <H3>Translate the @ syntax</H3>
    <Prose>Applying <Code>@logged</Code> to a definition is approximately <Code>mean = logged(mean)</Code> after the original function is created. logged is called during definition; wrapper runs during later calls. The nested wrapper retains a reference to function even after logged returns. This retained surrounding state is a closure.</Prose>
    <Prose><Code>*args</Code> captures positional arguments and <Code>**kwargs</Code> captures keyword arguments. Expanding both forwards them to the original call. Forgetting return would make successful calls return None. Catching an exception and quietly returning None would also change the contract, usually in a misleading way.</Prose>
    <Prose><Code>functools.wraps</Code> copies useful metadata such as the name and documentation and sets <Code>__wrapped__</Code> for inspection tools. It does not automatically validate arguments, make the wrapper's behaviour identical, or enforce the function's type hints. The output name stays mean instead of wrapper because we use wraps.</Prose>
    <Checkpoint prompt="What changes if the wrapper calls function(*args, **kwargs) but omits return?">
      <Prose>The calculation still runs and the logging still appears, but the caller receives None. Preserving side effects is not enough; test the returned value and exception behaviour too.</Prose>
    </Checkpoint>

    <H2>3. Configuration and stacking order</H2>
    <Prose>A configurable decorator needs one additional level: a factory receives configuration, a decorator receives the function, and the wrapper receives call arguments. Giving each layer a distinct job makes the nesting easier to read.</Prose>
    <PythonExample example={decoratorExamples.decoratorFactory}><Prose>The definition builds <Code>report = tagged("outer")(tagged("inner")(report))</Code>. Application is bottom-up; a later call enters outer then inner, reaches the body, and returns outward in reverse order. The printed result comes last because report() must finish before print receives its argument.</Prose></PythonExample>
    <DecoratorOrderLab />
    <PythonExample example={decoratorExamples.order}><Prose>Input 8 becomes 16 and then 10 with cap outside; reversing the order produces 8 and then 16. Input 3 happens to produce 6 either way, so one agreeing example would miss this difference. These wrappers deliberately change results: they would not be transparent logging wrappers.</Prose></PythonExample>
    <Prose>Order changes meaning. An authentication check outside a cache still runs before a cache hit is returned; a poorly designed outer cache could bypass an inner check. Keep correctness-critical policies explicit, and do not treat a decorator stack as decoration that can be reordered casually. The “after” prints in this simple tagged example do not run if the body raises—unlike the finally-based logger.</Prose>
    <details><summary>A decorator can register a plugin without wrapping calls</summary><Prose>A small importer may select a conversion function from a name in a data record. Registration connects a name to a function once, when its definition executes. It adds no wrapper to later calls. The duplicate check makes conflicting names visible instead of silently replacing behavior.</Prose><PythonExample example={decoratorExamples.registry}/><Prose>The identity result True is the useful surprise: decorator syntax did not replace the function object here. The map stores the same function for lookup by name. Real plugin systems also need explicit loading rules and trust boundaries; this local registry does not execute downloaded plugins or validate untrusted code.</Prose></details>

    <H2>4. Timing, caching and their limits</H2>
    <PythonExample example={decoratorExamples.decoratorTimer}><Prose>The fake clock deliberately reports a quarter second so you can reproduce the output. It is not a benchmark of sum. In normal use <Code>@timed()</Code> uses perf_counter and the elapsed output varies by machine and run. Injecting a clock makes the wrapper's calculation testable without sleeping.</Prose></PythonExample>
    <Prose>Finally records an elapsed interval on failure too. Logging/cleanup code must itself be reliable: a new exception in finally can replace the original one. Timing a generator function's call measures generator creation, not its later consumption. Similarly, calling an async function creates a coroutine; a wrapper must await it to measure its execution. This lesson's wrappers target ordinary synchronous functions.</Prose>
    <PythonExample example={decoratorExamples.decoratorCache}><Prose>The second call with 20 returns a cached result without running the body, so “compute 20” appears only once. The hit/miss counters make the reuse visible. maxsize bounds how many entries the least-recently-used cache keeps; clearing removes its stored results.</Prose></PythonExample>
    <Prose>Cache arguments must be hashable; passing a list raises TypeError. Cached results are shared objects, so mutating a returned list can affect later callers. Caching a function that reads a changing file, uses randomness or must always perform an action can give stale or skipped behaviour. A cache is not a persistence layer or a promise that concurrent misses execute only once.</Prose>
    <LessonTable caption="Before adopting a wrapper" headers={["Use", "Check first"]} rows={[
      ["Logging", "Do not expose secrets or enormous payloads. Preserve the error path."],
      ["Timing", "Measure the work you mean, and separate demonstration clocks from real measurements."],
      ["Caching", "Input identity, stale data, memory limits and mutable return values."],
      ["Retries", "Retry only defined transient failures, with limits; avoid duplicating irreversible actions."],
    ]} />

    <H2>5. Give a resource a clear lifetime</H2>
    <Prose>Opening a file creates an obligation to close it. A try/finally block can discharge that obligation, and a context manager packages the setup/cleanup pattern. The name after as receives the value returned by enter; it need not be the manager object itself.</Prose>
    <ContextLifetimeLab />
    <PythonExample example={decoratorExamples.contextClass}><Prose>The body reads one line, then fails. Python calls exit with ValueError and its details. Exit closes the text resource and returns False, so the original error continues to the outer handler. The final True verifies closure.</Prose></PythonExample>
    <LessonTable caption="What with does around the block" headers={["Phase", "Successful body", "Body raises an exception"]} rows={[
      ["Enter", "Call __enter__; bind its return value", "If enter itself fails, this manager's exit is not called"],
      ["Run body", "Execute statements, including return/break if present", "Unwind toward the manager"],
      ["Exit", "__exit__(None, None, None)", "__exit__(exception type, value, traceback)"],
      ["Continue", "Resume after the block or complete its return", "Propagate unless exit returns a truthy value"],
    ]} />
    <Prose>Returning True from exit suppresses the body's exception; it does not resume at the failing line. Execution continues after the with statement. Use suppression only when recovery is intentional. If setup partly acquires a resource and then enter raises, setup must clean up that partial acquisition itself. No Python cleanup scheme can promise execution after abrupt process termination or power loss.</Prose>
    <Prose>For an actual file, the normal pattern is <Code>with open("readings.txt", encoding="utf-8") as file:</Code>, with consumption inside the block. The file already provides a context manager; you do not need to write TextResource for it. File mode "w" creates or truncates output, so choose filenames and modes deliberately.</Prose>

    <H2>6. Write a generator-based context manager</H2>
    <PythonExample example={decoratorExamples.contextGenerator}><Prose>Before yield is acquisition. The yielded file becomes the as target. While suspended, the caller runs the block. On exit, the generator resumes and finally closes the file. The function must yield exactly once per context-manager invocation.</Prose></PythonExample>
    <Prose>An exception from the body is raised back into the generator at yield. If you catch it only to log it, re-raise it; otherwise the manager treats it as handled. A finally block is a good fit for unconditional cleanup. Create a fresh manager with <Code>text_resource(...)</Code> for each use; this generated manager instance is one-shot, not a reusable stream.</Prose>
    <Prose>Decorator syntax appears here because contextmanager transforms a generator function into a factory for context managers. This is not an ordinary data generator to iterate with for. Its single yield divides setup from cleanup rather than emitting a sequence of records.</Prose>

    <H2>7. Manage several resources safely</H2>
    <Prose>A report may read a variable number of inputs. If opening input 3 fails, inputs 1 and 2 still need closing. Keep a record of successful acquisitions, then unwind that record in reverse: the most recently entered resource leaves first. You can reason about this order without knowing how a stack data structure is implemented.</Prose>
    <ExitStackLab />
    <PythonExample example={decoratorExamples.contextStack}><Prose>A and B were acquired successfully. C fails before yielding, so its cleanup block was never entered. ExitStack unwinds the already-registered managers in reverse acquisition order: B, then A. The original acquisition error remains visible to the caller.</Prose></PythonExample>
    <Prose>For a fixed number of resources, multiple managers in one with statement are equivalent to nested blocks: enter left-to-right and exit right-to-left. ExitStack is useful when the number is dynamic. Its callback method can also register ordinary cleanup functions; register cleanup immediately after successful acquisition.</Prose>
    <LessonTable caption="Related contextlib helpers" headers={["Helper", "When it helps", "Caution"]} rows={[
      ["closing(obj)", "An object has close but not a context-manager protocol", "The block still needs to own that resource."],
      ["nullcontext(obj)", "Caller-owned resource needs no cleanup by this function", "Make the ownership policy explicit."],
      ["suppress(SpecificError)", "A narrowly defined failure is safe to ignore", "Do not hide all exceptions."],
      ["redirect_stdout(buffer)", "Capture printed output in a small test or script", "Changes global stdout; not a thread-local logging system."],
      ["AsyncExitStack / asynccontextmanager", "Asynchronous acquisition and cleanup", "Use async with and appropriate await behaviour."],
    ]} />

    <H2>8. Practise and check</H2>
    <H3>Independent investigation: restore a temporary setting</H3>
    <Prose>During one experiment, temporarily use another units label in a settings dictionary. Write <Code>temporary_value(settings, key, value)</Code> as a context manager. It must restore the previous binding after success or failure. If the key was absent, make it absent again. An existing None value must remain distinguishable from an absent key. Nested uses on the same key must restore the surrounding experiment's value before restoring the original.</Prose>
    <LessonTable caption="Checks for a temporary binding" headers={["Starting state / action", "Inside", "After exit"]} rows={[["units = C; temporarily F","F","C"],["absent debug; temporarily True","True","debug absent"],["limit = None; temporarily 10","10","limit present, None"],["outer F, inner K","K inside inner","F after inner, C after outer"],["body raises ValueError","changed binding until cleanup","binding restored; error still propagates"]]} />
    <details><summary>Hint: remember presence as well as value</summary><Prose>Create a unique sentinel with object(), remember settings.get(key, sentinel), then install the new binding. Yield the settings object inside try. In finally, restore the old value or remove the key. Do not catch and silently suppress the body's exception.</Prose></details>
    <details><summary>Complete solution and reasoning</summary><PythonExample example={decoratorExamples.restore}/><Prose>Each invocation stores its own original binding. The inner invocation remembers F, so its cleanup restores F; the outer remembers C. finally runs while the ValueError travels outward, and the outer handler still sees it. This changes one dictionary binding, not the objects referenced by other values. It is suitable for this local single-threaded experiment, not a global configuration mutation shared by concurrent requests.</Prose></details>
    <Checkpoint prompt="Change the initial units value to None, then repeat the nested experiment. Also delete a newly introduced debug key inside the block. What should cleanup do?"><Prose>The original units key must remain present with value None. A unique missing sentinel makes that distinguishable from absence. Removing debug during the body must still leave it absent afterward; the solution's pop handles an already-missing temporary key.</Prose></Checkpoint>
    <H3>Further transfer: calls, returns and visible output</H3>
    <details><summary>Hint for the counted decorator task</summary><Prose>Give the wrapper a calls attribute initialized at decoration time. Increment before the original call to count attempts, including failures. Return the original result, use wraps, and capture printed output separately from the returned value.</Prose></details>
    <Checkpoint prompt="Write a counted decorator that forwards keyword arguments, preserves metadata and exposes a calls counter. Capture a report's printed output without changing its returned result.">
      <PythonExample example={decoratorExamples.decoratorPractice} />
      <Prose>The counter records attempted calls because increment happens before the function executes. Moving it after a successful call would count successes instead. This simple mutable counter is not a concurrency-safe metrics system. redirect_stdout affects output, not the return value.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Change TextResource.__exit__ to return True. What disappears from the output, and is the resource still closed?">
      <Prose>The “caught: bad reading” line disappears because exit suppresses the ValueError. The exit message still appears and closed remains True. Suppression changes error handling, not whether the body resumes at the failed statement.</Prose>
    </Checkpoint>
    <Checkpoint prompt="A file is opened inside __enter__, then a second setup operation raises before __enter__ returns. Can you rely on __exit__ to close the file?">
      <Prose>No. The failed manager has not entered successfully. Protect partial setup with try/except cleanup or an acquisition stack. Already-entered outer managers will still unwind.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Write three checks for the logged wrapper before using it elsewhere.">
      <Prose>Check that a valid call returns the original result, that a failing call raises the same exception type, and that the leave action occurs on both paths. Also check the preserved name/documentation. The next lesson shows how to organise checks into an executable suite.</Prose>
    </Checkpoint>
    <Prose>Continue to <a href="/learn/topic/testing-debugging-dependency-management">Testing, Debugging &amp; Dependency Management</a> to turn these promises into repeatable evidence.</Prose>
    <Sources alternatives={<LearningResources>
      <li><a href="https://www.youtube.com/watch?v=T8CQwGIsrx4">Geir Arne Hjelle: Introduction to Decorators — PyCon 2020 video tutorial</a> (<a href="https://pycon-archive.python.org/2020/schedule/presentation/75/">official tutorial page</a>) — a longer guided introduction to functions as objects and constructing decorators. Use after the first two sections; its scope is older synchronous Python.</li>
      <li><a href="https://dabeaz-course.github.io/practical-python/Notes/07_Advanced_Topics/03_Returning_functions.html">David Beazley: Returning Functions</a> — written closure and wrapper exercises. Its retained-variable examples offer another way to reason about why a returned inner function can still refer to its original function.</li>
      <li><a href="https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager">Python contextlib guide: contextmanager</a> — compact worked setup/yield/cleanup examples and the exception rules. Read the ExitStack section after the resource-order investigation.</li>
    </LearningResources>}>
      <li><a href="https://docs.python.org/3/library/functools.html">Python: wraps, lru_cache and cache behaviour</a></li>
      <li><a href="https://docs.python.org/3/library/contextlib.html">Python: context managers, ExitStack and cleanup utilities</a></li>
      <li><a href="https://docs.python.org/3/reference/compound_stmts.html#the-with-statement">Python language reference: with statement semantics</a></li>
    </Sources>
  </div>,
};
