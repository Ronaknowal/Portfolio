import { Code, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from "../../components/lesson-labs/PythonExample";
import { CursorOwnershipLab, GeneratorFrameLab, PullPipelineLab } from "../../components/lesson-labs/IteratorLabs.jsx";


import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { iterationExamples } from "../iterator-examples.js";

export default {
  title: "Iterators, Iterables & Generators",
  readTime: "~34 min read + 60 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot">
    <LessonIntro prerequisites="Loops, functions, return and exceptions from Python Basics; object identity and simple classes from Object-Oriented Programming. File ownership is introduced here and developed in the next lesson."
      sections={[["1-ask-for-one-reading-at-a-time", "The problem"], ["2-the-iteration-protocol", "The protocol"], ["3-yield-pauses-instead-of-finishing", "Watch yield"], ["5-build-a-streaming-pipeline", "Build it"], ["8-write-your-own-iterable", "Custom objects"], ["10-practise-and-check", "Practise"]]}>
      Follow data from source to consumer, distinguish a reusable collection from a one-pass cursor, build a streaming pipeline and batches, and avoid exhaustion and resource-lifetime bugs.
    </LessonIntro>
    <H2>1. Ask for one reading at a time</H2>
    <Prose>For three temperatures, keeping a list is easy. For a large file, a live feed or an unknown number of records, constructing every intermediate list can waste memory and delay the first useful result. Often the next stage needs only the next item. Iteration separates the source of data from the code consuming it.</Prose>
    <Prose>Think of a collection as a book and an iterator as a reading position. A book can support several independent reading positions. A cursor advances as you read; asking again does not reset it. A generator is one convenient way to implement such a cursor in Python.</Prose>
    <LessonTable caption="Three terms, three different roles" headers={["Term", "Promise", "Example"]} rows={[
      ["Iterable", "iter(obj) can obtain an iterator", "A list, range, file, or generator"],
      ["Iterator", "next(obj) produces an item or raises StopIteration; iter(obj) returns itself", "iter([18, 21]), an open file, a generator"],
      ["Generator", "A particular iterator implemented by a generator function/expression", "A function with yield, or (f(x) for x in source)"],
    ]} />
    <Prose>Every generator is an iterator and every iterator is iterable, but a list is not itself an iterator. An iterable need not be reusable: a generator is iterable but one-pass. <Code>range</Code> is a reusable, compact sequence, not a generator. Being memory-efficient does not automatically make an object a generator.</Prose>
    <Prose>All programs use the standard library and were checked with Python 3.12.14. Save each independently as <Code>lesson.py</Code> and run <Code>python lesson.py</Code> in a terminal. The language features used work on Python 3.10+; the reference notes distinguish newer library additions. The examples print small collections so you can inspect them; do not convert a huge or infinite real-world stream into a list just to debug it.</Prose>

    <H2>2. The iteration protocol</H2>
    <Prose>Calling <Code>iter(values)</Code> asks the list for a cursor, while <Code>next(cursor)</Code> asks that cursor for one item. Assignment only adds another name for an object: <Code>b = a</Code> does not create another cursor. This is the object-identity rule from OOP applied to a changing reading position.</Prose>
    <CursorOwnershipLab />
    <PythonExample example={iterationExamples.iterProtocol}><Prose>The first next consumes 18. list(cursor) consumes the remaining 21 and 24. A second list call gets nothing because that same cursor is exhausted. The original list is still intact and can provide a new cursor. The optional second argument to next is returned on exhaustion instead of raising StopIteration.</Prose></PythonExample>
    <Prose>A for loop conceptually obtains <Code>iter(source)</Code>, repeatedly calls next, assigns each returned item to the loop variable, and stops on StopIteration. That exception is the normal end-of-iteration signal, not an ordinary item. Other exceptions are not swallowed by the loop; they propagate unless your code handles them.</Prose>
    <Prose>An iterator must remain exhausted once it has signalled exhaustion. Some iterables provide fresh independent iterators every time; others return themselves. Check the source's contract before planning multiple passes. Using <Code>next(cursor, None)</Code> is ambiguous if None could be real data; use a distinct sentinel or catch StopIteration when that distinction matters.</Prose>
    <Checkpoint prompt="You write total = sum(stream), then count = sum(1 for _ in stream). Why might count be zero?">
      <Prose>The first sum consumed a one-pass stream. Compute total and count in the same loop, create a fresh source for each pass, or deliberately materialise a bounded dataset. Do not assume a variable containing data can always be traversed twice.</Prose>
    </Checkpoint>

    <H2>3. Yield pauses instead of finishing</H2>
    <Prose>A normal function runs when called and returns one result. A function containing yield returns a generator object when called. Its body begins on the first request for an item. At yield it supplies a value and suspends, retaining local variables and its execution position. The next request resumes just after that suspension.</Prose>
    <PythonExample example={iterationExamples.iterYield}><Prose>“created” appears before “start” because constructing stream does not execute the generator body. The “finish” message appears only when the caller asks after the second yield. Merely receiving the last yielded item does not yet run the remaining body.</Prose></PythonExample>
    <GeneratorFrameLab />
    <PythonExample example={iterationExamples.frame}><Prose>The GEN_ state labels come from Python's inspection library. Two successful next calls still leave the generator suspended. A third request discovers the end and runs cleanup. The unstarted generator closes without printing another cleanup line because it never entered that try block.</Prose></PythonExample>
    <Prose><Code>return</Code> inside a generator ends iteration; reaching the end does the same. Do not manually raise StopIteration inside a generator to end it: an unhandled StopIteration escaping its body is converted to RuntimeError. Use return. Ordinary validation errors can emerge later, at consumption time, because execution was delayed.</Prose>
    <Checkpoint prompt="What gets printed if you create readings() but never call next or loop over it?">
      <Prose>Nothing from inside readings. Creation gives you the generator object; it does not run the body. Any print statements in the calling code still run normally.</Prose>
    </Checkpoint>

    <H2>4. Eager and lazy transformations</H2>
    <PythonExample example={iterationExamples.iterExpressions}><Prose>The list comprehension calls double for both values before “ready”. The generator expression delays calls to double until consumption. Asking for one item does one unit of work; list(lazy) then consumes what remains. Repeating list(lazy) returns an empty list.</Prose></PythonExample>
    <Prose>A generator expression uses parentheses; a list comprehension uses square brackets. A generator expression's leftmost iterable expression is evaluated immediately, but its item transformations are delayed. Lazy does not mean every subexpression waits, that the input was never allocated, or that errors cannot occur during construction.</Prose>
    <Prose>If the upstream source already holds a million values in a list, wrapping it in a generator does not remove that list. The generator may also retain references to upstream objects. Memory savings come from avoiding unnecessary materialised intermediate results, not from the syntax alone. Lazy code is not automatically faster: producing every item still does the work, and generator bookkeeping has a cost.</Prose>
    <LessonTable caption="Choose eager or lazy deliberately" headers={["Need", "Useful choice", "Trade-off"]} rows={[
      ["One pass over large input", "Generator or iterator pipeline", "No random indexing or automatic rewind."],
      ["Repeated access, indexing or sorting", "A bounded list", "Stores its elements in memory."],
      ["A running total/count", "A loop with accumulators", "Retains only the state needed for the summary."],
      ["First few values of a long source", "islice or an early-stopping loop", "Upstream work occurs only as far as consumption requires."],
    ]} />

    <H2>5. Build a streaming pipeline</H2>
    <Prose>A pipeline is a chain of small transformations. With these synchronous generators, the chain does no work on its own: the final consumer pulls, each stage asks its input for what it needs, and a result comes back. A filter may discard several source items before producing one output. “One next” therefore does not mean “one source read.”</Prose>
    <PullPipelineLab />
    <PythonExample example={iterationExamples.pull}><Prose>The first two results require lines 1, 2 and 3. Line 2 contributes work but no value. islice stops requesting after two results; it does not rewind or close the original stream. The final list resumes that stream and reads line 4. Change "24" to "bad": requesting one result still succeeds, but asking for the second reaches a ValueError.</Prose></PythonExample>
    <Prose>We will read text lines, remove blanks, convert numbers and compute a mean in one pass. StringIO supplies a tiny in-memory file so the example runs without downloading or creating a data file. The same functions accept an open text file because they depend only on iteration over lines.</Prose>
    <PythonExample example={iterationExamples.iterPipeline}><Prose>Each request pulls a value through the stages: the consumer asks numeric_readings, which asks non_empty_lines, which asks the file for another line. The blank line is skipped internally. The consumer receives 18, 24 and 30, so count is 3 and mean is 24.0. No separate list of all cleaned strings or parsed numbers is built.</Prose></PythonExample>
    <LessonTable caption="Trace the small input through each stage" headers={["Source line", "After strip/filter", "Value reaching the consumer"]} rows={[
      ["18 followed by newline", '"18"', "18.0"],
      ["Blank line", "Discarded", "No value yielded"],
      ["24 followed by newline", '"24"', "24.0"],
      ["30 followed by newline", '"30"', "30.0"],
    ]} />
    <Prose>For a real file, replace the with line with <Code>with open("readings.txt", encoding="utf-8") as file:</Code> and keep consumption inside the block. Iterating a text file yields lines; <Code>file.read()</Code> instead reads all remaining text at once. A single enormous line can still use substantial memory, so “line-by-line” is not an absolute memory bound.</Prose>
    <Prose><Code>with</Code> closes the file on block exit, including when an exception occurs. Creating a generator inside the block and returning it to consume later can leave it trying to read a closed file. The example prints True after the block to verify closure. For missing files, malformed numbers or changing input, define an explicit error policy; this pipeline lets conversion errors reach the caller instead of silently dropping bad measurements.</Prose>
    <Prose>The mean accumulator retains a total and a count rather than the observations. For ordinary finite inputs its extra state is essentially constant-size, apart from growth in integer representation. Our StringIO source still stores the entire example string; the demonstration shows incremental processing, not a measurement of large-file memory use.</Prose>

    <H2>6. Produce bounded batches</H2>
    <Prose>A downstream operation may need several items together, such as an API request or model batch. We want to accumulate at most one batch at a time, work with any iterable rather than just a sliceable list, and keep a final short batch rather than silently losing it.</Prose>
    <PythonExample example={iterationExamples.iterBatches}><Prose><Code>islice(cursor, size)</Code> requests at most size items from the shared cursor. Converting only that slice to a tuple forms one batch. An empty tuple means the source has ended. The final (6,) is a one-item tuple; its comma matters. The function rejects non-positive and non-integer sizes.</Prose></PythonExample>
    <Prose>Because this function contains yield, even its initial validation runs on first consumption, not merely when batches is called. The example forces consumption with list inside the try. If your API requires immediate validation, use an ordinary outer function to validate and return an inner generator.</Prose>
    <Prose>The batcher retains roughly one batch of references; a consumer that collects every batch into a list will still retain all of them. Decide explicitly whether an incomplete last batch is kept, padded or rejected. Python's standard library also has <Code>itertools.batched</Code> from Python 3.12; the implementation here makes the policy and the Python 3.10-compatible mechanism visible.</Prose>

    <H2>7. Small tools that compose</H2>
    <PythonExample example={iterationExamples.iterTools}><Prose>count can run forever; islice limits it to four items before list consumes it. chain concatenates sources without constructing a combined source list. pairwise yields overlapping adjacent pairs. map transforms items, filter keeps those passing a test, and enumerate adds positions. The lambda is a one-expression function returning whether its input is positive.</Prose></PythonExample>
    <Prose>zip is lazy too: it pairs inputs and normally truncates at the shortest. strict=True detects mismatched lengths during consumption, not necessarily at construction. Pairwise and strict zip are available in the Python 3.10+ baseline used here.</Prose>
    <LessonTable caption="Useful additional tools and the trap to remember" headers={["Tool", "Use", "Boundary"]} rows={[
      ["itertools.accumulate", "Yield running totals or another cumulative combination", "It still consumes the source once."],
      ["itertools.tee", "Make several cursors from one input", "Buffers values while consumers advance at different speeds; buffering may grow large."],
      ["itertools.groupby", "Group consecutive equal keys", "Not global SQL-style grouping; group iterators share the underlying source."],
      ["itertools.cycle", "Repeat a finite input", "Caches that input; not a no-storage solution."],
      ["sorted", "Order a finite iterable", "Eagerly collects values; do not apply to an infinite source."],
    ]} />
    <Prose>Prefer a comprehension over a complicated map/lambda expression when it is easier to read. Do not mutate a source list while relying on its iterator position. Also remember that lazy expressions can observe changes to names or objects made before consumption; use stable inputs or an explicit snapshot when that timing would be surprising.</Prose>
    <details className="lesson-deeper"><summary>Why two consumers can quietly turn a stream into stored history</summary>
      <Prose>Suppose a live dashboard reads quickly but an audit summary reads slowly. tee lets them consume the same one-pass source independently. To replay a value for the slower reader, tee must retain it after the faster reader advances. The gap between positions explains the storage requirement; tee has not made the original source rewindable.</Prose>
      <PythonExample example={iterationExamples.tee}><Prose>After a has read 18 and 21, b gets those values without another “source produced” line. They were retained for b. This shows replay behavior, not an exact buffer-size measurement: Python may allocate storage in blocks and retain extra references. A reader that never catches up can force growing storage. Consume the tee outputs rather than separately advancing their original source.</Prose></PythonExample>
      <Checkpoint prompt="A consumes a million records while b stays at the start. Can you justify a constant-memory claim for this pipeline?"><Prose>No. Those records must remain available to b. Either keep readers reasonably aligned, accept bounded materialization, or redesign the task as one pass that updates both summaries.</Prose></Checkpoint>
    </details>

    <H2>8. Write your own iterable</H2>
    <Prose>A generator is usually the shortest implementation. A custom iterator class is useful when the cursor also needs other methods or explicit inspectable state. The core contract is small: iter returns the cursor itself, and next advances or raises StopIteration.</Prose>
    <PythonExample example={iterationExamples.iterCustom}><Prose>Countdown stores one cursor's remaining count, so it is one-pass. CountdownSource stores the recipe and creates a fresh Countdown on every iter call, making repeated loops independent. The two uses of list(source) succeed; the second use of list(cursor) is empty.</Prose></PythonExample>
    <Prose>Special methods connect the class to the language's loop syntax; callers do not normally invoke __next__ directly. Our small countdown assumes an integer start and treats non-positive starts as empty. A production class should validate that input contract explicitly if it accepts arbitrary external values.</Prose>

    <H2>9. Delegation, cleanup and two-way generators</H2>
    <H3>Delegate a sub-iterator with yield from</H3>
    <PythonExample example={iterationExamples.iterDelegate}><Prose>yield from forwards items from each group. This flattens one level, not an arbitrarily nested structure. A string is also iterable, so supplying a string group would yield individual characters. The second example explicitly closes a started generator, causing its finally block to run.</Prose></PythonExample>
    <Prose>Breaking out of a for loop does not, by itself, guarantee prompt closure of a still-referenced generator. Do not rely on garbage-collection timing for files or locks. Own the resource in a with block, or use an explicit cleanup protocol. Closing a generator that never started does not first run its entire body to reach a finally block.</Prose>
    <H3>Optional deeper mechanism: send and a final return value</H3>
    <Prose>Most data pipelines only ask generators for values. A suspended yield can also receive a value from the caller through send. Learn this after ordinary iteration; it is not required to batch a file.</Prose>
    <PythonExample example={iterationExamples.iterSend}><Prose>The first next starts the generator and yields total 0. send(3) resumes the suspended yield expression with value 3, updates the total and runs to the next yield. send(4) does the same. Here None is a deliberate stop command, not a reading. The generator's return value is carried by StopIteration.value; it is not yielded as an extra item.</Prose></PythonExample>
    <Prose>A just-created generator must first be started with next or send(None) before you send a non-None value. <Code>throw</Code> can inject an exception at the suspension point; <Code>close</Code> requests termination via GeneratorExit. A generator should clean up and terminate rather than yield in response to that close request. Async generators use a different asynchronous protocol and belong with asynchronous programming, not this synchronous streaming path.</Prose>

    <H2>10. Practise and check</H2>
    <H3>Independent investigation: stop an alarm scan at the first useful result</H3>
    <Prose>A sensor source can be read only once. Write <Code>first_crossing(values, threshold)</Code> to return the first <Code>(index, value)</Code> strictly above the threshold. None ends the feed; zero is valid data. Return None if the source ends first. Crucially, do not consume readings after a crossing. Assume finite numeric readings or None; this task is about consumption, not validating every possible sensor encoding.</Prose>
    <LessonTable caption="Success checks before viewing the solution" headers={["Input / threshold", "Result", "Consumption check"]} rows={[["[0, 18, 24, 30] / 20","(2, 24)","30 remains in the same cursor"],["[0, None, 24] / 20","None","24 remains after the consumed sentinel"],["[] / 20","None","No exception"],["[20, 21] / 20","(1, 21)","Equality alone does not trigger"]]} />
    <details><summary>Hint: let demand define the stopping point</summary><Prose>Use one loop and enumerate for position. Test identity with None before comparing to the threshold. Return immediately when a result is known. Calling list(values) first would consume later readings before you could stop.</Prose></details>
    <details><summary>Complete solution and reasoning</summary><PythonExample example={iterationExamples.alarm}/><Prose>Each iteration consumes exactly one candidate. The function returns as soon as that candidate settles the question; no look-ahead occurs. It is an ordinary consumer function returning one result, not a generator yielding many. A real monitoring system also needs policies for stale readings, units and noisy values; a threshold crossing alone does not diagnose a physical condition.</Prose></details>
    <Checkpoint prompt="Change the source to [24, 0, None, 30] with threshold 20. What is returned, and what remains? Then perform two consecutive searches on that same cursor."><Prose>The first answer is (0, 24); remaining values are 0, None and 30. Without first materializing them, a second search consumes 0 then None and returns None, leaving 30. enumerate starts a new local index for each call; these are positions within each search, not durable sensor IDs.</Prose></Checkpoint>
    <details><summary>Further transfer: keep partial results as a stream</summary><Prose>For the tasks below, start with a one-pass cursor and verify empty input. Running means assume finite numeric readings; floating-point addition can accumulate rounding error on long streams. Large-scale numerical stability is a separate numerical-methods question.</Prose>
    <Checkpoint prompt="Write take_until_missing(values), which stops at the first None but keeps zero. Then write running_mean(values), yielding the mean after every new reading. Both must support empty input.">
      <PythonExample example={iterationExamples.iterPractice} />
      <Prose>The first function uses identity with None, not truthiness, so zero survives. Return stops the generator immediately. The second updates total/count once per value: the means of [18], [18, 24], and [18, 24, 30] are 18, 21 and 24. Neither solution needs to keep all prior values.</Prose>
    </Checkpoint>
    </details>
    <Checkpoint prompt="Your batch input contains seven items and size is three. How many batches should you see, and when is size=0 rejected?">
      <Prose>Three batches of sizes 3, 3 and 1. In the shown generator implementation, size=0 raises ValueError when you first consume it. Merely storing batches(source, 0) does not execute the validation.</Prose>
    </Checkpoint>
    <Checkpoint prompt="You create stream from an open file, exit the with block, then call list(stream). What should you change?">
      <Prose>Consume the stream while the file is open, or redesign the producer so resource ownership encloses the entire iteration lifetime with explicit cleanup. Moving just generator creation into with does not extend the file's lifetime.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Does replacing every list with a generator guarantee less memory use and identical behaviour?">
      <Prose>No. Iterators are one-pass and change when work/errors happen. The source may already be materialised, tee/cycle can buffer, and downstream consumers may collect all results. Choose the representation according to reuse, indexing, latency and resource-lifetime needs.</Prose>
    </Checkpoint>
    <H3>Ready for the next topic?</H3>
    <Prose>You should now be able to predict exactly when the generator body runs, build and consume a one-pass pipeline, explain its memory boundary, and close its resources safely. Next is <a href="/learn/topic/decorators-context-managers">Decorators &amp; Context Managers</a>, followed by testing/debugging and then the scientific Python libraries.</Prose>
    <Sources alternatives={<LearningResources>
      <li><a href="https://cs50.harvard.edu/python/weeks/9/">CS50 Python, David Malan: Week 9 lecture video</a> and <a href="https://cs50.harvard.edu/python/notes/9/#generators-and-iterators">Generators and Iterators notes</a> — a beginner alternative comparing a growing list with yield. The complete lecture covers other topics; use its generator section. Retain the precise distinction from this lesson: calling the generator function creates the iterator, and yield suspends synchronous execution. This is the 2022 course; newer APIs need current references.</li>
      <li><a href="https://dabeaz-course.github.io/practical-python/Notes/06_Generators/03_Producers_consumers.html">David Beazley: Producers, Consumers and Pipelines</a> — an intermediate written workshop extending generators into a stock-ticker stream. Exercises depend on earlier course files; work through its preceding generator setup before trying them.</li>
    </LearningResources>}>
      <li><a href="https://docs.python.org/3/tutorial/classes.html#iterators">Python tutorial: iterators and generators</a></li>
      <li><a href="https://docs.python.org/3/library/stdtypes.html#iterator-types">Python reference: the iterator contract</a></li>
      <li><a href="https://docs.python.org/3/library/itertools.html">Python reference: itertools tools and buffering behaviour</a></li>
      <li><a href="https://docs.python.org/3/reference/expressions.html#generator-expressions">Python language reference: generator expression timing</a></li>
      <li><a href="https://docs.python.org/3/reference/expressions.html#generator-iterator-methods">Python language reference: next, send, throw and close</a></li>
    </Sources>
  </div>,
};
