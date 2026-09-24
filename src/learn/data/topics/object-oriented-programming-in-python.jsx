import { Code, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import PythonExample from "../../components/lesson-labs/PythonExample";
import { OopBindingLab, OopLookupLab, OopValidationLab, OopCompositionLab, OopRecordDiagram } from "../../components/lesson-labs/oop-foundations-labs";
import { oopExamples } from "../oop-foundations-examples";
import { BoundMethodRetentionFigure } from "../../components/lesson-labs/BoundMethodFigure.jsx";

export default {
  title: "Object-Oriented Programming in Python",
  readTime: "~42 min read + 80 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot oop-lesson">
    <LessonIntro prerequisites="Python names, lists, dictionaries, functions, loops and exceptions. You only need small numbers; no previous class design or machine learning is assumed."
      sections={[["1-start-with-two-independent-logs", "Why objects"], ["2-follow-one-call-into-one-object", "Objects & self"], ["3-locate-state-before-changing-it", "Shared state"], ["4-make-valid-changes-the-easy-path", "Rules & properties"], ["5-connect-objects-through-small-contracts", "Composition"], ["6-distinguish-record-values-from-object-identity", "Data records"], ["9-build-and-test-an-independent-design", "Put it together"]]}>
      Two experiments record temperatures at different times. An update to one log must not silently change the other. Build those independent logs, see where Python sends a method call, protect their rules, and connect them to interchangeable reports. Finish by designing a small dataset split of your own.
    </LessonIntro>

    <H2>1. Start with two independent logs</H2>
    <Prose>You already know how to transform a list: a function can take [18, 24] and return its mean, 21. Now the task has a history. A morning log starts empty, receives readings, and answers questions about what it has collected. An evening log does the same job with different data. The data that persists between operations is called <strong>state</strong>.</Prose>
    <Prose>We want three things: each log owns its readings, all logs support the same operations, and invalid readings cannot sneak in through an ordinary update. Start with tools you already know. A dictionary holds each log's fields; a function receives the particular dictionary to change.</Prose>
    <PythonExample example={oopExamples.functions}><Prose>Each call to <Code>new_log</Code> evaluates a new <Code>[]</Code>, so each dictionary refers to a different list. Passing morning into add tells the function which log to change. Evening remains empty. This is a valid design for a small program.</Prose></PythonExample>
    <Prose>A <strong>class</strong> defines a kind of object, including its operations. An <strong>instance</strong> is one particular object made from that class. We will define ReadingLog once and make morning and evening instances. Each keeps its own data while obtaining the same behaviour from the class.</Prose>
    <Prose>The useful shift is from <Code>add(morning, 18)</Code> to <Code>morning.add(18)</Code>: name the object, then ask it to do something. A function accessed in this way is a <strong>method</strong>. The dot also lets us read an <strong>attribute</strong>, such as <Code>morning.name</Code>. An attribute is a value reached through an object; it can refer to data or provide behaviour.</Prose>
    <LessonTable caption="Keep the entities distinct" headers={["Entity", "Its job", "In our logs"]} rows={[
      ["Name", "Refers to an object", "morning, evening, alias"],
      ["Class", "Defines the kind and its shared behaviour", "ReadingLog"],
      ["Instance", "One object with its own identity", "The morning log, object A in the lab"],
      ["State", "Data describing an object now", "Its name and current readings"],
      ["Method call", "Runs an operation for a receiver", "morning.add(18)"],
    ]} />
    <Prose><strong>Run the examples:</strong> use Python 3.10 or later, with no extra packages. Each displayed program is complete and runs independently. Save it under its shown filename and run the accompanying command in a terminal. If your machine uses <Code>python3</Code> or <Code>py</Code> to select Python 3, substitute that command. The examples here were checked with Python 3.11.7.</Prose>
    <Checkpoint prompt="You only need to turn a list of Celsius temperatures into Fahrenheit once. What would a class add to the task?">
      <Prose>Probably very little. A function returning a converted list makes the input/output relationship clear. Persistent state, several related operations or a meaningful replaceable interface can justify a class; the label “object-oriented” is not itself a reason to add one.</Prose>
    </Checkpoint>

    <H2>2. Follow one call into one object</H2>
    <Prose>Here is the same task as a class. The first version accepts any value so that the reference mechanism stays visible. We will strengthen its input contract after understanding which object a method changes.</Prose>
    <PythonExample example={oopExamples.instances}><Prose>The initial morning mean is (18 + 24) / 2 = 21. The empty evening log returns None: no mean is available. Alias then reaches the same object as morning and adds 30, giving (18 + 24 + 30) / 3 = 24. The final booleans answer identity questions: alias and morning refer to one object; evening is different.</Prose></PythonExample>
    <H3>Read the class one mechanism at a time</H3>
    <Prose><Code>class ReadingLog:</Code> begins the definition. The indented functions define its methods. Calling <Code>ReadingLog("morning")</Code> creates an instance and calls <Code>__init__</Code> to initialise it. The double underscores mark a name with a meaning defined by Python. In ordinary classes, creation itself is handled by <Code>__new__</Code>; <Code>__init__</Code> sets up an already-created instance and must return None.</Prose>
    <Prose><Code>self</Code> names that instance inside the method. Python supplies it when we call an instance method through an object. The name is a convention, not a keyword and not a global “current object.” Each call gets its own local parameter. <Code>self.name = name</Code> stores the argument on that particular instance. <Code>self.values = []</Code> creates and stores a fresh list for that instance.</Prose>
    <Prose>Contrast <Code>values = []</Code> inside a function: that binds a local name which later calls do not automatically see. The <Code>self.</Code> tells Python to put or find the attribute on the object. In <Code>self.values.append(value)</Code>, first find self's values list, then ask that list to append the argument.</Prose>
    <OopBindingLab />
    <H3>A bound method remembers the receiver</H3>
    <Prose>Accessing <Code>morning.add</Code> produces a <strong>bound method</strong>: a pairing of the class's add function and the morning instance. Calling it supplies that instance as the function's first argument. For this ordinary method, <Code>morning.add(18)</Code> performs the same call as <Code>ReadingLog.add(morning, 18)</Code>. The class does not need a separate copy of the function for every log.</Prose>
    <Prose>That is also why <Code>alias = morning</Code> does not copy a log. It creates another reference to the existing object. Changing the object through either reference is visible through both. Calling ReadingLog again creates a new instance; assigning another name does not.</Prose>
    <Checkpoint prompt="After the complete example, run evening.add(99). What are both means? What changes if you use alias.add(99) instead?">
      <Prose>Evening becomes 99.0; morning stays 24.0. Using alias instead changes morning to (18 + 24 + 30 + 99) / 4 = 42.75, and evening remains empty. Identify the receiving object before doing the arithmetic.</Prose>
    </Checkpoint>
    <details className="oop-deeper"><summary>Deeper experiment: save a method, then reassign its old name</summary>
      <Prose>Predict whether the saved method follows the name morning when that name is reassigned to evening. Inspect <Code>__self__</Code>, the remembered receiver, and <Code>__func__</Code>, the underlying function.</Prose>
      <BoundMethodRetentionFigure />
      <PythonExample example={oopExamples.bound}><Prose>The saved method still refers to the original morning object. Reassigning the name morning does not rewire that saved reference. Its call changes the old object's list, while the name morning now reaches evening's empty list.</Prose></PythonExample>
    </details>

    <H2>3. Locate state before changing it</H2>
    <Prose>A class can also have attributes. A shared constant such as <Code>unit = "C"</Code> in the class body can be appropriate. A single shared readings list usually is not: it would mix independent experiments. The difference is <em>where the list is created and which attribute refers to it</em>.</Prose>
    <Prose>For the ordinary data attributes in this example, Python first looks for values on the instance. If it is absent there, lookup continues on the class and then its base classes. This is a useful starting model; properties and other descriptors have additional rules, which we will identify when we introduce a property.</Prose>
    <PythonExample example={oopExamples.shared}><Prose>Initially neither instance has its own values attribute. Both lookups reach the class list, and append mutates that same list. Assigning <Code>b.values = [99]</Code> creates an instance attribute on b that <strong>shadows</strong> the class attribute: later b.values finds the new list first. A's append still changes the class list. The final checks inspect each instance's attribute dictionary, <Code>__dict__</Code>, and confirm which one owns values.</Prose></PythonExample>
    <OopLookupLab />
    <H3>Choose what is shared deliberately</H3>
    <Prose>The repair for independent logs is the earlier initialiser: create a new list with <Code>self.values = []</Code> every time. Merely writing <Code>self.values.append(...)</Code> does not guarantee per-instance data; attribute lookup may still lead to a class list.</Prose>
    <Prose>There is another sharing decision when a caller supplies a list. <Code>self.values = supplied</Code> keeps that caller's list. <Code>self.values = list(supplied)</Code> makes a new outer list with the same elements. This is a <strong>shallow copy</strong>: appending to one outer list no longer changes the other, but a nested mutable element can still be shared. Decide what isolation your class promises instead of assuming a constructor always copies input.</Prose>
    <Checkpoint prompt="Repair this initialiser: def __init__(self, values=[]): self.values = values. Why is moving [] into a default argument insufficient?">
      <Prose>A default argument is evaluated when the function is defined, so calls that omit values reuse that one list. Use <Code>def __init__(self, values=None):</Code> and then <Code>self.values = [] if values is None else list(values)</Code>. This creates a new outer list for each instance, including when input is supplied. It still does not deep-copy nested objects.</Prose>
    </Checkpoint>

    <H2>4. Make valid changes the easy path</H2>
    <Prose>Our first class groups data and operations, but anyone can add text or infinity. The mean can then fail or mislead us. A useful class has a <strong>contract</strong>: what callers may pass, what an operation does, and what happens when it cannot do that job. An <strong>invariant</strong> is a condition that every supported valid state preserves.</Prose>
    <Prose>For this teaching log, stored readings must be finite floats. The public add method accepts int or float inputs, excluding bool; converts them to float storage; checks finiteness; then appends. Strings such as "24" are rejected instead of silently parsed. The file-reading lesson taught parsing external text: that conversion belongs at an explicit input boundary, before passing a measurement into this class.</Prose>
    <Prose>Validation must happen <em>before</em> mutation. A rejected addition leaves existing readings untouched. This example represents already-decoded temperatures in Celsius; checking finite numeric values does not prove that a sensor is accurate or that its range is scientifically plausible.</Prose>
    <OopValidationLab />
    <H3>Give callers a supported way to inspect state</H3>
    <Prose>The internal list will be called <Code>_values</Code>. The underscore communicates “implementation detail; use the supported interface.” The public <Code>values</Code> attribute will return a tuple snapshot. A tuple's membership cannot be changed, so a caller cannot append through the public result and bypass add's checks.</Prose>
    <Prose><Code>@property</Code> is a decorator: a line that changes how the following method is exposed. A property getter is called when the attribute is read, so callers write <Code>log.values</Code>, not <Code>log.values()</Code>. The getter below constructs the snapshot. No setter is provided, so assignment to <Code>log.values</Code> raises AttributeError.</Prose>
    <PythonExample example={oopExamples.validated}><Prose>The failed additions produce different exceptions for different contract failures, but none adds a reading. The saved snapshot remains (18.0,) after the internal list grows. The current snapshot is (18.0, 24.0), the count is 2 and the mean is 21.0.</Prose></PythonExample>
    <LessonTable caption="Why each check exists" headers={["Check or operation", "What it prevents or exposes", "What it does not establish"]} rows={[
      ["isinstance(value, bool)", "Booleans are a subclass of int, but True is not a reading in this contract.", "All integer-like scientific scalar types are not automatically accepted."],
      ["float(value), with OverflowError handled", "An arbitrarily large Python int may not fit in float storage.", "Conversion does not preserve every integer exactly."],
      ["math.isfinite(converted)", "NaN and positive/negative infinity are excluded.", "The reading is not thereby calibrated or within a valid device range."],
      ["Append after validation", "A failed add leaves the previous list unchanged.", "A multi-reading batch is not automatically all-or-nothing."],
      ["tuple(self._values)", "Public inspection does not expose the mutable list.", "Building a snapshot takes time proportional to its length."],
    ]} />
    <H3>Use Python's built-in operations when their meaning fits</H3>
    <Prose><Code>__len__</Code> supplies the count for <Code>len(log)</Code>. <Code>__repr__</Code> supplies a useful developer-facing string for <Code>repr(log)</Code>; the <Code>!r</Code> inside the formatted string shows the name's own representation, including quotes. This is a <strong>protocol</strong>: a defined set of operations through which an object participates in Python behaviour. You do not need to inherit a special “length class” to give your object a sensible length.</Prose>
    <Prose>If the object needs a separate user-facing display, <Code>__str__</Code> can provide it. Define these methods because their meaning helps a caller. Do not make arbitrary operations available merely because Python permits it.</Prose>
    <Checkpoint prompt="The latest log contains 18 and 24. You save old = log.values, then try log.add(float('nan')). What are old, log.values and len(log) after catching the exception?">
      <Prose>Old and the current snapshot are both (18.0, 24.0), and len(log) is 2. The finite-value gate raises before append. This tests the invariant and failure behaviour, not just the successful path.</Prose>
    </Checkpoint>
    <details className="oop-deeper"><summary>Deeper limits: encapsulation, setters, numeric range and batch operations</summary>
      <Prose>Python still lets determined callers access <Code>_values</Code>. Encapsulation means an explicit supported interface and maintained rules; it is not a security boundary. A double-leading underscore triggers name mangling to reduce subclass name collisions, not true secrecy.</Prose>
      <Prose>A property is a descriptor: an object whose hooks control attribute access. The simple “instance first, class next” diagram from the shared-list experiment is not the complete descriptor lookup algorithm. A property can take precedence over an ordinary instance-dictionary entry. A <Code>@name.setter</Code> method can validate assignments when replacement is part of the public contract; we intentionally omit that operation here.</Prose>
      <Prose>Our mean implementation is appropriate for these small temperature examples. Finite inputs alone do not ensure finite intermediate arithmetic: summing huge finite floats can overflow. Numerical range, rounding and robust aggregation require their own analysis. The input policy also intentionally excludes Decimal and some NumPy scalar types; supporting them requires a deliberate conversion and precision policy.</Prose>
      <Prose>If you later add <Code>add_many</Code>, decide whether a bad reading rejects the whole batch or whether earlier valid readings remain. Repeated calls to add naturally permit partial progress. To promise an all-or-nothing batch, validate and convert into temporary storage first, then extend the internal list after every input passes.</Prose>
    </details>

    <H2>5. Connect objects through small contracts</H2>
    <Prose>A report must show the same Celsius input in different formats. The report's job is to add a label. A separate formatter's job is to turn a Celsius number into temperature text. The report <em>has a formatter</em>; it is not a kind of formatter. Giving one object a reference to another object it uses is <strong>composition</strong>.</Prose>
    <Prose>Define the shared contract before making the implementations: for the small temperatures used here, <Code>format(celsius)</Code> accepts an already-validated Celsius number and returns a string with a numeric value and unit. CelsiusFormatter preserves the unit; FahrenheitFormatter uses F = C × 9/5 + 32. For 20°C, that is 20 × 9/5 + 32 = 68°F.</Prose>
    <Prose>The report does not need to know the conversion formula. It calls the operation promised by the collaborator and uses the returned text. Different objects responding to the same operation is <strong>polymorphism</strong>. Python often uses <strong>duck typing</strong>: rely on the supported behaviour rather than require a shared parent class.</Prose>
    <OopCompositionLab />
    <PythonExample example={oopExamples.composition}><Prose>Both formatters work with exactly the same Report code. The recording formatter in the second half is a small test double: it records the input it receives and returns predictable text. The assertions independently check that Report passes the original Celsius value and adds its label around the formatter's returned text. They do not re-test the conversion formula.</Prose></PythonExample>
    <H3>An interface is a promise about meaning</H3>
    <Prose>A method called format is not enough. A collaborator that secretly expects Fahrenheit input or returns a bare number violates this report's contract. Name, input interpretation, output type, errors and side effects all matter. Keep a formatter free of unexpected state changes unless its contract explicitly allows them.</Prose>
    <Prose>A class is useful for a formatter when it owns configuration or fits a wider interface. For this tiny example, passing a plain formatting function would also be a good design. Composition is the separation of responsibilities, not a requirement that every responsibility become a class.</Prose>
    <Checkpoint prompt="A colleague proposes CelsiusMorningLog, FahrenheitMorningLog, CelsiusEveningLog and FahrenheitEveningLog subclasses. How would you reduce this design?">
      <Prose>Keep the time-specific readings in ReadingLog instances. Keep the formatting choice in a separate collaborator. A report uses the chosen formatter with the data it is given. The two independent choices no longer require a new subclass for every combination.</Prose>
    </Checkpoint>
    <details className="oop-deeper"><summary>Deeper branch: typed interfaces and behavioural tests</summary>
      <Prose><Code>typing.Protocol</Code> describes a structural interface for a static type checker: a class can satisfy it by providing compatible operations without inheriting from it. An abstract base class can declare required abstract methods and prevent instantiation until those methods are implemented. Both help express a contract, but neither proves the implementation's semantic correctness.</Prose>
      <Prose>Test each responsibility at its boundary. For FahrenheitFormatter, check known conversions including 0°C → 32°F and 20°C → 68°F. For Report, use a predictable collaborator and check delegation and labelling. For ReadingLog, check independent instances, empty state, accepted values, rejected inputs and unchanged state after failure.</Prose>
      <Prose>The Python basics lesson introduced <Code>assert</Code>: it raises AssertionError when its condition is false. Here it is a lightweight reproducible test. Python can omit assertions under optimisation, so do not use assert as production validation for untrusted inputs. The later testing topic builds a maintained test suite around these same behavioural questions.</Prose>
    </details>

    <H2>6. Distinguish record values from object identity</H2>
    <Prose>Sometimes an object's main job is to carry named fields. A single reading might contain a Celsius value and tags. Writing a custom initialiser, display and field-by-field equality for every such record is repetitive. A <strong>dataclass</strong> can generate that routine code.</Prose>
    <Prose>In the following class, <Code>celsius: float</Code> and <Code>tags: list[str]</Code> are type annotations: declarations of intended field types for readers and tools. They do not make Python check those types at runtime. <Code>field(default_factory=list)</Code> tells the generated initialiser to call list for each missing tags argument, so each instance gets its own list.</Prose>
    <OopRecordDiagram />
    <PythonExample example={oopExamples.dataclass}><Prose>The generated equality compares fields for instances of the same type. Initially the two records have equal fields, so == is True, while is is False because there are two objects. After only a's tags change, their fields differ. With these default dataclass options, a mutable value record is unhashable: it cannot be used directly as a dictionary key or set element.</Prose></PythonExample>
    <Prose><strong>Identity</strong> asks “is this the very same object?”; <strong>equality</strong> asks “do these objects compare equal according to their type's rules?” Ordinary user-defined classes such as our ReadingLog retain the default identity-based equality unless a method changes it. Do not use <Code>is</Code> to compare numbers or text values; their representation and reuse can be implementation-dependent. Use <Code>is None</Code> when checking the unique None sentinel.</Prose>
    <Checkpoint prompt="If alias = a and alias.tags.append('checked'), which record changes? Would replacing the dataclass with a normal class automatically change the alias relationship?">
      <Prose>A changes through the alias; b remains independent. Aliasing is about references, regardless of whether the class was decorated as a dataclass. The decorator changes generated methods, not the meaning of assignment.</Prose>
    </Checkpoint>
    <details className="oop-deeper"><summary>Deeper experiment: frozen does not mean recursively immutable</summary>
      <Prose><Code>frozen=True</Code> blocks ordinary field reassignment. It does not freeze a mutable object stored inside a field, and does not validate a type annotation. Predict all three outputs before running this small counterexample.</Prose>
      <PythonExample example={oopExamples.frozen}><Prose>Appending changes the list object without reassigning the tags field. Directly reassigning celsius is blocked. Passing text still succeeds because no runtime validation was written. For an immutable record contract, choose immutable field values and add validation where needed, for example in <Code>__post_init__</Code>.</Prose></PythonExample>
      <Prose>A <strong>hash</strong> is a value used to locate keys in dictionaries and sets. Equal keys must have equal hashes, and a key's hash must remain stable while stored. Mutable equality-defining fields make that contract difficult. A frozen dataclass may receive a generated hash, but hashing can still fail if a participating field is unhashable. Do not add <Code>unsafe_hash=True</Code> simply to suppress an error.</Prose>
      <Prose>The common dataclass options in these examples work on Python 3.10+. More recent releases have changed details of generated equality for unusual values such as NaN; our finite simple records do not rely on those edge cases. Consult the documentation for your runtime before depending on generated-method internals.</Prose>
    </details>

    <H2>7. Use inheritance when a subtype keeps the promise</H2>
    <Prose><strong>Inheritance</strong> lets a class use and specialise behaviour from a base class. It fits when the specialised object can be substituted where the base kind is expected. A calibrated sensor can still support a sensor's interface while adding a correction offset. A report, however, is not a kind of sensor; composition fits that relationship better.</Prose>
    <details className="oop-deeper"><summary>Deeper branch: a complete subclass, super and substitution</summary>
      <Prose>The base Sensor promises a correction operation that takes a raw Celsius reading and returns a corrected Celsius reading. Its default correction is unchanged input. CalibratedSensor applies a fixed offset instead. This is an illustrative correction model, not a calibration procedure or a claim that every real sensor has a constant error.</Prose>
      <PythonExample example={oopExamples.inheritance}><Prose>The base initialiser establishes name; the subclass then adds offset. <Code>correct</Code> is overridden: Python selects the subclass's implementation for this instance. <Code>describe</Code> extends the base description using super. Adding −0.5 to 24 produces 23.5 in the same unit.</Prose></PythonExample>
      <Prose><Code>super()</Code> delegates according to the <strong>method resolution order</strong>, or MRO: the ordered class search used to resolve inherited behaviour. In this single-inheritance example, the next class is Sensor, followed by object. With multiple inheritance, “my parent” is an incomplete mental model; cooperative calls and compatible signatures matter.</Prose>
      <Prose>Substitution requires more than an <Code>isinstance</Code> result. If a subclass suddenly rejects inputs the base accepts, returns an unrelated type, changes units without saying so or introduces surprising side effects, a caller may break. Inherit a meaningful contract, not just convenient lines of code. Multiple inheritance, descriptors and metaclasses are further topics, not requirements for a well-designed small class.</Prose>
    </details>
    <Checkpoint prompt="A FahrenheitSensor subclass returns Fahrenheit from correct(raw), although the base Sensor promises a corrected Celsius value. What goes wrong when a caller adds the result to Celsius readings?">
      <Prose>The code can run while mixing units and producing an invalid result. The subclass breaks the output-meaning contract. Preserve Celsius inside the sensor interface and put Fahrenheit conversion at an explicit formatting/conversion boundary.</Prose>
    </Checkpoint>

    <H2>8. Choose methods by what information they need</H2>
    <Prose>Most methods in this lesson operate on a particular object and therefore receive self. Two other standard forms are useful once that mechanism is clear: a class method receives the class, and a static method receives no automatic first argument.</Prose>
    <details className="oop-deeper"><summary>Deeper branch: an alternative constructor and a related utility</summary>
      <PythonExample example={oopExamples.methods}><Prose>The instance method reads one object's celsius state. <Code>from_fahrenheit</Code> is an alternative constructor: convert the input, then call cls to create an instance of the receiving class. <Code>unit</Code> needs neither an instance nor a class; a module-level constant would also be reasonable here.</Prose></PythonExample>
      <LessonTable caption="What is automatically supplied?" headers={["Definition", "First argument supplied by Python", "Appropriate responsibility"]} rows={[
        ["def method(self, ...)", "The receiving instance", "Read or change this object's state."],
        ["@classmethod; def method(cls, ...)", "The receiving class", "Alternative construction or class-level behaviour."],
        ["@staticmethod; def method(...)", "Nothing", "A closely related utility independent of instance/class state."],
      ]} />
      <Prose>Like self, cls is a conventional parameter name. Using cls in an inherited factory can preserve the receiving subclass, provided that subclass supports the required constructor arguments. The <a href="/learn/topic/decorators-context-managers">decorators lesson</a> explains how these transformations work beneath the @ notation.</Prose>
    </details>

    <H2>9. Build and test an independent design</H2>
    <section className="oop-mission" aria-label="Independent dataset split investigation">
      <h3>Keep dataset membership independent; swap its report</h3>
      <p>A data project has a training split and an empty validation split. A split is a named collection of sample IDs, not the data files themselves. Design it without copying the reading log line for line: the valid values and useful operations are different.</p>
      <p><strong>Input:</strong> source = ["s1", "s2"], a train split built from it, and validation built from []. An ID must be a nonblank string; duplicates within one split are rejected. IDs are compared exactly: do not silently trim or change case.</p>
      <ol>
        <li>Create <Code>DatasetSplit(name, sample_ids)</Code>. Each split owns its membership list. Appending "s3" to the caller's source list must not change train.</li>
        <li>Provide <Code>add(sample_id)</Code>, read-only tuple snapshots through <Code>sample_ids</Code>, and <Code>len(split)</Code>. A rejected input must preserve the prior membership.</li>
        <li>Add "s4" to train. Reject a repeated "s1". Validation must remain empty. Explain which references are shared and which are independent.</li>
        <li>Build a CountFormatter and NamesFormatter with the same <Code>format(split)</Code> contract. A SplitReport receives a formatter and delegates to it. Support the empty case explicitly.</li>
        <li>Write assertions for independence, failure preservation and both report styles. Passing these checks demonstrates the requested behaviours, not every possible defect.</li>
      </ol>
      <details><summary>Hint: identify the invariant and the ownership boundary</summary><p>Build a fresh internal list and route constructor inputs through the same validation as later additions. Because the members are strings, a fresh outer list gives the membership isolation needed here. Validate type, blankness and duplication before append. Ask the split for its public snapshot and count when formatting.</p></details>
      <details><summary>Show one complete solution and expected output</summary>
        <PythonExample example={oopExamples.mission}><Prose>The constructor iterates over the supplied IDs into a fresh list; it does not keep the caller's list. Rejecting the duplicate happens before mutation. The report delegates without choosing between formatter types. NamesFormatter explicitly labels an empty collection instead of emitting an ambiguous trailing colon. A dictionary plus carefully designed functions could satisfy the same contract; this exercise practises the object boundaries just taught.</Prose></PythonExample>
      </details>
      <details><summary>Transfer: prove the design on changed inputs</summary>
        <p>Try duplicate IDs in the constructor. Try adding 7, "" and three spaces to an existing split. Save a snapshot before an accepted addition. Add a LinesFormatter that returns one ID per line or "(empty)". Then explain whether this class alone guarantees that training and validation sets never overlap.</p>
        <details><summary>Check your transfer reasoning</summary><p>Duplicate constructor input raises ValueError, so construction does not return a valid split to the caller. Adding 7 raises TypeError; blank text raises ValueError; an existing split keeps its prior contents after each rejection. A previously returned tuple snapshot remains unchanged after a later addition. LinesFormatter can use <Code>"\n".join(split.sample_ids) or "(empty)"</Code>, and requires no SplitReport changes.</p><p>Two individually valid splits may still contain the same ID. Cross-split disjointness is a different invariant owned by a dataset/partition operation that can inspect both. Do not claim data-leakage protection from a class that only enforces uniqueness within one list.</p></details>
      </details>
    </section>

    <H3>Choose the smallest design that keeps the contract clear</H3>
    <LessonTable caption="Practical starting points, not rigid rules" headers={["Need", "Start with", "Reason"]} rows={[
      ["Compute a result from inputs", "Function", "No persistent object state is necessary."],
      ["Carry a few named values", "Dictionary or dataclass", "Make the record easy to inspect."],
      ["Maintain state across related operations", "Class with a small public interface", "Put validation and supported changes at one boundary."],
      ["Replace one independent responsibility", "Composition with a function or object", "Change the collaborator without multiplying subclasses."],
      ["Specialise a genuine substitutable kind", "Inheritance", "Preserve the base contract while extending behaviour."],
    ]} />
    <Prose><strong>Before moving on:</strong> explain what self receives without saying “the class”; predict a shared-list bug by tracing lookup; distinguish identity from equality; reject a bad input without changing state; and swap a report's collaborator without editing the report. Revisit the relevant investigation if any answer still requires guessing.</Prose>
    <Prose>Next in this module, <a href="/learn/topic/iterators-iterables-generators">Iterators, Iterables &amp; Generators</a> shows how custom objects participate in loops and keep traversal state. Decorators and context managers then add behavior and control resource lifetimes; testing turns your contracts into repeatable checks. Later, Pandas moves from individually modelled records to tables. On a focused path, follow the reader's named Next link for its selected module topics.</Prose>
    <Sources>
      <li><a href="https://docs.python.org/3/tutorial/classes.html#method-objects">Python tutorial: bound methods, class/instance variables and inheritance</a></li>
      <li><a href="https://docs.python.org/3/reference/datamodel.html#customizing-attribute-access">Python data model: attribute access, descriptors and special methods</a></li>
      <li><a href="https://docs.python.org/3/library/functions.html#property">Python built-ins: property and its getter/setter interface</a></li>
      <li><a href="https://docs.python.org/3/library/dataclasses.html">Python dataclasses: generated equality, default factories, frozen and hash behaviour</a></li>
      <li><a href="https://docs.python.org/3/library/typing.html#typing.Protocol">Python typing: structural interfaces with Protocol</a></li>
      <li><a href="https://docs.python.org/3/library/abc.html">Python abstract base classes and abstract methods</a></li>
      <li><a href="https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement">Python language reference: assert and optimisation</a></li>
    </Sources>
  </div>,
};
