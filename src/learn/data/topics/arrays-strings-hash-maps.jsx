import { Code, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { ArrayMovementLab, TextUnitsLab, HashBucketLab } from "../../components/lesson-labs/ArrayMapFoundationsLabs";
import PythonExample from '../../components/lesson-labs/PythonExample';
import { arrayMapExamples } from "../array-map-foundations-examples";
import { ArrayAddressFigure } from "../../components/lesson-labs/SystemsMechanismFigures";
import { DsaPractice } from '../../components/lesson-labs/DsaPractice.jsx';
import arrayPractice from '../practice/arrays-strings-hash-maps.js';

export default {
  title:'Arrays, Strings & Hash Maps',readTime:'~40 min read + 75 min practice',hasIntegratedGuide:true,
  content:()=> <div className="lesson-pilot">
    <LessonIntro prerequisites="Python indexing, loops, functions and dictionaries. The earlier NumPy lesson helps distinguish typed arrays from Python lists. The optional custom-key example uses classes from OOP." sections={[["1-choose-how-you-want-to-find-something","Three access patterns"],["2-an-array-turns-a-position-into-an-address","Arrays"],["3-make-room-and-count-the-work","Movement and cost"],["4-ask-what-one-character-means","Text"],["5-a-hash-map-finds-a-candidate-then-checks-it","Hash maps"],["7-build-a-small-search-index","Applications"],["8-find-the-first-unique-event","Practice"],["guided-dsa-practice","Guided LeetCode practice"]]}>
      Choose a collection by the operations it needs to support. Trace an insertion, explain why visible text and byte counts differ, follow a colliding key through a map, and combine these ideas into a search index and an independent event-analysis task.
    </LessonIntro>
    <H2>1. Choose how you want to find something</H2>
    <Prose>An instrument produces readings, labels and event IDs. You might ask for the tenth reading, whether two labels match, or how often event 18 occurred. All three questions involve stored data, but their useful access patterns differ. A <strong>data structure</strong> arranges information so particular operations are convenient; its choice affects both the meaning of your program and the work it performs.</Prose>
    <LessonTable caption="Start with the question, then choose the structure" headers={['Question','Useful structure','What locates the answer?']} rows={[
      ['What is at position 9?','Array or dynamic array','An integer position in a sequence.'],
      ['What text was recorded?','String','An ordered sequence of text units with a defined encoding at I/O boundaries.'],
      ['How many times did event 18 occur?','Hash map; Python dict','A key associated with a value, not a position.'],
    ]}/>
    <Prose>These combine naturally: a list preserves event order, a string records a label, and a dictionary maps each event ID to its count. Do not replace a sequence with a map merely because maps offer quick lookups: repeated equal keys name one entry, so that replacement can discard occurrences.</Prose>
    <Prose>Run each complete block as a standalone Python 3.12+ script. Only the standard library is needed. Labs use small explicit models; their storage policy is stated, and their timings are not performance measurements of your computer.</Prose>

    <H2>2. An array turns a position into an address</H2>
    <Prose>Imagine equally sized slots placed consecutively. If the first begins at byte address 1000 and each slot occupies 8 bytes, index 0 begins at 1000, index 1 at 1008, and index 3 at 1024. The rule is <strong>slot address = base + index × slot width</strong>. You can calculate the destination without checking earlier values. Valid indices for a sequence of length n run from 0 through n − 1.</Prose>
    <ArrayAddressFigure/>
    <Prose>A fixed-size array reserves a fixed number of slots. A <strong>dynamic array</strong> manages a backing array and replaces it with a larger one when needed. Its <strong>length</strong> counts live elements; its <strong>capacity</strong> counts available slots. Spare capacity is an implementation detail, not additional list elements you may index.</Prose>
    <Prose>In CPython, a list's backing array stores references to objects. The slots are regularly spaced even when one refers to a string and another to an integer; the objects themselves can live elsewhere. A contiguous numeric NumPy array can instead store equal-width values directly. NumPy views can also have nontrivial strides, as you saw earlier. “Array” alone does not tell you the entire storage contract.</Prose>
    <Prose>The OS lesson explains another boundary: a contiguous virtual range need not occupy consecutive physical RAM frames. Our slot-address calculation describes the program's addressing view. It does not bypass virtual-memory translation.</Prose>
    <PythonExample example={arrayMapExamples.sequence}><Prose>Insert places X before the old index 1, moving later positions. Pop removes and returns the element now at index 2. The half-open slice 1:3 includes indices 1 and 2; for a Python list it creates a new outer list containing the selected references. Indexing past the end raises IndexError, while slice bounds are clipped. Negative Python indices count from the end; they are language conveniences over the same ordered sequence.</Prose></PythonExample>
    <Checkpoint prompt="After inserting X at index 1 in [A, B, C], does index 2 still identify C? Does the variable that refers to the C object need to change?">
      <Prose>Index 2 now holds B; C moved to index 3. A position is not an enduring element identity. Moving a reference in the list does not mutate the object it references or rebind an unrelated variable.</Prose>
    </Checkpoint>

    <H2>3. Make room and count the work</H2>
    <Prose>To insert X between A and B, B and C must move one slot right. Move from the right end first: copying B over C before saving C would lose information. Deletion does the opposite, closing a gap by moving later elements left. Appending with spare capacity needs no shift, but appending to a full backing array may require copying every live element.</Prose>
    <ArrayMovementLab/>
    <Prose>The lab counts element copies, shifts and insertion writes; it excludes allocation and clearing discarded slots. Its growth rule doubles capacity. CPython uses its own growth policy, so the exact capacities in this model are explanatory choices. During a resize the old backing array remains available until the new copy is ready; logical element order must survive that transition.</Prose>
    <H3>Cost means how work grows with input</H3>
    <Prose>If finding a slot takes a bounded amount of work as length grows, we call that constant time, written <Code>O(1)</Code>. If an operation may inspect or move a number of elements proportional to length n, we call that linear time, <Code>O(n)</Code>. These describe growth under a cost model, not a fixed number of nanoseconds. Counting operations first makes the notation less mysterious.</Prose>
    <LessonTable caption="A dynamic array, with bounded-cost element accesses" headers={['Operation','Why this much work?','Growth']} rows={[
      ['Read or replace a valid index','Calculate one slot; no searching.','O(1)'],
      ['Find an unknown value','May compare against every element.','O(n), assuming bounded-cost equality'],
      ['Insert or delete near the front','Move most later references.','O(n)'],
      ['Append with spare capacity','Store the new reference.','O(1) for this append'],
      ['Append requiring a new backing array','Copy the existing elements, then append.','O(n) for this append'],
      ['Copy a slice containing k elements','Create k references in a new list.','O(k) time and additional slots'],
    ]}/>
    <details><summary>Go deeper: how can append be constant on average if one append copies everything?</summary>
      <Prose>Follow a doubling array from capacity 1 to 2, 4 and 8. Reaching eight elements copies 1 + 2 + 4 = 7 old elements during growth, plus eight insertion writes. More generally the geometric copy total stays below twice the final length. Spread that total across all appends: work per append is bounded on average over the sequence. This is <strong>amortized O(1)</strong>; it does not promise each individual append is quick or rely on random inputs. A latency-sensitive system may still care about the occasional large resize. Formal analysis and alternative growth policies come later.</Prose>
    </details>
    <Checkpoint prompt="A full array has 1,000 items. Does inserting at its front become cheap just because the new allocation has many spare slots?">
      <Prose>No. Capacity handles where the result can fit; preserving order still requires moving or copying the existing elements into their new positions. Spare capacity makes future appends easier, not arbitrary front insertions.</Prose>
    </Checkpoint>

    <H2>4. Ask what one character means</H2>
    <Prose>Text has several useful units. A <strong>Unicode code point</strong> identifies a character or related text element. A user-perceived character can contain multiple code points, such as an e followed by a combining accent. An <strong>encoding</strong> turns code points into bytes for a file or network; UTF-8 uses a variable number of bytes per code point. Therefore a cursor step, Python string index and byte offset need not identify the same boundary.</Prose>
    <TextUnitsLab/>
    <Prose>Python str is an immutable sequence of Unicode code points. Len counts code points, not UTF-8 bytes or all user-perceived characters. A visually identical café can be represented using a single é code point or an e plus a combining acute accent. Exact string equality compares their code-point sequences. <strong>Normalization</strong> can put canonically equivalent spellings into a common representation; NFC composes where the normalization rules allow. It preserves the accent rather than deleting it.</Prose>
    <PythonExample example={arrayMapExamples.unicode}><Prose>The label policy explicitly ignores edge whitespace and letter case, then uses NFC. Casefold is intended for caseless matching and can change length; it is not just “lowercase every ASCII letter.” Save the raw label too if display, audit or round-trip fidelity matters. Passwords, identifiers and linguistically sensitive search may require different policies; applying this transformation everywhere would merge distinctions a system might need.</Prose></PythonExample>
    <Prose>Immutability means replacing a character produces new string content; it does not update the original string in place. For many pieces of generated text, collect fragments and use str.join to express one assembly operation. Repeated concatenation can repeatedly copy growing contents; interpreter optimizations vary, so do not infer an exact runtime from the syntax alone. Encode/decode at explicit boundaries, and do not cut arbitrary UTF-8 byte positions as if each were a character boundary.</Prose>
    <Checkpoint prompt="A service allows at most five UTF-8 bytes. Will every Python string with len(text) == 5 fit? Will NFC make every visible character one code point?">
      <Prose>Neither claim holds. Even café uses five UTF-8 bytes for four code points; many five-code-point strings need more than five bytes. Some visible characters remain multi-code-point sequences after NFC. A user-interface cursor may need grapheme-cluster segmentation beyond the normalization shown here.</Prose>
    </Checkpoint>

    <H2>5. A hash map finds a candidate, then checks it</H2>
    <Prose>A map associates keys with values. A <strong>hash function</strong> computes a number from a key; a hash table uses that number to narrow where it searches. In our small model, key modulo bucket count selects a bucket: 18 modulo 4 is 2. Different keys can select the same bucket. This is a <strong>collision</strong>, an expected case that the data structure must handle correctly.</Prose>
    <Prose>The model stores a short sequence of entries in each bucket, called separate chaining. To look up 18, select bucket 2, compare candidate keys for equality, and stop at the equal key. A hash match alone never establishes key equality. To set an existing key, replace its value; to set a missing key, add a new entry.</Prose>
    <HashBucketLab/>
    <Prose>Changing from four to five buckets spreads our particular keys differently. That demonstrates sensitivity to capacity and distribution, not a proof that five buckets is always better. Python dict uses a different collision-resolution layout. Its behavior agrees on key/value semantics, but this diagram does not reproduce its internal slots.</Prose>
    <PythonExample example={arrayMapExamples.keys}><Prose>Get returns a default for an absent key without inserting it. The counting assignment then writes a new count. Updating 14 changes its value without creating a second equal key. Python's equal numeric keys 1, True and 1.0 refer to the same dictionary entry; if event IDs must distinguish types, validate the input schema rather than assuming the dictionary will do that for you.</Prose></PythonExample>
    <H3>Key rules are part of correctness</H3>
    <Prose>A dictionary key must be <strong>hashable</strong>: its hash must remain stable during its lifetime, and equal objects must have equal hashes. Unequal objects may share a hash. Python lists are unhashable; a tuple is hashable only if its elements are hashable. Do not mutate the fields involved in a custom key's equality while it is stored in a map. Otherwise even a stable hash can leave key identity inconsistent with the table's existing entries.</Prose>
    <details><summary>Go deeper: force collisions in a real Python dictionary</summary>
      <PythonExample example={arrayMapExamples.collisions}/>
      <Prose>All these custom keys deliberately return hash zero. Equality still distinguishes their numbers, so no distinct entry disappears. Equal-key replacement retains three keys. The fixture never mutates number after construction; a production key type should enforce that contract. NotImplemented lets Python handle equality with other types. This example establishes correctness under collisions, not desirable performance or Python's exact comparison count.</Prose>
    </details>

    <H2>6. Choose with costs and ordering in mind</H2>
    <Prose>With a suitable hash distribution and controlled occupancy, hash-table lookup and update take expected constant time for bounded-size keys. They can degrade to linear work when many candidates collide. Computing or comparing a long string can itself require work proportional to its length; calling the whole lookup “one operation” hides that cost. <strong>Load factor</strong> is entries divided by buckets/slots under the table's model. Resizing reduces occupancy, but moving entries costs work too.</Prose>
    <LessonTable caption="Similar-looking questions can require different guarantees" headers={['Need','Good starting point','Check before choosing']} rows={[
      ['Preserve repeated readings and their positions','List/dynamic array','Front insertion and searching by value can be linear.'],
      ['Count or associate values by ID','Dictionary','Equal keys merge; expected cost is not a worst-case promise.'],
      ['Track membership without associated values','Set','No sequence-position or insertion-order guarantee.'],
      ['Display keys in sorted order','Sort the keys explicitly','Python dict preserves insertion order, which is not sorted order.'],
    ]}/>
    <Prose>Replacing an existing dictionary value does not move its key to a new insertion position. Deleting and reinserting a key places it at the end. If the application needs continuously ordered range queries, a sorted/tree-based design may suit it better than repeatedly sorting a map; trees are introduced later. The later <a href="/learn/topic/hashing-collision-resolution-amortized-analysis">Hashing, Collision Resolution & Amortized Analysis</a> lesson develops probing, resize policies and the assumptions behind the bounds.</Prose>

    <H2>7. Build a small search index</H2>
    <Prose>A useful combination is an <strong>inverted index</strong>: instead of asking every document whether it contains a word for every query, record which document IDs contain each word. The map finds a word; its associated set holds matching IDs. This is a first step toward search systems and also works for finding which experiments contain a tag or which logs mention an event.</Prose>
    <PythonExample example={arrayMapExamples.index}><Prose>Enumerate gives each document a position-based ID. Setdefault returns an existing posting set or installs an empty one; add includes the current ID. Document 0 contains red twice, but a set records its presence once. Set intersection, written &amp;, retains IDs present in both sets, so red AND fox returns document 0. Sorted is only for stable display. Try replacing fox with missing: the empty default set yields no matches.</Prose></PythonExample>
    <Prose>This deliberately small index splits on whitespace and distinguishes letter case and punctuation. It does not yet handle phrase positions, ranking, language-aware tokenization or updates to document IDs. Those features need explicit contracts; reusing a position-based ID after reordering documents would invalidate existing postings. For a changing collection, stable document IDs matter just as element identity mattered after array insertion.</Prose>

    <H2>8. Find the first unique event</H2>
    <Prose><strong>Independent task:</strong> given a reusable list of integer event IDs, return the first ID occurring exactly once, preserving original order. Return None if there is none. For [7, 2, 7, 9, 2, 4] the answer is 9, not the smallest ID and not the last unique ID. Include empty input, all repeated IDs and the legitimate answer zero. Design your own approach before opening either disclosure.</Prose>
    <details><summary>Hint: separate the two questions</summary><Prose>First find each ID's total frequency. Then ask which element of the original sequence is the earliest with frequency one. A set alone loses the distinction between one occurrence and multiple occurrences. A one-shot iterator would be exhausted by the first pass; this task deliberately accepts a reusable sequence.</Prose></details>
    <details><summary>Complete solution and why it works</summary><PythonExample example={arrayMapExamples.unique}/><Prose>After each first-pass iteration, counts contains exact frequencies for the processed prefix. At completion it contains frequencies for the entire input. The second pass visits original positions in order, so its first frequency-one match is precisely the requested answer. With expected constant-cost map operations on these bounded-size integer IDs, time is O(n) and extra space is O(k) for k distinct IDs. None is an absence marker under this integer-only contract; never use truthiness to discard a valid zero.</Prose></details>
    <Checkpoint prompt="What should [4, 1, 4, 2, 1, 3] return? Why is returning the first ID when you initially see it incorrect?">
      <Prose>Return 2. An ID that appears once in the prefix may repeat later, so you need enough information about the rest of the input before certifying it unique. If data arrives forever, the question needs a different completion or window contract.</Prose>
    </Checkpoint>
    <Prose>You can move on when you can trace shifted positions, state which text unit you count, explain why collisions do not overwrite unequal keys, and choose a structure by its operations. Next in the opening curriculum, <a href="/learn/topic/linked-lists-stacks-queues">Linked Lists, Stacks & Queues</a> replaces moving array slots with changing links and develops last-in/first-out and first-in/first-out behavior.</Prose>
    <DsaPractice practice={arrayPractice}/>
    <Sources alternatives={<LearningResources>
      <li><a href="https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-2-data-structures-and-dynamic-arrays/">MIT 6.006: Data Structures and Dynamic Arrays — lecture video</a>. A second explanation of sequence interfaces and resizing. Watch after the movement lab; its mathematical runtime analysis is a deeper pass, not a prerequisite for the opening explanation.</li>
      <li><a href="https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-4-hashing/">MIT 6.006: Hashing — lecture video</a> and <a href="https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/ce9e94705b914598ce78a00a70a1f734_MIT6_006S20_lec4.pdf">matching lecture notes</a>. Develop the dictionary interface and collision assumptions; universal hashing is an optional advanced continuation.</li>
      <li><a href="https://docs.python.org/3/howto/unicode.html">Python's Unicode HOWTO — written tutorial</a>. Revisit encoding, combining characters and normalization with additional examples. Useful when a visually identical label fails an exact comparison.</li>
    </LearningResources>}>
      <li><a href="https://docs.python.org/3/tutorial/datastructures.html">Python tutorial: data structures</a> — list operations, sets, dictionary use and iteration.</li>
      <li><a href="https://docs.python.org/3/library/stdtypes.html#mapping-types-dict">Python dictionary contract</a> — hashable keys, equal numeric keys, defaults and insertion order.</li>
      <li><a href="https://docs.python.org/3/faq/design.html#how-are-lists-implemented-in-cpython">Python FAQ: CPython list implementation</a> — variable-length arrays of references; implementation-specific storage rather than a rule for every Python runtime.</li>
      <li><a href="https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize">Python normalization API</a> — canonical and compatibility forms; NFC is the policy used here.</li>
    </Sources>
  </div>,
};
