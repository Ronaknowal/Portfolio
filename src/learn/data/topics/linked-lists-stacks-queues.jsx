import { Code, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { DsaPractice } from '../../components/lesson-labs/DsaPractice.jsx';
import linkedPractice from '../practice/linked-lists-stacks-queues.js';
import { LinkedReversalLab, BracketStackLab, CircularQueueLab } from "../../components/lesson-labs/LinkedFoundationsLabs";
import PythonExample from '../../components/lesson-labs/PythonExample';
import { linkedExamples } from "../linked-foundations-examples";

export default {
  title:'Linked Lists, Stacks & Queues',readTime:'~40 min read + 90 min practice',hasIntegratedGuide:true,
  content:()=> <div className="lesson-pilot">
    <LessonIntro prerequisites="Arrays, Strings & Hash Maps for sequence operations and growth costs; Python functions and OOP for the small Node and RingQueue classes. No pointer arithmetic or recursion is required." sections={[["1-separate-the-rule-from-the-storage","Behavior versus storage"],["2-follow-references-instead-of-positions","Linked lists"],["3-reverse-links-without-losing-the-rest","Reversal"],["4-a-stack-remembers-what-must-finish-first","Stacks"],["5-a-queue-preserves-arrival-order","Queues"],["6-combine-lookup-with-recency","LRU application"],["7-remove-one-node-with-a-clear-contract","Practice"],["guided-dsa-practice","Guided LeetCode practice"]]}>
      Follow node references, preserve the unprocessed part while reversing links, detect incorrectly nested brackets, and operate a bounded queue across wraparound. Then use these ideas for recency-based caching and a moving window of readings.
    </LessonIntro>
    <H2>1. Separate the rule from the storage</H2>
    <Prose>You undo the most recent edit first, but serve the oldest waiting print job first. Both store items, yet they promise different removal orders. A <strong>stack</strong> removes the last item added; a <strong>queue</strong> removes the first. These describe behavior. A <strong>linked list</strong> describes a storage organization: each node holds a value and references to other nodes.</Prose>
    <LessonTable caption="An interface says what happens; an implementation says how" headers={['Idea','Removal/access rule','Possible implementation']} rows={[
      ['Stack: last in, first out (LIFO)','Push at the top; pop the newest item.','End of a dynamic array, or head of a singly linked list.'],
      ['Queue: first in, first out (FIFO)','Enqueue at the back; dequeue from the front.','Linked head/tail nodes, circular array or deque.'],
      ['Linked sequence','Follow references to reach the next element.','Nodes allocated independently, with next and possibly previous links.'],
    ]}/>
    <Prose>Choosing a stack does not require choosing linked nodes. Python's list is useful as a stack; collections.deque supports efficient operations at both ends. We will implement small structures to understand their invariants, then use library structures for applications. All complete programs use only Python 3.12+ and its standard library.</Prose>

    <H2>2. Follow references instead of positions</H2>
    <Prose>In a singly linked list, a <strong>node</strong> contains a value and one next reference. The <strong>head</strong> reference identifies the first node; None means no node. The last node's next is None. Nodes do not have to be adjacent in memory. A reference leads to the next node wherever it is stored.</Prose>
    <div className="nt-flow"><span>head → A<small>value 4 · next B</small></span><span>Node B<small>value 7 · next C</small></span><span>Node C<small>value 9 · next None</small></span></div>
    <Prose>A, B and C are node identities in our drawings, not array indices or actual memory addresses. Changing A.next changes one relationship; it does not move B's stored value into A. Two nodes can hold equal values and still be different nodes. As in OOP, two references can also reach the same object, so a mutation may be visible through both names.</Prose>
    <H3>What does a local edit require?</H3>
    <Prose>To insert X after a known node A, first set X.next to A.next, then set A.next to X. This preserves the old suffix. To remove the node after a known predecessor A, set A.next to A.next.next, after checking that the node exists. Removing the first node instead updates head. These local changes are constant in number, but finding A by value or position can still require walking the whole list.</Prose>
    <LessonTable caption="State the condition behind each cost" headers={['Operation on a singly linked list','Cost','Condition']} rows={[
      ['Read element at position i','O(i + 1)','Start at head and follow links; there is no slot-address formula.'],
      ['Find a value','O(n) worst case','Equality has bounded cost; target may be absent.'],
      ['Insert after a known node','O(1) link changes','Already have the node; count allocation separately if relevant.'],
      ['Remove after a known predecessor','O(1) link changes','Already have the predecessor and check its successor exists.'],
      ['Append using a maintained tail','O(1) link changes','Maintain both head and tail correctly, including the empty case.'],
      ['Remove the last node','O(n) in general','A tail reference alone does not identify its predecessor.'],
    ]}/>
    <Prose>A <strong>doubly linked list</strong> also stores previous links, enabling direct unlinking of a known node when both directions and endpoints are maintained. A <strong>circular list</strong> links the end back into the sequence, so traversal needs a termination rule other than “until None.” These are useful variants, not automatic upgrades: extra references and independently allocated nodes cost memory, and following links can have worse locality than visiting consecutive array slots. Measure real workloads before claiming one is faster.</Prose>
    <Checkpoint prompt="A singly linked list stores a million items, and you only know a target's value. Is deletion O(1)?">
      <Prose>The local relink can be O(1), but discovering the matching node and its predecessor may take O(n). A claim about one step is not a claim about the complete operation. Also define whether you remove the first match or every match.</Prose>
    </Checkpoint>

    <H2>3. Reverse links without losing the rest</H2>
    <Prose>Reversing [4, 7, 9] in place should make the same nodes reachable in the order [9, 7, 4]. No new value nodes are needed. The trap is overwriting a next reference before remembering where the remaining sequence starts. The lab displays all original nodes for diagnosis, including nodes no longer reachable from head.</Prose>
    <LinkedReversalLab/>
    <H3>Keep a fact true after every iteration</H3>
    <Prose>An <strong>invariant</strong> is a fact that remains true as an algorithm progresses. Here previous identifies the already reversed prefix, and current identifies the untouched suffix. At the loop boundary, these two chains account for every original node exactly once. Initially the reversed prefix is empty and current is the original head.</Prose>
    <div className="nt-flow"><span>Save following<small>Remember the remaining suffix</small></span><span>Reverse current.next<small>Attach this node to the prefix</small></span><span>Advance both references<small>Prefix grows; suffix shrinks</small></span></div>
    <Prose>When current becomes None, the suffix is empty and previous reaches the whole reversed list. That explains both why the loop stops and why it returns previous. Testing examples helps; the invariant explains the general mechanism.</Prose>
    <PythonExample example={linkedExamples.reverse}><Prose>Node stores a value and next reference. Build prepends input values in reverse order to construct their original order. Values walks nodes and tracks identities to diagnose accidental cycles. Reverse itself uses only a few references: O(n) time and O(1) additional working space. The display helper's result list and diagnostic set do use O(n) extra space; do not attribute their cost to the reversing function.</Prose></PythonExample>
    <Checkpoint prompt="What happens if the caller runs reverse(head) but ignores the returned reference? What if you set current.next = previous before saving the old next?">
      <Prose>The caller's old head still reaches the old first node, which is now the last; it does not become the new head automatically. Overwriting next before saving it can lose the remaining suffix. A one-node test might hide both mistakes, so test a multi-node case and check every original node remains reachable exactly once.</Prose>
    </Checkpoint>

    <H2>4. A stack remembers what must finish first</H2>
    <Prose>In ([…]), the square bracket must close before the surrounding parenthesis can close. The most recently opened unfinished group is the next one that needs attention: exactly the stack rule. <strong>Push</strong> adds a new unfinished opening; <strong>pop</strong> removes and returns the top. <strong>Peek</strong> inspects the top without removing it. Popping an empty stack needs an explicit error or rejection policy.</Prose>
    <BracketStackLab/>
    <Prose>The invariant is that the stack contains precisely the unmatched openings in the processed prefix, with the newest on top. On an opening, push it. On a closing, require a nonempty stack whose top is the matching opening; then pop. At the end, require an empty stack. Merely counting equal numbers of opening and closing symbols cannot detect the crossed nesting in ([)].</Prose>
    <PythonExample example={linkedExamples.brackets}><Prose>List.append and pop at the end implement the stack. The partner dictionary maps each closing symbol to the required opening. Python's short-circuit or avoids popping when the stack is empty. Characters outside these six bracket symbols are ignored. Each symbol is visited once; time is O(n) with amortized list-end operations, and extra space is O(d) for maximum nesting depth d, which can reach n.</Prose></PythonExample>
    <Prose>This is a bracket recognizer, not a complete Python parser: quotes, comments and escape sequences change whether a visible bracket is syntax. For actual code, tokenize or parse under the language's rules first. The same newest-unfinished-first idea also appears in function-call return state and undo histories. An undo system usually needs stored inverse actions or earlier state, not merely a stack of action names.</Prose>
    <Checkpoint prompt="Why is an empty input balanced, but (() is not? Would a queue of openings correctly accept ([]) ?">
      <Prose>Empty input has no unmatched obligations. (() leaves an opening on the stack at the end. A FIFO queue would compare the ] with the oldest opening (, rejecting a valid nesting; the required matching order is last in, first out.</Prose>
    </Checkpoint>

    <H2>5. A queue preserves arrival order</H2>
    <Prose>A work queue accepts A, then B, then C. Serving A should leave B before C. Enqueue adds at the back and dequeue removes the front. A Python list can express this with append and pop(0), but each front removal shifts remaining references. A deque avoids that repeated movement for end operations. Deque is short for “double-ended queue,” so it can also support stack behavior.</Prose>
    <H3>Reuse fixed slots with a circular buffer</H3>
    <Prose>A bounded queue can reserve an array once. Store head, the physical index of the oldest live item, and size, the number of live items. The next insertion index is <Code>(head + size) % capacity</Code>. Modulo gives the remainder: for capacity 4, index 4 wraps to 0 and index 5 wraps to 1. Logical FIFO order can cross the end of the physical array without moving stored items.</Prose>
    <CircularQueueLab/>
    <Prose>The invariants are 0 ≤ size ≤ capacity and: logical item i occupies (head + i) modulo capacity. Empty means size is zero; full means size equals capacity. In both cases the next insertion index equals head, so indices alone cannot distinguish them under this representation. Size carries that distinction.</Prose>
    <PythonExample example={linkedExamples.ring}><Prose>Put refuses a full queue before changing state. Get saves the oldest value, clears its slot to release the stored reference, advances head with wraparound, and reduces size. None may be a legitimate queued value because emptiness is determined by size, not slot contents. Construction allocates O(capacity) slots; each put/get makes a bounded number of changes, O(1), without resizing.</Prose></PythonExample>
    <Prose>The correct full-queue policy depends on what is stored. A job processor might reject or wait for space; a live plot might intentionally discard its oldest reading. Python deque(maxlen=k) automatically discards an item from the opposite end on a full append, which is useful for recent-history buffers but can silently lose jobs if you assume this lab's rejection policy. A queue's storage alone does not coordinate concurrent producers and consumers; blocking, locking and process communication are separate contracts.</Prose>
    <Checkpoint prompt="Capacity is 3, head is 2 and size is 2. Which slots hold the live items, and where does the next insertion go? Does a priority queue promise this same removal order?">
      <Prose>The live items are in slots 2 then 0; the insertion goes into slot (2 + 2) % 3 = 1. A priority queue removes by priority rather than arrival order; its tie policy needs definition. Heaps develop that different contract later.</Prose>
    </Checkpoint>

    <H2>6. Combine lookup with recency</H2>
    <Prose>A cache with room for two results receives requests a, b, a, c. When c arrives, removing b keeps the more recently used a. This is <strong>least recently used (LRU)</strong> eviction. A map can find an entry by key quickly, while a doubly linked recency list can move a known entry to the newest end without searching for its predecessor. Together they support lookup and recency updates; neither structure alone provides both capabilities efficiently.</Prose>
    <PythonExample example={linkedExamples.lru}><Prose>OrderedDict supplies efficient reordering operations, so we use it instead of writing production cache internals. A hit moves the existing key to the newest end. A miss inserts a value and, if capacity is exceeded, removes the oldest. Requesting a again protects it from the next eviction; frequency and recency are different policies. Unknown source keys raise KeyError before a result is inserted.</Prose></PythonExample>
    <Prose>Recency is useful when recently used data is likely to be needed again, but a one-time scan can evict otherwise valuable entries. This toy source is fixed: eviction does not address stale data, expiration, synchronization or failures while computing a value. Those belong in the later <a href="/learn/topic/caching-strategies-semantic-exact-kv-cache-sharing">caching strategies</a> topic. The example's purpose is to show why a lookup map and an ordered structure complement each other.</Prose>

    <H2>7. Remove one node with a clear contract</H2>
    <Prose><strong>Independent task:</strong> remove only the first node whose value equals a target, preserving every other node and its order. Return the possibly changed head. Input is an acyclic singly linked list. Test [4, 7, 7, 9] with target 7, then remove 4 from the result; also test an absent target and an empty list. Do not convert the input to an array and rebuild it: the task is to preserve remaining node identities.</Prose>
    <details><summary>Hint: make the head case look like every other case</summary><Prose>A temporary sentinel node can point to the real head. It is not a data item in the returned list. Starting previous at the sentinel means even removing the first real node can use previous.next = current.next. Otherwise you may write a separate head case; both approaches are valid when their contracts are checked.</Prose></details>
    <details><summary>Complete solution and invariant</summary><PythonExample example={linkedExamples.remove}/><Prose>Before each comparison, previous.next is current and no earlier examined node matched the target. If current matches, changing one link removes exactly that node from the returned chain, and immediate return protects later equal values. If the loop ends, nothing matched and the original sequence remains. The sentinel is omitted by returning sentinel.next. Search is O(n), with O(1) additional space for references and one sentinel.</Prose></details>
    <Checkpoint prompt="Which original node identities should remain after removing 7 from [4, 7, 7, 9]? Does removing a node from one list guarantee no other variable can still access it?">
      <Prose>The original first, third and fourth nodes remain, in that order. The second node is no longer reachable through the returned list, but another reference can still reach it. Unlinking and making an object unreachable everywhere are different events.</Prose>
    </Checkpoint>

    <H2>8. Maintain a recent window of readings</H2>
    <Prose><strong>Transfer task:</strong> for readings [2, 4, 8, 10], report the mean of at most the most recent three values after each arrival. Early windows use the available count, so the first two means are 2 and 3. Keep a deque and a running total; when a full window receives another reading, remove the oldest contribution before adding the new one. This pattern supports a sensor dashboard without retaining its entire history.</Prose>
    <details><summary>Hint: keep the sum consistent with membership</summary><Prose>After every arrival, total must equal the sum of the live window. Eviction changes both membership and total. Divide by the current count, not always the maximum width. For empty input, no mean exists, so return no outputs.</Prose></details>
    <details><summary>Complete solution and changed-window check</summary><PythonExample example={linkedExamples.window}/><Prose>When 10 arrives at width 3, discard 2: the new window [4, 8, 10] has sum 22 and mean 22/3. Width 2 instead ends with [8, 10], mean 9. The task assumes finite numeric readings; missing/nonfinite values need an explicit policy. Floating-point running totals can accumulate rounding error in long streams, so this is not a numerical-accuracy guarantee for every dataset. Update work is O(1) per item; window storage is O(width), while this batch helper also stores O(n) returned means.</Prose></details>
    <Prose>You are ready to continue when you can explain which references an edit changes, state a loop invariant, choose LIFO versus FIFO by the task, and distinguish a bounded queue's full/empty states. Next in this module, <a href="/learn/topic/trees-binary-search-trees">Trees &amp; Binary Search Trees</a> extends linked relationships into branching structures. <a href="/learn/topic/complexity-analysis-recursion">Complexity Analysis &amp; Recursion</a> later makes operation-growth analysis more systematic. The reader's Next link includes planned lessons and follows your selected module order.</Prose>
    <DsaPractice practice={linkedPractice}/>
    <Sources alternatives={<LearningResources>
      <li><a href="https://www.youtube.com/watch?v=aV8LlSmd1E8">CS50x 2025, Lecture 5: Data Structures — YouTube lecture</a> with <a href="https://cs50.harvard.edu/x/2025/notes/5/">illustrated lecture notes</a> and <a href="https://cs50.harvard.edu/x/2025/weeks/5/">transcript and alternate player</a>. A visual second pass through nodes, lists, queues and stacks. It uses C pointers and allocation; focus on the diagrams first if C is unfamiliar. Python's reference and memory-management rules remain the ones used in this lesson.</li>
      <li><a href="https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks">Python tutorial: stacks and queues</a>. Short runnable alternatives using list and deque. Compare why the recommended end operations differ.</li>
      <li><a href="https://docs.python.org/3/library/collections.html#deque-recipes">Python deque recipes</a>. Continue the recent-window investigation with additional queue-based patterns; some recipes require iterator concepts introduced later.</li>
    </LearningResources>}>
      <li><a href="https://docs.python.org/3/library/collections.html#collections.deque">Python deque reference</a> — end-operation behavior, bounded eviction and indexing limitations.</li>
      <li><a href="https://docs.python.org/3/library/collections.html#collections.OrderedDict">OrderedDict reference and LRU recipes</a> — move_to_end, oldest removal and the distinction from ordinary insertion-ordered dict.</li>
      <li><a href="https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-2-data-structures-and-dynamic-arrays/">MIT 6.006: data-structure interfaces</a> — sequence representations and why an operation's implementation matters.</li>
    </Sources>
  </div>,
};
