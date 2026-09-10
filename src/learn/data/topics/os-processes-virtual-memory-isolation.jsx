import { Code, CodeBlock, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { SchedulingLab, AddressTranslationLab, CopyOnWriteLab } from "../../components/lesson-labs/OsFoundationsLabs";
import PythonExample from '../../components/lesson-labs/PythonExample';
import { osExamples } from "../os-foundations-examples";
import { SeparateAddressFigure } from "../../components/lesson-labs/SystemsMechanismFigures";

export default {
  title:'OS Processes, Virtual Memory & Isolation',readTime:'~40 min read + 80 min practice',hasIntegratedGuide:true,
  content:()=> <div className="lesson-pilot">
    <LessonIntro prerequisites="Linux's parent/child, files and exit-status model; Python variables, functions and modules. No C, assembly or operating-system course is assumed." sections={[["1-a-program-needs-a-place-to-run","Processes"],["2-share-a-cpu-without-sharing-everything","Scheduling"],["3-give-each-process-its-own-address-map","Address spaces"],["4-translate-and-check-an-access","Paging"],["6-share-storage-with-a-clear-contract","Private/shared"],["8-build-and-diagnose-a-worker","Practice"]]}>
      Explain how two programs keep separate state while sharing a machine. Trace CPU time, translate a virtual address, distinguish a recoverable page fault from an invalid access, and communicate with a worker without assuming its variables update yours.
    </LessonIntro>
    <H2>1. A program needs a place to run</H2>
    <Prose>You launch an analysis while an editor and another analysis are already running. Each needs a current instruction, working values, access to files and a way to report completion. A <strong>program</strong> is the instructions and supporting data; a <strong>process</strong> is one running instance with its own execution context and resources. Starting the same program twice normally creates two processes, not two names for one calculation.</Prose>
    <div className="nt-flow"><span>Program + input<small>What should be computed?</small></span><span>OS creates a process<small>Identity, mappings and resources</small></span><span>CPU executes instructions<small>State changes until waiting or exit</small></span><span>Parent collects result<small>Data and exit status are distinct</small></span></div>
    <LessonTable caption="What must be kept for each running instance?" headers={['Entity','Plain meaning','Why it matters']} rows={[
      ['PID','An operating-system process identifier','Identify a particular living instance; IDs can be reused later.'],
      ['Instruction position and registers','Where execution will continue and small CPU-held working values','Pause and resume without restarting the program.'],
      ['Address space','The virtual memory ranges and access rules the process can use','The same numerical address can mean different storage in different processes.'],
      ['Open resources','References to files, pipes, sockets and related kernel objects','Some resources may intentionally be shared or inherited.'],
      ['Execution state','Ready, running, waiting, stopped or exited','Existing does not mean currently consuming a CPU.'],
    ]}/>
    <Prose>The kernel is the privileged part of the operating system. Application instructions usually run in a restricted user mode. A <strong>system call</strong> is a controlled request for kernel work, such as reading a file or creating a process. Entering the kernel is not automatically switching to another process; a request can finish and return to the same caller.</Prose>
    <Prose>The examples use Python's standard library. Save a whole block as a script and run it with Python 3.12 or later. The ordinary subprocess examples are portable; the explicit fork/mmap example is labelled Linux-only. Their small invented inputs require no downloads or administrative access. A browser diagram never launches a real process.</Prose>
    <PythonExample example={osExamples.child}><Prose><Code>sys.executable</Code> selects the Python interpreter already running the parent. The argument list runs that executable directly; -c supplies its program text. <Code>subprocess.run</Code> waits, captures stdout/stderr and checks the exit status. JSON is the explicit message format. The child setting value to 99 does not rebind the parent's value; the parent learns 99 only by reading the returned message. PIDs vary, so the check compares their identities rather than promising fixed numbers.</Prose></PythonExample>
    <H3>Creating a process and replacing its program are different operations</H3>
    <LessonTable caption="A common Unix launch lifecycle; Python hides the platform-specific launch details" headers={['Operation','Effect','Identity and state']} rows={[
      ['fork','Create a child from the calling process.','The child has a new PID and initially inherited contents/resources under their sharing rules.'],
      ['exec','Replace the current process image with another program.','On success the PID remains, but the old program image is replaced. It does not return to the replaced code.'],
      ['exit, then wait','The child finishes; its parent collects termination status.','On Unix, an unreaped exited child can retain a small status record as a zombie; it is not still running the finished program.'],
    ]}/>
    <Prose>A shell can create a child and have that child replace its program with the requested executable. Python's subprocess API provides a portable launch interface and may use platform-specific alternatives; do not assume every call literally follows one fork implementation. Waiting and collecting output are also distinct operations: the status reports how the process ended, while stdout carries the result's data.</Prose>
    <Checkpoint prompt="Two processes run the same Python file. Does assigning score = 10 in one automatically change score in the other?">
      <Prose>No. Ordinary process memory is separate. Use an explicit communication mechanism or deliberately shared storage when information must cross that boundary. Shared files are still possible, so process separation is not a promise that their external effects cannot interact.</Prose>
    </Checkpoint>

    <H2>2. Share a CPU without sharing everything</H2>
    <Prose>A CPU core has a limited number of hardware execution contexts. To make progress on more runnable work, the OS chooses a context, lets it execute, then may save it and restore another. This is a <strong>context switch</strong>. Saving registers and instruction position does not mean copying the process's entire memory on every switch; its address-space mappings remain available. Real systems schedule runnable threads; this introductory model gives each process just one thread, so each has one instruction stream to follow.</Prose>
    <Prose>A ready process could execute if selected. A waiting process cannot yet continue because an event, such as I/O completion, is missing. A stopped process has been explicitly suspended. Linux Basics introduced stopped versus exited; here distinguish both from the ordinary ready/waiting transitions of scheduling.</Prose>
    <SchedulingLab/>
    <Prose>A <strong>time slice</strong> limits how long one selected context runs before another can be considered. The lab rotates ready processes in round-robin order. With one core, alternating progress is concurrency; instructions from A and B are not executing simultaneously in the model. Multiple cores can provide parallel execution. Changing the schedule can change response times even when final independent results agree.</Prose>
    <Checkpoint prompt="A spends most of its time waiting for input. Would doubling its allowed CPU slice necessarily make its result arrive twice as fast?">
      <Prose>No. More CPU time does not complete the missing I/O event. First identify what resource limits progress. Scheduling also has overhead and competing goals; the zero-overhead toy timeline cannot establish real speedups or universal fairness.</Prose>
    </Checkpoint>

    <H2>3. Give each process its own address map</H2>
    <Prose>A memory address is a number used to locate a byte. Programs ordinarily issue <strong>virtual addresses</strong>; hardware translates them using mappings maintained by the OS. RAM is divided into physical storage locations. Virtual addresses let processes use convenient independent layouts while the system chooses where accessible contents live.</Prose>
    <LessonTable caption="A conceptual process address space; actual layout varies" headers={['Region','What it commonly contains','Important boundary']} rows={[
      ['Code and read-only data','Machine instructions and constants','Writing may be forbidden even when reading/execution is allowed.'],
      ['Writable data and heap','Longer-lived data and dynamically allocated objects','Allocated virtual space is not necessarily all resident in RAM.'],
      ['Stack regions','Call-related execution data','Each thread normally has stack state; Python frames are interpreter objects, not a simple diagram of one native stack slot per variable.'],
      ['Mapped libraries, files or shared regions','Code/data brought into the address space through mappings','Mappings can be private, shared or protected.'],
      ['Unmapped ranges','No currently permitted access','A numerical address is not automatically a usable allocation.'],
    ]}/>
    <Prose>This table is a conceptual map, not a fixed order of addresses or a scale drawing. Address-space layout randomization, libraries, allocator choices and architecture change actual locations. You need not know a Python object's raw address to use it safely; the low-level map explains the mechanism beneath references and allocations.</Prose>

    <H2>4. Translate and check an access</H2>
    <Prose>Paging divides virtual memory into equal-sized <strong>pages</strong> and physical memory into equally sized <strong>frames</strong>. A page-table entry maps a virtual page to a frame and records access information. The offset within a page stays unchanged: only the page's location is substituted.</Prose>
    <div className="foundation-rule">For a toy page size of 16 bytes, virtual address 22 is 1 × 16 + 6. Page number = 1; offset = 6. If page 1 maps to frame 3, physical address = 3 × 16 + 6 = 54. If another process maps it to frame 5, the same virtual 22 reaches physical 86.</div>
    <SeparateAddressFigure/>
    <AddressTranslationLab/>
    <PythonExample example={osExamples.translation}><Prose>Divmod returns quotient and remainder together. The function first finds a valid mapping, then checks write permission, then residency. A missing frame requests fault handling; the function does not fabricate a physical address or allocate a real OS page. This deliberately small translator makes the same arithmetic inspectable without any unsafe pointer access.</Prose></PythonExample>
    <Checkpoint prompt="With 16-byte pages, what changes between virtual byte 31 and byte 32 in A's map?">
      <Prose>31 is page 1, offset 15, reaching physical 63. 32 is page 2, offset 0, so it needs that page's separate mapping; here the valid nonresident page requires fault service. A page boundary changes the lookup even though the virtual byte numbers are consecutive.</Prose>
    </Checkpoint>

    <H2>5. A fault is an event to interpret</H2>
    <Prose>A <strong>page fault</strong> transfers control to the kernel because an access cannot complete under the current translation/protection state. Some faults are expected: allocate an initially unused anonymous page, bring back absent data, or handle a private copy-on-write write. Others represent an invalid address or forbidden operation. The kernel examines the mapping and access before deciding whether to service and retry or report a failure.</Prose>
    <LessonTable caption="Three events that are often incorrectly treated as identical" headers={['Event','What is missing or wrong?','Possible consequence']} rows={[
      ['TLB miss','A translation is absent from the CPU’s translation cache','A page-table walk may find a resident permitted page; no disk I/O is implied.'],
      ['Page fault on a valid nonresident page','The access needs OS service before proceeding','Zero-fill, file-backed data retrieval or other fault handling, depending on the mapping.'],
      ['Unmapped or forbidden access','The program lacks a valid permitted mapping for this operation','A normal unhandled access may end the process, such as via SIGSEGV on Linux.'],
    ]}/>
    <Prose>The <strong>TLB</strong>, or translation lookaside buffer, caches translations to avoid a full page-table walk on every access. It is distinct from a cache holding ordinary data bytes. Multilevel page tables reduce the need to allocate entries for every possible virtual address; the hardware-specific walk is a deeper implementation topic.</Prose>
    <details className="nt-deeper"><summary>Deeper branch: virtual size, residency and memory pressure</summary>
      <Prose>Virtual memory is not another name for swap. A process can have a large mapped address range with only some pages resident. RSS, resident set size, describes resident pages attributed to the process, but shared pages can appear in several processes' totals. Adding RSS values can therefore double-count shared storage. Proportional accounting such as PSS divides shared residency across users under its accounting rules.</Prose>
      <Prose>Under pressure, the system may reclaim caches, evict eligible pages, use configured swap, fail an allocation or trigger an out-of-memory policy. A high fault count alone does not prove disk activity; minor and major fault counters distinguish different classes of service. Inspect trends and the relevant workload instead of equating “large virtual size” with “all RAM consumed.”</Prose>
      <CodeBlock language="bash">{`# Linux inspection recipe: output depends on your actual process.
cat /proc/self/maps
cat /proc/self/status`}</CodeBlock>
      <Prose>These commands inspect the process opening /proc/self—in this case cat, not your earlier Python script. To inspect the intended script, read /proc/self from inside it or use its verified PID. Maps shows virtual ranges and mapping permissions; it does not directly reveal every physical frame. No fixed output is promised for this recipe.</Prose>
    </details>

    <H2>6. Share storage with a clear contract</H2>
    <Prose>Separate virtual maps can intentionally reach the same physical frame. This can avoid duplicating read-only code or data. <strong>Copy-on-write</strong> preserves private-write behavior while deferring a copy until a relevant write requires it. A normal fork starts a child with the parent's private contents; Linux can initially share their physical pages under copy-on-write rules. The maps and process identities remain distinct.</Prose>
    <CopyOnWriteLab/>
    <H3>A real private/shared comparison on Linux</H3>
    <Prose>Run the following complete script in Linux or WSL as an ordinary user. Mmap creates a region whose bytes can be accessed directly; -1 requests anonymous storage rather than a file descriptor. The flags explicitly choose private versus shared behavior. The child inherits both mappings at fork. Waiting for that child to exit ensures its update precedes the parent's inspection.</Prose>
    <PythonExample example={osExamples.mappings}><Prose>The parent still reads A through the private mapping and S through the shared mapping. The child changed its private view to P. This verifies the observable mapping contract, not an exact physical-frame count. <Code>os._exit</Code> ends the forked child without running copied Python finalizers or flushing inherited buffers; the parent remains responsible for its own cleanup. Use this single-threaded standalone fixture, not fork from a notebook with unknown active threads.</Prose></PythonExample>
    <Prose>This mechanism is useful for workers that read a large inherited dataset but alter only a small part. Private writes may increase memory consumption, and runtime activity can dirty pages too. “Fork is cheap” is therefore not a blanket memory/performance guarantee. Deliberate shared memory can avoid repeated copies, but readers and writers need a protocol so they do not observe a partially updated multi-field result.</Prose>

    <H2>7. Know what the boundary does and does not isolate</H2>
    <LessonTable caption="Different isolation layers answer different questions" headers={['Layer','Provides','Does not automatically provide']} rows={[
      ['Process address space','Controlled mappings for ordinary memory accesses','Isolation from shared files, granted resources, kernel defects or every side channel.'],
      ['Threads within a process','Separate execution contexts, usually with separate stacks','Separate ordinary heap address spaces; shared state needs synchronization.'],
      ['Container mechanisms','Configured views of resources and accounting/limits','A separate host kernel for each normal container, or correct permissions merely from packaging.'],
      ['Virtual machine','A guest kernel running over virtual hardware','A proof that every host/guest boundary or application policy is correct.'],
    ]}/>
    <Prose>Memory permissions and filesystem mode bits are different checks. A process may be unable to write its code page yet still have permission to overwrite an output file. Restricting one channel does not revoke the other. Namespaces change which resources are visible; resource controls constrain usage. Good boundaries depend on what is shared and what authority is granted.</Prose>
    <Checkpoint prompt="A worker cannot modify the parent's ordinary Python list. Does that make it safe to grant the worker write access to the parent's report directory?">
      <Prose>No. Its file writes are a separate effect authorized through filesystem access. Decide which inputs, outputs and capabilities the worker needs; process separation is only one part of that design.</Prose>
    </Checkpoint>

    <H2>8. Build and diagnose a worker</H2>
    <Prose><strong>Independent task:</strong> accept text inputs 3, bad and 0. Run a worker that squares an integer. Valid input must return a result; invalid input must return a nonzero status and explanatory stderr. The parent must retain successful zero rather than treating it as failure. Predict which data crosses the process boundary and how the parent detects failure before opening the solution.</Prose>
    <details><summary>Hint: define three channels</summary><Prose>Use stdin for the input, stdout for a successful result, and stderr plus nonzero exit status for a diagnostic. Capture them separately. A timeout needs separate exception handling in a production caller; do not silently treat it as a valid numerical result.</Prose></details>
    <details><summary>Complete solution and explanation</summary><PythonExample example={osExamples.worker}/><Prose>The worker's local value never becomes a parent variable through ordinary assignment. The parent parses stdout only after successful status. An explicit ok flag distinguishes zero from failure. The five-second timeout bounds waiting; TimeoutExpired is intentionally surfaced to the caller rather than converted to invented output.</Prose></details>
    <Checkpoint prompt="Change the inputs to -2 and an empty string. What should be returned? Now change the translator's page size to 32 while keeping A page 1 mapped to frame 3: where does virtual 38 go?">
      <Prose>The worker returns 4 for -2 and status 2 with invalid integer for an empty string. With 32-byte pages, 38 is page 1 offset 6, so its physical address is 3 × 32 + 6 = 102. The mapping table stores page/frame numbers; their byte interpretation depends on page size.</Prose>
    </Checkpoint>
    <Prose>You are ready to move on when you can distinguish waiting from running, translate an address with its permissions, explain private/shared writes, and identify the explicit result channel. Next in this module, <a href="/learn/topic/threads-concurrency-locks-deadlocks">Threads, Concurrency, Locks &amp; Deadlocks</a> develops coordination when workers deliberately share state. The DSA module then starts with <a href="/learn/topic/arrays-strings-hash-maps">Arrays, Strings &amp; Hash Maps</a>; <a href="/learn/topic/cpu-architecture-cores-caches-simd-pipelining">CPU architecture</a> develops hardware details. The reader's Next link follows your selected route.</Prose>
    <Sources alternatives={<LearningResources>
      <li><a href="https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf">OSTEP — The Abstraction: The Process</a> and <a href="https://pages.cs.wisc.edu/~remzi/OSTEP/vm-paging.pdf">Paging: Introduction</a> · Free illustrated chapters by Remzi and Andrea Arpaci-Dusseau. Read the process states after the scheduling lab and the page/offset example after translation; some low-level examples use C.</li>
      <li><a href="https://www.youtube.com/watch?v=f1Hpjty3TT8">MIT 6.S081 — Lecture 4: Page Tables</a>, Frans Kaashoek · Video with <a href="https://pdos.csail.mit.edu/6.828/2020/lec/l-vm.txt">lecture notes</a>. A deeper follow-on after this page: it uses the educational xv6 kernel, C and RISC-V details. Its 2020 implementation context is not a description of every current Linux machine.</li>
    </LearningResources>}>
      <li><a href="https://docs.python.org/3/library/subprocess.html">Python subprocess: argument lists, captured streams, status and timeout</a></li>
      <li><a href="https://man7.org/linux/man-pages/man2/fork.2.html">Linux fork contract</a>, <a href="https://man7.org/linux/man-pages/man2/execve.2.html">execve</a> and <a href="https://man7.org/linux/man-pages/man2/waitpid.2.html">waitpid</a> — creation, image replacement, status collection and copy-on-write notes.</li>
      <li><a href="https://man7.org/linux/man-pages/man2/mmap.2.html">Linux mmap: private/shared mappings and protection</a></li>
      <li><a href="https://docs.kernel.org/filesystems/proc.html">Linux proc: process maps, residency and accounting fields</a></li>
      <li><a href="https://pdos.csail.mit.edu/6.1810/2025/lec/l-vm.txt">MIT page-table notes: translation, permissions and caching</a></li>
    </Sources>
  </div>,
};
