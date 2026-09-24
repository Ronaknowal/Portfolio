// Authoring blueprint: OS Processes, Virtual Memory & Isolation.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Connect a running program to saved execution state, scheduling, address translation and explicit communication; distinguish private memory from deliberately shared effects.",
  "outcomes": [
    "Distinguish a program, process, execution context and resource",
    "Trace waiting and time-sliced progress without imagining whole-memory copies on every switch",
    "Translate a virtual address and distinguish resident access, demand service and invalid permissions",
    "Predict private versus shared writes and diagnose a worker using separate result and error channels"
  ],
  "prerequisites": [
    "Linux Basics, Filesystems & Processes",
    "Python Basics: Types, Control Flow, Functions & Modules"
  ],
  "sequence": [
    "Start two instances of a program and define PID, saved registers and resources",
    "Run a complete subprocess example that reports child state explicitly",
    "Trace one CPU, ready order, saved counters, waits and time-slice boundaries",
    "Introduce virtual address-space regions with layout and runtime caveats",
    "Split an address into page/offset, consult rights/residency and calculate the physical destination",
    "Distinguish TLB misses, demand faults, invalid access and I/O; optionally inspect Linux memory accounting",
    "Compare copy-on-write fault/copy/remap with shared-write behavior",
    "Run explicit private/shared anonymous mappings in a single-threaded Linux fork fixture",
    "Compare process, thread, container and VM boundaries without promising complete security",
    "Independently define stdin/stdout/stderr/status contracts for a numerical worker, retaining zero and reporting invalid input",
    "Transfer page arithmetic to a changed page size and bridge to array storage"
  ],
  "visual": {
    "type": "CPU timeline and saved process state",
    "question": "Which process can run next, and which counter changes?",
    "interaction": "Change time slice and I/O request; step execution, waits, next instruction and ready queue."
  },
  "visuals": [
    {
      "type": "Page-to-frame route and permission gates",
      "question": "Why can the same virtual address reach different values?",
      "interaction": "Choose process, address and read/write; step page split, mapping, fault handling and final destination."
    },
    {
      "type": "Shared-frame and private-copy arrows",
      "question": "Will A see B write 9?",
      "interaction": "Compare explicit shared storage with private copy-on-write, stepping fault, copy/remap and resulting values."
    }
  ],
  "practice": {
    "task": "Run a child and a translator, verify Linux private/shared effects, then design a worker for valid, invalid, zero, negative and empty inputs.",
    "success": "Arithmetic matches divmod, worker status controls result parsing and native Linux preserves the private view while exposing the shared update. No toy timeline is described as a measured Linux scheduler."
  },
  "misconceptions": [
    "A running process need not be using a CPU now",
    "A context switch need not copy all RAM",
    "A TLB miss is not necessarily a page fault or disk read",
    "Separate virtual maps can share physical storage",
    "Process memory isolation does not revoke filesystem authority"
  ],
  "sources": [
    "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf",
    "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-paging.pdf",
    "https://pdos.csail.mit.edu/6.1810/2025/lec/l-vm.txt",
    "https://man7.org/linux/man-pages/man2/fork.2.html",
    "https://man7.org/linux/man-pages/man2/mmap.2.html",
    "https://docs.python.org/3/library/subprocess.html",
    "https://docs.kernel.org/filesystems/proc.html"
  ],
  "depth": "core",
  "reviewFocus": "Three causal investigations, four complete programs and an independent worker. Page size, scheduling, data and allocation assumptions must stay explicit. Native fork evidence concerns observable mapping semantics, not exact physical sharing. OSTEP written alternatives and MIT page-table video/notes deepen the lesson; annotate C/RISC-V and older xv6 details.",
  "designRecord": "docs/teaching/systems-and-structures-design.md"
};
