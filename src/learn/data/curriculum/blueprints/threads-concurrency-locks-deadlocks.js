// Authoring blueprint: Threads, Concurrency, Locks & Deadlocks.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Reason about shared-state invariants, waiting predicates and lifecycle; expose a lost update, repair it and complete bounded concurrent work with explicit outcomes.",
  "prerequisites": [
    "OS Processes, Virtual Memory & Isolation",
    "Python Basics: Types, Control Flow, Functions & Modules",
    "Decorators & Context Managers"
  ],
  "outcomes": [
    "Distinguish concurrent progress, parallel execution and shared object identity",
    "Construct a lost update and protect the complete invariant with one shared lock",
    "Explain predicate checks, wait/reacquire and early or stolen notifications",
    "Draw a two-lock wait-for cycle and use one resource ordering",
    "Collect results/errors and complete a bounded queue with cooperative shutdown"
  ],
  "sequence": [
    "Map thread-local execution state to shared process objects",
    "Separate start, join and result collection",
    "Schedule reads, computations and writes to expose a lost update",
    "Force native reads with a Barrier, then verify a protected counter",
    "Choose critical-section boundaries from invariants",
    "Trace condition state through empty, published and stolen-item schedules",
    "Run late/early publication using stored state and wait_for",
    "Construct a two-lock cycle and remove circular waiting with stable ordering",
    "Run ordered account transfers with total preservation",
    "Collect finite file tasks and explicit parsing errors through futures",
    "Request cooperative cancellation and explain GIL/free-threaded runtime boundaries",
    "Independently validate a bounded stream with index/error/stop accounting and changed workloads"
  ],
  "visual": {
    "type": "Shared counter and worker-local state",
    "question": "How can two increments produce one?",
    "interaction": "Schedule either worker one step at a time; toggle a whole-operation lock and inspect blocked attempts."
  },
  "visuals": [
    {
      "type": "Predicate, lock ownership and notification timeline",
      "question": "Why is notification insufficient to consume?",
      "interaction": "Compare empty notification, ordinary publication and a competing consumer."
    },
    {
      "type": "Wait-for graph and lock ownership",
      "question": "Which worker can reach release?",
      "interaction": "Choose opposite or global lock order; schedule acquisitions and detect the two-worker cycle."
    }
  ],
  "practice": {
    "task": "Build a bounded sensor validator that returns one indexed result/error per input and terminates every worker; test empty, bad, zero, nonfinite and changed-worker-count cases.",
    "success": "No accepted work disappears; completion accounting includes stop markers; observed results match a sequential oracle and no owned worker remains."
  },
  "misconceptions": [
    "A local reference may reach a shared object",
    "A GIL is not a multi-operation transaction",
    "Locking only writes can preserve stale reads",
    "Notification does not reserve an item or release the lock",
    "A timeout does not cancel a running thread",
    "Lock ordering does not establish fairness"
  ],
  "sources": [
    "https://docs.python.org/3/library/threading.html",
    "https://docs.python.org/3/library/concurrent.futures.html",
    "https://docs.python.org/3/library/queue.html",
    "https://docs.python.org/3/howto/free-threading-python.html",
    "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks.pdf",
    "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-cv.pdf",
    "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-bugs.pdf"
  ],
  "depth": "core",
  "designRecord": "docs/teaching/bash-and-concurrency-design.md",
  "reviewFocus": "Three bounded causal investigations, six native programs and independent changed-input work. Force the race rather than relying on sleep or incidental bytecode behavior. Test all legal small schedules against invariants; never launch an unbounded deadlock. Compare current free-threading docs with the actually tested standard CPython runtime. Keep queue capacity, work completion and thread termination distinct. Module-order continuation replaces prior global difficulty/prerequisite reshuffling."
};
