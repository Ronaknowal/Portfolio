// Authoring blueprint: SQL, Relational Data & Transactions for ML.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Build one traceable feature row per sensor, preserve missingness and time boundaries, and make a two-statement change atomic.",
  "outcomes": [
    "Define row grain, keys and foreign-key relationships",
    "Trace join multiplicity, NULL extension, predicate placement and aggregation",
    "Write parameterized cutoff features and execute explicit commit/rollback"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules"
  ],
  "sequence": [
    "Identify sensor and observation grains before SQL syntax",
    "Create a disposable SQLite fixture with explicit constraints",
    "Read filtering and null predicates as questions about records",
    "Trace every join output to source rows and vary duplicate keys",
    "Compare ON with WHERE before aggregation",
    "Extract one feature row per sensor at an explicit time cutoff",
    "Trace committed and pending states across a failed two-update transaction",
    "Consider indexes, window functions and production boundaries as deeper branches",
    "Solve a changed-sensor feature request and repair three independent errors"
  ],
  "visual": {
    "type": "Linked join rows",
    "question": "Why did three sensors become four output rows?",
    "interaction": "Select each result row to highlight its sources; vary join, cutoff, predicate placement and an unconstrained duplicate key."
  },
  "visuals": [
    {
      "type": "Transaction state trace",
      "question": "Can a failed transfer lose two credits?",
      "interaction": "Compare one explicit transaction with separately committed statements, including failed and successful second updates."
    }
  ],
  "practice": {
    "task": "Add a measured zero and a sensor with no observations, extract cutoff features, and repair incorrect WHERE/count/imputation choices.",
    "success": "One row per sensor, correct measured/observation counts and means, no future observations, no invented zeros; rollback boundary explained."
  },
  "misconceptions": [
    "A join can multiply rows even when one key is unique",
    "NULL comparison is not ordinary equality",
    "COUNT(*) differs from counting a nullable joined column",
    "A failed statement does not universally roll back prior statements",
    "Parameter placeholders represent values, not SQL identifiers"
  ],
  "sources": [
    "https://www.sqlite.org/lang_select.html",
    "https://www.sqlite.org/lang_aggfunc.html",
    "https://docs.python.org/3/library/sqlite3.html",
    "https://www.sqlite.org/lang_transaction.html",
    "https://www.sqlite.org/foreignkeys.html"
  ],
  "depth": "core",
  "reviewFocus": "Check actual SQLite joins and independent committed-state reads; distinguish logical query order from physical execution and engine-specific isolation/type behavior.",
  "designRecord": "docs/teaching/data-foundations-reimplementation.md"
};
