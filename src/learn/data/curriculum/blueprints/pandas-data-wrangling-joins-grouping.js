// Authoring blueprint: Pandas: Data Wrangling, Joins & Grouping.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Build a checked order report by following row identity through alignment, conversion, joins and grouped reductions; transfer those ideas to sensor calibration and device uptime.",
  "outcomes": [
    "Define row grain and inspect labels, types and missingness before transforming a table",
    "Trace every join output to source records and enforce the intended cardinality",
    "Distinguish aggregate from transform, missing from zero, and backward time matching from information availability",
    "Build and diagnose an independent grouped report with an explicit denominator"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules",
    "NumPy: Arrays, Broadcasting & Vectorization"
  ],
  "sequence": [
    "Begin with one order per row and introduce DataFrame, Series, index and nullable dtype",
    "Select by label, position and a checked mask",
    "Trace label assignment versus positional assignment and distinguish CoW from a plain alias",
    "Preserve raw tokens through conversion, nullable comparison and selection",
    "Parse dates, inspect duplicates and validate identifiers",
    "Trace matching and unmatched join records with cardinality validation",
    "Follow group membership into size, count, sum, mean and a row-aligned transform",
    "Reshape and compute sorted grouped windows with explicit aggregation policies",
    "Explore sorted backward calibration matching, tolerance and availability assumptions",
    "Build a complete order report and solve a new device-uptime task before reading the solution"
  ],
  "visual": {
    "type": "Label-to-source mapping",
    "question": "Which fee reaches order b when rows move or label a is absent?",
    "interaction": "Switch label/position assignment, order and missing labels; select a destination to reveal the source arrow and total."
  },
  "visuals": [
    {
      "type": "Conversion gates",
      "question": "Can converting successfully hide invalid input?",
      "interaction": "Step raw text, nullable values, selection mask and surviving row identities; compare known and positive policies."
    },
    {
      "type": "Join lineage",
      "question": "Why did this output row survive or multiply?",
      "interaction": "Select left/inner/outer, duplicate lookup and validation; select an output to trace both origins."
    },
    {
      "type": "Group contributors and return shape",
      "question": "Why do two rows have mean 10 instead of 5?",
      "interaction": "Inspect size versus known-value count; compare missing-key/zero policies and aggregate versus transformed output."
    },
    {
      "type": "Pivot coordinate and source inspector",
      "question": "What if two records request the same output cell?",
      "interaction": "Select a cell to reveal source records; compare unique, duplicate and missing coordinates with pivot, explicit sum and mean. Block duplicate pivot output instead of inventing a partial result."
    }
  ],
  "practice": {
    "task": "Create a region order report, vary calibration tolerance, then independently summarize device observations and active fractions while checking duplicate IDs.",
    "success": "IDs, row counts, join cardinality and denominator survive every step; zero remains measured and all-missing groups remain unknown; changed-input predictions match real Pandas."
  },
  "misconceptions": [
    "Equal length does not establish label alignment",
    "Coercion is not validation",
    "Many-to-one describes right-key uniqueness, not equal table sizes",
    "An all-missing group cannot establish a zero measurement",
    "A backward event-time match alone does not prove availability at prediction time"
  ],
  "sources": [
    "https://pandas.pydata.org/docs/user_guide/indexing.html",
    "https://pandas.pydata.org/docs/user_guide/copy_on_write.html",
    "https://pandas.pydata.org/docs/user_guide/missing_data.html",
    "https://pandas.pydata.org/docs/user_guide/merging.html",
    "https://pandas.pydata.org/docs/user_guide/groupby.html",
    "https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html"
  ],
  "depth": "core",
  "reviewFocus": "Retain 15 complete runnable examples and the five focused investigations, including pivot coordinate collisions; these counts describe implemented coverage, not a quota. Read docs/teaching/SCIENTIFIC-VISUAL-REVIEW.md for the visual follow-up. Verify installed Pandas 3 semantics independently. Explain current APIs alongside the older Data School video playlist and official tutorials; no requirement for a video quota. Revisit scope/title during writing and read destination notes.",
  "designRecord": "docs/teaching/next-three-reimplementation.md"
};
