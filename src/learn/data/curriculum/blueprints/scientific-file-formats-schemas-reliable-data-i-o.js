// Authoring blueprint: Scientific File Formats, Schemas & Reliable Data I/O.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Preserve IDs, missing values, units and array structure across a validated file round trip, then publish a complete update.",
  "outcomes": [
    "Distinguish decoding, parsing, conversion and schema/domain validation",
    "Choose a format using data structure and access patterns",
    "Verify a semantic round trip and predict the effect of a failed write"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules",
    "NumPy: Arrays, Broadcasting & Vectorization"
  ],
  "sequence": [
    "Inspect leading-zero IDs, quoted commas, missing values and measured zero",
    "Follow bytes into text, fields, values and accepted records",
    "Specify names, types, units, ranges and uniqueness",
    "Run a complete CSV importer and JSON envelope round trip",
    "Compare text, typed table and array formats",
    "Verify NPY dtype, shape, values and separate metadata",
    "Compare direct writes with staged validation and replacement",
    "Study chunking, manifests and schema evolution as deeper branches",
    "Import and diagnose an independent changed batch"
  ],
  "visual": {
    "type": "Schema gates",
    "question": "Which stage should reject this file?",
    "interaction": "Select malformed numbers, nonfinite values, mixed units or duplicate IDs and step from raw text to validation."
  },
  "visuals": [
    {
      "type": "Publication trace",
      "question": "What does a fresh reader see after a failed write?",
      "interaction": "Compare direct and staged writing with success/failure while published and temporary files remain visible."
    }
  ],
  "practice": {
    "task": "Import a changed batch with zero and a blank, verify counts and mean, reject a repeated ID and design an array/metadata round-trip check.",
    "success": "All IDs and units survive; missing values are distinct from zero; wrong keys and meaning changes are detected before publication."
  },
  "misconceptions": [
    "A file extension does not validate content",
    "Conversion alone does not validate a measurement",
    "Equal flattened numbers can conceal changed axis meaning",
    "Atomic replacement is not a complete durability or multi-file transaction protocol"
  ],
  "sources": [
    "https://docs.python.org/3/library/csv.html",
    "https://docs.python.org/3/library/json.html",
    "https://numpy.org/doc/stable/reference/generated/numpy.load.html",
    "https://docs.python.org/3/library/os.html#os.replace",
    "https://arrow.apache.org/docs/python/parquet.html",
    "https://docs.h5py.org/en/stable/high/dataset.html"
  ],
  "depth": "core",
  "reviewFocus": "Keep filesystem/concurrency limits explicit. Verify actual temporary-file behavior and numeric array round trips; no unsupported Parquet/HDF5 benchmark claims.",
  "designRecord": "docs/teaching/data-foundations-reimplementation.md"
};
