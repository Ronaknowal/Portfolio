export const decoratorTrace = {
  title: "Follow two wrappers around one call",
  question: "Predict the next event in report(). Decoration has already built outer(inner(report)); the original body runs only once.",
  headers: ["Location", "Action", "Next destination"],
  steps: [
    { code: "report()", rows: [["outer wrapper", "Print outer before", "inner wrapper"]], output: "outer before", note: "The name report refers to the outermost wrapper." },
    { code: "function(*args, **kwargs)", rows: [["inner wrapper", "Print inner before", "original function"]], output: "outer before\ninner before", note: "The outer wrapper's saved function is the inner wrapper." },
    { code: 'print("body"); return 21', rows: [["original report", "Print body; return 21", "inner wrapper"]], output: "outer before\ninner before\nbody", note: "The original function runs once and returns its result." },
    { code: 'print("inner", "after"); return result', rows: [["inner wrapper", "Pass result 21 back", "outer wrapper"]], output: "outer before\ninner before\nbody\ninner after", note: "Returns unwind through the layers in reverse order." },
    { code: 'print("outer", "after"); return result', rows: [["outer wrapper", "Pass result 21 back", "caller"]], output: "outer before\ninner before\nbody\ninner after\nouter after", note: "The wrapper must return the result if it wants to preserve the caller's contract." },
    { code: 'print("result", report())', rows: [["caller", "Print returned value", "finished"]], output: "outer before\ninner before\nbody\ninner after\nouter after\nresult 21", note: "The call completes before the outer print receives its result." },
  ],
};
