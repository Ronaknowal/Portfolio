export const debuggingTrace = {
  title: "Diagnose the failing mean before changing code",
  question: "This is a guided reasoning trace, not a live debugger. Follow the small failing input and make one evidence-based change.",
  headers: ["Evidence", "Observed", "Interpretation"],
  steps: [
    { code: "wrong_mean([18, 24])", rows: [["Contract", "Expected 21.0", "Both readings must contribute"], ["Result", "Actual 9.0", "A real counterexample"]], note: "Write the expected result independently: (18 + 24) / 2." },
    { code: "total = 0", rows: [["values", "[18, 24]", "Two readings"], ["total", "0", "No items processed"]], note: "The initial state is correct, so do not change it speculatively." },
    { code: "total += value", rows: [["value", "18", "First iteration"], ["total", "18", "Only one reading included"]], note: "The sum is correct so far. The question is whether execution reaches the second iteration." },
    { code: "return total / len(values)", rows: [["return", "18 / 2 = 9.0", "Exits the function now"]], note: "Return is inside the loop. It exits before 24 is visited; changing the denominator would hide the real bug." },
    { code: "Move return after the loop", rows: [["iterations", "18 then 24", "total becomes 42"], ["return", "42 / 2 = 21.0", "The regression case now passes"]], note: "Make the smallest correction, then rerun the original counterexample." },
    { code: "Run ordinary, empty, zero and non-finite cases", rows: [["Suite in the next example", "4 test methods", "0 failures, 0 errors"]], note: "The corrected implementation uses fsum and explicit validation. A passing counterexample alone is not enough." },
  ],
};
