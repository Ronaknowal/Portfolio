# Incoming discovery for Sets, Logic, Relations & Proof Techniques

## 2026-09-11 — Equation transformations as implication versus equivalence

- Status: adapted in the Math44 implementation; final author/independent verification remains separately recorded.
- Origin: [Algebra design](../ALGEBRA-FUNCTIONS-LESSON-DESIGN.md), stable `algebra-functions-exponentials-logarithms`.
- Current local treatment: subtracting a number or dividing by a nonzero number is reversible; squaring x=−2 yields x²=4 but admits the extra candidate2 in reverse; multiplying by zero loses the original constraint. A rational cancellation keeps its original excluded input.
- Proposed destination and benefit: Sets/Logic owns the next formal language of implication/equivalence and counterexamples. Reuse or adapt these now-familiar cases to make the direction of a logical arrow concrete before quantified proofs.
- Required reasoning: a valid forward operation does not establish a reversible operation; domain and nonzero assumptions belong in the statement. Avoid claiming that every manipulation that preserves truth preserves the full solution set.
- Disposition: Math44 section3 now introduces the arrow through original x=−2 versus squared x²=4 solution sets, then explains nonzero division, multiplication by zero and the retained x≠1 cancellation domain. `ImplicationFigure` makes the added+2 candidate visible and states that restricting x≤0 restores equivalence in this specific example. Section5 revisits reversible transformations as a proof-discovery versus proof-validation distinction. This adapts the useful bridge without treating Algebra43 as a required prerequisite.
- Implementation: `src/learn/data/topics/sets-logic-relations-proof-techniques.jsx`, `ImplicationFigure` in `SetsLogicLabs.jsx`, and [Math44 design](../SETS-LOGIC-LESSON-DESIGN.md). Math44 native/model and final1440/390/320 actual-font browser checks pass; see [author verification](../SETS-LOGIC-VERIFICATION.md). Independent review is separate. No Algebra43 body was changed.

