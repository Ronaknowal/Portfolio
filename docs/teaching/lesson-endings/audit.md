# Recurring lesson endings: source audit

28 September 2026. Baseline inspection of all 234 manifest-owned lesson sources, with Randomized Linear Algebra as the requested example. [The inventory](inventory.json) records source paths, final H2 candidates, later subheadings, helper use, local practice helpers and disclosure labels. Counts describe source constructs, not rendered exercise totals or teaching completeness. No scientific checks, manuscript revisions or ledger changes were undertaken.

| Baseline pattern | Scope | Minimal treatment |
| --- | ---: | --- |
| `Sources` in `lesson-labs/LessonElements.jsx` | 128 topics; 118 pass `alternatives` | Existing props provide an explicit further-learning/technical-reference split. Use independently labeled sections, preserving every link and annotation. |
| `LearningResources` in `LessonInvestigation.jsx` | 14 topics, all inside `Sources.alternatives` | Retain its list and explanatory subheading within further learning. |
| `Checkpoint` in `LessonElements.jsx` | 359 instances across 107 topics | Preserve in-body checkpoints. Give final-exercise treatment only inside a marked practice area. |
| Local exercise helpers | 89 topic-local definitions | Mark existing roots/disclosures rather than rewriting their APIs. Eighty use `Practice`; others use `Exercise`, `PracticeTask`, `ConcentrationExercise`, `FamilyExercise`, `CausalExercise`, or `StateFamilyCheckpoint`. |
| `DsaPractice` in `lesson-labs/DsaPractice.jsx` | 22 topics | Preserve local practice, external LeetCode, optional extensions and readiness as distinct roles, with existing IDs and datasets. |
| Authored endings without `Sources` | 106 topics, including all 42 DL topics | Explicitly group actual blocks. Do not classify an entire tail from its last heading. |

Forty-eight older lessons end in `Self-check exercises` without native details elements in that section. Some worked answers are always-visible `Callout` blocks; others use prose or lists. Across that group there are 59 source Callout instances. `content/Callout.jsx` also serves teaching notes throughout lessons, so a global rewrite would exceed the ending scope.

## Randomized Linear Algebra

In `src/learn/data/topics/randomized-linear-algebra.jsx`, H2 “8. Practise with new failures and new data” begins at line 144. Exercises A–D have H3 titles at lines 145, 152, 156 and 160. Each combines a visible prompt, a separate hint disclosure, and a `Checkpoint` holding its solution. Exercise A includes `RunnableExample` in the answer: preserve that complete program/result view.

“Ready to continue” at line 164 is a distinct H3 and paragraph, containing readiness criteria and the actual next-topic link. Give it a readiness/continuation wrapper separate from the exercises. `Sources` at line 167 has two alternative-learning paragraphs and three technical-reference list items. Keep their reading/viewing limits and version/provenance annotations while separating the groups visually.

No ending-specific selectors were found in `randomized-linear-algebra-labs.css`. Presentation currently relies on shared `lesson-check`, `lesson-deeper`, `lesson-sources` and heading styles. Semantic practice and exercise boundaries can unify it without changing a calculation or lab.

## Deliberate migration cases

- `independent-component-analysis-ica.jsx:34`: local `Practice` returns a fragment. It needs an exercise root rather than a class added to an existing root. Readiness and references already have separate H2s at lines 409 and 417.
- `numerical-pdes-grids-finite-elements-stability.jsx`: local `Section` generates its H2 from a `title` prop. Use the call's title/ID rather than the single dynamic H2 template. Active Learning and End-to-End Supervised Learning also have nested heading wrappers.
- `gaussian-processes-gp.jsx`: practice is section 7, followed by substantive deeper applications in sections 8–9 and references in section 10. Do not sweep the deeper applications into an ending wrapper.
- `convnext-modern-cnn-designs.jsx:451`: readiness, next-topic prose, a mixed paper/tutorial/alternative-reading list, and an evidence/provenance paragraph share one heading. Preserve the latter's connection to what it qualifies.
- `sequence-to-sequence-encoder-decoder.jsx:507`: a mixed list contains an alternative implementation, papers, lecture/notes, dataset/schema attribution and API documentation; a next-topic paragraph follows. A paper URL alone does not determine an entry's learning purpose.
- `ring-attention-sequence-parallelism.jsx:466`: readiness, next-topic prose, extension guidance, mixed references and an evidence-boundary paragraph share “Continue learning”. Keep distinct subgroups and their annotations.
- `active-learning.jsx:224`: alternatives, survey/API references, deeper papers and next-topic material are separate paragraphs under a combined H2. Preserve `active-section-10` and its heading fragment.
- Combined hint/solution disclosures occur in Active Learning, RBM, Spectral Normalization and Hyena. Style them honestly; do not invent a hint by splitting sentences automatically.
- `tokenization.jsx:1000` illustrates older visible Callout answers. Scope any neutral exercise treatment to the closing practice and preserve other explanatory callouts.

## Migration contract

Use explicit semantic markers for practice, further learning, references and readiness/next, with exercise and feedback markers inside practice. Retain authored nodes and their order. Share typography, spacing, neutral/amber surfaces, focus and disclosure behavior while preserving topic-specific body layouts. Avoid runtime text classification, “last N nodes” wrapping and blanket styling of all details elements.

Preserve H2 text, authored numbers and existing IDs when wrapping sections. New unnumbered resource headings must not renumber other sections. Baseline `Sources` is an aside: adding H2s inside it would still exclude them from the reader TOC. Use ordinary semantic sections for its new further-learning/reference headings. Avoid duplicate authored headings and invented resource tags, dates, counts or readiness claims.

Keep prompts visible and independently accessible hints/solutions where those are authored separately. Use native details/summary; preserve combined feedback when no separate hint exists. Do not hide a lab's live output or alter its example to make endings look alike.

Verify text/link/ID conservation, distinct section purposes, actual TOC inclusion, working disclosures/programs, keyboard focus and 320px layout. Representative cases: Randomized Linear Algebra, a local-helper lesson, ICA, DSA, Numerical PDE, Gaussian Processes, ConvNeXt/Seq2Seq/Ring, Active Learning and an older visible-answer lesson. Reuse unchanged scientific evidence.

## Explicit-boundary review

The subsequent layout plan uses exact authored titles, not the baseline inventory's last-four-heading candidate region. The inventory's `migrationReview` records the 23 initially unmatched lessons and their concrete mappings. Most use ordinary H2s with independent-task titles; Numerical PDE instead needs its local `Section` with `id="npde-practice"` and title “15. Produce a changed solver report”. NumPy section 11 remains substantive body teaching between practice in section 10 and readiness in section 12.

Review removed inappropriate ending classifications from Gaussian Processes sections 8–9, PCA section 10, Time Series section 7, K-means section 14 and Real Analysis section 14. These are substantive deeper teaching or worked capstones. Real Analysis's actual continuation starts at H3 “Make the method reusable”. Embedding Models' “11. Further reading” is resources, not readiness. Dynamical Systems and Stochastic Processes now start their practice wrappers at the actual “Independent practice” H3, preserving the preceding application teaching as body content.

Explicit next-step H3s were added for Python Basics, OOP, Iterators, Pandas, Reductions, Network Flow and Decision Theory. Decision Theory also has an otherwise empty H3 “References and other ways to learn” directly before `Sources`; keep it outside exercise grouping and inspect the combined heading rhythm after `Sources` gains its two normal H2s. Existing mixed ConvNeXt, Seq2Seq, Ring Attention and Active Learning tails still require content-aware visual grouping rather than fabricated link categories.

Itô Calculus's “10. Apply, test and practise” starts with substantive applications. Its explicit custom range begins at the first of 11 sibling `Practice` calls and stops before the `Prose` prefix “Continue in module order:”; that paragraph is continuation, not another task. Headingless custom ranges and Numerical PDE's local `Section` wrapper need their bespoke generator preservation, because generic direct-H2/H3 extraction cannot recover those boundaries.

`Sources` was changed in `src/learn/components/lesson-labs/LessonElements.jsx`: `alternatives` becomes “Further learning”, children become “Technical references”, and each body gets `lesson-resource-list`. The optional-reference note and all supplied nodes remain. A source check confirmed that the file parses and `Checkpoint` is byte-for-byte unchanged from the captured baseline. All 349 exact-heading plan entries resolved uniquely against the 234 baseline sources at review time. Full conservation and browser receipts belong to the integration owner; this source review does not claim fresh scientific verification.

CSS review identified the 14 nested `LearningResources` lists (`.nt-resources > ul`) as needing explicit resource-list styling coverage. Mixed lists within next-step sections also need an explicit list wrapper rather than a blanket rule that could restyle readiness criteria. Keep native focus/disclosure behavior and retain topic-specific inner representations.

## Independent implementation review

Reviewed `scripts/lib/lesson-ending-layout.mjs`, `src/learn/components/lesson-endings.css`, `LessonElements.jsx`, the exact-title plan and the three updated authoring documents. The layout transformer is authoring-only: it parses once per operation, edits source ranges, reparses before returning, and rejects missing/ambiguous titles. Adjacent-range ordering now puts the previous close before the next open. Native disclosures and explicit semantic sections preserve browser keyboard behavior; there is no new browser state, observer or content scan. Shared CSS remains within the reader and marked endings, with neutral/amber colors, visible focus and a narrow-screen adjustment. The 14 nested learning-resource lists are now covered.

The custom-range fail-safe was exercised against the current Itô source: generic regeneration throws a clear preservation error instead of dropping its headingless range. The code standard documents this limit. The inventory remains about 646 KB, retaining grouped helper/disclosure evidence rather than full repeated source snippets.

Both review findings are resolved and independently rechecked:

- **Resolved — practice section heading became an exercise.** `groupPracticeExercises` now skips a practice wrapper's identifying H3. A focused fixture with “Independent practice” plus two `Practice` components now produces two exercise wrappers, leaves the heading outside both, and remains unchanged on a second pass. This covers the narrowed Dynamical Systems/Stochastic Processes groups.
- **Resolved — URL-based resource styling in readiness.** The `href`-based CSS selector is removed. Ring Attention's exact-title entry uses `resourceList: true`, which renders `data-lesson-resource-list`; `endingLayoutOf` retains the flag. The code standard documents explicit use for actual resource lists without reclassifying readiness material.

No unresolved implementation findings remain in this bounded source review. Source fixtures also confirm the custom-range preservation error. The integration owner's eight authoring fixtures and all-topic conservation proof provide the broader recorded check; this review separately inspected the fixes above.

Browser/build results are intentionally owned by the integration receipt. This review did not repeat numerical checks or revise ledger checkpoints.
