# Curriculum expansion and handoff verification

Scope note: this records the curriculum expansion, regrouping and navigation increment before the first-five lesson implementation. Its before/after counts and route-order assertions are historical results for that increment. Current status is in [the authoring handoff](LESSON-AUTHORING-HANDOFF.md); subsequent lesson, brief, prerequisite and route changes are recorded in [the first-five implementation](FIRST-FIVE-REIMPLEMENTATION.md).

9 September 2026. Implementation and technical review completed for the requested curriculum expansion and durable authoring guidance. This record does not mark the new plans as published lessons or claim user approval of this increment. The user-approved Linux lesson remains the teaching reference.

## Delivered scope

| Measure | Before | After |
| --- | ---: | ---: |
| Unique topic IDs | 1,002 | 1,218 |
| Module entries, including shared topics | 1,006 | 1,222 |
| Modules | 28 | 28 |
| Guided paths | 5 | 7 |
| Registered published lessons | 187 | 187 |
| Stored individual teaching briefs | 0 in the catalogue schema | 275 |

The 216 additions comprise 65 GPU topics, 66 neural-engineering topics and 85 cross-domain topics. The 275 briefs cover every addition, all 32 original GPU topics, all 26 original neuroscience topics, and the approved Linux design. The complete GPU module has 97 individually planned topics; the neural module has 92. Supporting prerequisites are shared rather than duplicated.

326 topics have recorded prerequisite reviews. **943 older topics still need individual design**, using their module guidance and the current standard. These are explicit planning gaps, not completed bespoke plans. The review covered catalogue scope and selected prerequisite connections; it did not fact-check every existing lesson or certify all unknown dependencies.

## Authoring handoff and instruction cleanup

- Added workspace and repository `AGENTS.md` entry points to [the current handoff](LESSON-AUTHORING-HANDOFF.md).
- Updated [the authoritative teaching standard](LESSON-TEACHING-STANDARD.md) around Linux's approved learning decisions, complete examples, independent practice, research and separate verification/acceptance states.
- Added [domain-specific teaching strategies](docs/teaching/DOMAIN-PLAYBOOK.md) and [the full topic-design workflow](docs/teaching/TOPIC-DESIGN-BRIEF.md). Structure, visual type and lab count remain topic-dependent.
- Replaced six current-looking pilot/batch reports with short redirects. Their evidence is preserved under [the historical archive](docs/archive/lesson-rollout/README.md), with explicit non-authoritative notices. Updated Linux's recorded user acceptance.
- Replaced the stale curriculum plan with the current architecture and research map. Older outer-workspace plans are identified as historical in their README and the workspace entry point.
- Identified `StructuredLesson` as a legacy helper, not the new authoring template. General reading guidance no longer assumes every page has the same intuition anchor or suggests accuracy can wait until a second pass.

Read [the handoff](LESSON-AUTHORING-HANDOFF.md) for current scope. Earlier rewritten lessons remain deferred. This increment did not rewrite Linux, Bash or the other full lesson bodies.

## Catalogue and reader behavior

Neural Engineering and GPU Engineering have dedicated paths. Their individually planned dependencies introduce the specific shared skills needed. LLM and Embodied Intelligence retain shared foundations and relevant supporting GPU/neural core material. Difficulty staging and explicit dependencies prevent an entire advanced maths module from automatically preceding a domain introduction.

Module readers also honor recorded prerequisites. Following the user's grouping and navigation feedback, the sidebar keeps each real module together with unnumbered topics. Previous/Next name the immediately adjacent topics in the recorded sequence, including planned material. Opening a path or module resumes its first unfinished entry in that sequence; the URL resolves to that topic so completion never silently advances it. The earlier contiguous-section grouping created 472 groups in the full syllabus; it has been replaced with 28 module groups. Compact header counts use the same selected-topic scope for totals and completion. Full-module links preserve the current topic when it belongs to that module, or deliberately open the first selected module topic. Shared lessons remain visible in every applicable module and catalogue filter using the same ID and progress. Hub counts include all modules contributing prerequisites and match the reader through `getLearningRoute`. Existing topic IDs and progress keys are preserved. An old path link to a topic outside a newly focused route redirects to its canonical topic page.

Planned pages show purpose, outcomes, prerequisite links, learning sequence, a proposed investigation, independent practice and research starting points. Activities and solutions remain clearly unpublished. Planned entries display “Syllabus outline” rather than an invented reading time and cannot be marked as completed lessons. Misconception notes remain in authoring briefs for later explanation rather than being displayed as unsupported statements.

## Checks actually completed

| Check | Result and limits |
| --- | --- |
| `node scripts/verify-curriculum.mjs` | Passed after regrouping: 28 modules, 1,218 unique topics, 275 briefs and seven routes; all 1,002 original IDs preserved; required brief fields; new/specialist brief coverage; exact recorded prerequisites; no missing dependencies or cycles; complete sidebar coverage; one group per real module; valid shared memberships; within-module step order; small independent graph fixtures; focused-path regression checks. Unknown dependencies remain unreviewed. |
| `node scripts/build-curriculum-inventory.mjs` | Passed and regenerated [the Markdown inventory](docs/curriculum/CURRICULUM-INVENTORY.md) and [full JSON plans](docs/curriculum/curriculum-inventory.json). Topic-specific CLI output was also exercised with a GPU capstone. |
| `scripts/review-curriculum.cjs` | Passed at 1440 px and 390 px: seven path cards, both new route entry points, GPU/neural planned outlines, displayed sequences and prerequisite/source links, keyboard disclosure, disabled planned completion, sidebar/current-topic and Next navigation, module navigation, old path-link fallback, original Linux's four labs and completion toggle, an older unplanned outline, no page errors or document overflow in the inspected outlines. |
| `scripts/review-module-grouping.cjs` | Passed at 1440 px and 390 px: exactly 28 full-syllabus module groups; all 1,218 unique topics and 1,222 module memberships rendered in preserved within-module order without global step numbers or a separate scope row; shared lessons remain in applicable modules; compact selected/completed counts and full-module links; module entry follows prerequisite order; keyboard expansion and cross-module Next; no page errors or document overflow. |
| `scripts/review-reader-sequence.cjs` | Passed at 1440 px and 390 px: exact resolved module/topic/publication counts and canonical first entry for all seven paths; completion does not change the page; completed counts update; named Previous/Next retain a published → planned → published sequence; keyboard navigation; no next-published shortcut; path/module resume stops at the first unfinished entry; no page errors or overflow. |
| `scripts/review-catalogue-membership.cjs` | Passed at 1440 px and 390 px: exact unique counts and IDs for all 28 module filters, including all four shared topics in their secondary modules; composed search; consistent Module terminology; no page errors or overflow. |
| Visual inspection | Reviewed desktop/mobile captures for both new specialist outlines and the hub. Verified readable mobile reflow, clear planned status, section hierarchy and source disclosure. These are author inspections, not a novice study or complete assistive-technology audit. |
| `npm.cmd run build` | Passed on the final code. Existing JSX `>` warnings in the Bayesian-networks lesson remain, along with the large shared lesson-bundle warning (approximately 10.07 MB minified). This increment does not solve the existing bundle architecture. |
| Active handoff links and policy scan | Local links checked and superseded root reports inspected. No broken links in the checked active handoff/research documents after adding this record. No conflicting fixed lab count or stale active rollout queue found in the reviewed instructions. |

Evidence is under [scratch/curriculum-review](scratch/curriculum-review): build output, browser results, six screenshots, the pre-expansion snapshot, a topic-brief CLI sample and the handoff link audit. The browser script uses the already available Playwright runtime via `PLAYWRIGHT_PACKAGE`; no application dependency was added.

The browser check was corrected to wait for the newly selected topic, rather than accepting a still-visible previous selection during navigation. This avoids treating the URL update alone as proof that React has rendered the destination.

## Topic conservation and grouping correction

The complete pre-regrouping snapshot is [before-regrouping.json](scratch/curriculum-review/before-regrouping.json). Exact comparisons confirmed that all 1,218 current IDs, every module's topic memberships and all seven prerequisite-aware path sequences are unchanged. These comparisons passed again after the count/sequence UI correction, recorded in [sequence-conservation.json](scratch/curriculum-review/sequence-conservation.json). No topic was removed, merged away or renamed to implement either navigation correction. The four extra memberships are intentional shared references using the same topic IDs.

| Route | Earlier fragmented groups | Real module groups now |
| --- | ---: | ---: |
| ML Foundations | 93 | 6 |
| LLM Engineer | 188 | 13 |
| Embodied Intelligence | 117 | 10 |
| Neural Engineering | 91 | 7 |
| GPU Engineering | 76 | 8 |
| Research & Revision | 151 | 12 |
| Complete Curriculum | 472 | 28 |

Paths can include supporting modules through prerequisite links. A partial module's header shows how many topics are in this route and how many of those are completed. Its full-module link gives the complete topic count and access to all of them; there is no separate “All topics in this module” row. The module catalogue itself is unchanged: for example GPU has 97 topics, neural engineering 92, mathematics 57 and LLMs 61.

Results are saved in [regrouping-results.json](scratch/curriculum-review/regrouping-results.json) and [module-grouping-browser.json](scratch/curriculum-review/module-grouping-browser.json). The final desktop and phone captures `module-groups-1440.png` and `module-groups-390.png` were visually inspected. The application build and original curriculum browser checks passed again after the grouping correction.

The follow-up UI correction removes publication-driven entry and skip guidance, replaces sidebar numbering with status markers, names adjacent topics in the footer, and aligns hub/module/catalogue counts. [Sequence results](scratch/curriculum-review/reader-sequence-browser.json) and [catalogue results](scratch/curriculum-review/catalogue-membership-browser.json) record the new checks. Desktop/phone captures `compact-sidebar-*`, `sequence-footer-*` and `route-counts-*` were inspected for the updated layout. The active handoff, teaching standard and curriculum plan now specify these behaviors; old numbering and publication-shortcut instructions were replaced. The inventory's `resolvedModuleIds` and `moduleCount` distinguish resolved path coverage from the initial `trackIds` selection.

## Research and remaining work

Primary-source maps, dates, access limits and scope decisions are saved in the [GPU plan](docs/curriculum/GPU-ENGINEERING-PLAN.md), [neural plan](docs/curriculum/NEURAL-ENGINEERING-PLAN.md) and [26-track cross-domain review](docs/curriculum/CROSS-DOMAIN-COVERAGE.md). Proposed exercises are original design work. No GPU performance experiments, neural experiments or new lesson-runtime validations are claimed by these planning records.

Future authors must verify their particular claims, versions and outputs; finish individual plans where missing; implement the authorized lesson slice; and record computational, browser, visual and user review separately. Catalogue breadth is not proof that every niche topic exists, every older lesson meets the standard, or a reader has acquired supervised professional skills.

Nothing was deployed. The local hub can be opened at [127.0.0.1:5173/learn](http://127.0.0.1:5173/learn) while the development server is running.
