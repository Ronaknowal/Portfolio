# Visuals And Labs

Part of [$lesson-authoring](../SKILL.md). Repository paths below are relative to the selected application checkout; see [the repository adapter](portfolio-adapter.md). Load only the sections needed for the selected mode.

- [5. Visuals and labs: no fixed count](#5-visuals-and-labs-no-fixed-count)
- [Let the concept determine the visual form](#let-the-concept-determine-the-visual-form)
- [Inline diagrams are part of the explanation](#inline-diagrams-are-part-of-the-explanation)
- [Choose the representation for the hurdle](#choose-the-representation-for-the-hurdle)
- [Graphs and simulations need traceable evidence](#graphs-and-simulations-need-traceable-evidence)
- [Visual layout is a separate completion check](#visual-layout-is-a-separate-completion-check)

## 5. Visuals and labs: no fixed count

**There is no requirement for exactly one lab, at least one lab, or a maximum number of labs per lesson.** Support every major conceptual hurdle using the representation that teaches it best. Several small labs and diagrams are appropriate when a topic contains several distinct mechanisms. A clear static diagram may fully explain a relationship without interaction.

### Let the concept determine the visual form

Keep the learning philosophy, accessibility, navigation and control behavior dependable across the site. Choose the illustration's geometry, visual vocabulary, terminology and interactions for the specific concept and learner question, including different choices within one lesson. Consistency does not require every topic to use the same lab arrangement, text panel, slider or stepper. Variation has no quota either: reuse a familiar representation when it still exposes the mechanism well.

Before choosing a component, identify the actual objects, relationships and operations the learner needs to understand, and what is hard to see in prose or code. Choose the form that makes those features inspectable. Use the subject's useful conventional representations and terms, introducing their plain meaning and visual legend first. For example, token boundaries and pair merges suggest symbol strips; reference sharing suggests names, objects and arrows; concurrent execution suggests separate lanes and synchronization points. Changing a panel's title or colors does not adapt it to a different mechanism. An existing table or text trace remains appropriate when the task is best understood that way.

For an interactive lab, let the learner act on meaningful entities or parameters and see the corresponding transformation. A tokenization trainer can connect an editable corpus, the selected pair, its frequency, changed word segmentations and merge history. A guided trace and an open investigation can serve different purposes in the same lesson. This is a reference for designing around a topic's operations, not a required layout or an endorsement of every detail in the existing BPE implementation.

Keep connected views consistent with the same model state. Make it clear when edited input is applied; labels, counts, highlights and outputs must describe the active state. Use controls named for the action when helpful, such as merging a pair or advancing one instruction. Do not require unrestricted editing, animation or a full simulator when a smaller investigation teaches the question. Share useful controls and rendering primitives, but adapt or build a representation when the available generic panel obscures the concept. Record the learning reason for that choice in the brief; novelty alone is not a reason.

### Inline diagrams are part of the explanation

Plan visual support throughout the reading flow, including compact, immediately visible diagrams between paragraphs. Do not treat completing the interactive labs as completing visual teaching. An inline diagram helps the learner see a structure, correspondence, contrast or transformation while reading; an interactive investigation helps them explore how it changes. Use either or both when their teaching jobs warrant it.

The user's tokenization comparison is a useful presentation reference: the same input is shown as character units, an unknown-word result and subword units in three labelled rows. Boundaries and contrasting representations are visible without operating a control. Reference this visual pattern, not the surrounding page as a blanket accuracy endorsement or a compulsory token-chip style for every subject.

For a new concept, ask whether the reader currently has to assemble its relationships mentally from prose or code. When a picture would make those relationships materially clearer, place it at that point in the explanation. Useful forms include aligned before/after states, labelled values or cells, reference arrows, nested ownership, branching control flow, data transformations, miniature timelines and same-input comparisons. Introduce the objects and labels before asking the learner to manipulate them. Connect the picture to the exact example and explain what to notice in a short caption or nearby sentence.

A static diagram's design contract is its learner question, concrete example/state, meaningful visual encoding, placement, interpretation and limits. Verify its labels, values, arrows and correspondences against the example. Controls, step/reset behavior and simulated state are additional requirements only when the diagram is interactive. Prefer the representation the concept needs; there is no requirement for SVG, animation, a full lab panel or a particular renderer. The existing React/HTML/CSS/SVG approach supports these figures.

Review the lesson in ordinary reading order without operating every lab. Check that important structures and contrasts are visible where introduced, and that their meaning does not depend on reconstructing several code blocks or discovering a later lab state. A useful, already visible lab illustration can satisfy this need; do not duplicate it automatically. Record any substantive visual gap and its proposed placement. Counts of labs, code examples, tables or colored boxes are not evidence that this reading pass succeeds. Keep tables when they explain a comparison well, and omit diagrams that merely decorate or restate a clear sentence.

Use consistent labels and restrained color to track identity or correspondence, with text/position/shape cues as well as color. Preserve readable labels, correct grouping and logical order on narrow screens. A row of boxes containing paragraph summaries is not a mechanism diagram unless its arrangement actually reveals the relationship being taught.

### Choose the representation for the hurdle

| What the learner needs to understand | Useful representation |
| --- | --- |
| Identity, containment, shared references | Labeled objects and connecting arrows; distinguish a name from the object it refers to. |
| A changing process or hidden state | Stepped execution with the current instruction, before/after state, and changed items highlighted. |
| A parameter's effect | A focused experiment with a labeled control, comparable outputs, and an explanation of the change. |
| Distributions and uncertainty | Meaningful axes, reference lines, correctly shaded probability areas, and repeated samples where relevant. |
| Shapes, broadcasting, reductions | Aligned dimensions, labeled row/column/cell correspondence, and an explicit resulting shape. |
| Joins and grouping | Trace source rows into result rows; show unmatched records, multiplicity, and aggregation. |
| Data or control movement | Directed flows with labeled inputs, outputs, order, and failure paths. |
| Competing methods | Same-input comparisons with assumptions, outcome differences, and when to choose each. |
| Exact values or accessible inspection | Tables alongside the conceptual representation. Tables are sometimes the primary representation, especially for tabular data. |

For every lab, specify a short teaching contract:

- **Question:** what particular uncertainty should this resolve for the learner?
- **Starting point:** what entities and initial state are shown, and what must the learner already know?
- **Live starting result:** what useful output and intermediate state are visible before any interaction?
- **Control:** what changes and what is deliberately held fixed?
- **Visible consequence:** which states, correspondences, or quantities change and why?
- **Check:** what observation or follow-up task would show understanding?
- **Boundary:** what is simulated, simplified, fixed, unsupported, or outside the model?
- **Fixture check:** which alternative settings were run to confirm that the chosen data actually exhibit the contrast the surrounding prose promises, and which null case shows the contrast is absent when it should be?

Implement the interaction as **manipulate → observe live → explain the mechanism → compare → decide → transfer**. Start with a useful preset and its complete current result. Use sliders, keyboard-accessible direct manipulation, cell edits, switches or other controls appropriate to the topic. Prefer changing one meaningful factor at a time, with visible intermediate effects and synchronized final outputs. Offer before/after comparison or a pinned baseline when that clarifies a tradeoff, plus meaningful reset/back controls. Explain unchanged results as well as changed ones. If a simulation has randomness, keep the seed fixed during parameter comparisons and make drawing fresh data a separate explicit action.

**Do not include a prediction feature in a lab, even optionally.** No predicted-answer field, prediction choices, commitment step, correctness score, answer masking or “Check prediction” reveal. Remove those states and misleading instructions when updating older work; merely making the old gate optional is insufficient. Independent practice with hints and worked solutions may remain separate from exploration. Model predictions, attention gates and train/test masking are subject matter and are unaffected by this interaction rule.

Step, play/pause and bounded run controls remain useful for a process whose intermediate states matter. They advance the actual algorithm or simulation, not permission to view an answer. Keep the current state and its output visible throughout, and expose instantaneous parameter effects where meaningful. If a costly calculation needs an explicit run, show which input configuration produced the current result, label pending edits honestly and cancel obsolete work; never substitute grading for a computational boundary. Inspect a prerecorded experiment with clearly labeled record selectors; do not imply a new model was trained live.

**Make the affordance honest, including inline figures.** A thumb on a track, draggable point or editable-looking cell must work as advertised. When manipulating it helps reveal the mechanism, implement real pointer/touch and keyboard interaction with visible current values and synchronized consequences. Keep the coordinate scale stable during a drag; changing an error must change its encoded length/area, not disappear into automatic rescaling. Otherwise draw a clearly read-only annotation (for example a tick rather than a slider thumb), identify the actual control and explain what stays fixed. Do not call a preset-only illustration a freely editable lab. A static diagram may remain static when that serves its explanatory purpose.

Review controls throughout the article, not just inside the main lab wrapper. Check enabled, selected, focused and disabled styling against the site's dark/amber theme; native gray buttons or unstyled selects are defects. Exercise real dragging and keyboard changes, inspect the resulting numbers/plots/captions, and review desktop and narrow layouts. Automated control-value changes alone do not establish that a drag target, touch target or visible explanation works.

For every exposed range, test intermediate values as well as presets, and test both endpoints through the actual native control. A validator that accepts only old presets must not sit behind a continuous slider. Paired slider/number editors must show the same exact model value; do not introduce an arbitrary snapping grid that silently changes defaults or typed values. Bind each field to its own accessible label, particularly when a label also contains an output readout. Preserve the learner's inspection position when recomputing a trace where that position remains valid. At an initial state that cannot change, explain the invariant and offer a clearly identified full replay/preview or meaningful inspection step; do not make a control look ineffective by resetting to an unchanged starting frame. Previews must identify which later states are computed, observed or hypothetical.

Three requirements turn a guided trace into an investigation and are checked at review:

1. **The consequence is available immediately.** Opening the lab shows a useful current result. Valid edits update the linked mechanism, diagram, numbers and explanation together without an answer submission. The learner can trace input → operation → intermediate state → output → practical decision. Check continuous changes, boundary/null cases, reset, and quick repeated edits; no prediction UI or correctness feedback may remain.
2. **The learner acts on the topic's entities.** At least one control changes something the prose has not already resolved: choosing which items start a process, placing or selecting a threshold, changing an input rather than picking among two or three labeled presets whose outcomes are described beside them. Presets remain useful as suggested starting points.
3. **Every fixture demonstrates what it is placed beside.** Before accepting a dataset or example for a lab, run the alternatives the lab offers and confirm they differ in the way the adjacent explanation claims. A table that promises four rules behave differently is contradicted by a fixture on which all four agree. When the natural running example cannot show the contrast, add a second fixture built to show it, and check that its critical values are exactly representable so a promised tie is a real tie.

Connect representations directly: selecting an edge may highlight its matrix entries; stepping a line may move a reference arrow; choosing a result cell may reveal its input operands. Use stable labels and color meaning. Explain whether layout position, line thickness, area, or color encodes a quantity or only organizes the drawing.

Do not treat a changing table, a “Next” button, decorative animation, or a screenshot of code as proof that the mechanism is intuitive. Ask whether a first-time learner can explain what changed without reconstructing everything mentally. Retain exact tables as inspection/accessible companions when adding spatial explanations.

Give each additional lab a distinct job. Put it beside the concept it teaches. Split crowded controls or unrelated questions into focused activities; offer an optional integrated investigation after the parts are understood. Share UI components where useful, without forcing every concept into the same visualization.

### Graphs and simulations need traceable evidence

Code-rendered charts can teach relationships clearly; rendering code does not establish the truth of their values or interpretation. Identify what each quantitative figure represents: an exact calculation, an analytic model, a simulation, measured data, externally sourced data or an illustrative sketch. Put the relevant assumptions and status next to the figure, and retain its equation, generator, measurement record or source locator in the authoring evidence.

- Check values independently where appropriate, along with units, scales, log transforms, ranges, normalization, rounding and correspondence to the text. Distinguish probability from density, observed samples from interpolation, and supported ranges from extrapolation. Show variability or uncertainty when the claim depends on it.
- Real performance comparisons need applicable sourced measurements or a reproducible benchmark: record the task, corpus/input sizes, algorithm settings, software versions, hardware, parallelism, timing boundaries and variability. The caption and prose must not generalize beyond those conditions.
- An illustrative sketch can explain a hypothetical relationship, but invented points must not imply measured seconds, real-product rankings or established scaling laws. Use explicit model assumptions and generic series for a hypothetical example, or replace it with verified data. The word “illustrative” does not justify nearby unsupported empirical claims.
- **The claimed feature must be perceptible in the rendered figure.** If the prose says a curve bends, flattens, peaks or separates, a reader must be able to see that at the size the figure renders, on desktop and on a phone. Check the range of the plotted values against the axis: when one value dwarfs the rest, a linear axis turns the interesting region into a flat line, and a logarithmic axis, a second panel or a relative measure is needed. Include the natural baseline point (the one-component, zero-change or untreated case) when it anchors the comparison. Show repeated runs as individual marks when their disagreement is part of the message. A caption admitting that the numbers are easier to read in the table is a sign the figure has failed.
- Full-width SVG figures scale their text with their width. Inspect each figure at desktop width as well as phone width, and constrain the figure's maximum width or text size so labels stay in proportion at both.

### Visual layout is a separate completion check

A numerically correct figure can still teach incorrectly when a heading crosses another component's bar, a label is clipped, or several stages collapse into one crowded image. Passing model tests, counting figures and checking document overflow do not establish that the visual itself works. The GMM allocation figure reported by the user on 14 September 2026 is a concrete example: all marks were inside one SVG, but a component heading intersected the preceding component's update.

- For a new or rewritten lesson, inspect **every distinct inline figure**, plus the informative states of its investigations, with the actual fonts at desktop and narrow widths. For a targeted repair, inspect every affected figure/representation and reuse unchanged evidence. Identify reviewed figures/states explicitly; selected screenshots must not be reported as a complete visual pass. At responsive breakpoints, also inspect the composition between the endpoints. Include increased text size when the visual contains substantial annotation.
- Keep explanations, section headings, legends and multi-line numeric readouts in reflowing HTML when their position does not encode data. Reserve SVG coordinates for geometry, short labels and scale-bound marks. Separate stages or stack panels when necessary; do not force all steps into a fixed-height drawing or shrink text until it fits.
- Check the interior, not just the outside bounds: text against text, labels against bars/arrows/points, labels within their intended nodes, and clear separation between successive stages. Use computed glyph bounds and targeted regression checks to catch known failures, then inspect the rendered result. Geometry flags need interpretation; an intentional node label or background grid intersection is not automatically an error.
- Include built-in presets that materially change the geometry: coincident components/points, repeated measurements, a zero or boundary value, and a completed trace where relevant. Labels that are distinct in the opening state may coincide after an update. Record the specific preset/state tested; a default-state scan is not coverage of those teaching cases.
- Preserve the mathematical meaning while changing layout: comparable panels retain the same scale and units; circles/angles require equal physical axis scale; bar lengths/areas still encode the declared quantity. Never stretch axes, clamp quantitative positions or hide explanatory content merely to remove a layout warning.

Follow the engineering layout contract (repository path: `docs/engineering/LEARNING-CODE-STANDARD.md#diagram-layout-and-svg-legibility`). A repeatable triage tool supports this pass but cannot certify all interactive states, typography or teaching quality. Fix confirmed findings and repeat affected checks; reuse unrelated passing evidence.

Validate guided traces, trainer outputs and their prose against the stated algorithm/conventions as well. A convincing visual is a design reference until its model and claims have been checked; an attractive lab does not establish the accuracy of its entire lesson.
