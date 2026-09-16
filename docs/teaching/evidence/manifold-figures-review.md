# Manifold learning inline figures — author visual review

Reviewed 14 September 2026. All nine prescribed figures are implemented and visually inspected. This is the figure author's review; the separate lab author's figure review supplies independent scrutiny. No figure-owned findings remain open.

## Scope and method

Reviewed the complete saved manuscript and visual specifications, the current lesson design, teaching standard, and internal-layout requirements. The figures follow the saved packet. They use topic-specific SVG and HTML, exact constructed fixtures or saved native coordinates, named evidence categories, equal metric axes, persistent observation identities, and wrapping HTML captions/legends.

The production preview at `http://127.0.0.1:4183/learn/path/full-curriculum/t-sne-umap-manifold-learning?module=classical-ml` was inspected in headless Microsoft Edge with actual Google Space Grotesk/JetBrains Mono and KaTeX fonts loaded. `document.fonts.ready` was awaited. Smooth scrolling was disabled; fixed/sticky elements outside `main.reader-content` were hidden for diagnostic captures. The review does not infer correctness from those shell elements being hidden.

All nine figures were captured and opened at 1366, 780, 390, and 320 CSS pixels, plus 780 pixels with `document.documentElement.style.fontSize = '200%'`. The last case is explicit root text enlargement, not a claim of actual browser zoom. All 45 states received visual inspection. The final shared EqualPlot change affected F6 only; its five states and expanded source-306 state were recaptured and reopened after the final shared build.

[Source-bound layout evidence](manifold-figures-layout.json) records source/build hashes, screenshot hashes, capture times, dimensions, loaded fonts, and each SVG inspection. There were zero page errors and zero SVG layout-triage candidates. The automated checks supplement, rather than replace, screenshot inspection.

## Figure coverage and placement

| Figure | Export and placement | Relationship verified |
| --- | --- | --- |
| F1 | `ManifoldRouteFigure`, §2 | Seven fixed U identities, six unit graph edges, ambient shortcut2, cumulative route6; equal input coordinate scales. |
| F2 | `ManifoldNeighborFigure`, §3 | Exact original/proposed1D coordinates, original versus mapped directed neighbors, all false/missing choices, R1=0 and T1=0.5. |
| F3 | `ManifoldProbabilityFigure`, §4 | Unnormalized Gaussian weights versus row probabilities, shared0–1 axes, all three values retained including the small far-neighbor probability, base2 entropy/perplexity. |
| F4 | `ManifoldForceFigure`, §5 | A's three signed negative-gradient contributions and their total on aligned signed axes; full initial gradient and actual learning-rate0.5 update. The packet explicitly specifies no growing KL curve. |
| F5 | `ManifoldFuzzyGraphFigure`, §6 | Self-excluded calibration sum2, directed0.5/0.25 memberships, fuzzy union0.625, then candidate separations with the graph weight fixed. |
| F6 | `ManifoldDigitsFigure`, §7 | The same300 measured observations in native PCA/t-SNE maps, shared source identity, pixel evidence, exact-coordinate disclosure, global R10/T10/C10. The initial illustration withholds local neighbor-count answers for InvestigationD. |
| F7 | `ManifoldMdsFigure`, §9.1 | Exact distance/Gram matrices and centered line at−7/3,−1/3,8/3; recovered gaps2,3,5 and B_AB=7/9. |
| F8 | `ManifoldLleFigure`, §9.3 | Reconstruction weights2/3 and1/3 preserve1 from0/3 and2 from0/6 on common-scale axes; general negative-weight caveat retained. |
| F9 | `ManifoldTopologyFigure`, §9.5 | Square versus identified projection points; four IDs retained; edges, filled triangles and tetrahedron distinguished; β1 exists exactly on[1,√2). |

## Retained screenshots

Each row links all five inspected states. The images show the actual rendered page, not reconstructed diagrams.

| Figure |1366px|780px|390px|320px|780px/root text200%|
|---|---|---|---|---|---|
|F1|[desktop](screenshots/manifold-figures/f1-desktop.png)|[intermediate](screenshots/manifold-figures/f1-intermediate.png)|[phone](screenshots/manifold-figures/f1-phone.png)|[narrow](screenshots/manifold-figures/f1-narrow.png)|[enlarged](screenshots/manifold-figures/f1-enlarged-text.png)|
|F2|[desktop](screenshots/manifold-figures/f2-desktop.png)|[intermediate](screenshots/manifold-figures/f2-intermediate.png)|[phone](screenshots/manifold-figures/f2-phone.png)|[narrow](screenshots/manifold-figures/f2-narrow.png)|[enlarged](screenshots/manifold-figures/f2-enlarged-text.png)|
|F3|[desktop](screenshots/manifold-figures/f3-desktop.png)|[intermediate](screenshots/manifold-figures/f3-intermediate.png)|[phone](screenshots/manifold-figures/f3-phone.png)|[narrow](screenshots/manifold-figures/f3-narrow.png)|[enlarged](screenshots/manifold-figures/f3-enlarged-text.png)|
|F4|[desktop](screenshots/manifold-figures/f4-desktop.png)|[intermediate](screenshots/manifold-figures/f4-intermediate.png)|[phone](screenshots/manifold-figures/f4-phone.png)|[narrow](screenshots/manifold-figures/f4-narrow.png)|[enlarged](screenshots/manifold-figures/f4-enlarged-text.png)|
|F5|[desktop](screenshots/manifold-figures/f5-desktop.png)|[intermediate](screenshots/manifold-figures/f5-intermediate.png)|[phone](screenshots/manifold-figures/f5-phone.png)|[narrow](screenshots/manifold-figures/f5-narrow.png)|[enlarged](screenshots/manifold-figures/f5-enlarged-text.png)|
|F6|[desktop](screenshots/manifold-figures/f6-desktop.png)|[intermediate](screenshots/manifold-figures/f6-intermediate.png)|[phone](screenshots/manifold-figures/f6-phone.png)|[narrow](screenshots/manifold-figures/f6-narrow.png)|[enlarged](screenshots/manifold-figures/f6-enlarged-text.png)|
|F7|[desktop](screenshots/manifold-figures/f7-desktop.png)|[intermediate](screenshots/manifold-figures/f7-intermediate.png)|[phone](screenshots/manifold-figures/f7-phone.png)|[narrow](screenshots/manifold-figures/f7-narrow.png)|[enlarged](screenshots/manifold-figures/f7-enlarged-text.png)|
|F8|[desktop](screenshots/manifold-figures/f8-desktop.png)|[intermediate](screenshots/manifold-figures/f8-intermediate.png)|[phone](screenshots/manifold-figures/f8-phone.png)|[narrow](screenshots/manifold-figures/f8-narrow.png)|[enlarged](screenshots/manifold-figures/f8-enlarged-text.png)|
|F9|[desktop](screenshots/manifold-figures/f9-desktop.png)|[intermediate](screenshots/manifold-figures/f9-intermediate.png)|[phone](screenshots/manifold-figures/f9-phone.png)|[narrow](screenshots/manifold-figures/f9-narrow.png)|[enlarged](screenshots/manifold-figures/f9-enlarged-text.png)|

The [additional F6 source306 state](screenshots/manifold-figures/f6-selected-source306-phone.png) inspects the t-SNE tab, optional digit colors, wrapping ten-digit legend, real digit2 tile, full8×8 integer pixel grid, and intact nine-decimal coordinate disclosure. Source306 intentionally differs from the storage index.

## Findings and closure

1. **Narrow F4 decimals broke across lines.** The [before capture](screenshots/manifold-figures/f4-narrow-before.png) shows the defect even though an overflow-only check passed. F4 and F6 exact-coordinate tables now switch to label/value rows in narrow containers and keep numeric tokens intact. The final narrow/phone/enlarged captures were reopened and checked.
2. **Own SVG labels reduced excessively on narrow containers.** Geometry-only responsive typography now raises glyph sizes without affecting the shared plot. Axes and labels remain separate from strokes; no enlarged-font collisions appeared. The shared F6 plot subsequently received a wider left gutter and mobile typography from the lab author. Its final320px minimum computed rendered SVG font is10.929px; full negative ticks remain legible and unclipped. Independent shared-geometry evidence measures13px glyph bounds and9px inner left clearance at320px.
3. **F6 select name included every option.** An explicit concise accessible name now identifies source-row inspection. Source306 selection, tab switching, color overlay, and exact disclosure were exercised.
4. **Page min-content expansion at780px.** The parent fixed `reader-content`'s automatic minimum width. All normal viewport captures now have zero document overflow.
5. **Root text enlargement exposed a separate 4px source-list overflow.** The unbroken token `standard/modified/Hessian` reaches 783.766px in a 780px viewport. An optional lab checkbox label also exceeded its own narrow grid cell. In-page `overflow-wrap:anywhere` trials remove both, yielding scrollWidth 780. The parent incorporated scoped list-item and checkbox wrapping plus `min-width:0` into `manifold-lesson.css` and added this exact enlarged-text regression. These elements are outside the nine figures. The final [parent browser evidence](manifold-browser.json) binds the incorporated CSS and its recheck; these earlier screenshots truthfully retain their capture-time document-overflow measurement.

The figures render computed relationships rather than decorative stand-ins. Figure arithmetic uses the shared pure models/fixtures; actual-digit coordinates and metrics remain from the preserved native data. The saved independent tiny-gradient error in F4 is labeled as saved evidence and is not conflated with the later JavaScript finite-difference measurement.

No manuscript, lesson ledger, design document, or unrelated application source was edited by this figure task. The source freeze is `ManifoldFigures.jsx` SHA256 `d95fdffdbd8b3c2021e7faa6f70ee74ce94360a32244a55b7d808c84c63218f3`, `manifold-figures.css` SHA256 `c3ae1be13d6c7cd29b0786766e6398318b74d33b19e53602158e60dc3f1a3402`.
