# Authoring notes: Non-Negative Matrix Factorization (NMF)

Canonical topic ID: `non-negative-matrix-factorization-nmf`

## 2026-09-12 — Connect different meanings of a latent component

- Status: resolved on 2026-09-14; see the closing entry below.
- Origin: the content-first work for Classical ML positions 16–18: [GMM](../GMM-LESSON-DESIGN.md), [manifold learning](../MANIFOLD-LEARNING-LESSON-DESIGN.md) and [ICA](../ICA-LESSON-DESIGN.md).
- Destination and ownership: NMF is the next module entry, position 19. Its additive factors provide the natural contrast with GMM responsibilities, embedding coordinates and ICA sources. This discovery note does not authorize rewriting it now.
- Existing coverage inspected: the original NMF introduction and core intuition, plus the convergence/reference/troubleshooting passages in `src/learn/data/topics/non-negative-matrix-factorization-nmf.jsx` at commit `8c5da59f18516be77c29d5aeeafca3decca4f738`. This was a scoped boundary check, not a complete lesson review. The current prose repeatedly implies nonnegativity forces recognizable physical parts and calls unnormalized factors topic distributions. Later convergence language also needs the distinctions below.
- Learning benefit: a learner can state what the chosen model constrains and what evidence is needed to interpret a factor, rather than treating every hidden coordinate as the same kind of object.
- Proposed treatment: begin from the actual matrix entries and reconstruction `X ≈ WH`, with observation rows, nonnegative activation columns and nonnegative component rows. Additivity is guaranteed by the constraints; recognizable parts, sparsity, independence, uniqueness and a physiological interpretation require further conditions or evidence. Explain any normalization before giving factors a probability interpretation. Link back to ICA's assumptions and GMM's normalized conditional responsibilities. These contrasts need a short local bridge, not a repeated full lesson.
- Concrete proposed example, derived by direct multiplication: let `X = [[2,1,3],[1,2,3],[3,3,6]]`. Both `W1 = [[2,1],[1,2],[3,3]]`, `H1 = [[1,0,1],[0,1,1]]` and `W2 = [[1.5,.5],[.5,1.5],[2,2]]`, `H2 = [[1.25,.25,1.5],[.25,1.25,1.5]]` reconstruct X exactly. All factors are nonnegative, yet H2's rows are mixtures of H1's rows, beyond mere scaling/permutation. A two-ray cone and an aligned additive reconstruction can make the ambiguity visible. Confirm this fixture and design an independent variation during authorized authoring; no runtime figure has been built here.
- Accuracy checks to carry forward: distinguish objective values decreasing toward a limit from parameter convergence, stationarity, local optimality and global optimality. For bound-constrained NMF, a zero gradient is not the whole stationarity contract at the boundary, and stationarity alone is not a local-minimum certificate. Teach zero locking and the exact update/initialization assumptions when introducing multiplicative updates. An auxiliary-function resemblance to EM does not transfer every convergence statement from one algorithm to another.
- Evidence inspected 12 September 2026: [Donoho & Stodden, NIPS 2003](https://papers.nips.cc/paper_files/paper/2003/file/1843e35d41ccf6e63273495ba42df3c1-Paper.pdf), introduction and §§2–3 on interpretation and nonuniqueness; [Lee & Seung, NIPS 2000](https://papers.nips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf), algorithm abstract and original paper located; [Lin, 2007](https://www.csie.ntu.edu.tw/~cjlin/papers/multconv.pdf), abstract and §§I–II on objective monotonicity, KKT conditions and the need for a distinct convergence argument. Full proofs and current library semantics remain for the NMF author to inspect.
- Prerequisites/boundaries: matrix multiplication, nonnegative weighted sums and the preceding lessons' distinctions between constraints and interpretation. Retain the existing useful applications but verify their actual assumptions; do not claim a successful decomposition identifies true sources automatically.
- Resolution: researched and incorporated into the authorized content-only revision on 12 September 2026; see [the complete manuscript](../drafts/non-negative-matrix-factorization-nmf/lesson.md), especially §§1,3,4,7, and [its source/author record](../drafts/non-negative-matrix-factorization-nmf/design.md). Both proposed products were checked exactly; the manuscript distinguishes additivity, normalization, stronger factor ambiguity, objective decrease and boundary stationarity. Status remains open until implementation consumes the packet; this is not a production implementation claim.
- Implementation/verification links: implemented on 2026-09-14; see the closing entry below.

## 2026-09-14 — Implementation consumed the packet; the note closes

- Status: resolved.
- Phase two implemented this topic from the saved content packet. The latent-meaning bridge, the two
  exact factorizations, the normalization contract and the convergence distinctions this note asked
  for are all on the published page, and each is now checked rather than asserted.
- Where each proposal landed: the additive reconstruction `X ≈ WH` with observations in rows opens
  §1 and drives the first inline figure and the mixture investigation; the interpretation contract —
  additivity is guaranteed, parts, sparsity, independence, uniqueness and probabilities are not —
  is a single callout in §1, stated once for the whole lesson; the GMM/t-SNE/ICA/NMF contrast sits
  immediately beneath it; normalization before any probability reading is §4, with the exact
  `[.5,0,.5]`/`[0,.5,.5]`, activations `[4,2]`, total 6 and proportions `[2/3,1/3]`; zero locking,
  boundary stationarity and the "an epsilon changes the algorithm" caution are §3 with a dedicated
  static contrast figure, and the nonnegative first-order conditions are derived in §7.
- Both proposed products were confirmed exactly and are now a runtime model check, not a claim:
  `W1 H1 = W2 H2 = X`, with a two-cone figure whose containment is computed from cone coordinates
  rather than drawn by eye. The scale-invariance strip beside it shows a doubled H row and a halved
  W column leaving the product alone.
- Verification: `scripts/verify-nmf-models.mjs` (21 grouped checks), `scripts/verify-nmf-data.py`
  (224 checks, every recorded value in `calculated-inputs.json` recomputed), `scripts/verify-nmf-examples.py`
  (3 displayed programs executed, 25 oracle assertions, recorded output proven against a fresh run)
  and `scripts/verify-nmf-browser.cjs` (12 cases, 29 screenshots at five widths). Evidence under
  `docs/teaching/evidence/nmf-*.json`; the phase-two record is in
  [the design file](../drafts/non-negative-matrix-factorization-nmf/design.md#phase-two-implementation--14-september-2026).
- Nothing here authorizes work on another topic, and the phase ledger is closed by the integration
  owner rather than by this note.
