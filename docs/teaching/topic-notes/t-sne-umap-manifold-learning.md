# Authoring notes: t-SNE, UMAP & Manifold Learning

Canonical topic ID: t-sne-umap-manifold-learning

## 2026-09-11 — Separate a learned layout from verified input topology

- Status: open
- Origin: scoped design for [Topology & Topological Data Analysis](../TOPOLOGY-TDA-LESSON-DESIGN.md).
- Destination and ownership: this lesson teaches dimensionality reduction and its evaluation; it is the right home for auditing a projection's neighborhood and topological distortions. TDA will teach its own metric/filtration dependence and link this distinction, without rewriting the manifold-learning body.
- Existing coverage: inspected the actual lesson sections1–2. It already warns about t-SNE cluster sizes and distances, but then asserts that plotted close points were necessarily close in the input. It also calls UMAP faster at every scale and assigns its graph weights a literal probability of a true topological connection. These statements need scoped investigation and qualification; the plot is an optimized representation, not an exact preservation certificate.
- Learning benefit: distinguish a geometric object, the input distance matrix, the constructed neighbor graph and the learned low-dimensional coordinates. A hole drawn in a projection may be created, lost or stretched by the projection; persistence computed before and after it answers different questions.
- Proposed treatment: retain useful algorithms and figures, but add a matched-input comparison that identifies false neighbors and missing neighbors, with a defined trustworthiness/continuity or distance measure. Compare filtration output in input space and projected space on a deliberately small example; state what was actually preserved. Do not use the image's apparent separation as proof. If performance comparisons remain, name implementations, data, parameters, hardware and measured scope.
- Prerequisites/boundaries: introduce distance and neighborhood conventions, stochastic optimization and what topology summary is being compared. A small controlled projection counterexample belongs here; full persistent-homology derivation belongs to TDA. This note does not prescribe a universal best embedding.
- Evidence: [scikit-learn manifold guide](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne), section2.2.9, inspected10September2026 UTC, documentation1.9.1: nonconvex objective, initialization/parameter dependence and non-guaranteed global preservation. [UMAP FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html#how-umap-can-go-wrong), inspected same date, documentation0.5.8: disconnected/far-point normalization can create misleading similarities and destroy structure. The FAQ itself contains dated broad claims elsewhere; this is not a blanket endorsement.
- Resolution: not yet assessed by the destination author. No native UMAP/t-SNE experiment or benchmark was performed during this scoped discovery.
- Implementation/verification links: none yet.
