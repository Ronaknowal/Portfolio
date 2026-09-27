# Independent teaching review

long-context-sequence-models-transformer-xl-griffin-perceiver

Read the complete current canonical lesson, all eight practices and scratch/library implementation explanations, including relative position, RG-LRU gates, Perceiver IO/AR, detached gradients and resource counts. Checked the new isolated distance example against Transformer-XL section 3.3 and the distinction between global and query-dependent preferences. The full lesson keeps cache, state and retained input bank separate and preserves declared small-model limitations. No correction found; current source-bound generator and trained outputs retained.

## Actual checks

Independent source reading: [Transformer-XL, section 3.3](https://arxiv.org/pdf/1901.02860). Checked the separate query-dependent and global positional terms and lower-layer detached-memory construction against the new example.

- Independent complete lesson reading and concept-transition assessment: passed.
- Current author source bindings and JSX parse: passed.
- Relative preference reverses with query, not common translation: passed.
- RG-LRU gate isolates decay from zero writes: passed.
- Memory output, budget and detached derivative arithmetic: passed.

Reproducible command: `node scripts/verify-concept-intuition-independent-representation.mjs long-context-sequence-models-transformer-xl-griffin-perceiver`. The arithmetic checks were authored independently of the topic author's verifier. Exact reviewed source hashes are in independent-checks.json. No new blocking finding. Browser paint and interaction checks remain separate with root; this record does not claim screenshots or a new fitted-data campaign.
