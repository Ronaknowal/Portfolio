# Independent teaching review

state-space-models-s4-mamba-mamba-2

Read the complete current canonical lesson, all nine practices, native and specialist-library explanations, and advanced HiPPO, DPLR, SSD, S5 and Mamba-3 branches. Checked the transposed FFT table after the browser finding: times 0,1,2,3 carry values 1,2,1,2 into length-three slots 0,1,2,0. Independently inspected Mamba-3 section 3.1 and appendix A.2 for the lambda-dependent accuracy condition and Tri Dao SSD algorithm article for chunk/state decomposition. The new Woodbury example retains coupling and is correctly scoped. No content correction found; no GPU execution or refitting claimed.

## Actual checks

Independent source reading: [Mamba-3, section 3.1 and appendix A.2](https://arxiv.org/html/2603.15569v1), especially the condition on the interpolation weight for the stated order of approximation; [Tri Dao’s SSD algorithm explanation](https://tridao.me/blog/2024/mamba2-part3-algorithm/), including local outputs, chunk states, state passing and incoming-state output contributions. These readings support the mathematical distinctions rather than importing reported speedups.

- Independent complete lesson reading and concept-transition assessment: passed.
- Current author source bindings and JSX parse: passed.
- Independent explicit DFT exposes circular alias and correct padding: passed.
- Woodbury example agrees with direct two-by-two solve: passed.
- SSD recurrence versus independently expanded influence: passed.
- Endpoint write rule and bounded-state bytes: passed.

Reproducible command: `node scripts/verify-concept-intuition-independent-representation.mjs state-space-models-s4-mamba-mamba-2`. The arithmetic checks were authored independently of the topic author's verifier. Exact reviewed source hashes are in independent-checks.json. No new blocking finding. Browser paint and interaction checks remain separate with root; this record does not claim screenshots or a new fitted-data campaign.
