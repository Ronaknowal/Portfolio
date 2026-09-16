# Splice-junction data and experiment provenance

Retrieved 13 September 2026 from the [official UCI dataset 69 page](https://archive.ics.uci.edu/dataset/69/molecular%2Bbiology%2Bsplice%2Bjunction%2Bgene%2Bsequences), whose current license is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: primate GenBank 64.1 examples; donors Geoffrey Towell, Michiel Noordewier and Jude Shavlik. Preserve attribution and license with redistributed data. The original `splice.names` file was read in full and contains no conflicting use restriction. These are historical public DNA windows, not new patient records.

Official download: `https://archive.ics.uci.edu/static/public/69/molecular+biology+splice+junction+gene+sequences.zip`. Archive SHA-256: `3e7ce5dcbeec8c221f57dda495611b9d6ec9525551f445419f5c74cc38067e4e`. Only two byte-original members were retained:

| File | SHA-256 |
| --- | --- |
| splice.data | ebbae10c85d3c285e2a2489a37e42bc2ee8d91d15005e391a9962fa3389a114f |
| splice.names | dd67d612a9c57a1230fbeed1e6e0a2130559f419b01765799627236bc7f6fff1 |

There are 3,190 rows, each containing a class, instance identifier and 60 characters. Raw class counts are EI 767, IE 768 and N 1,655. Character counts: C 50,300; G 50,245; T 46,308; A 44,487; N 56; D 2; R 1; S 1. The metadata's 62 attributes include the label and identifier. Array indices 0–29 correspond to −30…−1 and indices 30–59 to +1…+30. Ambiguity symbols D=A/G/T, N=A/C/G/T, R=A/G and S=C/G remain separate tokens. Input N means an ambiguous base; output class N means neither boundary.

Original metadata §3b interchanges donor/acceptor terminology; §4 gives the explicit EI=exon/intron and IE=intron/exon directions used here. The manuscript does not copy the misleading description of “superfluous DNA” being removed. Splicing processes RNA; the linked NHGRI background supports that correction.

## Processing and data roles

One-based source IDs identify raw lines. Group rows by exact DNA string: there are 3,005 distinct strings. One group, IDs 1022 (IE) and 1969 (N), has conflicting labels; exclude both. Keep the first source occurrence of every other sequence, removing 184 additional repeat rows and leaving 3,004 examples. Retained class counts are EI 679, IE 672 and N 1,653. The complete duplicate-group list is in `splice-results.json.data.duplicate_groups`.

Before deduplication, parse the source-record prefix as text before `-DONOR-`, `-ACCEPTOR-` or `-NEG-`. Union prefixes connected by identical DNA strings. Use the resulting connected groups for role assignment, retaining the visible prefix relationships among different windows. This is a specified identifier rule, not a validated gene-family or patient identifier. Homology and other biological relationships can remain across apparently different groups.

`GroupShuffleSplit(test_size=.30, random_state=1991)` splits fit groups from the remainder. A second `GroupShuffleSplit(test_size=.5, random_state=1992)` divides the remainder into validation and assessment. Sizes are 2,128 / 460 / 416 rows and 1,008 / 216 / 216 groups. Corresponding EI/IE/N counts are [488,488,1152], [106,101,253] and [85,83,248]. Exact source IDs, group names and role lists are saved. The author checked that groups never overlap across roles. Every model uses these same roles and fixed encoding; identifiers are not input features.

## Executed experiment and retained outputs

`splice_models.py` is complete instructional code. Four fits were declared before the run: linear-29, gated-29, ungated-29 and gated-71. All use 80 epochs, Adam at .003, batches of 128 and gradient clipping at 1. The earliest minimum validation cross-entropy selects the checkpoint. The gated model has width 16 and two blocks, each with one long filter, normalization and residual feed-forward updates. The ungated comparison retains nonlinear feed-forward layers but replaces q and k with ones. Nominal parameter counts are 1,443 for the linear model and 6,771 for each sequence model; unused projection branches in the ungated model mean nominal counts are not equal effective capacity.

`splice-results.json` retains the protocol, all 80 validation measurements per fit, selected epochs, role confusion matrices and metrics, and post-fit short-filter/gate-removal interventions. `splice-fits.npz` holds selected weights, buffers and role logits as arrays compatible with `allow_pickle=False`. Assessment data did not choose epochs; all fixed candidates are reported. Post-fit interventions are diagnostics, not separately retrained models or a selection for an assessment winner.

`author_calculations.py` reloads every fit and checks all 460 saved validation logits per fit (maximum difference 0), role-group separation, direct/FFT output and gradient agreement, prefix causality and two actual sequence-edit fixtures. `author-results.json` retains these checks, source IDs, strings, logits, probabilities, the all-N response and actual fitted filters. Source row 3 is the worked example; source row 4 is the fresh investigation. Their synthetic AA edits have no measured biological label; the original label does not transfer to an edited sequence.

Native versions: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, scikit-learn 1.9.1, scipy 1.18.1; two CPU threads. All four fits and bounded author calculations actually executed. Position coordinates use a fixed reference length of 60. Prefix/future-edit float32 differences are approximately 2–3e−6; the direct/FFT double-precision gradient difference is 1.42e−14. No large pretrained HyenaDNA or StripedHyena download, million-token benchmark, browser lab, rendered review or formal phase-two campaign was performed.

The website and original JSX remain unchanged. Preserve this packet for implementation. Temporary primary-paper text can be removed after the research record closes. External paper or repository licenses do not automatically apply to future pretrained weights; inspect a chosen artifact's license if later using one.
