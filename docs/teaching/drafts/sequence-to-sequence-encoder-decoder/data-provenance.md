# English inflection extract: provenance and evaluation contract

Retrieved13 September2026. Dataset: **UniMorph English** repository contributors. The pinned repository README names **Wikipedia** as its source and links **Creative Commons Attribution–ShareAlike3.0**. UniMorph publications discuss Wiktionary extraction more generally; this packet preserves the repository's actual attribution instead of silently substituting that broader description.

- Repository/source revision: [66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b](https://github.com/unimorph/eng/tree/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b).
- [Pinned README](https://github.com/unimorph/eng/blob/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b/README.md).
- [License: CC BY-SA3.0](https://creativecommons.org/licenses/by-sa/3.0/). Retain attribution, source/revision, changes and this license with the adapted data, including any phase-two subset. No endorsement implied.
- [Project schema](https://unimorph.github.io/schema/) and [UniMorph4.0 paper](https://aclanthology.org/2022.lrec-1.89/).

Source file `eng`:18,022,905bytes,652,477 nonempty rows,0 malformed rows,385 repeated exact triples. SHA256:

`20a191cefdc7cad6fa74b00f49d6f658684f17b14541aae372e5a3d5a8c15c67`

Each source row is lemma, surface form, feature bundle. Select three verb bundles: `V;PST`→past; `V;V.PTCP;PRS`→present participle; `V;PRS;3;SG`→third-person-singular present. The compact names are task inputs; this is not a redesign of the UniMorph schema.

The filter accepts lowercaseASCII lemmas3–8 characters and forms3–12. Within those filtered records there are14,174 lemma groups,0 multiple-form ambiguities and14,162 complete groups with one form for each of the three requests. This is a statement about the filtered records, not proof that every English lemma has one acceptable form or that the full source is ambiguity-free.

Order eligible lemmas by SHA256 of `seq2seq-inflection-v1:` plus the lemma. Select the first600, initially placing the first450 in training and150 in development. All three requests for a spelling stay together. Source duplicate triples collapse into one record with all original one-based row numbers retained.

A second audit groups selected spellings connected by any shared target form. A group is assigned to training if any member was in the initial training set. This moves all three `work` entries because training `worke` shares `worked`/`working`. It produces **451 training lemmas/1,353 examples and149 development lemmas/447 examples**, with0 cross-partition lemma strings and0 cross-partition surface forms.

This is conservative: a shared surface form can connect genuinely different lexemes, and aliases without a shared selected form can remain undetected. It protects the demonstrated overlap without claiming a perfect linguistic identity graph. It does not use model correctness to choose membership.

The final file `english-inflections.csv` is89,626bytes, SHA256:

`eb56afb2415f267586809803c67574c4fff7998dbf02a70551fd784cf86dbe14`

Columns are partition, lemma, compact request, original UniMorph bundle, form and source row numbers. There is no learned preprocessing from development data. The32-token vocabulary is declared from the task alphabet, not inferred from held-out examples. Encoder inputs include request and sourceEOS; target output includes EOS. No statistical normalization of characters is performed.

The initial literal-lemma-only split was fitted before the shared-form audit. A compact `split-integrity-repair.json` retains the superseded protocol and aggregate results solely as an author audit. All three final fits were rerun after repairing the data split, keeping architecture, optimizer, seeds,1,200-update budget and rule baseline unchanged. **Only final-split results belong in the lesson or browser.** Old model arrays were replaced, not included as another experiment for learners.

The training set estimates parameters. The development set is inspected for checkpoints, error slices and greedy/beam comparisons; it is not an untouched test. The following attention lesson may reuse this precise split with a declared architecture change. Any such reuse remains development and does not permit claiming a new independent final test.

Exact match uses the retained source form plus natural EOS. Character error rate sums edit operations over total reference characters, excludingEOS and padding. The lexicon includes rare/historical entries and is not sampled by frequency in contemporary text. A plausible alternative spelling can score wrong against a single recorded form; preserve original records and inspect provenance before interpreting such errors.

`prepare-inflection-data.py` recreates the final extraction from the pinned public file and calculates its hashes/group audit. It is an optional author/data reproduction tool; the learner's training program reads the already supplied small CSV offline.
