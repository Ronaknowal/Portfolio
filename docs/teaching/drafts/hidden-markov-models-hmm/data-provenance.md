# HMM data and author calculation provenance

Content-first packet, checked 12 September 2026. No browser artifacts or production model are implemented.

## Constructed mechanism data

The Rainy/Sunny and Walk/Shop/Clean model is an explicitly constructed numerical teaching example, not measured weather and not an exact example attributed to Rabiner. All probabilities, trellises, posterior marginals, expected counts, duration values and iteration histories come from the complete [NumPy program](hmm-experiments.py). Its JSON contains exact numerical calculations, not measured runtime or real weather predictions.

Executed using Python 3.12.14 and NumPy 2.3.5 in the existing shared environment. Twelve independent length-30 recordings were sampled from the stated model with seed71. Three seeds3/7/19 received40 exact EM updates; uniform initialization received3. The generating-model score and every initial/updated objective are retained. No display curve was fitted to a desired story.

The program exhaustively enumerates all16 paths for the four-observation fixture, checks evidence and maximum-path parity, checks both margins of pair posteriors, verifies unchanged filtering under a changed future, compares log-space and scaling for400 rare observations, checks a genuinely impossible event, and checks objective nondecrease up to1e−8. Additional bounded calls verified dataset-duplication count/parameter behavior and length-one transition retention. No existing real fits were repeated for these additions.

## Real licensed sequence extract

Reused the CRF packet's two files byte-for-byte. This is a local copy of an already retrieved, source-bound extract, not a claim of a second raw-data download:

- ewt-sequences.json:89,597 bytes, SHA256 e801e665c4e2a6fa002e04ebb52b00b4fbc1421c7f029494565c0ecbaa016a4d.
- data-sources.json:624 bytes, SHA256 823bc17d95f6bc22a80ce857b6c53841fb6872aa04c10769feb5521c3f5a032a.

[data-sources.json](data-sources.json) records exact r2.16 train/dev/test URLs and original SHA256 values. Those source hashes come from the prior CRF retrieval; the local copy hashes were independently checked here.

Source: [UD English EWT r2.16](https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.16), published May2025. The release tag, not an outdated README heading, identifies the version. Read the complete relevant rights/summary metadata and citation in its [README](https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/README.md).

Attribution: Natalia Silveira, Timothy Dozat, Marie-Catherine de Marneffe, Samuel Bowman, Miriam Connor, John Bauer and Christopher D. Manning, “A Gold Standard Dependency Corpus for English,” LREC2014. Annotations/database rights are distributed under CC BY-SA4.0; Stanford annotation copyright2013–2021 and underlying text rights/notices (including Google2012, Yahoo2011, University of Pennsylvania2012 and original authors/public-domain material where applicable) remain applicable. Retain attribution, release, notices, link to the license and identification of this extract/coarse-label transformation when distributing it. Do not describe all underlying text as newly owned by this project.

Extraction keeps integer-ID syntactic word rows, excluding multiword range rows and decimal empty nodes. It preserves original sentence ID, tokens and UPOS, selecting the first120/40/40 sentences of length3–15 in the official train/dev/test files. All200 sentence IDs are distinct. Token counts1188/341/370 are checked. Short-first extraction is deliberately biased and is an instructional resource, not a representative benchmark.

## Declared fitting protocol

Map NOUN/PROPN to0, VERB/AUX to1, all others to2; preserve originals. Lowercase and retain training words occurring≥2, plus one unknown category, giving146 symbols. There are140 unknown development tokens. Estimate supervised initial, transition, emission and lexical occupancy counts with smoothing0.1 or1.0. Each probability set supports a lexical decoder and a Viterbi decoder: two count-based fits, four configurations.

Evaluate only train/development as documented. Select highest development correct-token count; ties prefer the lexical decoder then lower smoothing. Stored configuration1.0/HMM gives270/341 tokens and9/40 whole sentences; matching lexical gives266/341 and8/40. Ten repaired versus six broken decisions are retained through source-ID-aligned rows. Majority-Other is216/341. All40 development predictions, scores and fitted parameters are present. The40 reserved test sentences are not scored by this experiment; educational reuse elsewhere does not make them a fresh global research holdout.

Do not refit from development labels, alter vocabulary from development words, or relabel a posterior path as known latent truth. The model is supervised on coarse tags, not unsupervised discovery of parts of speech.

## Optional native examples and deferred evidence

[hmmlearn-examples.py](hmmlearn-examples.py) is complete, AST-parsed and authored against the0.3.3 documentation/source. hmmlearn is absent from the shared environment, so it is unexecuted. Fixed categorical assertions describe mathematically computed expected results, not claimed native stdout. Gaussian data are explicitly constructed and will be generated by the program; no native fitted values are reported now.

Before phase-two display, execute that exact program in a compatible environment, verify categorical parity and sequence boundaries, review actual Gaussian fit behavior and confirm decoder score semantics. Then implement and independently review browser models, diagrams, grading, accessibility and performance. No phase-two evidence is implied by these author calculations.

