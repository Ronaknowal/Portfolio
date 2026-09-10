export default {
  summary: 'Connect an actual lossy encoder and decoder to the minimum information required at a chosen error budget, then derive and compute binary, finite-alphabet and Gaussian rate-distortion limits.',
  outcomes: ['Encode and decode a complete small block code and measure its average and worst-case error', 'Distinguish an information lower bound, finite code length and an asymptotic coding theorem', 'Derive the binary rate-distortion curve including biased sources and its zero-rate endpoint', 'Compute a finite-alphabet tradeoff with checked probability tables and optimization bounds', 'Allocate distortion across independent Gaussian components with explicit rate and error units', 'Choose a distortion measure and assess actual payload size, fidelity and relevant failures'],
  prerequisites: ['Entropy, Cross-Entropy & KL Divergence', 'Mutual Information & Information Bottleneck'],
  sequence: ['Build a three-bit compressor', 'Define rate, distortion and the joint-law constraint', 'Derive binary limits and explain block coding', 'Compute finite rate-distortion tradeoffs', 'Derive Gaussian limits and distortion allocation', 'Choose fidelity and practical coding measures', 'Solve changed-constraint practice and a reproducible codec task'],
  visual: {
    type: 'Bit-string codebook and reconstruction ledger, calculated frontier, evolving conditional-probability matrix and Gaussian variance reservoirs',
    question: 'Which information crosses the bitstream, which error is permitted, and how close is this actual method to a theoretical limit?',
    interaction: 'Change source bias, codebook, error budget, optimization step and distortion allocation; inspect actual reconstructed symbols, probability sums, bounds and units.'
  },
  practice: {
    task: 'Build and verify a changed codebook, repair an invalid curve, diagnose a support-limited optimizer, allocate a Gaussian error budget and design an audio evaluation protocol.',
    success: 'Decoding uses only the declared payload and shared codebook; finite averages, endpoints, objective bounds, rate units and application failure checks are correct.'
  },
  misconceptions: ['A test channel is an implemented compressor', 'One minus binary entropy is valid for every distortion budget', 'A finite exact zero-error fixed-length code always reaches source entropy', 'The output marginal is fixed as in ordinary transport', 'A zero initialization can recover a missing reconstruction symbol', 'Latent dimensions or a training loss equal actual encoded bits', 'Matching the output distribution guarantees fidelity to the input', 'An average error constraint protects every rare case'],
  sources: ['https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/resources/mit6_441s16_chapter_23/', 'https://www.cs.cmu.edu/~aarti/Class/10704_Spring15/lecs/lec16.pdf', 'https://proceedings.mlr.press/v97/blau19a.html', 'https://nptel.ac.in/courses/117101053'],
  depth: 'specialist',
  designRecord: 'docs/teaching/RATE-DISTORTION-LESSON-DESIGN.md',
  reviewFocus: 'Fixed-length versus entropy-coded rates, iid/separable and endpoint assumptions, backward binary/Gaussian constructions, finite optimizer support and certified error, Gaussian component units, and actual fidelity versus marginal-distribution criteria.'
};
