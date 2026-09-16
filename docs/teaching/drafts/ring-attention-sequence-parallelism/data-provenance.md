# Ring Attention: data and calculated evidence

Research-and-writing checkpoint, 13 September 2026. No GPU execution, new fitting campaign, browser implementation or hardware benchmark took place in this topic.

## Real input and frozen model

Dataset: [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), DOI 10.24432/C5GC82. Credit Daniel Baptista Dias, Sarajane Marques Peres and Helton Hideraldo Bíscaro. License: CC BY 4.0. Retain attribution when publishing transformed arrays. The original data contains 360 rows: 90 numbers representing 45 normalized two-dimensional hand-centroid points, followed by a movement-class label from 1–15. These are movement features, not raw video, complete sign-language translations or identity/session annotations.

The packet's movement_libras.data is an exact copy of the preserved Self-Attention data. SHA256: 97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d. No network refetch or invented rows. The UCI schema/license was verified during the current authorized research; the frozen source packet retains names/archive provenance.

movement-attention-model.json stores all weights, architecture metadata, original example probabilities and the SHA256 of its source model file in the Self-Attention packet. The source is unchanged. The selected model is attention-2, seed 101: a 2→24 stem with tanh; bias-free 24→24 Q/K/V/output projections; two heads of width 12; residual addition; mean pooling; and a 24→15 classifier. Total parameters: 2,751. Inputs use the fixed transform 2x−1. There is no dropout, explicit position encoding, causal mask, normalization layer or hidden training-time state.

The earlier study removed 30 exact duplicate trajectories, keeping earliest source IDs and compatible labels, leaving 330 distinct examples. Per-class RNG seed 73 produced 220 fitting, 50 validation and 60 assessment trajectories, with four assessment examples per class. Four model kinds each used seeds 101/102/103 and 200 full-batch Adam epochs, learning rate .01, weight decay 0. Checkpoints were selected by highest validation macro-F1, then lower cross-entropy, then earlier epoch. The seed-101 two-head visual model was declared there in advance. This inherited experiment was not refitted here, and its assessment results were not used for new model selection.

The new systems comparison uses source 77 (class 4, inherited worked probe) and fresh source 20 (class 1). It computes the full model in NumPy float64 from saved float32 parameter values, holding weights and input fixed between dense and partitioned execution. Four owners hold 12/11/11/11 positions. Source 77 predicts class 5; source 20 predicts class 2. Both errors remain visible.

Actual edits are source 77/frame 23/x + .10 and source 20/frame 10/x − .15, clipped to [0,1]. Code uses zero-based array indices; source/frame labels in prose are one-based. Before/after points, Q/K/V, attention weights, dense outputs, probabilities and owner traces are retained. Reversed ring direction is computed separately. Labels do not enter inference.

## Calculation contracts

ring-attention-reference.py is a complete executable NumPy CPU reference for one sequence, equal Q/K width, equal MHA head count, arbitrary value width, no dropout/additional score bias and a global boolean square mask. Its empty-query convention is output 0 and LSE −∞. Ownership must be an exact nonempty partition; uneven sizes work, while P > L yields an empty owner and is rejected. The program allocates full tensors and local score blocks. Its central backward sums model required remote reductions without transporting messages. Inputs must satisfy the documented finite-array/shape contract; browser input validation is specified separately.

attention-partition-study.py executes:

- Stable scalar merging, changed records and a common score offset.
- Fixed RNG-91 asymmetric Q/K/V and upstream arrays of shape 2×7×3.
- Dense, causal, packed-causal and one-empty-row masks, with three uneven owners.
- Reversal and rank-renaming nulls.
- Direct derivatives, Torch automatic derivatives for valid-row cases and selected central finite differences with step 1e−5.
- Deleted document metadata, changed value and incomplete gradient-accumulation contrasts.
- Complete frozen real-model inference and point edits.

The JSON stores finite reported results; empty-row infinities are not serialized as numeric answers. Maximum float64 dense/ring differences are below 2.23e−16 for the constructed cases and 2.23e−15 for selected real outputs. Direct/autograd gradient discrepancies are below 6.67e−16; selected finite-difference errors are below 1.85e−11. The maximum output changes for deleted packed-document metadata and the fresh value edit are 1.7043724677 and 1.0618276716. Incomplete key-gradient accumulation differs by .8648028024.

These checks compare distinct derivative formulations/estimators, not a helper against itself. Source 77 NumPy probabilities differ from the inherited float32 probe by at most 4.86410143e−8. The source-model training suite was not reopened.

systems-calculations.py enumerates L16/P4 and fresh L12/P3 contiguous, striped and paired-chunk zigzag ownership. It counts global causal pairs and all cells of any tile containing a valid pair. Tile sizes are 1/2/4; a synchronized round is charged its maximum rank work. This is a declared toy scheduler, not FlashAttention's kernel behavior. It also computes the labeled L8/H4/P2 Ulysses transformation, target-boundary and weighted-loss examples, rotary-dot counterexample, bytes, FLOPs and timelines.

Hypothetical timing parameters are B = 1, Hq = 8, Hkv = 2, d = 64, two-byte stored values, effective compute 100 TFLOP/s, effective one-direction bandwidth 50 GB/s and latency 2µs. Only two GEMMs are counted. C = 4c²Hqd/F and D = latency + 2cHkvd·bytes/R. For P devices, compute-only time is PC, serial time is PC+(P−1)D and ideal overlap is C+(P−1)max(C,D). No measured hardware name or performance ranking is attached.

The forward memory inventory lists Q, current KV, next KV, FP32 numerator, two FP32 row statistics, distinct output and one 128² score tile per head. Kernels may alias/fuse arrays; training adds other storage. This is not measured allocator peak memory.

## Reproduction and retention

Executed environment: Python 3.12.14, NumPy 2.3.5 and Torch 2.14.0+cpu, with one Torch thread. Run the reference and the two study programs with the complete packet present. No downloads are required.

Data, model, semantic programs and JSON results are necessary handoff assets. Phase two should extract only the model/examples required by the current lab, retaining provenance and checking parity, then load them on demand. The full author evidence is not an eager browser dependency. No generated images, videos, temporary report copies or duplicate environments were added.

Lesson results are independently calculated arithmetic, inherited fixed-model evidence, newly calculated frozen-model inference, hypothetical cost models or separately cited research. Formal independent review and browser/backend verification remain deferred.
