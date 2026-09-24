# Loss functions: implementation-depth revision

Canonical ID: `loss-functions-ce-mse-focal-contrastive-triplet`.

The existing full packet, labs and real digit experiment remain. The new **Build the objectives, then control the library** section closes the gap between formulas/library calls and an explicit, inspectable gradient-to-update implementation. The canonical additions are `public/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/loss-mechanisms.py` and the corresponding manuscript section. The topic generator produces the body and lazily displayed code from these sources. The program is loaded only when its disclosure opens.

| Mechanism | Ownership and bridge |
| --- | --- |
| MSE, MAE, Huber and quantile | New NumPy objectives + explicit input derivatives; torch standard function/composed quantile comparison |
| Stable CE, BCE and focal | New stable log-domain computation + explicit derivative; exact raw logits and mean conventions matched |
| Squared triplet and paired InfoNCE | New value/gradient implementation, including cosine-normalization derivative; matched custom-distance module/normalized candidate CE |
| Pair-contrastive objective and semi-hard miner | Reuse exposed tensor-primitive functions in existing `loss-experiments.py` rather than duplicate the owner |
| Parameter update | New same-state affine classifier manual SGD step compared with `nn.Linear`, CE and SGD; weight orientation stated |
| Autodiff engine | Reuse preceding Backpropagation lesson with a direct route; no new engine |
| Smoothed labels | Explained independent extension, hint and full solution; compare epsilon zero and .2 and gradient row-sum invariant |

The routines expose their finite-input/dtype assumptions and supported reduction/target contracts. Dense CE is O(NC) time/storage including returned gradients and avoids allocating a one-hot target matrix. Full paired InfoNCE is O(N²D) time and O(N²) score storage; the text explains why changing candidates and exact chunking are different interventions. No runtime benchmark or universal optimality is claimed. Existing real-data fits are not rerun because their source and outputs are unchanged.

Author checks: both displayed NumPy/PyTorch routes executed in the existing CPU environment; loss and derivative parity, extreme logits including correctly and incorrectly classified tails, and a matched SGD update. Exact output is served beside the script. Independent review and production integration are recorded separately; this record is not itself their evidence.

Primary API contracts consulted 21 September 2026: [CE](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) and [custom-distance triplet](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TripletMarginWithDistanceLoss.html). Formula derivations and scope are local. The scalar log-sum-exp primitive is permitted; an opaque loss function is not used in the NumPy objective.
