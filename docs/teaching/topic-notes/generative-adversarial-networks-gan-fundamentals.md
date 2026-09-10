# Authoring notes: Generative Adversarial Networks (GAN Fundamentals)

Canonical topic ID: `generative-adversarial-networks-gan-fundamentals`.

## 2026-09-10 — Separate the population discrepancy from practical generator updates

- Status: open
- Origin: [f-Divergences & IPMs design](../F-DIVERGENCES-IPMS-LESSON-DESIGN.md), mathematical critic bridge.
- Destination and ownership rationale: the GAN topic is planned and has no individual brief/body at this scoped review. Its training-loop prerequisites make it the appropriate owner for actual finite networks, discriminator capacity, alternating updates and gradient diagnostics. The mathematics topic derives exact population relationships without a neural training lab.
- Idea and learning benefit: distinguish the unrestricted population discriminator optimum, a trained empirical discriminator, the minimax generator loss and the commonly used non-saturating loss. A small loss number or a mathematical optimum should not be mistaken for a calibrated divergence or proof of converged generation.
- Proposed treatment: trace one tiny distribution through discriminator fitting and a generator update; show generator gradients for both losses and inspect held-out samples. If WGAN/f-GAN variants are owned by another exact topic, route those implementation details there while retaining the correct bridge here.
- Explanation/example: for equal source priors, D*(x)=p/(p+q), and the maximized population log objective equals −ln4+2JS under the half-sum JS convention. The non-saturating loss −E_Q lnD is a different generator objective. For singular moving atoms, exact JS is constant off equality while W1 depends on physical displacement, but a restricted approximate critic and its regularization can still fail; avoid blanket useful-gradient promises.
- Prerequisites and boundaries: probability/log loss, parameter gradients and alternating training; conceptual f-divergence/IPM link available. Compare actual gradient calculations against an independent finite difference, not only decreasing displayed objectives.
- Evidence: original [GAN paper section4.1](https://arxiv.org/pdf/1406.2661), [f-GAN section2](https://arxiv.org/pdf/1606.00709), [WGAN section2](https://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) read 10 September 2026 within that mathematical scope. Current library interfaces and implementation behavior remain for the future author to verify.
- Resolution: not yet assessed by the destination author; no GAN implementation is claimed here.
- Implementation/verification links: [f-Divergences/IPMs source verification](../F-DIVERGENCES-IPMS-VERIFICATION.md); destination work remains open.
