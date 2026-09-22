# Authoring notes: BERT & T5 (Encoder & Encoder-Decoder Landmarks)

Canonical topic ID: bert-t5-encoder-encoder-decoder-landmarks

## 2026-09-22 — Connect encoder landmarks to modern typed decision systems

- Status: open
- Origin: [Typed decision-model curriculum plan](../../curriculum/TYPED-DECISION-MODELS-PLAN.md).
- Destination and ownership rationale: this landmark lesson should explain why encoder-only representations remain useful alongside generative encoder-decoder/decoder models. The [Contextual Embeddings owner](contextual-embeddings-elmo-bert-variants.md) owns detailed ModernBERT/mmBERT mechanisms; the new typed-decision topic owns the custom head and calibrated decision workflow.
- Idea and learning benefit: compare representation extraction and direct candidate scoring with text generation for the same task, without suggesting one architecture universally replaces another.
- Existing coverage: planned BERT/T5 landmark; explicit decision-model bridge is newly proposed, not completed teaching.
- Proposed treatment: add a concise modern continuation using primary sources, link the mechanism owner and `/learn/projects/typed-decision-model`, and explain why non-autoregressive inference can fit structured classification tasks while output-schema compliance remains distinct from correctness.
- Explanation/example: follow the same document through an encoder head and a text-producing model; compare the objects each interface returns and the evaluation evidence each requires. Do not invent comparative latency or quality numbers.
- Prerequisites and boundaries: teach BERT/T5 first; reuse rather than duplicate modern encoder code. Jev's undisclosed architecture is not evidence that it is a BERT derivative.
- Evidence: [ModernBERT paper](https://arxiv.org/abs/2412.13663), [mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base), [Laya repository](https://github.com/NandhaKishorM/laya), and [Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev), inspected 22 September 2026.
- Resolution: not yet reviewed.
- Implementation/verification links: none yet.
