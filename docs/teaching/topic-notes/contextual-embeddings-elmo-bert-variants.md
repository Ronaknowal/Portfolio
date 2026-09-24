# Authoring notes: Contextual Embeddings (ELMo, BERT Variants)

Canonical topic ID: contextual-embeddings-elmo-bert-variants

## 2026-09-22 — ModernBERT and mmBERT as reusable decision-model backbones

- Status: open
- Origin: [Typed decision-model curriculum plan](../../curriculum/TYPED-DECISION-MODELS-PLAN.md).
- Destination and ownership rationale: this topic owns modern bidirectional representation mechanisms and pretrained-encoder use. The typed decision-model topic owns its task-conditioned option head and downstream decision pipeline; the landmark topic owns architectural/historical comparison.
- Idea and learning benefit: explain which properties of a bidirectional encoder matter when a learner reads representations at option markers, adapts a backbone, or selects a multilingual checkpoint.
- Existing coverage: the catalogue has this planned BERT-variants owner; the prior coverage review did not find an explicit ModernBERT/mmBERT treatment. This note does not imply that a manuscript has been written.
- Proposed treatment: compare BERT with ModernBERT from primary sources, covering positional/masking/local-global attention and implementation efficiency only after verifying the chosen model/version. Explain tokenizer, hidden-state/attention-mask contracts and how advertised context differs from a downstream checkpoint's configured input budget. Treat multilingual representation quality as empirical, not guaranteed by the family name.
- Explanation/example: reuse earlier transformer components to inspect bidirectional token representations, then connect a standard encoder's hidden states to a small custom marker or pooling head. Test padding, candidate positions and a changed language, retaining clear limits for small fixtures.
- Prerequisites and boundaries: links to Self-Attention, Transformer Block Architecture and Positional Encodings should reuse actual explained programs. Fine-tuning/calibration of a typed decision pipeline belongs to `typed-decision-models-calibrated-neural-decision-systems` and its project.
- Evidence: [ModernBERT paper](https://arxiv.org/abs/2412.13663), [mmBERT-base model card](https://huggingface.co/jhu-clsp/mmBERT-base), and [Laya repository](https://github.com/NandhaKishorM/laya), inspected 22 September 2026. Pin exact source and checkpoint versions when writing; no models were executed for this planning note.
- Resolution: not yet reviewed by the lesson author; include, adapt or explicitly reroute with reasons during topic preflight.
- Implementation/verification links: none yet.
