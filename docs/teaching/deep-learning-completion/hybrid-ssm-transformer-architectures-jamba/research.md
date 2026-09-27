# Implementation research — 27 September 2026

Original config rechecked: https://huggingface.co/ai21labs/Jamba-v0.1/blob/main/config.json fields178–225. Current implementation sections read: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/jamba/modeling_jamba.py attention140–186, recurrence261–302, normalization449–520 and router612–642. These source checks confirm the retained top-k probability convention and internal RMS normalization; they do not claim full-model execution.

Public card https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.7 usage102–180 still has differently ordered model identifiers. No gated terms accepted or large weights downloaded. The dated claim and explicit revision requirement remain. Hybrid allocator documentation https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ was located through its Mamba allocation/prefix-cache cases; no new universal backend-support claim inferred.
