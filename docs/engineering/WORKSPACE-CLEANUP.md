# Workspace cleanup — 11 September 2026

The user requested removal of obsolete working material and instructions that caused unnecessary repeated checking before continuing the first ten Classical ML topics.

## Completed cleanup

- Removed **113 loose scratch files**: 112 unreferenced one-off editing utilities and the Decision Trees lesson draft, proven byte-identical to its installed production source. Their 277,911 bytes are recoverable from one 136,369-byte ZIP in [the historical archive](../archive/workspace-cleanup/README.md).
- Preserved active work, the shared Python environment, original-source archives, verification outputs, referenced screenshots, nonidentical drafts and required datasets. The full scratch inventory contained 597 top-level entries and about 2.72 GB; the Python runtime alone accounted for about 1.55 GB. This cleanup reduces obsolete working instructions, rather than claiming most of that necessary storage was reclaimed.
- Shortened the active handoff to current scope, teaching contracts, relevant links and a concrete resume checkpoint. Detailed historical results remain in their existing records.
- Added explicit bounded verification and scratch-retention rules. Passing checks on unchanged source are reused. Repairs trigger affected checks; completed modules and old integration runs stay closed unless a relevant change or concrete concern justifies reopening them.
- Updated the ten-topic ledger with the already recorded author-review states and the outstanding regression model amendment. No lesson was marked independently reviewed or integrated by cleanup.

## Verification and recovery

The exact [cleanup manifest](../teaching/evidence/workspace-cleanup-2026-09-11.json) records removed file hashes, archive hash, duplicate-source proof and retained active paths. All **1,930 existing source, tooling and configuration files** matched their pre-cleanup fingerprints. No application source, topic, publication mapping or curriculum order changed during cleanup. No lesson tests or application build were rerun for these documentation/file-retention changes.

Automatic approval review rejected the initial broader deletion because it included drafts and verification material that might still be needed. That broader deletion did not run. The completed alternative was narrowed to editing utilities and the proven duplicate, with a byte-verified recovery archive created before removal. Verification artifacts and other uncertain files remain in place.

Continue from [the current implementation checkpoint](../../CLASSICAL-ML-SUPERVISED-IMPLEMENTATION.md#resume-checkpoint--after-workspace-cleanup-11-september-2026), not from archived utilities or scratch folder names. This record is a completed cleanup report, not a recurring task.

## Additional image and workspace cleanup

The user's follow-up explicitly requested removal of unnecessary images and temporary material outside scratch, followed by resuming implementation. The [additional manifest](../teaching/evidence/working-artifact-cleanup-2026-09-11.json) records **2,222 removed obsolete screenshots (187,565,742 bytes)** and **35 removed temporary files outside scratch**. All 5,327 images matched to evidence/source references were retained, along with active Classical ML material. Additional uncertain/unreferenced images were not automatically deleted merely for lacking a direct reference.

Outside cleanup covered the old `tmp_moe` working scripts/data, empty `tmp_vit`, `tmp_*.py`, `scratch_neural_ode.py`, the zero-byte `npm` redirect artifact, CatBoost training logs and the outer `_curation_scratch` / `_synth_scratch` working files. Their exact small contents are recoverable from `docs/archive/workspace-cleanup/2026-09-11-legacy-topic-working-files.zip` (57,667 bytes). Shared Python/Node dependencies, source, actual runtime images, build configuration and active review data were preserved.

The new [working-artifact retention policy](WORKING-ARTIFACT-RETENTION.md) makes cleanup part of topic completion: keep selected final evidence and remove superseded captures, installed duplicate drafts and temporary tool outputs. The optional image audit only identifies candidates; it is not a periodic task or authorization to delete every unreferenced file.
