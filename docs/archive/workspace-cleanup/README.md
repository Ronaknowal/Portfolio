# Archived obsolete editing utilities

The 11 September 2026 cleanup removed 112 unreferenced one-off editing utilities and one byte-identical installed lesson draft from `scratch/`. Their exact bytes are preserved in `2026-09-11-editing-utilities.zip` to make the narrowed cleanup recoverable. These are historical tools, not current authoring instructions. Do not run an archived installer against current lesson source.

The [cleanup manifest](../../teaching/evidence/workspace-cleanup-2026-09-11.json) records each original relative path, byte count and SHA-256 digest, plus the archive digest. To inspect a historical file, read only its named ZIP entry. If restoration is necessary, restore only that entry inside the repository after checking that its path has no parent traversal and would not overwrite current work. There is no reason to expand the entire archive as part of a normal authoring session.

Current source, active work, original-source baselines, verification outputs, referenced screenshots and the Python runtime remain at their existing paths.

The subsequent `2026-09-11-legacy-topic-working-files.zip` contains 35 obsolete temporary scripts/data/log files formerly outside `scratch/`, including two outer workspace working directories. Its entries are relative to the outer workspace, unlike the first archive's repository-relative entries. The [additional cleanup manifest](../../teaching/evidence/working-artifact-cleanup-2026-09-11.json) records their absolute original paths and hashes. Restore only a specific needed entry after validating its destination; never blindly extract over current work.
