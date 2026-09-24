# Discoveries for a topic's future author

This is the persistent routing mechanism required by [the teaching standard](../../../LESSON-TEACHING-STANDARD.md). Use it for useful coverage gaps, applications, facts, examples or title/scope suggestions discovered while working on another topic. It does not create a new rollout queue or authorize rewriting the destination now. No topic audit or new content recommendation was performed when establishing this mechanism.

## Write to the destination, then make a decision when authoring it

1. For an existing destination, resolve its actual stable ID in the live catalogue and create or update `<stable-topic-id>.md` in this directory. The filename is its identity; a future display-title change must not strand the note. Read existing entries first and merge duplicates while retaining their sources and reasoning. Several related topics may be linked, but choose one primary teaching owner and record the local bridges.
2. If no suitable destination exists or ownership is genuinely unresolved, use [UNASSIGNED.md](UNASSIGNED.md). Record a proposed module/topic, alternatives considered and the prerequisite relationship. Do not invent a canonical ID for a nonexistent lesson. When ownership is resolved, move the full note to the destination and retain an inbox pointer and disposition.
3. Persist the note when discovered, including halfway through writing or verification. Update the originating design record with a link. Notes belong in the repository, not only in chat, generated inventory output, a source comment or an archived report.
4. Before writing any topic, run `node scripts/build-curriculum-inventory.mjs --topic "Exact topic title or stable ID"`. Its `authoringNotes` includes the destination Markdown and unresolved inbox, with paths and an explicit missing-file state. Read notes relevant to this topic and its prerequisites. Directly opening the topic source does not replace this step.
5. Treat a note as a reasoned proposal to investigate. Check current coverage, sources, level and learner benefit; decide to include, adapt, reroute, defer with a concrete reason, or reject with evidence. Explain the decision in the entry. Do not paste an unsupported claim because an earlier author proposed it. Already resolved entries remain useful context, not instructions to implement twice.
6. On delivery, leave unresolved entries visible and summarize their destinations in the increment record. For implemented ideas, link the actual lesson section, revised brief/design and relevant verification. Note handling is complete only when the disposition is saved, not when someone has merely read it.

## Per-topic file and entry format

Use one file per existing stable topic ID, with a title and canonical ID at the top. Append a separately headed entry for each distinct idea. Keep the decision text with the entry rather than deleting the discovery after implementation. Do not create empty files for every curriculum topic.

Copy and fill only when an actual discovery occurs:

```markdown
# Authoring notes: Destination topic title

Canonical topic ID: exact-existing-topic-id

## YYYY-MM-DD — Short descriptive idea

- Status: open
- Origin: source topic ID and link to its design record or lesson section
- Destination and ownership rationale: why this topic teaches it better; other homes considered
- Idea and learning benefit: what the learner will be able to explain, predict, use or connect
- Existing coverage: absent / mentioned / planned / taught / verified, with exact checked locations
- Proposed treatment: placement in the flow, core/deeper depth, scope/title implications
- Explanation/example: setup, mechanism mapping, useful result or consequence, possible visual/practice
- Prerequisites and boundaries: what must be introduced; limits or reasons not to put this in the origin
- Evidence: source URLs/locators/date/version and what remains unverified; distinguish a teaching scenario from a documented real-world use
- Resolution: not yet reviewed
- Implementation/verification links: none yet
```

Use status `open`, `implemented`, `adapted`, `rerouted`, `deferred` or `rejected`. `Adapted` means the revised idea has actually been implemented; otherwise keep it open. `Deferred` retains a concrete reason and revisit condition. `Rerouted` names the receiving stable ID and links its saved entry. Rejection requires the pedagogical or technical reason; implementation requires evidence links. A title suggestion remains a suggestion until the receiving author evaluates the final scope and implements compatible identity handling if needed.

For content-first work, keep implementation status separate from **Content disposition: prepared**. That disposition links the complete manuscript section, instructional program and relevant specification, and states the execution/integration checks still deferred. It does not mean merely writing an instruction for the next agent to add the missing teaching. A known missing core mechanism or library route must be written before its content disposition or topic content checkpoint is complete. The note can remain implementation-open until phase two consumes that prepared material.

A useful application note describes the mechanism and instructional benefit, not merely “add a fun use of this.” Several applications may be warranted; none is required simply to populate a section. Follow the standard's accuracy, adequate-detail and no-quota rules.
