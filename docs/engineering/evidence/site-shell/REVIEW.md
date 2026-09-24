# Personal hub integration review

23 September 2026. Implemented the user-selected personal hub with Portfolio and Learn. Site structure and future additions are documented in [SITE-ARCHITECTURE.md](../../SITE-ARCHITECTURE.md).

**Home presentation refined after this integration:** the user requested selective inspiration from Aleksa Gordić's homepage. The current home replaces the large general-purpose headline and entrance panels with a personal name/introduction, compact Portfolio/Learn directory, nested learning shortcuts, direct social links and a denser selected-entry list. The original shell structure stays intact. See [the current refinement receipt](personal-home-refinement.json) for the exact home source hashes, production dependency boundary and scoped in-app-browser checks. The three screenshots and 23-check receipt below describe the initial design and remain a bounded comparison record, not certification of the newer home pixels. No user acceptance is inferred.

The refined page was visually inspected at default desktop size and in tablet/mobile layouts. All three added learning shortcuts were opened and their destination headings confirmed; the profile anchor, Back to top and keyboard skip were also checked. Production build and JSX parsing pass. The unchanged route/section-switcher implementation reuses the earlier integration evidence. The existing automated site-shell script now additionally asserts the personal heading and nested shortcut destinations; it was updated but not rerun as part of this browser-driven refinement.

## Verified

- Production build succeeds. Catalogue generation reports zero changed generated files: 1,461 topics, 231 published lessons, 486 separate outlines. No teaching body, curriculum sequence or delivery-ledger edits were required.
- The focused [browser receipt](browser-review.json) records 23 passing checks and the exact tested source hashes. Coverage includes 1440, 768, 390 and 320px layouts; home entrances; the shared section disclosure; keyboard skip, Escape/focus return and dismissal; all seven historical portfolio anchors; the current sequence lesson and project training route; preserved lesson/project storage; unknown URLs; and a deliberately failed portfolio import followed by successful reload.
- Home fetched only the shared application entry and its stylesheet: about 78.3 kB compressed JavaScript and 2.8 kB compressed CSS in the local production browser measurement, excluding fonts and HTML. It fetched no curriculum, lesson, lab, project-body or portfolio-animation chunk. The existing large shared learning chunk still produces Vite's >500 kB warning; it is outside the home entry's dependency tree.
- All learning/reader rules moved from `src/index.css` to `src/learn/learning-base.css` were verified unchanged by comparing the concatenated files to the original stylesheet, normalizing line endings only. JSX parsing and the scoped whitespace check pass.
- Visually inspected [desktop home](home-desktop.png), [mobile home](home-mobile.png), and [mobile section switcher](section-switcher-mobile.png). The layout retains neutral black/charcoal and amber, readable flow, intact entrance links, and an unclipped navigation disclosure.

## Scope and limits

This review covers the new home, site navigation, route compatibility and loading boundaries. The existing portfolio body was moved, not rewritten; its longstanding internal presentation and project descriptions are not newly certified here. Its long mobile section menu scrolls within its own row. Existing lesson computational evidence remains valid and was not rerun for a navigation change.

Research notes, articles and standalone tools have documented destination contracts, not empty published pages. The user's visual acceptance of the new home remains separate from implementation verification. No deployment or commit is implied.

Retain only these three final screenshots and the report. Task-specific exploratory captures under `scratch/site-home-review/` were removed after inspection; existing shared font fixtures and unrelated work are preserved.
