// Retain authored numbers because prose may refer to them. An unnumbered
// implementation or reference section must not shift the other section numbers.
export function sectionLabel(text) {
  const label = text.replace(/\s+/g, " ").trim();
  const match = label.match(/^(\d+(?:\.\d+)*)(?:[.):]|\s+[—–-])\s+(.+)$/u);
  return match ? { number: match[1], title: match[2] } : { number: null, title: label };
}

export function headingId(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

export function collectLessonSections(article) {
  const headings = [...article.querySelectorAll("h2")].filter(heading =>
    !heading.closest("details, aside, nav, [data-lab], .lesson-lab, .neural-lab, .lesson-load-error"));
  const used = new Set();
  return headings.map((heading, index) => {
    const { number, title } = sectionLabel(heading.textContent);
    const baseId = heading.id || headingId(heading.textContent) || `lesson-section-${index + 1}`;
    let id = baseId;
    let suffix = 2;
    while (used.has(id) || (article.ownerDocument.getElementById(id) && article.ownerDocument.getElementById(id) !== heading)) {
      id = `${baseId}-${suffix++}`;
    }
    used.add(id);
    heading.id = id;
    heading.tabIndex = -1;
    heading.classList.add("reader-section-heading");
    return { id, title, number };
  });
}
