// Normalize separators without conflating names such as HLL and HLL++.
// Index only compact catalogue metadata, never lesson bodies or full briefs.
export function normalizeTopicQuery(value) {
  return value.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase()
    .replace(/[^\p{L}\p{N}+#]+/gu, " ").trim();
}

export function createTopicSearchIndex(topics) {
  return new Map(topics.map(topic => [topic.id, {
    title: normalizeTopicQuery(topic.title),
    subtopics: (topic.subtopics || []).map(label => ({ label, text: normalizeTopicQuery(label) })),
  }]));
}

export function findTopicMatch(entry, normalizedQuery) {
  if (!normalizedQuery || entry.title.includes(normalizedQuery)) return { matches: true, subtopic: null };
  const subtopic = entry.subtopics.find(value => value.text.includes(normalizedQuery));
  return { matches: Boolean(subtopic), subtopic: subtopic?.label || null };
}
