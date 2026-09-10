import { slugify } from "../topic-id.js";

function topicTitle(topic) {
  return typeof topic === "string" ? topic : topic.title;
}

// Derive tracks from track-definitions with both flat topicIds and section structure
export const buildTracks = trackDefinitions => trackDefinitions.map((definition) => ({
  id: definition.id,
  title: definition.title,
  description: definition.description,
  topicIds: definition.sections.flatMap((section) =>
    section.topics.map((topic) => slugify(topicTitle(topic)))
  ),
  sections: definition.sections.map((section) => ({
    name: section.name,
    topicIds: section.topics.map((topic) => slugify(topicTitle(topic))),
  })),
}));
