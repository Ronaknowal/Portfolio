import { trackDefinitions } from "./track-definitions";
import { slugify } from "./topics/index";

function topicTitle(t) {
  return typeof t === "string" ? t : t.title;
}

// Derive tracks from track-definitions with both flat topicIds and section structure
export const tracks = trackDefinitions.map((def) => ({
  id: def.id,
  title: def.title,
  description: def.description,
  topicIds: def.sections.flatMap((s) =>
    s.topics.map((t) => slugify(topicTitle(t)))
  ),
  sections: def.sections.map((s) => ({
    name: s.name,
    topicIds: s.topics.map((t) => slugify(topicTitle(t))),
  })),
}));
