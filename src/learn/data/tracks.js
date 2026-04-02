import { trackDefinitions } from "./track-definitions";
import { slugify } from "./topics/index";

// Derive tracks from track-definitions, generating topicIds from section topics
export const tracks = trackDefinitions.map((def) => ({
  id: def.id,
  title: def.title,
  description: def.description,
  topicIds: def.sections.flatMap((s) =>
    s.topics.map((t) => slugify(typeof t === "string" ? t : t.title))
  ),
}));
