import { trackDefinitions } from "../track-definitions";

// --- Custom content imports (topics with rich JSX content + visualizers) ---
import attentionContent from "./attention";
import backpropContent from "./backprop";
import tokenizationContent from "./tokenization";
import byteLevelContent from "./byte-level-tokenization";

// Map custom content by the slugified title they correspond to in track-definitions
const customContent = {
  "attention-mechanism-bahdanau-luong": attentionContent,
  "backpropagation-automatic-differentiation": backpropContent,
  "byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram": tokenizationContent,
  "byte-level-tokenization-token-free-models": byteLevelContent,
};

// --- Slugify: title → URL-safe ID ---
export function slugify(str) {
  return str
    .toLowerCase()
    .replace(/[()]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// --- Process all track definitions into flat topic objects ---
const topicMapBuilder = {};
let globalOrder = 0;

for (const track of trackDefinitions) {
  for (const section of track.sections) {
    for (const topicDef of section.topics) {
      const title = typeof topicDef === "string" ? topicDef : topicDef.title;
      const level = typeof topicDef === "string" ? "foundation" : (topicDef.level || "foundation");
      const id = slugify(title);
      if (!topicMapBuilder[id]) {
        const custom = customContent[id];
        topicMapBuilder[id] = {
          id,
          title: custom ? custom.title : title,
          category: track.id,
          level,
          section: section.name,
          readTime: custom ? custom.readTime : "5 min",
          order: globalOrder++,
          content: custom ? custom.content : null, // null = use PlaceholderContent
        };
      }
    }
  }
}

// Map of all topics keyed by ID for O(1) lookup
export const topicMap = topicMapBuilder;

// All topics sorted by global order (tracks in definition order, topics in section order)
export const allTopicsOrdered = Object.values(topicMap).sort((a, b) => a.order - b.order);

// All topic IDs in master sequence
export const allTopicIds = allTopicsOrdered.map((t) => t.id);

// All categories (track IDs with labels)
export const categories = trackDefinitions.map((t) => ({
  id: t.id,
  label: t.title,
}));
