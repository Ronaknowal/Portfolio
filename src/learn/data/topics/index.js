import { trackDefinitions } from "../track-definitions";

// --- Custom content imports (topics with rich JSX content + visualizers) ---
import attentionContent from "./attention";
import backpropContent from "./backprop";
import tokenizationContent from "./tokenization";
import byteLevelContent from "./byte-level-tokenization";
import vocabularyMultilingualContent from "./vocabulary-multilingual";
import multimodalContent from "./multimodal-tokenization";
import dynamicContent from "./dynamic-tokenization";
// Pre-Training section
import causalLMContent from "./causal-language-modeling";
import maskedLMContent from "./masked-language-modeling";
import dedupContent from "./data-curation-deduplication";
import scalingLawsContent from "./scaling-laws";
import curriculumContent from "./curriculum-data-mixing";
import fp8Content from "./fp8-training";
import moeContent from "./moe-training";
import multimodalPretrainingContent from "./multimodal-pretraining";
import curationPipelinesContent from "./data-curation-pipelines";
import syntheticDataContent from "./synthetic-data-pretraining";

// Map custom content by the slugified title they correspond to in track-definitions
const customContent = {
  "attention-mechanism-bahdanau-luong": attentionContent,
  "backpropagation-automatic-differentiation": backpropContent,
  "byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram": tokenizationContent,
  "byte-level-tokenization-token-free-models": byteLevelContent,
  "vocabulary-design-multilingual-tokenization": vocabularyMultilingualContent,
  "multimodal-tokenization-visual-audio-video": multimodalContent,
  "dynamic-tokenization-adat-boundlessbpe-litetoken": dynamicContent,
  // Pre-Training
  "causal-language-modeling-next-token-prediction": causalLMContent,
  "masked-language-modeling-bert-style": maskedLMContent,
  "data-curation-deduplication-minhash-bloom-filters": dedupContent,
  "scaling-laws-kaplan-chinchilla-beyond": scalingLawsContent,
  "curriculum-learning-data-mixing-strategies": curriculumContent,
  "fp8-training-low-precision-pre-training": fp8Content,
  "moe-training-expert-load-balancing": moeContent,
  "multimodal-pre-training-vision-encoders-cross-modal-alignment": multimodalPretrainingContent,
  "data-curation-pipelines-curator-models-quality-filtering": curationPipelinesContent,
  "synthetic-data-generation-for-pre-training": syntheticDataContent,
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
