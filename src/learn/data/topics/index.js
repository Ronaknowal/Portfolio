import attention from "./attention";
import backprop from "./backprop";
import tokenization from "./tokenization";

// Map of all topics keyed by ID for O(1) lookup
export const topicMap = {
  [attention.id]: attention,
  [backprop.id]: backprop,
  [tokenization.id]: tokenization,
};

// Category order for sorting the "All Topics" master list
const categoryOrder = ["fundamentals", "nlp", "vision", "generative", "training", "evaluation"];

// All topics sorted by category order, then by manual `order` field
export const allTopicsOrdered = Object.values(topicMap).sort((a, b) => {
  const catA = categoryOrder.indexOf(a.category);
  const catB = categoryOrder.indexOf(b.category);
  if (catA !== catB) return catA - catB;
  return a.order - b.order;
});

// All topic IDs in master sequence (used as the hidden "All Topics" track)
export const allTopicIds = allTopicsOrdered.map((t) => t.id);

// All categories that have at least one topic
export const categories = [...new Set(allTopicsOrdered.map((t) => t.category))];
