// Small shared navigation data. Lesson bodies and authoring plans are separate.
import { allTopicsOrdered, tracks } from "./generated/navigation.js";
export { allTopicsOrdered };
export { slugify } from "./topic-id.js";
export const topicMap = Object.fromEntries(allTopicsOrdered.map(topic => [topic.id, topic]));
export const allTopicIds = allTopicsOrdered.map(topic => topic.id);
export const categories = tracks.map(track => ({ id: track.id, label: track.title }));
