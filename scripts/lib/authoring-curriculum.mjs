import { trackDefinitions } from "../../src/learn/data/track-definitions.js";
import { topicCatalogue } from "../../src/learn/data/curriculum/topic-catalogue.js";
import { buildTracks } from "../../src/learn/data/curriculum/track-structure.js";
import { trackGroups, learningPaths } from "../../src/learn/data/curriculum/learning-paths.js";
import { createLearningNavigation } from "../../src/learn/data/curriculum/navigation.js";
export { trackGroups, learningPaths };
export const tracks = buildTracks(trackDefinitions);
export const { getTracks, getPathTopicIds, getPathNavigationGroups, getLearningRoute } = createLearningNavigation({ tracks, topicCatalogue, trackGroups });
