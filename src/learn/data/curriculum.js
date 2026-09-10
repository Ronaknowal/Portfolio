import { tracks } from "./tracks.js";
import { topicMap } from "./catalogue.js";
import { trackGroups } from "./curriculum/learning-paths.js";
import { createLearningNavigation } from "./curriculum/navigation.js";
export { trackGroups, learningPaths } from "./curriculum/learning-paths.js";
export const { getTracks, getPathTopicIds, getPathNavigationGroups, getLearningRoute } = createLearningNavigation({ tracks, topicCatalogue: topicMap, trackGroups });
