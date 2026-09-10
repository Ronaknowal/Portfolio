import { orderWithPrerequisites as orderDependencies } from "./topic-dependencies.js";
import { trackDefinitions } from "../track-definitions.js";
import { slugify } from "../topic-id.js";

// Plain data, independently inspectable without importing React lesson content.
export const topicCatalogue = {};
for (const track of trackDefinitions) {
  for (const section of track.sections) {
    for (const value of section.topics) {
      const topic = typeof value === "string" ? { title: value, level: "foundation" } : value;
      const id = slugify(topic.title);
      const prerequisiteTitles = topic.blueprint?.prerequisites ?? topic.prerequisites;
      if (!topicCatalogue[id]) {
        topicCatalogue[id] = {
          ...topic, id, trackId: track.id, section: section.name,
          trackIds: [track.id],
          prerequisiteStatus: prerequisiteTitles ? "recorded" : "individual-review-required",
          prerequisiteIds: (prerequisiteTitles || []).map(slugify),
          designStatus: topic.blueprint ? "topic-brief-ready" : "individual-design-required",
        };
      } else if (!topicCatalogue[id].trackIds.includes(track.id)) {
        topicCatalogue[id].trackIds.push(track.id);
      }
    }
  }
}


export const orderWithPrerequisites = (ids, catalogue = topicCatalogue) => orderDependencies(ids, catalogue);
