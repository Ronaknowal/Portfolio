import { orderWithPrerequisites } from "./topic-dependencies.js";

// Shared by the compact browser catalogue and full authoring tools.
export function createLearningNavigation({ tracks, topicCatalogue, trackGroups }) {
  function getTracks(trackIds) {
    return trackIds
      .map((id) => tracks.find((track) => track.id === id))
      .filter(Boolean);
  }

  const pathOrderCache = new Map();
  function getPathTopicIds(pathOrTrackIds) {
    const path = Array.isArray(pathOrTrackIds) ? { trackIds: pathOrTrackIds } : pathOrTrackIds;
    const { trackIds, focusTrackIds, coreTrackIds = [], backgroundMode = "foundations" } = path;
    const key = JSON.stringify([trackIds, focusTrackIds, coreTrackIds, backgroundMode]);
    if (!pathOrderCache.has(key)) {
      const ids = getTracks(trackIds).flatMap((track) => track.topicIds.filter((id) => {
        const topic = topicCatalogue[id];
        if (!focusTrackIds || focusTrackIds.includes(track.id)) return true;
        if (coreTrackIds.includes(track.id)) return (topic.depth ?? topic.blueprint?.depth) === "core" || topic.level === "foundation";
        if (backgroundMode === "prerequisites") return false;
        return topic.level === "foundation";
      }));
      // Dependencies determine membership, not reading order. The catalogue's
      // section/topic order owns progression; difficulty never reshuffles it.
      const included = orderWithPrerequisites(ids, topicCatalogue);
      const groups = getPathNavigationGroups(included, { moduleOrder: trackIds });
      pathOrderCache.set(key, [...new Set(groups.flatMap(group => group.topicIds))]);
    }
    return pathOrderCache.get(key);
  }

  // Selected modules come first in their declared order. Additional prerequisite
  // modules follow in catalogue order. Within each, use the actual syllabus order.
  // Shared lessons keep all memberships and one stable ID/completion record.
  function getPathNavigationGroups(topicIds, { moduleOrder = [] } = {}) {
    const included = new Set(topicIds);
    const assigned = new Map();
    const preferred = new Set(moduleOrder.length ? moduleOrder : topicIds.map(id => topicCatalogue[id].trackId));
    for (const id of topicIds) {
      const topic = topicCatalogue[id];
      const memberships = topic.trackIds.filter(trackId => preferred.has(trackId));
      for (const moduleId of memberships.length ? memberships : [topic.trackId]) {
        if (!assigned.has(moduleId)) assigned.set(moduleId, new Set());
        assigned.get(moduleId).add(id);
      }
    }
    const orderedModules = [...new Set([...moduleOrder, ...trackGroups.flatMap(group => group.trackIds)])];
    return getTracks(orderedModules).filter(track => assigned.has(track.id)).map(track => ({
      id: track.id, label: track.title, totalTopicCount: new Set(track.topicIds).size,
      topicIds: track.topicIds.filter(id => included.has(id) && assigned.get(track.id).has(id)),
    }));
  }

  // Use the resolved route for both hub counts and reader navigation, including
  // modules that contribute prerequisites outside the initial selection.
  function getLearningRoute(pathOrTrackIds) {
    const moduleOrder = Array.isArray(pathOrTrackIds) ? pathOrTrackIds : pathOrTrackIds.trackIds;
    const topicIds = getPathTopicIds(pathOrTrackIds);
    const navigationGroups = getPathNavigationGroups(topicIds, { moduleOrder });
    // Occurrences preserve a shared lesson's local position when opened from a
    // second module. Unique topicIds still own counts, resume and saved progress.
    const steps = navigationGroups.flatMap(group => group.topicIds.map(topicId => ({ topicId, moduleId: group.id })));
    return { topicIds, navigationGroups, steps, moduleCount: navigationGroups.length };
  }

  return { getTracks, getPathTopicIds, getPathNavigationGroups, getLearningRoute };
}
