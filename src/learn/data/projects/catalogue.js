// Metadata only. Project code, labs and teaching load on its own route.
import typedDecisionModel from "./typed-decision-model/metadata.js";

export const projects = [typedDecisionModel];
export const projectMap = Object.fromEntries(projects.map(project => [project.id, project]));

export function getProject(projectId) {
  return Object.hasOwn(projectMap, projectId) ? projectMap[projectId] : null;
}

export function projectStageUrl(projectId, stageId) {
  const base = `/learn/projects/${encodeURIComponent(projectId)}`;
  return stageId ? `${base}/${encodeURIComponent(stageId)}` : base;
}
