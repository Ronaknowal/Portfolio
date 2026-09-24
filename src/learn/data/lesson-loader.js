import { lessonModules } from "./generated/lesson-imports.js";
const outlineModules = import.meta.glob("./generated/outlines/*.json", { import: "default" });
const requestedResources = new Map();

// Import only the current topic. Successful visits reuse the same module;
// rejected promises are evicted so a later attempt can recover where supported.
export function loadTopicResource(topic) {
  if (topic.status !== "published" && !topic.hasOutline) return Promise.resolve(null);
  if (requestedResources.has(topic.id)) return requestedResources.get(topic.id);
  const loader = topic.status === "published" ? lessonModules[topic.id] : outlineModules[`./generated/outlines/${topic.id}.json`];
  const request = Promise.resolve().then(() => {
    if (!loader) throw new Error(`Missing topic resource: ${topic.id}`);
    return loader();
  }).then(module => {
    if (topic.status !== "published") return module;
    if (typeof module.default?.content !== "function") throw new Error(`Invalid lesson export: ${topic.id}`);
    return module.default;
  }).catch(error => {
    requestedResources.delete(topic.id);
    throw error;
  });
  requestedResources.set(topic.id, request);
  return request;
}
