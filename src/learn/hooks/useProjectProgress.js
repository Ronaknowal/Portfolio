import { useSyncExternalStore } from "react";

const STORAGE_KEY = "learning-project-progress-v1";
const listeners = new Set();
let snapshot;

function readStoredProgress() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    const entries = value && typeof value === "object" && !Array.isArray(value)
      ? Object.entries(value).filter(([, complete]) => complete === true)
      : [];
    return { progress: Object.fromEntries(entries), storageUnavailable: false };
  } catch {
    // Keep this visit's milestones even if storage is denied or becomes full.
    return { progress: snapshot?.progress || {}, storageUnavailable: true };
  }
}

function getSnapshot() {
  if (!snapshot) snapshot = readStoredProgress();
  return snapshot;
}

function publish(next) {
  snapshot = next;
  listeners.forEach(listener => listener());
}

function receiveStorageChange(event) {
  if (event.key === STORAGE_KEY || event.key === null) {
    publish(readStoredProgress());
  }
}

function subscribe(listener) {
  listeners.add(listener);
  if (listeners.size === 1) {
    window.addEventListener("storage", receiveStorageChange);
    // Another tab may have changed storage while no project view was mounted.
    if (!getSnapshot().storageUnavailable) publish(readStoredProgress());
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) {
      window.removeEventListener("storage", receiveStorageChange);
    }
  };
}

function toggleStage(projectId, stageId) {
  const current = getSnapshot();
  const { progress } = current.storageUnavailable ? current : readStoredProgress();
  const key = `${projectId}/${stageId}`;
  const next = { ...progress };
  if (next[key] === true) delete next[key];
  else next[key] = true;

  let storageUnavailable = false;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    storageUnavailable = true;
  }
  publish({ progress: next, storageUnavailable });
}

export default function useProjectProgress() {
  const { progress, storageUnavailable } = useSyncExternalStore(subscribe, getSnapshot);
  const isStageComplete = (projectId, stageId) => progress[`${projectId}/${stageId}`] === true;
  const completedCount = project => project.stages.filter(
    stage => isStageComplete(project.id, stage.id),
  ).length;
  const nextUnfinishedStage = project => project.stages.find(
    stage => !isStageComplete(project.id, stage.id),
  );

  return { isStageComplete, completedCount, nextUnfinishedStage, toggleStage, storageUnavailable };
}
