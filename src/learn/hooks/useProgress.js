import { useState, useCallback } from "react";

const STORAGE_KEY = "kd-progress";

function loadProgress() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveProgress(data) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

export default function useProgress() {
  const [completed, setCompleted] = useState(loadProgress);

  const markComplete = useCallback((topicId) => {
    setCompleted((prev) => {
      const next = { ...prev, [topicId]: true };
      saveProgress(next);
      return next;
    });
  }, []);

  const markIncomplete = useCallback((topicId) => {
    setCompleted((prev) => {
      const next = { ...prev };
      delete next[topicId];
      saveProgress(next);
      return next;
    });
  }, []);

  const toggleComplete = useCallback((topicId) => {
    setCompleted((prev) => {
      const next = { ...prev };
      if (next[topicId]) {
        delete next[topicId];
      } else {
        next[topicId] = true;
      }
      saveProgress(next);
      return next;
    });
  }, []);

  const isComplete = useCallback(
    (topicId) => completed[topicId] === true,
    [completed]
  );

  const trackProgress = useCallback(
    (topicIds) => {
      const done = topicIds.filter((id) => completed[id]).length;
      return { done, total: topicIds.length, percent: topicIds.length > 0 ? done / topicIds.length : 0 };
    },
    [completed]
  );

  return { isComplete, markComplete, markIncomplete, toggleComplete, trackProgress };
}
