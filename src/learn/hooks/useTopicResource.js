import { useEffect, useState } from "react";
import { loadTopicResource } from "../data/lesson-loader.js";

export default function useTopicResource(topic) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState({ topicId: topic.id, status: "loading", resource: null });
  useEffect(() => {
    let active = true;
    setState({ topicId: topic.id, status: "loading", resource: null });
    loadTopicResource(topic).then(
      resource => { if (active) setState({ topicId: topic.id, status: "ready", resource }); },
      error => { if (active) setState({ topicId: topic.id, status: "error", resource: null, error }); },
    );
    // Dynamic imports cannot be cancelled. Ignore completion of an old route.
    return () => { active = false; };
  }, [topic, attempt]);
  const current = state.topicId === topic.id ? state : { status: "loading", resource: null };
  return { ...current, attempt, retry: () => setAttempt(value => value + 1) };
}
