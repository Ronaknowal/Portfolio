import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { colors, fonts, levelColors, levelLabels } from "../styles";
import { allTopicsOrdered, categories } from "../data/catalogue";
import { topicMap as topicCatalogue } from "../data/catalogue.js";
import LevelBadge from "./LevelBadge";

const LEVELS = ["foundation", "intermediate", "advanced", "frontier"];
const RESULTS_PER_PAGE = 100;

export default function TopicList({ isComplete }) {
  const [query, setQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("all");
  const [activeLevel, setActiveLevel] = useState("all");
  const [activeStatus, setActiveStatus] = useState("all");
  const [visibleCount, setVisibleCount] = useState(RESULTS_PER_PAGE);
  const navigate = useNavigate();

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return allTopicsOrdered.filter((topic) => {
      const matchesQuery = !normalizedQuery || topic.title.toLowerCase().includes(normalizedQuery);
      const matchesCategory = activeCategory === "all" || topicCatalogue[topic.id]?.trackIds.includes(activeCategory);
      const matchesLevel = activeLevel === "all" || topic.level === activeLevel;
      const matchesStatus = activeStatus === "all" || topic.status === activeStatus;
      return matchesQuery && matchesCategory && matchesLevel && matchesStatus;
    });
  }, [activeCategory, activeLevel, activeStatus, query]);

  useEffect(() => {
    setVisibleCount(RESULTS_PER_PAGE);
  }, [query, activeCategory, activeLevel, activeStatus]);

  const visibleTopics = filtered.slice(0, visibleCount);

  return (
    <section aria-label="Topic catalogue">
      <div className="catalogue-controls">
        <label className="sr-only" htmlFor="topic-search">Search the curriculum</label>
        <input
          id="topic-search"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search topics, e.g. Kalman, JAX, backprop..."
          className="catalogue-search"
        />

        <div className="catalogue-selects">
          <label className="catalogue-select-label">
            <span>Module</span>
            <select aria-label="Module" value={activeCategory} onChange={(event) => setActiveCategory(event.target.value)}>
              <option value="all">All modules</option>
              {categories.map((category) => (
                <option key={category.id} value={category.id}>{category.label}</option>
              ))}
            </select>
          </label>
          <label className="catalogue-select-label">
            <span>Availability</span>
            <select value={activeStatus} onChange={(event) => setActiveStatus(event.target.value)}>
              <option value="all">Published + planned</option>
              <option value="published">Published lessons</option>
              <option value="planned">Planned lessons</option>
            </select>
          </label>
        </div>
      </div>

      <div className="catalogue-filter-row" aria-label="Filter by difficulty">
        <button
          type="button"
          onClick={() => setActiveLevel("all")}
          className={`catalogue-filter ${activeLevel === "all" ? "is-active" : ""}`}
        >
          All levels
        </button>
        {LEVELS.map((level) => (
          <button
            key={level}
            type="button"
            onClick={() => setActiveLevel(level)}
            className={`catalogue-filter ${activeLevel === level ? "is-active" : ""}`}
            style={activeLevel === level ? { color: levelColors[level], borderColor: `${levelColors[level]}66` } : undefined}
          >
            {levelLabels[level]}
          </button>
        ))}
      </div>

      <p className="catalogue-result-count" aria-live="polite">
        {filtered.length.toLocaleString()} matching topic{filtered.length === 1 ? "" : "s"}
        {filtered.length > visibleTopics.length ? ` · showing first ${visibleTopics.length}` : ""}
      </p>

      <ul className="topic-results">
        {visibleTopics.map((topic) => {
          const done = isComplete(topic.id);
          return (
            <li key={topic.id}>
              <button
                type="button"
                onClick={() => navigate(`/learn/topic/${topic.id}`)}
                className="topic-result"
                data-topic-id={topic.id}
                aria-label={`Read ${topic.title}, ${topic.status} lesson`}
              >
                <span className="topic-result-main">
                  <span className="topic-result-title">{topic.title}</span>
                  <span className={`topic-status topic-status--${topic.status}`}>{topic.status}</span>
                  <span className="topic-result-time">{topic.readTime}</span>
                </span>
                <span className="topic-result-meta">
                  <LevelBadge level={topic.level} />
                  <span className={done ? "topic-done" : "topic-not-done"} aria-label={done ? "Complete" : "Not complete"}>
                    {done ? "✓" : "○"}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>

      {visibleTopics.length === 0 && (
        <p className="catalogue-empty">Nothing matches those filters. Try a broader term or include planned lessons.</p>
      )}

      {visibleCount < filtered.length && (
        <button
          type="button"
          className="catalogue-more"
          onClick={() => setVisibleCount((count) => count + RESULTS_PER_PAGE)}
        >
          Show 100 more
        </button>
      )}
    </section>
  );
}
