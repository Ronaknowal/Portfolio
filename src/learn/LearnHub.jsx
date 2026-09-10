import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { colors, fonts, navLinkStyle } from "./styles";
import { allTopicsOrdered, topicMap } from "./data/catalogue";
import { getLearningRoute, getTracks, learningPaths, trackGroups } from "./data/curriculum";
import useProgress from "./hooks/useProgress";
import TrackCard from "./components/TrackCard";
import TopicList from "./components/TopicList";

const tabs = [
  { id: "paths", label: "Guided paths" },
  { id: "tracks", label: "Modules" },
  { id: "catalogue", label: "Search catalogue" },
];

export default function LearnHub() {
  const [activeTab, setActiveTab] = useState("paths");
  const navigate = useNavigate();
  const { isComplete, trackProgress } = useProgress();

  useEffect(() => {
    if (!document.querySelector('link[href*="JetBrains+Mono"]')) {
      const link = document.createElement("link");
      link.href = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap";
      link.rel = "stylesheet";
      document.head.appendChild(link);
    }
  }, []);

  const availability = useMemo(() => {
    const published = allTopicsOrdered.filter((topic) => topic.status === "published").length;
    return { published, planned: allTopicsOrdered.length - published, total: allTopicsOrdered.length };
  }, []);

  return (
    <div className="learn-shell" style={{ fontFamily: fonts.sans }}>
      <nav className="learn-nav" aria-label="Primary navigation">
        <Link to="/" style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.gold, fontWeight: 600, letterSpacing: 1, textDecoration: "none" }}>
          ronak.ai
        </Link>
        <div className="learn-nav__links">
          <Link to="/" style={navLinkStyle}>portfolio</Link>
          <Link to="/learn" style={{ ...navLinkStyle, color: colors.gold }}>learn</Link>
        </div>
      </nav>

      <main className="learn-main">
        <p className="learn-kicker">EPOCH ∞ — KNOWLEDGE DISTILLATION</p>
        <h1 className="learn-heading">A curriculum that meets you where you are.</h1>
        <p className="learn-intro">
          Follow a route when you need structure, search the catalogue when you know the question, or use the complete syllabus for deliberate revision. Published lessons and planned material are always labelled separately.
        </p>
        <div className="availability-note" aria-label="Curriculum availability">
          <span><strong>{availability.published}</strong> published lessons</span>
          <span>{availability.planned} planned lessons</span>
          <span>{availability.total.toLocaleString()} topics across {trackGroups.flatMap((group) => group.trackIds).length} modules</span>
        </div>

        <div className="learn-tabs" role="tablist" aria-label="Learning views">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              className="learn-tab"
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === "paths" && (
          <section className="path-grid" aria-label="Guided learning paths">
            {learningPaths.map((path) => {
              const { topicIds: pathTopicIds, moduleCount } = getLearningRoute(path);
              const publishedCount = pathTopicIds.filter((id) => topicMap[id]?.status === "published").length;
              return (
                <article key={path.id} data-path-id={path.id} className={`path-card ${path.id === "full-curriculum" ? "path-card--full" : ""}`}>
                  <p className="path-card__eyebrow">{path.eyebrow}</p>
                  <h2 className="path-card__title">{path.title}</h2>
                  <p className="path-card__description">{path.description}</p>
                  <p className="path-card__meta">{moduleCount} modules · {pathTopicIds.length.toLocaleString()} topics · {trackProgress(pathTopicIds).done} completed</p>
                  <p className="path-card__availability">{publishedCount} published · {pathTopicIds.length - publishedCount} planned</p>
                  {path.milestones && (
                    <details className="path-milestones">
                      <summary>Explore this learning route</summary>
                      <ol>{path.milestones.map((milestone) => <li key={milestone}>{milestone}</li>)}</ol>
                      <p>Start with shared foundations, then follow this field's core and specialist material. Supporting modules contribute selected foundations and linked prerequisites. The full route is still being written.</p>
                    </details>
                  )}
                  <button type="button" className="path-card__action" onClick={() => navigate(`/learn/path/${path.id}`)}>
                    {path.id === "full-curriculum" ? "Open complete syllabus" : "Follow this path"} →
                  </button>
                </article>
              );
            })}
          </section>
        )}

        {activeTab === "tracks" && (
          <section aria-label="Learning modules">
            {trackGroups.map((group) => (
              <section key={group.id} className="track-group">
                <p className="track-group__eyebrow">{group.label}</p>
                <h2 className="track-group__heading">{group.description}</h2>
                <div className="track-stack">
                  {getTracks(group.trackIds).map((track) => {
                    const progress = trackProgress(track.topicIds);
                    const completedIds = new Set(track.topicIds.filter((id) => isComplete(id)));
                    return <TrackCard key={track.id} track={track} progress={{ ...progress, completedIds }} />;
                  })}
                </div>
              </section>
            ))}
          </section>
        )}

        {activeTab === "catalogue" && <TopicList isComplete={isComplete} />}
      </main>

      <footer style={{ padding: "32px", textAlign: "center", borderTop: "1px solid #0f0f0f" }}>
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: "#444", lineHeight: 2 }}>
          knowledge distillation in progress — still training
        </div>
      </footer>
    </div>
  );
}
