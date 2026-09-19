import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { tracks } from "../data/tracks";
import { topicMap } from "../data/catalogue";
import PlaceholderContent from "./PlaceholderContent";
import LevelBadge from "./LevelBadge";
import LessonGuide from "./LessonGuide";
import useTopicResource from "../hooks/useTopicResource.js";
import LessonBoundary, { LessonLoadError } from "./LessonBoundary.jsx";
import "./topic-content.css";

export default function TopicContent({ topic, context, track, currentModule, previousStep, nextStep, isComplete, toggleComplete, basePath }) {
  const navigate = useNavigate();
  const done = isComplete(topic.id);
  const otherTracks = tracks.filter((candidate) => candidate.topicIds.includes(topic.id) && (!track || candidate.id !== track.id));
  const prevId = previousStep?.topicId;
  const nextId = nextStep?.topicId;
  const { status, resource, retry, attempt } = useTopicResource(topic);
  const [renderFailed, setRenderFailed] = useState(false);
  const retryLesson = () => { setRenderFailed(false); retry(); };
  const lesson = topic.status === "published" ? resource : null;
  const Content = lesson?.content;
  useEffect(() => {
    if (status !== "ready" || !window.location.hash) return;
    const frame = requestAnimationFrame(() => {
      try { document.getElementById(decodeURIComponent(window.location.hash.slice(1)))?.scrollIntoView(); }
      catch { /* A malformed URL fragment must not prevent reading the lesson. */ }
    });
    return () => cancelAnimationFrame(frame);
  }, [topic.id, status]);

  const handleNav = (step) => {
    navigate(`${basePath}/${step.topicId}?module=${encodeURIComponent(step.moduleId)}`);
    window.scrollTo(0, 0);
  };

  return (
    <main className="reader-content">
      <nav className="reader-breadcrumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => navigate("/learn")}>Learn</button>
        <span aria-hidden="true">→</span>
        <button type="button" onClick={() => navigate(context.href)}>{context.title}</button>
        <span aria-hidden="true">→</span>
        <span aria-current="page">{topic.title}</span>
      </nav>

      {otherTracks.length > 0 && (
        <div className="reader-related" aria-label="Topic modules">
          {otherTracks.map((candidate) => (
            <button key={candidate.id} type="button" onClick={() => navigate(`/learn/track/${candidate.id}/${topic.id}`)}>
              {track ? "Also in" : "Module"}: {candidate.title}
            </button>
          ))}
        </div>
      )}

      <header className="reader-header">
        <h1>{topic.title}</h1>
        <div className="reader-header__meta">
          <LevelBadge level={topic.level} size="normal" />
          <span className={`topic-status topic-status--${topic.status}`}>{topic.status}</span>
          {topic.depth && <span>{topic.depth === "core" ? "Core study" : `${topic.depth === "frontier" ? "Frontier" : "Specialist"} branch`}</span>}
          <span>{topic.readTime}</span>
          {currentModule && <span>{currentModule.label} · {currentModule.topicIds.indexOf(topic.id) + 1} of {currentModule.topicIds.length} topics on this route</span>}
        </div>
      </header>

      {Content && topic.prerequisiteIds?.length > 0 && <details className="reader-prerequisites">
        <summary>Before this lesson · {topic.prerequisiteIds.length} prerequisite {topic.prerequisiteIds.length === 1 ? "topic" : "topics"}</summary>
        <p>Review these if their ideas are unfamiliar. The reading sequence follows the module contents; these links let you revisit supporting concepts.</p>
        <ul>{topic.prerequisiteIds.map(id => <li key={id}><a href={`/learn/topic/${id}`}>{topicMap[id].title}</a>{isComplete(id) ? " · completed" : ""}</li>)}</ul>
      </details>}

      {Content && !topic.hasIntegratedGuide && <LessonGuide topic={topic} />}

      <div className="reader-article" aria-busy={status === "loading"}>
        {status === "loading" ? <p className="lesson-loading" role="status">{topic.status === "published" ? "Loading lesson…" : "Loading syllabus outline…"}</p>
        : status === "error" ? <LessonLoadError onRetry={retry} />
        : <LessonBoundary key={`${topic.id}:${attempt}`} onRetry={retryLesson} onError={() => setRenderFailed(true)}>{Content ? (
          lesson.sections ? (
            lesson.sections.map((section, index) => {
              const SectionContent = section.content;
              return (
                <section key={section.id || index} id={section.id} className="reader-article__section">
                  {section.title && <h2>{section.title}</h2>}
                  <SectionContent />
                </section>
              );
            })
          ) : <Content />
        ) : (
          <PlaceholderContent title={topic.title} blueprint={resource} prerequisiteIds={topic.prerequisiteIds} subtopics={topic.subtopics} />
        )}</LessonBoundary>}
      </div>

      <footer className="reader-footer">
        <button
          type="button"
          className={`reader-complete ${done ? "is-complete" : ""}`}
          disabled={topic.status === "planned" || status !== "ready" || renderFailed}
          onClick={() => toggleComplete(topic.id)}
        >
          {topic.status === "planned" ? "Lesson not yet published" : done ? "✓ Completed" : "○ Mark as complete"}
        </button>
        <div className="reader-footer__nav" aria-label="Lesson navigation">
          {prevId && <button type="button" className="reader-footer__previous" onClick={() => handleNav(previousStep)}>
            <span className="reader-footer__direction">← Previous</span>
            <span>{topicMap[prevId].title}</span>
          </button>}
          {nextId && <button type="button" className="reader-footer__next" onClick={() => handleNav(nextStep)}>
            <span className="reader-footer__direction">Next in sequence →</span>
            <span>{topicMap[nextId].title}</span>
          </button>}
        </div>
      </footer>
    </main>
  );
}
