import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { tracks } from "../data/tracks";
import { topicMap } from "../data/catalogue";
import PlaceholderContent from "./PlaceholderContent";
import LevelBadge from "./LevelBadge";
import LessonGuide from "./LessonGuide";
import { LessonOpeningContext } from "./LessonOpening.jsx";
import useTopicResource from "../hooks/useTopicResource.js";
import LessonBoundary, { LessonLoadError } from "./LessonBoundary.jsx";
import "./topic-content.css";
import "./lesson-endings.css";
import { LessonCodeProvider, LessonCodeDownloads } from './content/LessonCodeDownloads.jsx';
import './content/lesson-code.css';
import { projects } from "../data/projects/catalogue.js";

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
  const articleRef = useRef(null);
  const [openingTargets, setOpeningTargets] = useState({});
  const openingSlots = useMemo(() => Object.fromEntries(
    ["summary", "route", "prerequisites", "exploration"].map(kind => [kind, node => {
      setOpeningTargets(previous => previous[kind] === node ? previous : { ...previous, [kind]: node });
    }])
  ), []);
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
    <main className="reader-content" id="learning-main" tabIndex={-1}>
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

      {projects.filter(project => project.relatedTopicIds?.includes(topic.id)).map(project => (
        <div className="related-projects" key={project.id}>
          <span>BUILD WITH THIS</span>
          <Link to={`/learn/projects/${project.id}`}>{project.title} ↗</Link>
        </div>
      ))}
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

      <LessonCodeProvider key={topic.id} topicId={topic.id}>
      <LessonOpeningContext.Provider value={openingTargets}>
      {Content && !renderFailed && <LessonGuide topic={topic} articleRef={articleRef} content={Content}
        prerequisiteTopics={(topic.prerequisiteIds || []).map(id => topicMap[id]).filter(Boolean)}
        isComplete={isComplete} slots={openingSlots} />}

      <div className="reader-article" ref={articleRef} aria-busy={status === "loading"}>
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
      </LessonOpeningContext.Provider>
      {Content && !renderFailed && <LessonCodeDownloads articleRef={articleRef} content={Content} />}
      </LessonCodeProvider>

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
