import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { colors, fonts } from "../styles";
import { topicMap } from "../data/catalogue";
import ProgressBar from "./ProgressBar";

export default function TrackCard({ track, progress }) {
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();
  const { done, total, percent } = progress;
  const published = track.topicIds.filter((id) => topicMap[id]?.status === "published").length;

  return (
    <article className="track-card">
      <div className="track-card__header">
        <div>
          <div className="track-card__eyebrow">
            MODULE · {total} TOPICS · {track.sections?.length || 0} SECTIONS
          </div>
          <h3 className="track-card__title">{track.title}</h3>
        </div>
        <div className="track-card__progress">{done}/{total} complete</div>
      </div>

      <p className="track-card__description">{track.description}</p>
      <div className="track-card__availability">
        <span className="topic-status topic-status--published">{published} published</span>
        <span className="topic-status topic-status--planned">{total - published} planned</span>
      </div>
      <ProgressBar percent={percent} />

      <div className="track-card__actions">
        <button type="button" className="track-card__start" onClick={() => navigate(`/learn/track/${track.id}`)}>
          {done > 0 ? "Resume module" : published > 0 ? "Start module" : "View syllabus"} →
        </button>
        <button
          type="button"
          className="track-card__expand"
          onClick={() => setExpanded((value) => !value)}
          aria-expanded={expanded}
        >
          {expanded ? "Hide outline" : "View outline"}
        </button>
      </div>

      {expanded && track.sections && (
        <div className="track-card__outline">
          {track.sections.map((section) => {
            const sectionDone = section.topicIds.filter((id) => progress.completedIds?.has(id)).length;
            return (
              <section key={section.name} className="track-section">
                <div className="track-section__header">
                  <span>{section.name}</span>
                  <span>{sectionDone}/{section.topicIds.length}</span>
                </div>
                <div className="track-section__topics">
                  {section.topicIds.map((id) => {
                    const topic = topicMap[id];
                    if (!topic) return null;
                    const complete = progress.completedIds?.has(id);
                    return (
                      <button
                        key={id}
                        type="button"
                        onClick={() => navigate(`/learn/track/${track.id}/${topic.id}`)}
                        className={`track-topic ${complete ? "is-complete" : ""}`}
                      >
                        {complete ? "✓ " : ""}{topic.title}
                      </button>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>
      )}
    </article>
  );
}
