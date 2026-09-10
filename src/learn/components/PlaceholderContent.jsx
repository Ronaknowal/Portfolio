import { Link } from "react-router-dom";
import { topicMap as topicCatalogue } from "../data/catalogue.js";

export default function PlaceholderContent({ title, blueprint, prerequisiteIds = [] }) {
  return (
    <section className="planned-lesson" aria-label="Planned lesson">
      <p className="planned-lesson__eyebrow">SYLLABUS ENTRY · NOT YET PUBLISHED</p>
      <h2>{title}</h2>
      <p>
        {blueprint ? blueprint.summary : "This topic is part of the syllabus. Its full teaching plan and lesson are still to come."}
      </p>
      {blueprint && <p className="planned-lesson__notice">The outline below describes the planned lesson. The explanations, interactive activities and worked solutions are not yet published.</p>}
      {prerequisiteIds.length > 0 && (
        <div className="syllabus-block">
          <h3>Build on these ideas</h3>
          <ul className="syllabus-prerequisites">{prerequisiteIds.map((id) => <li key={id}><Link to={`/learn/topic/${id}`}>{topicCatalogue[id]?.title || id}</Link></li>)}</ul>
        </div>
      )}
      {blueprint && (
        <>
          <div className="syllabus-block">
            <h3>What you will learn to do</h3>
            <ul>{blueprint.outcomes.map((outcome) => <li key={outcome}>{outcome}</li>)}</ul>
          </div>
          <div className="syllabus-block">
            <h3>The learning sequence</h3>
            <ol className="syllabus-sequence">{blueprint.sequence.map((step) => <li key={step}>{step}</li>)}</ol>
          </div>
          <div className="syllabus-block">
            <h3>A planned investigation</h3>
            <p className="syllabus-question">{blueprint.visual.question}</p>
            <p>{blueprint.visual.interaction}</p>
          </div>
          <div className="syllabus-block">
            <h3>Try it independently</h3>
            <p>{blueprint.practice.task}</p>
            <p><strong>Success looks like:</strong> {blueprint.practice.success}</p>
          </div>
          <details className="syllabus-details">
            <summary>Research starting points</summary>
            <ul className="syllabus-sources">{blueprint.sources.map((url, index) => <li key={url}><a href={url} target="_blank" rel="noreferrer">Reference {index + 1} · {new URL(url).hostname}</a></li>)}</ul>
          </details>
        </>
      )}
    </section>
  );
}
