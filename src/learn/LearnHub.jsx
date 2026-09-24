import { useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { allTopicsOrdered, topicMap } from "./data/catalogue";
import { getLearningRoute, getTracks, learningPaths, trackGroups } from "./data/curriculum";
import { projects, projectStageUrl } from "./data/projects/catalogue.js";
import useProgress from "./hooks/useProgress";
import useProjectProgress from "./hooks/useProjectProgress.js";
import LearningNav from "./components/LearningNav.jsx";
import TrackCard from "./components/TrackCard";
import TopicList from "./components/TopicList";
import "./learning-workspace.css";

const publishedCount = allTopicsOrdered.filter(topic => topic.status === "published").length;
const moduleCount = new Set(trackGroups.flatMap(group => group.trackIds)).size;
const groupNames = {
  start: "Foundations",
  models: "Models & intelligence",
  systems: "Systems & engineering",
  embodied: "Brains & embodied intelligence",
  "professional-finance": "Financial research",
  reference: "Quantum & reference",
};
const pageHeadings = {
  paths: {
    title: "A direction for your curiosity.",
    description: "Connected routes through the curriculum, from your first concepts to specialist work. The sequence stays intact as new lessons are published.",
  },
  projects: {
    title: "Build it. Test it. Understand it.",
    description: "End-to-end projects with a researcher's workflow: a concrete question, working implementations, careful experiments, and results you can explain.",
  },
  modules: {
    title: "Explore the foundations and frontiers.",
    description: "Each module keeps its own teaching sequence. Open its outline to find the right depth or follow it from the beginning.",
  },
  catalogue: {
    title: "Follow the question.",
    description: "Search concepts, algorithms, and subtopics across the full curriculum. Published lessons and planned outlines are labelled separately.",
  },
};

function ProjectPreview({ project }) {
  const preview = project.preview;
  if (!preview) {
    return (
      <div className="project-feature__diagram">
        <span className="workspace-eyebrow">THE BUILD SEQUENCE</span>
        <ol>
          {project.stages.map(stage => <li key={stage.id}>{stage.title}</li>)}
        </ol>
        {project.outcome && <p>{project.outcome}</p>}
      </div>
    );
  }

  return (
    <div className="project-feature__diagram" role="img" aria-label={preview.accessibleDescription}>
      <span className="workspace-eyebrow">{preview.eyebrow}</span>
      <div className="model-flow__input">
        {preview.inputs.map(label => <span key={label}>{label}</span>)}
      </div>
      <span className="model-flow__arrow" aria-hidden="true">↓</span>
      <div className="model-flow__core">
        <span className="model-flow__glyph" aria-hidden="true">{preview.core.symbol || "→"}</span>
        <strong>{preview.core.title}</strong>
        <small>{preview.core.detail}</small>
      </div>
      <span className="model-flow__arrow" aria-hidden="true">↓</span>
      <div className="model-flow__output">
        {preview.outputs.map(label => <span key={label}>{label}</span>)}
      </div>
      <p>{preview.caption}</p>
    </div>
  );
}

function ProjectFeature({ project, compact = false }) {
  const { completedCount, nextUnfinishedStage } = useProjectProgress();
  const done = completedCount(project);
  const unfinishedStage = nextUnfinishedStage(project);
  const actionLabel = !done ? "Open project" : unfinishedStage ? "Continue project" : "Revisit project";
  const destination = projectStageUrl(project.id, done ? unfinishedStage?.id : undefined);

  return (
    <article className={`project-feature${compact ? " project-feature--compact" : ""}`}>
      <div className="project-feature__copy">
        <p className="workspace-eyebrow">{project.kind} <span>·</span> {project.status}</p>
        <h2><Link to={projectStageUrl(project.id)}>{project.title}</Link></h2>
        <p>{project.description}</p>
        <div className="workspace-meta">
          <span>{project.stages.length} stages</span><span>{done} completed</span><span>{project.level}</span>
        </div>
        <Link className="workspace-button" to={destination}>
          {actionLabel}<span aria-hidden="true">↗</span>
        </Link>
      </div>
      <ProjectPreview project={project} />
    </article>
  );
}

function FieldDirectory() {
  return (
    <section className="field-directory" aria-labelledby="fields-title">
      <div className="workspace-section-heading">
        <div>
          <p className="workspace-eyebrow">THE KNOWLEDGE BASE</p>
          <h2 id="fields-title">Find your field.</h2>
        </div>
        <Link to="/learn/modules" className="workspace-text-link">All {moduleCount} modules ↗</Link>
      </div>
      <div className="field-list">
        {trackGroups.map((group, index) => {
          const ids = [...new Set(getTracks(group.trackIds).flatMap(track => track.topicIds))];
          const available = ids.filter(id => topicMap[id]?.status === "published").length;
          return (
            <Link className="field-row" to={`/learn/modules?group=${group.id}`} key={group.id}>
              <span className="field-row__number">{String(index + 1).padStart(2, "0")}</span>
              <div><h3>{groupNames[group.id] || group.label}</h3><p>{group.description}</p></div>
              <span className="field-row__meta">
                {group.trackIds.length} {group.trackIds.length === 1 ? "module" : "modules"}
                <br />{available} published
              </span>
              <span className="field-row__arrow" aria-hidden="true">↗</span>
            </Link>
          );
        })}
      </div>
    </section>
  );
}

function PathDirectory({ trackProgress }) {
  return (
    <section className="workspace-paths" aria-label="Guided learning paths">
      <Link className="complete-syllabus" to="/learn/path/full-curriculum">
        <div>
          <span className="workspace-eyebrow">THE COMPLETE MAP</span>
          <h2>Browse the full curriculum</h2>
          <p>
            {moduleCount} modules · {allTopicsOrdered.length.toLocaleString()} topics · {trackProgress(allTopicsOrdered.map(topic => topic.id)).done} completed
          </p>
        </div>
        <span aria-hidden="true">↗</span>
      </Link>
      {learningPaths.filter(path => path.id !== "full-curriculum").map((path, index) => {
        const route = getLearningRoute(path);
        const published = route.topicIds.filter(id => topicMap[id]?.status === "published").length;
        const done = trackProgress(route.topicIds).done;
        return (
          <article className="workspace-path" data-path-id={path.id} key={path.id}>
            <span className="workspace-path__number">{String(index + 1).padStart(2, "0")}</span>
            <div>
              <p className="workspace-eyebrow">{path.eyebrow}</p>
              <h2><Link to={`/learn/path/${path.id}`}>{path.title}</Link></h2>
              <p>{path.description}</p>
              <p className="workspace-meta">{route.moduleCount} modules · {route.topicIds.length} topics · {done} completed</p>
              {path.milestones && (
                <details>
                  <summary>See the learning sequence</summary>
                  <ol>{path.milestones.map(milestone => <li key={milestone}>{milestone}</li>)}</ol>
                </details>
              )}
            </div>
            <div className="workspace-path__action">
              <span>{published} published<br />{route.topicIds.length - published} planned</span>
              <Link className="workspace-text-link" to={`/learn/path/${path.id}`}>
                {done ? "Continue path" : "Explore path"} ↗
              </Link>
            </div>
          </article>
        );
      })}
    </section>
  );
}

function WorkspaceHero() {
  return (
    <header className="workspace-hero">
      <div>
        <p className="workspace-eyebrow">A WORKSPACE FOR DEEP TECHNICAL LEARNING</p>
        <h1>Understand deeply.<br /><em>Build with evidence.</em></h1>
        <p className="workspace-lead">
          From the first principles to the systems you can build, test, and improve. Follow the ideas. Work through the details. Make something that holds up.
        </p>
        <div className="workspace-hero__actions">
          <Link className="workspace-button" to="/learn/paths">Find a learning path <span aria-hidden="true">→</span></Link>
          <Link className="workspace-text-link" to="/learn/projects">Learn by building ↗</Link>
        </div>
      </div>
      <aside className="research-loop">
        <p className="workspace-eyebrow">THE RESEARCH HABIT</p>
        <ol>
          <li><span>01</span><div><strong>Understand</strong><p>Intuition, mechanisms, and mathematics.</p></div></li>
          <li><span>02</span><div><strong>Implement</strong><p>From scratch, then with the right tools.</p></div></li>
          <li><span>03</span><div><strong>Investigate</strong><p>Baselines, ablations, and failure cases.</p></div></li>
        </ol>
        <Link to="/learn/path/ml-foundations">New to the field? Start with foundations →</Link>
      </aside>
    </header>
  );
}

function ProjectPrinciples() {
  return (
    <section className="project-principles">
      <p className="workspace-eyebrow">WHAT YOU TAKE AWAY</p>
      <div>
        <article>
          <span>01 / IMPLEMENTATION</span><h2>Something you can run.</h2>
          <p>Reproduce the mechanism, understand the library route, and know which parts you can change.</p>
        </article>
        <article>
          <span>02 / INVESTIGATION</span><h2>Evidence you can inspect.</h2>
          <p>Compare against baselines, isolate changes, and explain failures alongside successes.</p>
        </article>
        <article>
          <span>03 / CONNECTIONS</span><h2>A deeper understanding.</h2>
          <p>Revisit the exact concepts each stage needs, then bring them back to the system you are building.</p>
        </article>
      </div>
    </section>
  );
}

function ModuleDirectory({ requestedGroup, trackProgress, isComplete }) {
  const groups = requestedGroup ? [requestedGroup] : trackGroups;
  return (
    <>
      <nav className="workspace-filters" aria-label="Module fields">
        <Link to="/learn/modules" aria-current={!requestedGroup ? "page" : undefined}>All fields</Link>
        {trackGroups.map(group => (
          <Link key={group.id} to={`/learn/modules?group=${group.id}`} aria-current={requestedGroup?.id === group.id ? "page" : undefined}>
            {groupNames[group.id] || group.label}
          </Link>
        ))}
      </nav>
      {groups.map(group => (
        <section key={group.id} className="workspace-module-group">
          <div className="workspace-section-heading">
            <div><p className="workspace-eyebrow">{group.label}</p><h2>{group.description}</h2></div>
          </div>
          <div className="track-stack">
            {getTracks(group.trackIds).map(track => (
              <TrackCard
                key={track.id}
                track={track}
                progress={{ ...trackProgress(track.topicIds), completedIds: new Set(track.topicIds.filter(isComplete)) }}
              />
            ))}
          </div>
        </section>
      ))}
    </>
  );
}

export default function LearnHub({ view = "explore" }) {
  const { isComplete, trackProgress } = useProgress();
  const [params] = useSearchParams();
  const requestedGroup = trackGroups.find(group => group.id === params.get("group"));
  const featuredProject = projects[0];
  const pageHeading = pageHeadings[view];

  useEffect(() => {
    document.title = `${view === "explore" ? "Learn" : view[0].toUpperCase() + view.slice(1)} · ronak.sh`;
    window.scrollTo(0, 0);
  }, [view, requestedGroup]);

  return (
    <div className="learn-shell learning-workspace">
      <LearningNav active={view === "modules" ? "explore" : view} />
      <main id="learning-main" className="workspace-main" tabIndex={-1}>
        {view === "explore" ? (
          <>
            <WorkspaceHero />
            <div className="workspace-availability">
              <span><strong>{publishedCount}</strong> published lessons</span>
              <span>{allTopicsOrdered.length - publishedCount} planned</span>
              <span>{moduleCount} modules</span>
              <Link to="/learn/catalogue">Search the knowledge base ↗</Link>
            </div>
            <div className="workspace-section-heading">
              <div><p className="workspace-eyebrow">LEARNING THROUGH BUILDING</p><h2>A question. A model. Your experiment.</h2></div>
              <Link className="workspace-text-link" to="/learn/projects">Explore projects ↗</Link>
            </div>
            {featuredProject && <ProjectFeature project={featuredProject} compact />}
            <FieldDirectory />
          </>
        ) : (
          <>
            <header className="workspace-page-heading">
              <p className="workspace-eyebrow">LEARN / {view === "catalogue" ? "KNOWLEDGE BASE" : view.toUpperCase()}</p>
              <h1>{view === "modules" && requestedGroup ? groupNames[requestedGroup.id] || requestedGroup.label : pageHeading?.title}</h1>
              <p>{pageHeading?.description}</p>
            </header>
            {view === "paths" && <PathDirectory trackProgress={trackProgress} />}
            {view === "projects" && (
              <>
                {featuredProject ? <ProjectFeature project={featuredProject} /> : <p>Project guides are being prepared. Explore the learning paths while they take shape.</p>}
                <ProjectPrinciples />
                {projects.slice(1).map(project => <ProjectFeature key={project.id} project={project} compact />)}
              </>
            )}
            {view === "modules" && <ModuleDirectory requestedGroup={requestedGroup} trackProgress={trackProgress} isComplete={isComplete} />}
            {view === "catalogue" && <TopicList isComplete={isComplete} />}
          </>
        )}
      </main>
      <footer className="workspace-footer">
        <Link to="/learn">ronak.sh / learn</Link>
        <span>Concepts → implementations → evidence.</span>
        <Link to="/learn/path/full-curriculum">Complete curriculum ↗</Link>
      </footer>
    </div>
  );
}
