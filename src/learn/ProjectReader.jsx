import { Component, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import LearningNav from "./components/LearningNav.jsx";
import { getProject, projectStageUrl } from "./data/projects/catalogue.js";
import { loadProject } from "./data/projects/loader.js";
import { topicMap } from "./data/catalogue.js";
import useProjectProgress from "./hooks/useProjectProgress.js";
import "./project-reader.css";

function ProjectError({ onRetry }) {
  return (
    <section className="project-error" role="alert">
      <h2>This stage could not be opened.</h2>
      <p>Your saved milestones are still available. Try again, or reload to fetch the current version.</p>
      <button type="button" onClick={onRetry}>Try again</button>
      <button type="button" onClick={() => window.location.reload()}>Reload page</button>
    </section>
  );
}

class ProjectBoundary extends Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error) {
    console.error("Project stage failed", error);
    this.props.onError();
  }

  render() {
    return this.state.failed
      ? <ProjectError onRetry={this.props.onRetry} />
      : this.props.children;
  }
}

function ProjectStage({ project, stage, children }) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState({ status: "loading", content: null });
  const [renderFailed, setRenderFailed] = useState(false);

  useEffect(() => {
    let active = true;
    setState({ status: "loading", content: null });
    loadProject(project.id).then(
      content => {
        if (!active) return;
        const hasStage = Object.hasOwn(content, stage.id) && typeof content[stage.id] === "function";
        setState({ status: hasStage ? "ready" : "error", content });
      },
      () => {
        if (active) setState({ status: "error", content: null });
      },
    );
    return () => { active = false; };
  }, [project.id, stage.id, attempt]);

  const retry = () => {
    setRenderFailed(false);
    setState({ status: "loading", content: null });
    setAttempt(value => value + 1);
  };
  const Content = state.content?.[stage.id];

  return (
    <>
      {state.status === "loading" && (
        <p className="project-load-state" role="status">Opening this project stage…</p>
      )}
      {state.status === "error" && <ProjectError onRetry={retry} />}
      {state.status === "ready" && (
        <ProjectBoundary key={attempt} onError={() => setRenderFailed(true)} onRetry={retry}>
          <div className="project-stage-content"><Content /></div>
        </ProjectBoundary>
      )}
      {children(state.status === "ready" && !renderFailed)}
    </>
  );
}

function ProjectNotFound({ project }) {
  return (
    <div className="learn-shell learning-workspace">
      <LearningNav active="projects" />
      <main className="workspace-main" id="learning-main" tabIndex={-1}>
        <h1>{project ? "Stage not found" : "Project not found"}</h1>
        <p className="workspace-lead">This address does not match a published project stage.</p>
        <Link
          className="workspace-button"
          to={project ? projectStageUrl(project.id) : "/learn/projects"}
        >
          Back to {project ? "project" : "projects"} →
        </Link>
      </main>
    </div>
  );
}

function ProjectOutline({ project, stage, completedCount, isStageComplete, storageUnavailable }) {
  return (
    <aside className="project-outline">
      <Link to="/learn/projects" className="workspace-text-link">← All projects</Link>
      <p className="workspace-eyebrow">{project.kind} · {project.level}</p>
      <h2>{project.title}</h2>
      <p className="project-progress">{project.stages.length} stages · {completedCount} completed</p>
      <progress max={project.stages.length} value={completedCount} aria-label="Project milestones completed" />
      <nav aria-label="Project stages">
        <ol>
          {project.stages.map((item, index) => {
            const complete = isStageComplete(project.id, item.id);
            return (
              <li key={item.id}>
                <Link to={projectStageUrl(project.id, item.id)} aria-current={stage.id === item.id ? "step" : undefined}>
                  <span className="project-stage-number" aria-hidden="true">
                    {complete ? "✓" : String(index + 1).padStart(2, "0")}
                  </span>
                  <span>
                    {item.title}
                    {complete && <span className="sr-only">, completed</span>}
                  </span>
                </Link>
              </li>
            );
          })}
        </ol>
      </nav>
      <p className="project-local-note">
        {storageUnavailable
          ? "Browser storage is unavailable. Your milestones last for this visit only."
          : "Your milestone checklist is saved in this browser."}
      </p>
    </aside>
  );
}

function ProjectPrerequisites({ project }) {
  const prerequisites = (project.prerequisiteIds || []).map(id => topicMap[id]).filter(Boolean);
  const relatedTopics = (project.relatedTopicIds || []).map(id => topicMap[id]).filter(Boolean);
  if (!prerequisites.length && !relatedTopics.length) return null;

  return (
    <details className="project-prerequisites">
      <summary>Concepts to bring into this project</summary>
      <p>Use these lessons for the mechanisms behind the build. Planned topics open their scope, not a finished lesson.</p>
      <ul>
        {prerequisites.map(topic => (
          <li key={topic.id}>
            <Link to={`/learn/topic/${topic.id}`}>
              {topic.title}<span>{topic.status}</span>
            </Link>
          </li>
        ))}
      </ul>
      {relatedTopics.map(topic => (
        <p key={topic.id}>
          <Link to={`/learn/topic/${topic.id}`}>
            Related topic: {topic.title} <span>({topic.status})</span>
          </Link>
        </p>
      ))}
    </details>
  );
}

export default function ProjectReader() {
  const { projectId, stageId } = useParams();
  const project = getProject(projectId);
  const { isStageComplete, completedCount, toggleStage, storageUnavailable } = useProjectProgress();
  const stage = stageId
    ? project?.stages.find(item => item.id === stageId)
    : project?.stages[0];
  const index = project?.stages.indexOf(stage) ?? -1;

  useEffect(() => {
    document.title = !project
      ? "Project not found · Learn"
      : !stage
        ? "Stage not found · Learn"
        : `${stage.title} · Research projects`;
    window.scrollTo(0, 0);
  }, [project, stage]);

  if (!project || !stage) return <ProjectNotFound project={project} />;

  const done = completedCount(project);
  const complete = isStageComplete(project.id, stage.id);
  const previousStage = project.stages[index - 1];
  const nextStage = project.stages[index + 1];

  return (
    <div className="learn-shell learning-workspace">
      <LearningNav active="projects" />
      <div className="project-reader">
        <ProjectOutline
          project={project}
          stage={stage}
          completedCount={done}
          isStageComplete={isStageComplete}
          storageUnavailable={storageUnavailable}
        />
        <main className="project-main" id="learning-main" tabIndex={-1}>
          <nav className="project-breadcrumb" aria-label="Breadcrumb">
            <Link to="/learn">Learn</Link><span aria-hidden="true">/</span>
            <Link to="/learn/projects">Projects</Link><span aria-hidden="true">/</span>
            <span>{project.title}</span>
          </nav>
          <header className="project-stage-header">
            <p className="workspace-eyebrow">
              STAGE {String(index + 1).padStart(2, "0")} / {String(project.stages.length).padStart(2, "0")}
              <span> · {project.status}</span>
            </p>
            <h1>{stage.title}</h1>
            <p>{stage.summary}</p>
            <div className="project-deliverable">
              <span>YOU WILL PRODUCE</span><p>{stage.deliverable}</p>
            </div>
          </header>
          <ProjectPrerequisites project={project} />
          <ProjectStage key={`${project.id}/${stage.id}`} project={project} stage={stage}>
            {ready => (
              <footer className="project-stage-footer">
                <label className="project-completion">
                  <input
                    type="checkbox"
                    checked={complete}
                    disabled={!ready}
                    onChange={() => toggleStage(project.id, stage.id)}
                  />
                  <span>I have completed this stage’s deliverable</span>
                </label>
                {storageUnavailable && (
                  <p role="status">Browser storage is unavailable. Progress will last for this visit only.</p>
                )}
                <div className="project-stage-navigation">
                  {previousStage ? (
                    <Link to={projectStageUrl(project.id, previousStage.id)}>← {previousStage.title}</Link>
                  ) : (
                    <Link to="/learn/projects">← Project library</Link>
                  )}
                  {nextStage ? (
                    <Link className="workspace-button" to={projectStageUrl(project.id, nextStage.id)}>Next stage →</Link>
                  ) : (
                    <Link className="workspace-button" to="/learn/projects">Back to projects ↗</Link>
                  )}
                </div>
                {done === project.stages.length && (
                  <p className="project-complete-note">
                    All milestones checked. Keep your code, results and limitations together so someone else can reproduce your work.
                  </p>
                )}
              </footer>
            )}
          </ProjectStage>
        </main>
      </div>
    </div>
  );
}
