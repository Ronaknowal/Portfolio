import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { topicMap } from "../data/catalogue";
import ProgressBar from "./ProgressBar";

function TopicItem({ id, isCurrent, done, onClick }) {
  const topic = topicMap[id];
  if (!topic) return null;
  const statusLabel = done ? "Completed" : topic.status === "published" ? "Published" : "Planned";
  return (
    <button
      type="button"
      onClick={() => onClick(id)}
      className={`reader-topic ${isCurrent ? "is-current" : ""} ${done ? "is-complete" : ""}`}
      aria-current={isCurrent ? "page" : undefined}
      data-topic-id={id}
    >
      <span>{topic.title}</span>
      <span className={`reader-topic__status reader-topic__status--${topic.status}`} role="img" aria-label={statusLabel} title={statusLabel}>{done ? "✓" : topic.status === "published" ? "●" : "○"}</span>
    </button>
  );
}

export default function ReaderSidebar({ contextLabel, contextType, navigationGroups, topicIds, currentTopicId, currentModuleId, isComplete, trackProgress, basePath }) {
  const navigate = useNavigate();
  const currentGroup = currentModuleId;
  const [openGroups, setOpenGroups] = useState(() => new Set(currentGroup ? [currentGroup] : []));
  const progress = trackProgress(topicIds);
  const groupsRef = useRef(null);

  useEffect(() => {
    if (!currentGroup) return;
    setOpenGroups((groups) => groups.has(currentGroup) ? groups : new Set([...groups, currentGroup]));
  }, [currentGroup]);

  useEffect(() => {
    const container = groupsRef.current;
    const selected = container?.querySelector('[aria-current="page"]');
    if (!selected) return;
    const bounds = container.getBoundingClientRect();
    const item = selected.getBoundingClientRect();
    // Scroll only the contents panel; never pull the lesson page back to the
    // sidebar on phones when moving to a new reading step.
    if (item.top < bounds.top || item.bottom > bounds.bottom) {
      container.scrollTop += item.top - bounds.top - 12;
    }
  }, [currentTopicId, currentGroup, openGroups]);

  const handleTopicClick = (topicId, moduleId) => {
    navigate(`${basePath}/${topicId}?module=${encodeURIComponent(moduleId)}`);
    window.scrollTo(0, 0);
  };

  const toggleGroup = (groupId) => {
    setOpenGroups((groups) => {
      const next = new Set(groups);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  };

  return (
    <aside className="reader-sidebar" aria-label="Course contents">
      <div className="reader-sidebar__header">
        <div className="reader-sidebar__type">{contextType}</div>
        <div className="reader-sidebar__title">{contextLabel}</div>
        <div className="reader-sidebar__progress">{progress.done} of {progress.total} complete</div>
        <div style={{ marginTop: 7 }}><ProgressBar percent={progress.percent} /></div>
      </div>

      <div className="reader-sidebar__groups" ref={groupsRef}>
        {navigationGroups.map((group) => {
          const isOpen = openGroups.has(group.id);
          const groupDone = group.topicIds.filter((id) => isComplete(id)).length;
          return (
            <section key={group.id} className="reader-group" data-module-id={group.id}>
              <button
                type="button"
                className="reader-group__toggle"
                onClick={() => toggleGroup(group.id)}
                aria-expanded={isOpen}
              >
                <span>{isOpen ? "▾" : "▸"} {group.label}</span>
                <span className="reader-group__size">
                  <span>{group.topicIds.length} {group.topicIds.length === 1 ? "topic" : "topics"}</span>
                  <span className="reader-group__completed">{groupDone} completed</span>
                </span>
              </button>
              {isOpen && (
                <div className="reader-group__topics">
                  {group.topicIds.length < group.totalTopicCount && <button type="button" className="reader-group__full" onClick={() => navigate(`/learn/track/${group.id}/${group.topicIds.includes(currentTopicId) ? currentTopicId : group.topicIds[0]}`)}>View all {group.totalTopicCount} module topics →</button>}
                  {group.topicIds.map((id) => (
                    <TopicItem
                      key={id}
                      id={id}
                      isCurrent={id === currentTopicId && group.id === currentGroup}
                      done={isComplete(id)}
                      onClick={id => handleTopicClick(id, group.id)}
                    />
                  ))}
                </div>
              )}
            </section>
          );
        })}
      </div>
    </aside>
  );
}
