import { useNavigate } from "react-router-dom";
import { colors, fonts } from "../styles";
import { topicMap } from "../data/topics/index";
import ProgressBar from "./ProgressBar";

export default function ReaderSidebar({ track, topicIds, currentTopicId, isComplete, trackProgress, basePath }) {
  const navigate = useNavigate();

  const handleTopicClick = (topicId) => {
    navigate(`${basePath}/${topicId}`);
    window.scrollTo(0, 0);
  };

  const progress = trackProgress(topicIds);

  return (
    <div
      style={{
        width: 220,
        flexShrink: 0,
        borderRight: `1px solid ${colors.border}`,
        padding: "16px 14px",
        overflowY: "auto",
        height: "calc(100vh - 57px)",
        position: "sticky",
        top: 57,
      }}
    >
      {/* Track header */}
      <div style={{ marginBottom: 16, paddingBottom: 12, borderBottom: "1px solid #111" }}>
        <div style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.gold, letterSpacing: 1, marginBottom: 4 }}>
          {track ? "TRACK" : "ALL TOPICS"}
        </div>
        {track && (
          <div style={{ fontFamily: fonts.sans, fontSize: 13, color: colors.textPrimary, fontWeight: 500, marginBottom: 6 }}>
            {track.title}
          </div>
        )}
        <div style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.textMuted }}>
          {progress.done} of {progress.total} completed
        </div>
        <div style={{ marginTop: 6 }}>
          <ProgressBar percent={progress.percent} />
        </div>
      </div>

      {/* Topic list */}
      <div style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.textDark, letterSpacing: 1, marginBottom: 8 }}>
        TOPICS
      </div>
      {topicIds.map((id) => {
        const topic = topicMap[id];
        if (!topic) return null;
        const isCurrent = id === currentTopicId;
        const done = isComplete(id);
        return (
          <div
            key={id}
            onClick={() => handleTopicClick(id)}
            style={{
              padding: "6px 8px",
              borderRadius: 3,
              fontFamily: fonts.mono,
              fontSize: 11,
              color: isCurrent ? colors.gold : done ? colors.green : colors.textMuted,
              background: isCurrent ? `${colors.gold}14` : "transparent",
              borderLeft: isCurrent ? `2px solid ${colors.gold}` : "2px solid transparent",
              cursor: "pointer",
              marginBottom: 2,
              transition: "all 0.2s ease",
            }}
          >
            <span style={{ marginRight: 6 }}>
              {isCurrent ? "▸" : done ? "✓" : "○"}
            </span>
            {topic.title}
          </div>
        );
      })}
    </div>
  );
}
