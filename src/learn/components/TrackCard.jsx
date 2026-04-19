import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { colors, fonts } from "../styles";
import { topicMap } from "../data/topics/index";
import ProgressBar from "./ProgressBar";
import LevelBadge from "./LevelBadge";

export default function TrackCard({ track, progress }) {
  const [hovered, setHovered] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();
  const { done, total, percent } = progress;

  const handleCardClick = () => {
    navigate(`/learn/track/${track.id}`);
  };

  const handleTopicClick = (e, topicId) => {
    e.stopPropagation();
    navigate(`/learn/track/${track.id}/${topicId}`);
  };

  const handleToggleExpand = (e) => {
    e.stopPropagation();
    setExpanded((prev) => !prev);
  };

  return (
    <div
      onClick={handleCardClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        border: `1px solid ${hovered ? colors.cardHoverBorder : colors.border}`,
        borderRadius: 6,
        padding: 16,
        background: hovered ? colors.cardHoverBg : colors.cardBg,
        cursor: "pointer",
        transition: "all 0.3s ease",
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
        <div>
          <div style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.gold, letterSpacing: 1, marginBottom: 4 }}>
            DEEP DIVE · {total} TOPICS · {track.sections?.length || 0} SECTIONS
          </div>
          <div style={{ fontFamily: fonts.sans, fontSize: 18, fontWeight: 600, color: colors.textPrimary }}>
            {track.title}
          </div>
        </div>
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: done > 0 ? colors.green : colors.textMuted }}>
          {done}/{total} {done > 0 && "✓"}
        </div>
      </div>

      {/* Description */}
      <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textMuted, marginBottom: 10 }}>
        {track.description}
      </div>

      {/* Progress bar */}
      <ProgressBar percent={percent} />

      {/* Expandable sections with topics */}
      <div
        onClick={handleToggleExpand}
        style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid #111", cursor: "pointer" }}
      >
        <div style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.textDim, marginBottom: 8 }}>
          {expanded ? "▾" : "▸"} SECTIONS & TOPICS
        </div>
        {expanded && track.sections && (
          <div onClick={(e) => e.stopPropagation()}>
            {track.sections.map((section, si) => {
              const sectionDone = section.topicIds.filter((id) => progress.completedIds?.has(id)).length;
              return (
                <div key={si} style={{ marginBottom: si < track.sections.length - 1 ? 14 : 0 }}>
                  {/* Section header */}
                  <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 6,
                    paddingBottom: 4,
                    borderBottom: `1px solid #111`,
                  }}>
                    <span style={{
                      fontFamily: fonts.mono,
                      fontSize: 9,
                      color: colors.gold,
                      letterSpacing: 0.5,
                      textTransform: "uppercase",
                    }}>
                      {section.name}
                    </span>
                    <span style={{ fontFamily: fonts.mono, fontSize: 8, color: colors.textDim }}>
                      {sectionDone}/{section.topicIds.length}
                    </span>
                  </div>

                  {/* Topic chips in this section */}
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                    {section.topicIds.map((id) => {
                      const topic = topicMap[id];
                      if (!topic) return null;
                      const isDone = progress.completedIds?.has(id);
                      return (
                        <span
                          key={id}
                          onClick={(e) => handleTopicClick(e, id)}
                          style={{
                            padding: "2px 6px",
                            borderRadius: 3,
                            fontFamily: fonts.mono,
                            fontSize: 8,
                            color: isDone ? colors.green : colors.textMuted,
                            background: isDone ? `${colors.green}11` : "transparent",
                            border: `1px solid ${isDone ? `${colors.green}33` : colors.border}`,
                            cursor: "pointer",
                            transition: "all 0.2s ease",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 4,
                          }}
                        >
                          {isDone ? "✓ " : ""}{topic.title}
                        </span>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
