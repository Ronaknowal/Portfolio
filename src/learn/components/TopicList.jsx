import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { colors, fonts, levelColors, levelLabels } from "../styles";
import { allTopicsOrdered, categories } from "../data/topics/index";
import { tracks } from "../data/tracks";
import LevelBadge from "./LevelBadge";

const LEVELS = ["foundation", "intermediate", "advanced", "frontier"];

export default function TopicList({ isComplete }) {
  const [activeCategory, setActiveCategory] = useState("all");
  const [activeLevel, setActiveLevel] = useState("all");
  const navigate = useNavigate();

  let filtered = activeCategory === "all"
    ? allTopicsOrdered
    : allTopicsOrdered.filter((t) => t.category === activeCategory);

  if (activeLevel !== "all") {
    filtered = filtered.filter((t) => t.level === activeLevel);
  }

  // Build a reverse lookup: topicId → track titles
  const topicToTracks = {};
  for (const track of tracks) {
    for (const id of track.topicIds) {
      if (!topicToTracks[id]) topicToTracks[id] = [];
      topicToTracks[id].push(track.title);
    }
  }

  // Find track label for category ID
  const categoryLabel = (catId) => {
    const cat = categories.find((c) => c.id === catId);
    return cat ? cat.label : catId;
  };

  return (
    <div>
      {/* Category filter pills */}
      <div style={{ display: "flex", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
        <span
          onClick={() => setActiveCategory("all")}
          style={{
            padding: "3px 10px",
            border: `1px solid ${activeCategory === "all" ? `${colors.gold}44` : colors.border}`,
            borderRadius: 3,
            fontFamily: fonts.mono,
            fontSize: 9,
            color: activeCategory === "all" ? colors.gold : colors.textMuted,
            cursor: "pointer",
            textTransform: "uppercase",
            transition: "all 0.2s ease",
          }}
        >
          all ({allTopicsOrdered.length})
        </span>
        {categories.map((cat) => {
          const isActive = activeCategory === cat.id;
          const count = allTopicsOrdered.filter((t) => t.category === cat.id).length;
          return (
            <span
              key={cat.id}
              onClick={() => setActiveCategory(cat.id)}
              style={{
                padding: "3px 10px",
                border: `1px solid ${isActive ? `${colors.gold}44` : colors.border}`,
                borderRadius: 3,
                fontFamily: fonts.mono,
                fontSize: 9,
                color: isActive ? colors.gold : colors.textMuted,
                cursor: "pointer",
                textTransform: "uppercase",
                transition: "all 0.2s ease",
                maxWidth: 200,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {cat.label.length > 25 ? cat.label.slice(0, 22) + "..." : cat.label} ({count})
            </span>
          );
        })}
      </div>

      {/* Level filter pills */}
      <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
        <span
          onClick={() => setActiveLevel("all")}
          style={{
            padding: "3px 10px",
            border: `1px solid ${activeLevel === "all" ? `${colors.gold}44` : colors.border}`,
            borderRadius: 3,
            fontFamily: fonts.mono,
            fontSize: 9,
            color: activeLevel === "all" ? colors.gold : colors.textMuted,
            cursor: "pointer",
            textTransform: "uppercase",
          }}
        >
          all levels
        </span>
        {LEVELS.map((lvl) => {
          const isActive = activeLevel === lvl;
          const lvlColor = levelColors[lvl];
          return (
            <span
              key={lvl}
              onClick={() => setActiveLevel(lvl)}
              style={{
                padding: "3px 10px",
                border: `1px solid ${isActive ? `${lvlColor}44` : colors.border}`,
                borderRadius: 3,
                fontFamily: fonts.mono,
                fontSize: 9,
                color: isActive ? lvlColor : colors.textMuted,
                cursor: "pointer",
                textTransform: "uppercase",
                transition: "all 0.2s ease",
              }}
            >
              {levelLabels[lvl]}
            </span>
          );
        })}
      </div>

      {/* Topic rows */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {filtered.map((topic) => {
          const done = isComplete(topic.id);
          const inTracks = topicToTracks[topic.id];
          return (
            <div
              key={topic.id}
              onClick={() => navigate(`/learn/topic/${topic.id}`)}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "8px 12px",
                border: `1px solid ${colors.border}`,
                borderRadius: 4,
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = colors.cardHoverBorder;
                e.currentTarget.style.background = colors.cardHoverBg;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = colors.border;
                e.currentTarget.style.background = "transparent";
              }}
            >
              <div style={{ minWidth: 0, flex: 1 }}>
                <span style={{ fontFamily: fonts.sans, fontSize: 13, color: colors.textSecondary }}>
                  {topic.title}
                </span>
                <span style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.textDim, marginLeft: 8 }}>
                  {topic.readTime}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                <LevelBadge level={topic.level} />
                <span style={{ fontSize: 12, color: done ? colors.green : colors.textDark }}>
                  {done ? "✓" : "○"}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
