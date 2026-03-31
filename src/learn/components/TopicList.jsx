import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { colors, fonts } from "../styles";
import { allTopicsOrdered, categories } from "../data/topics/index";
import { tracks } from "../data/tracks";

export default function TopicList({ isComplete }) {
  const [activeCategory, setActiveCategory] = useState("all");
  const navigate = useNavigate();

  const filtered =
    activeCategory === "all"
      ? allTopicsOrdered
      : allTopicsOrdered.filter((t) => t.category === activeCategory);

  // Build a reverse lookup: topicId → track titles
  const topicToTracks = {};
  for (const track of tracks) {
    for (const id of track.topicIds) {
      if (!topicToTracks[id]) topicToTracks[id] = [];
      topicToTracks[id].push(track.title);
    }
  }

  return (
    <div>
      {/* Category filter pills */}
      <div style={{ display: "flex", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
        {["all", ...categories].map((cat) => {
          const isActive = activeCategory === cat;
          const count =
            cat === "all"
              ? allTopicsOrdered.length
              : allTopicsOrdered.filter((t) => t.category === cat).length;
          return (
            <span
              key={cat}
              onClick={() => setActiveCategory(cat)}
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
              }}
            >
              {cat} ({count})
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
              <div>
                <span style={{ fontFamily: fonts.sans, fontSize: 13, color: colors.textSecondary }}>
                  {topic.title}
                </span>
                <span style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.textDim, marginLeft: 8 }}>
                  {topic.category} · {topic.readTime}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                {inTracks && (
                  <span style={{ fontFamily: fonts.mono, fontSize: 8, color: colors.textMuted }}>
                    in: {inTracks.join(", ")}
                  </span>
                )}
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
