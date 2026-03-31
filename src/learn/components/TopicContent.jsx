import { useNavigate } from "react-router-dom";
import { colors, fonts } from "../styles";
import { tracks } from "../data/tracks";

export default function TopicContent({ topic, track, topicIds, currentIndex, isComplete, toggleComplete, basePath }) {
  const navigate = useNavigate();
  const done = isComplete(topic.id);

  // Find other tracks that contain this topic
  const otherTracks = tracks.filter(
    (t) => t.topicIds.includes(topic.id) && (!track || t.id !== track.id)
  );

  // Previous / Next
  const prevId = currentIndex > 0 ? topicIds[currentIndex - 1] : null;
  const nextId = currentIndex < topicIds.length - 1 ? topicIds[currentIndex + 1] : null;

  const handleNav = (id) => {
    navigate(`${basePath}/${id}`);
    window.scrollTo(0, 0);
  };

  const Content = topic.content;

  return (
    <div style={{ flex: 1, padding: "20px 32px", overflowY: "auto", minHeight: "calc(100vh - 57px)" }}>
      {/* Breadcrumb */}
      <div style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.textDim, marginBottom: 16 }}>
        <span
          onClick={() => navigate("/learn")}
          style={{ color: colors.textMuted, cursor: "pointer" }}
        >
          Learn
        </span>
        <span style={{ color: colors.textDark }}> → </span>
        {track ? (
          <>
            <span
              onClick={() => navigate(`/learn/track/${track.id}`)}
              style={{ color: colors.textMuted, cursor: "pointer" }}
            >
              {track.title}
            </span>
            <span style={{ color: colors.textDark }}> → </span>
          </>
        ) : (
          <>
            <span style={{ color: colors.textMuted }}>All Topics</span>
            <span style={{ color: colors.textDark }}> → </span>
          </>
        )}
        <span style={{ color: colors.gold }}>{topic.title}</span>
      </div>

      {/* "Also in" badges */}
      {otherTracks.length > 0 && (
        <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
          {otherTracks.map((t) => (
            <span
              key={t.id}
              onClick={() => navigate(`/learn/track/${t.id}/${topic.id}`)}
              style={{
                fontFamily: fonts.mono,
                fontSize: 8,
                color: colors.textMuted,
                padding: "2px 6px",
                border: `1px solid ${colors.border}`,
                borderRadius: 2,
                cursor: "pointer",
                transition: "color 0.2s ease",
              }}
              onMouseEnter={(e) => (e.target.style.color = colors.gold)}
              onMouseLeave={(e) => (e.target.style.color = colors.textMuted)}
            >
              Also in: {t.title}
            </span>
          ))}
        </div>
      )}

      {/* Topic header */}
      <h1 style={{ fontFamily: fonts.sans, fontSize: 28, fontWeight: 600, color: colors.textPrimary, margin: "0 0 6px" }}>
        {topic.title}
      </h1>
      <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginBottom: 28 }}>
        {topic.category} · {topic.readTime} · Topic {currentIndex + 1} of {topicIds.length}
      </div>

      {/* Article content (JSX) — supports single content or sections */}
      {topic.sections ? (
        topic.sections.map((section, i) => {
          const SectionContent = section.content;
          return (
            <div key={section.id || i} id={section.id} style={{ marginBottom: i < topic.sections.length - 1 ? 40 : 0 }}>
              {section.title && (
                <h2 style={{
                  fontFamily: fonts.sans,
                  fontSize: 22,
                  fontWeight: 600,
                  color: colors.textPrimary,
                  margin: "0 0 16px 0",
                  paddingTop: i > 0 ? 20 : 0,
                  borderTop: i > 0 ? `1px solid ${colors.border}` : "none",
                }}>
                  {section.title}
                </h2>
              )}
              <SectionContent />
            </div>
          );
        })
      ) : (
        <Content />
      )}

      {/* Bottom bar */}
      <div
        style={{
          marginTop: 40,
          paddingTop: 16,
          borderTop: `1px solid ${colors.border}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        {/* Mark complete toggle */}
        <div
          onClick={() => toggleComplete(topic.id)}
          style={{
            padding: "8px 16px",
            border: `1px solid ${colors.green}44`,
            borderRadius: 4,
            fontFamily: fonts.mono,
            fontSize: 11,
            color: colors.green,
            background: done ? `${colors.green}11` : "transparent",
            cursor: "pointer",
            transition: "all 0.2s ease",
          }}
        >
          {done ? "✓ Completed" : "○ Mark as Complete"}
        </div>

        {/* Prev / Next */}
        <div style={{ display: "flex", gap: 16 }}>
          {prevId && (
            <span
              onClick={() => handleNav(prevId)}
              style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textMuted, cursor: "pointer" }}
            >
              ← Previous
            </span>
          )}
          {nextId && (
            <span
              onClick={() => handleNav(nextId)}
              style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.gold, cursor: "pointer" }}
            >
              Next →
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
