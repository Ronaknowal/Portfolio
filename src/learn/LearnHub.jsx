import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { colors, fonts, epochLabelStyle, sectionTitleStyle, subtitleStyle, navLinkStyle } from "./styles";
import { tracks } from "./data/tracks";
import useProgress from "./hooks/useProgress";
import TrackCard from "./components/TrackCard";
import TopicList from "./components/TopicList";

export default function LearnHub() {
  const [activeTab, setActiveTab] = useState("tracks");
  const { isComplete, trackProgress } = useProgress();

  // Load Google Fonts (same as Portfolio)
  useEffect(() => {
    if (!document.querySelector('link[href*="JetBrains+Mono"]')) {
      const link = document.createElement("link");
      link.href =
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap";
      link.rel = "stylesheet";
      document.head.appendChild(link);
    }
  }, []);

  return (
    <div style={{ background: colors.bg, color: colors.textPrimary, minHeight: "100vh", fontFamily: fonts.sans }}>
      {/* Nav */}
      <nav
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10001,
          padding: "16px 32px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "rgba(5,5,5,0.8)",
          backdropFilter: "blur(12px)",
          borderBottom: "1px solid #111",
        }}
      >
        <Link to="/" style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.gold, fontWeight: 600, letterSpacing: 1, textDecoration: "none" }}>
          ronak.ai
        </Link>
        <div style={{ display: "flex", gap: 28 }}>
          <Link to="/" style={navLinkStyle} onMouseEnter={(e) => (e.target.style.color = colors.gold)} onMouseLeave={(e) => (e.target.style.color = colors.textMuted)}>
            portfolio
          </Link>
          <Link to="/learn" style={{ ...navLinkStyle, color: colors.gold }}>
            learn
          </Link>
        </div>
      </nav>

      {/* Main content */}
      <div style={{ maxWidth: 800, margin: "0 auto", padding: "120px 32px 80px" }}>
        {/* Header */}
        <div style={epochLabelStyle}>EPOCH ∞ — KNOWLEDGE DISTILLATION</div>
        <h1 style={sectionTitleStyle}>Knowledge Distillation</h1>
        <p style={subtitleStyle}>Compressing what I know into digestible representations. Learn alongside me.</p>

        {/* Tab bar */}
        <div style={{ display: "flex", gap: 0, marginBottom: 24, borderBottom: `1px solid ${colors.border}` }}>
          {["tracks", "all topics"].map((tab) => {
            const isActive = activeTab === tab;
            return (
              <div
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: "8px 20px",
                  fontFamily: fonts.mono,
                  fontSize: 11,
                  color: isActive ? colors.gold : colors.textMuted,
                  borderBottom: isActive ? `2px solid ${colors.gold}` : "2px solid transparent",
                  cursor: "pointer",
                  textTransform: "uppercase",
                  transition: "color 0.2s ease",
                  letterSpacing: 1,
                }}
              >
                {tab}
              </div>
            );
          })}
        </div>

        {/* Tab content */}
        {activeTab === "tracks" ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {tracks.map((track) => {
              const progress = trackProgress(track.topicIds);
              // Add completedIds set for TrackCard to use
              const completedIds = new Set(track.topicIds.filter((id) => isComplete(id)));
              return (
                <TrackCard
                  key={track.id}
                  track={track}
                  progress={{ ...progress, completedIds }}
                />
              );
            })}
          </div>
        ) : (
          <TopicList isComplete={isComplete} />
        )}
      </div>

      {/* Footer */}
      <footer style={{ padding: "40px 32px", textAlign: "center", borderTop: "1px solid #0a0a0a" }}>
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: "#222", lineHeight: 2 }}>
          knowledge distillation in progress — still training
        </div>
      </footer>
    </div>
  );
}
