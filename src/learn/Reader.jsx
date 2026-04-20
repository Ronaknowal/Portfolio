import "katex/dist/katex.min.css";
import { useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { colors, fonts, navLinkStyle } from "./styles";
import { topicMap, allTopicIds } from "./data/topics/index";
import { tracks } from "./data/tracks";
import useProgress from "./hooks/useProgress";
import ReaderSidebar from "./components/ReaderSidebar";
import TopicContent from "./components/TopicContent";

export default function Reader() {
  const { trackId, topicId } = useParams();
  const navigate = useNavigate();
  const { isComplete, toggleComplete, trackProgress } = useProgress();

  // Load Google Fonts
  useEffect(() => {
    if (!document.querySelector('link[href*="JetBrains+Mono"]')) {
      const link = document.createElement("link");
      link.href =
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap";
      link.rel = "stylesheet";
      document.head.appendChild(link);
    }
  }, []);

  // Determine context: track-based or all-topics
  const track = trackId ? tracks.find((t) => t.id === trackId) : null;
  const topicIds = track ? track.topicIds : allTopicIds;
  const basePath = track ? `/learn/track/${track.id}` : "/learn/topic";

  // Resolve current topic
  let currentTopicId = topicId;
  if (!currentTopicId && track) {
    // Find first incomplete topic in track, or first topic
    currentTopicId = topicIds.find((id) => !isComplete(id)) || topicIds[0];
  }
  if (!currentTopicId) {
    currentTopicId = topicIds[0];
  }

  const topic = topicMap[currentTopicId];
  const currentIndex = topicIds.indexOf(currentTopicId);

  // Redirect if topic not found
  useEffect(() => {
    if (!topic) {
      navigate("/learn", { replace: true });
    }
  }, [topic, navigate]);

  if (!topic) return null;

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

      {/* Reader layout */}
      <div style={{ display: "flex", paddingTop: 57 }}>
        <ReaderSidebar
          track={track}
          topicIds={topicIds}
          currentTopicId={currentTopicId}
          isComplete={isComplete}
          trackProgress={trackProgress}
          basePath={basePath}
        />
        <TopicContent
          topic={topic}
          track={track}
          topicIds={topicIds}
          currentIndex={currentIndex}
          isComplete={isComplete}
          toggleComplete={toggleComplete}
          basePath={basePath}
        />
      </div>
    </div>
  );
}
