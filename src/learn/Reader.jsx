import { useEffect } from "react";
import { useParams, useNavigate, useSearchParams, Link } from "react-router-dom";
import { colors, fonts, navLinkStyle } from "./styles";
import { topicMap } from "./data/catalogue";
import { tracks } from "./data/tracks";
import { getLearningRoute, getTracks, learningPaths, trackGroups } from "./data/curriculum";
import useProgress from "./hooks/useProgress";
import ReaderSidebar from "./components/ReaderSidebar";
import TopicContent from "./components/TopicContent";

const fullTrackIds = trackGroups.flatMap((group) => group.trackIds);

export default function Reader() {
  const { trackId, pathId, topicId } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { isComplete, toggleComplete, trackProgress } = useProgress();

  useEffect(() => {
    if (!document.querySelector('link[href*="JetBrains+Mono"]')) {
      const link = document.createElement("link");
      link.href = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap";
      link.rel = "stylesheet";
      document.head.appendChild(link);
    }
  }, []);

  const track = trackId ? tracks.find((item) => item.id === trackId) : null;
  const path = pathId ? learningPaths.find((item) => item.id === pathId) : null;
  const contextTracks = path ? getTracks(path.trackIds) : track ? [track] : getTracks(fullTrackIds);
  const { topicIds, navigationGroups, steps } = getLearningRoute(path || (track ? [track.id] : fullTrackIds));
  const basePath = path ? `/learn/path/${path.id}` : track ? `/learn/track/${track.id}` : "/learn/topic";
  const context = path
    ? { title: path.title, href: `/learn/path/${path.id}`, type: "GUIDED PATH" }
    : track
      ? { title: track.title, href: `/learn/track/${track.id}`, type: "MODULE" }
      : { title: "Complete curriculum", href: "/learn/topic", type: "ALL TOPICS" };

  let currentTopicId = topicId;
  if (!currentTopicId) {
    // Publication does not change the teaching sequence. Resume at the first
    // unfinished entry, including planned topics and recorded prerequisites.
    currentTopicId = topicIds.find((id) => !isComplete(id)) || topicIds[0];
  }

  const topic = topicMap[currentTopicId];
  const currentIndex = topicIds.indexOf(currentTopicId);
  const topicIsInContext = Boolean(topic) && currentIndex >= 0;
  const requestedModule = searchParams.get("module");
  const selectedStep = steps.findIndex(step => step.topicId === currentTopicId && step.moduleId === requestedModule);
  const stepIndex = selectedStep >= 0 ? selectedStep : steps.findIndex(step => step.topicId === currentTopicId);
  const currentModuleId = steps[stepIndex]?.moduleId;
  const currentModule = navigationGroups.find(group => group.id === currentModuleId);

  useEffect(() => {
    if ((pathId && !path) || (trackId && !track)) {
      navigate("/learn", { replace: true });
    } else if (!topicIsInContext) {
      // A focused route can change without invalidating an old topic link.
      navigate(topic ? `/learn/topic/${topic.id}` : "/learn", { replace: true });
    } else if (!topicId) {
      // Give entry URLs a stable lesson so marking it complete cannot silently
      // switch the page before the learner chooses Next.
      navigate(`${basePath}/${currentTopicId}`, { replace: true });
    }
  }, [navigate, path, pathId, topic, topicIsInContext, track, trackId, topicId, basePath, currentTopicId]);

  if (!topicIsInContext || !contextTracks.length) return null;

  return (
    <div className="learn-shell" style={{ fontFamily: fonts.sans }}>
      <nav className="learn-nav" aria-label="Primary navigation">
        <Link to="/" style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.gold, fontWeight: 600, letterSpacing: 1, textDecoration: "none" }}>
          ronak.ai
        </Link>
        <div className="learn-nav__links">
          <Link to="/" style={navLinkStyle}>portfolio</Link>
          <Link to="/learn" style={{ ...navLinkStyle, color: colors.gold }}>learn</Link>
        </div>
      </nav>

      <div className="reader-layout">
        <ReaderSidebar
          contextLabel={context.title}
          contextType={context.type}
          navigationGroups={navigationGroups}
          topicIds={topicIds}
          currentTopicId={currentTopicId}
          currentModuleId={currentModuleId}
          isComplete={isComplete}
          trackProgress={trackProgress}
          basePath={basePath}
        />
        <TopicContent
          key={topic.id}
          topic={topic}
          context={context}
          track={track}
          topicIds={topicIds}
          currentIndex={currentIndex}
          currentModule={currentModule}
          previousStep={steps[stepIndex - 1]}
          nextStep={steps[stepIndex + 1]}
          isComplete={isComplete}
          toggleComplete={toggleComplete}
          basePath={basePath}
        />
      </div>
    </div>
  );
}
