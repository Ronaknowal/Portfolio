import { Component, Suspense, lazy, useEffect, useLayoutEffect } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import Home from "../home/Home.jsx";
import SiteHeader from "../shared/layout/SiteHeader.jsx";
import { portfolioAnchors } from "../portfolio/navigation.js";

const Portfolio = lazy(() => import("../portfolio/Portfolio.jsx"));
const ArticlesIndex = lazy(() => import("../articles/ArticlesIndex.jsx"));
const ArticleReader = lazy(() => import("../articles/ArticleReader.jsx"));
const LearnHub = lazy(() => import("../learn/LearnHub.jsx"));
const Reader = lazy(() => import("../learn/Reader.jsx"));
const ProjectReader = lazy(() => import("../learn/ProjectReader.jsx"));

function HomeEntry() {
  const { hash, search } = useLocation();
  if (portfolioAnchors.some(anchor => hash === `#${anchor}`)) return <Navigate to={`/portfolio${search}${hash}`} replace />;
  return <Home />;
}

function MissingPage() {
  useEffect(() => { document.title = "Page not found · ronak.sh"; }, []);
  return <><SiteHeader /><main id="site-main" className="site-recovery" tabIndex={-1}><span className="site-eyebrow">404 / Page not found</span><h1>Let’s find your way back.</h1><p>This address doesn’t point to a published page. Start at home to explore the portfolio or learning workspace.</p><Link to="/">Back to home →</Link></main></>;
}

class SiteRouteBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return <><SiteHeader /><main id="site-main" className="site-recovery" tabIndex={-1}><h1>This page couldn’t load.</h1><p>Try reloading the page. Your saved learning progress stays on this device.</p><button type="button" onClick={() => window.location.reload()}>Reload page →</button><Link to="/">Back to home</Link></main></>;
    return this.props.children;
  }
}

function RouteScroll() {
  const { pathname, hash } = useLocation();
  useLayoutEffect(() => {
    if (!hash) {
      window.scrollTo({ top: 0, behavior: "instant" });
      return;
    }
    let anchor;
    try { anchor = decodeURIComponent(hash.slice(1)); } catch { return; }
    const scrollToAnchor = () => {
      const element = document.getElementById(anchor);
      if (!element) return false;
      element.scrollIntoView({ behavior: "instant", block: "start" });
      return true;
    };
    if (scrollToAnchor()) return;
    // A direct anchor may arrive before its lazy section has mounted.
    const observer = new MutationObserver(() => { if (scrollToAnchor()) observer.disconnect(); });
    observer.observe(document.getElementById("root"), { childList: true, subtree: true });
    const timeout = window.setTimeout(() => observer.disconnect(), 10000);
    return () => { observer.disconnect(); window.clearTimeout(timeout); };
  }, [pathname, hash]);
  return null;
}

export default function SiteRoutes() {
  const { pathname } = useLocation();
  return (
    <>
      <RouteScroll />
      <SiteRouteBoundary key={pathname}>
        <Suspense fallback={<div className="site-loading" role="status">Opening page…</div>}>
          <Routes>
            <Route path="/" element={<HomeEntry />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/articles" element={<ArticlesIndex />} />
            <Route path="/articles/:slug" element={<ArticleReader />} />
            <Route path="/learn" element={<LearnHub />} />
            <Route path="/learn/paths" element={<LearnHub view="paths" />} />
            <Route path="/learn/modules" element={<LearnHub view="modules" />} />
            <Route path="/learn/catalogue" element={<LearnHub view="catalogue" />} />
            <Route path="/learn/projects" element={<LearnHub view="projects" />} />
            <Route path="/learn/projects/:projectId/:stageId?" element={<ProjectReader />} />
            <Route path="/learn/path/:pathId/:topicId?" element={<Reader />} />
            <Route path="/learn/track/:trackId/:topicId?" element={<Reader />} />
            <Route path="/learn/topic/:topicId" element={<Reader />} />
            <Route path="*" element={<MissingPage />} />
          </Routes>
        </Suspense>
      </SiteRouteBoundary>
    </>
  );
}
