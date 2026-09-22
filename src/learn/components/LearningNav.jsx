import { useEffect } from "react";
import { Link } from "react-router-dom";
import "../learning-workspace.css";

const sections = [
  { id: "explore", label: "Explore", href: "/learn" },
  { id: "paths", label: "Paths", href: "/learn/paths" },
  { id: "projects", label: "Projects", href: "/learn/projects" },
];

export default function LearningNav({ active = "explore" }) {
  useEffect(() => {
    if (document.querySelector('link[href*="JetBrains+Mono"]')) return;
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
  }, []);

  return (
    <>
      <a className="learning-skip" href="#learning-main">Skip to content</a>
      <nav className="learn-nav workspace-nav" aria-label="Primary navigation">
        <div className="workspace-brand">
          <Link to="/" className="workspace-logo">ronak.ai</Link>
          <span aria-hidden="true">/</span>
          <Link to="/learn" className="workspace-wordmark">learn</Link>
        </div>
        <div className="workspace-nav__sections">
          {sections.map(section => (
            <Link
              key={section.id}
              to={section.href}
              aria-current={active === section.id ? "page" : undefined}
            >
              {section.label}
            </Link>
          ))}
          <Link
            to="/learn/catalogue"
            className="workspace-search"
            aria-label="Search the knowledge base"
            aria-current={active === "catalogue" ? "page" : undefined}
          >
            <svg width="15" height="15" viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <circle cx="8.5" cy="8.5" r="5.5" stroke="currentColor" strokeWidth="1.5" />
              <path d="m13 13 4 4" stroke="currentColor" strokeWidth="1.5" />
            </svg>
            <span>Search</span>
          </Link>
        </div>
      </nav>
    </>
  );
}
