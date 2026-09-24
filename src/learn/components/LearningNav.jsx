import { Link } from "react-router-dom";
import SiteHeader from "../../shared/layout/SiteHeader.jsx";
import "../learning-base.css";
import "../learning-workspace.css";

const sections = [
  { id: "explore", label: "Explore", href: "/learn" },
  { id: "paths", label: "Paths", href: "/learn/paths" },
  { id: "projects", label: "Projects", href: "/learn/projects" },
];

export default function LearningNav({ active = "explore" }) {
  return (
    <SiteHeader section="learn" skipTarget="learning-main" className="site-header--learn">
        <nav className="workspace-nav__sections" aria-label="Learning navigation">
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
        </nav>
    </SiteHeader>
  );
}
