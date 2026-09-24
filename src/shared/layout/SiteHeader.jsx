import { useEffect, useId, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { siteSections } from "../../app/navigation.js";
import "./site-shell.css";

export default function SiteHeader({ section = "home", children, skipTarget = "site-main", className = "" }) {
  const [isOpen, setIsOpen] = useState(false);
  const switcherRef = useRef(null);
  const triggerRef = useRef(null);
  const panelId = useId();
  const { pathname, hash } = useLocation();
  const currentSection = siteSections.find(item => item.id === section);

  useEffect(() => { setIsOpen(false); }, [pathname, hash]);

  useEffect(() => {
    if (!isOpen) return;
    const dismissOutside = event => {
      if (!switcherRef.current?.contains(event.target)) setIsOpen(false);
    };
    const dismissEscape = event => {
      if (event.key !== "Escape") return;
      setIsOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", dismissOutside);
    document.addEventListener("focusin", dismissOutside);
    document.addEventListener("keydown", dismissEscape);
    return () => {
      document.removeEventListener("pointerdown", dismissOutside);
      document.removeEventListener("focusin", dismissOutside);
      document.removeEventListener("keydown", dismissEscape);
    };
  }, [isOpen]);

  return (
    <>
      <a className="site-skip" href={`#${skipTarget}`}>Skip to content</a>
      <header className={`site-header ${className}`}>
        <div className="site-brand">
          <Link to="/" className="site-logo" aria-label="ronak.sh — Home">ronak.sh<span aria-hidden="true">↗</span></Link>
          {section !== "home" && (
            <div className="site-switcher" ref={switcherRef}>
              <span className="site-brand__separator" aria-hidden="true">/</span>
              <button ref={triggerRef} type="button" className="site-switcher__trigger" aria-label={`Browse site, current section: ${currentSection.label}`} aria-expanded={isOpen} aria-controls={panelId} onClick={() => setIsOpen(open => !open)}>
                <span>{currentSection.label}</span>
                <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true"><path d="m2 3.5 3 3 3-3" stroke="currentColor" strokeWidth="1.3" /></svg>
              </button>
              {isOpen && (
                <nav id={panelId} className="site-switcher__panel" aria-label="Site sections">
                  <p>Explore ronak.sh</p>
                  {siteSections.map(item => (
                    <Link key={item.id} to={item.href} aria-current={item.id === section ? "true" : undefined} onClick={() => setIsOpen(false)}>
                      <span>{item.label}<span aria-hidden="true">{item.id === section ? "•" : "↗"}</span></span>
                      <small>{item.description}</small>
                    </Link>
                  ))}
                </nav>
              )}
            </div>
          )}
        </div>
        {children || (
          <nav className="site-header__links" aria-label="Primary navigation">
            {siteSections.filter(item => item.id !== "home").map(item => <Link key={item.id} to={item.href} aria-current={item.id === section ? "page" : undefined}>{item.label}<span aria-hidden="true">↗</span></Link>)}
          </nav>
        )}
      </header>
    </>
  );
}
