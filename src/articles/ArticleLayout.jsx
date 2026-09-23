import { Link } from "react-router-dom";
import SiteHeader from "../shared/layout/SiteHeader.jsx";
import "./articles.css";

export default function ArticleLayout({ children, reader = false }) {
  return <div className="articles-page">
    <SiteHeader section="articles">
      <nav className="articles-local-nav" aria-label="Articles navigation"><Link to="/articles" aria-current={reader ? undefined : "page"}>All articles</Link></nav>
    </SiteHeader>
    <main id="site-main" className={reader ? "articles-main articles-main--reader" : "articles-main"} tabIndex={-1}>{children}</main>
    <footer className="articles-footer"><Link to="/">← Back to home</Link><span>Ronak Sharma <span aria-hidden="true">/</span> Articles</span></footer>
  </div>;
}
