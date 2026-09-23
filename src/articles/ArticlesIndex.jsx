import { useEffect, useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import ArticleLayout from "./ArticleLayout.jsx";
import { articles } from "./generated/catalogue.js";
import { formatArticleDate } from "./article-format.js";

export default function ArticlesIndex() {
  const [params, setParams] = useSearchParams();
  const query = params.get("q") || "";
  const tag = params.get("tag") || "";
  const tags = useMemo(() => [...new Set(articles.flatMap(article => article.tags))].sort(), []);
  const filtered = useMemo(() => articles.filter(article => (!tag || article.tags.includes(tag)) &&
    `${article.title} ${article.summary} ${article.tags.join(" ")}`.toLowerCase().includes(query.trim().toLowerCase())), [query, tag]);
  useEffect(() => { document.title = "Articles · ronak.ai"; }, []);
  function updateFilter(key, value) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value); else next.delete(key);
    setParams(next, { replace: true });
  }
  return <ArticleLayout>
    <header className="articles-intro"><p className="site-eyebrow">Writing / Ronak Sharma</p><h1>Ideas worth<br /><span>staying with.</span></h1><p>A place for longer explanations, observations from building, and questions that deserve a closer look.</p></header>
    <section className="articles-collection" aria-labelledby="articles-list-title">
      <div className="articles-collection-heading"><h2 id="articles-list-title">Articles</h2><span>{articles.length} published</span></div>
      {articles.length > 0 ? <>
        <form className="articles-filters" role="search" onSubmit={event => event.preventDefault()}>
          <label>Search articles<input type="search" value={query} onChange={event => updateFilter("q", event.target.value)} placeholder="Find an idea or subject" /></label>
          {tags.length > 0 && <label>Subject<select value={tag} onChange={event => updateFilter("tag", event.target.value)}><option value="">All subjects</option>{tags.map(item => <option key={item}>{item}</option>)}{tag && !tags.includes(tag) && <option value={tag}>{tag}</option>}</select></label>}
        </form>
        <p className="articles-result-count" role="status">{filtered.length} {filtered.length === 1 ? "article" : "articles"}{query || tag ? " found" : ""}</p>
        {filtered.length ? <ol className="articles-list">{filtered.map(article => <li key={article.slug}>
          <div className="article-list-meta"><time dateTime={article.publishedAt}>{formatArticleDate(article.publishedAt)}</time><span>{article.readingMinutes} min read</span></div>
          <div><h3><Link to={`/articles/${article.slug}`}>{article.title}<span aria-hidden="true">↗</span></Link></h3><p>{article.summary}</p>{article.tags.length > 0 && <ul className="article-tags" aria-label="Subjects">{article.tags.map(item => <li key={item}>{item}</li>)}</ul>}</div>
        </li>)}</ol> : <div className="articles-empty"><h3>No matching articles.</h3><p>Try another phrase or subject.</p><button type="button" onClick={() => setParams({})}>Clear filters →</button></div>}
      </> : <div className="articles-empty"><span className="articles-empty-mark" aria-hidden="true">[ — ]</span><div><h3>No articles published yet.</h3><p>In the meantime, explore the learning library or take a look at what I’ve been building.</p><div className="articles-empty-links"><Link to="/learn">Explore Learn →</Link><Link to="/portfolio">Visit Portfolio →</Link></div></div></div>}
    </section>
  </ArticleLayout>;
}
