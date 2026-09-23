import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ArticleLayout from "./ArticleLayout.jsx";
import { articles } from "./generated/catalogue.js";
import { articleLoaders } from "./generated/loaders.js";
import { formatArticleDate } from "./article-format.js";

const markdownComponents = {
  a({ node: _node, href, children, ...attributes }) {
    if (!href) return <span>{children}</span>;
    if (href.startsWith("/") && !href.startsWith("//")) return <Link to={href} {...attributes}>{children}</Link>;
    // Preserve GFM footnote IDs, back-reference labels and other semantic attributes.
    return <a href={href} {...attributes} rel="noreferrer">{children}</a>;
  },
  img({ src, alt, title }) { return <img src={src} alt={alt || ""} title={title} loading="lazy" decoding="async" />; },
  table({ children }) { return <div className="article-table-scroll" role="region" aria-label="Scrollable table" tabIndex={0}><table>{children}</table></div>; },
  pre({ children }) { return <pre tabIndex={0} aria-label="Code example">{children}</pre>; },
};

function ArticleBody({ article }) {
  const [body, setBody] = useState(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    setFailed(false);
    articleLoaders[article.slug]().then(value => { if (active) setBody(value); }).catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [article.slug, attempt]);
  return <>
    <header className="article-heading"><Link className="article-back" to="/articles">← All articles</Link><p className="site-eyebrow">Article / {article.tags.join(" · ") || "Writing"}</p><h1>{article.title}</h1><p className="article-summary">{article.summary}</p>
      <div className="article-byline"><span>{article.author}</span><time dateTime={article.publishedAt}>{formatArticleDate(article.publishedAt)}</time><span>{article.readingMinutes} min read</span>{article.updatedAt && article.updatedAt !== article.publishedAt && <span>Updated <time dateTime={article.updatedAt}>{formatArticleDate(article.updatedAt)}</time></span>}</div>
    </header>
    {failed ? <div className="articles-empty" role="alert"><div><h2>The article couldn’t load.</h2><p>Check your connection and try again.</p><button type="button" onClick={() => setAttempt(value => value + 1)}>Try again →</button></div></div> : body === null ? <p className="article-loading" role="status">Loading article…</p> : <article className="article-prose"><Markdown remarkPlugins={[remarkGfm]} components={markdownComponents} skipHtml>{body}</Markdown></article>}
    {body !== null && !failed && <nav className="article-end" aria-label="After this article"><Link to="/articles">← Browse all articles</Link><a href="#site-main">Back to top ↑</a></nav>}
  </>;
}

export default function ArticleReader() {
  const { slug } = useParams();
  const article = articles.find(item => item.slug === slug);
  useEffect(() => { document.title = `${article?.title || "Article not found"} · ronak.ai`; }, [article]);
  return <ArticleLayout reader>{article ? <ArticleBody key={slug} article={article} /> : <div className="article-missing"><p className="site-eyebrow">404 / Article not found</p><h1>This article isn’t available.</h1><p>It may not be published, or the address may have changed.</p><Link to="/articles">Browse published articles →</Link></div>}</ArticleLayout>;
}
