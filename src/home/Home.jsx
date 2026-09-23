import { useEffect } from "react";
import { Link } from "react-router-dom";
import SiteHeader from "../shared/layout/SiteHeader.jsx";
import "./home.css";

const highlights = [
  {
    type: "Guided build",
    title: "Build a typed decision model",
    description: "From a question to a trained model. Build the mechanism, test its limits, and make decisions you can inspect.",
    href: "/learn/projects/typed-decision-model",
    detail: "Code · Experiments · Evaluation",
  },
  {
    type: "Interactive lesson",
    title: "What makes a neuron learn?",
    description: "Start with perceptrons and activation functions. Move the controls, see the mechanism, then implement it yourself.",
    href: "/learn/path/full-curriculum/perceptrons-neurons-activation-functions?module=deep-learning-fundamentals",
    detail: "Deep learning · First principles",
  },
  {
    type: "Portfolio project",
    title: "Bloospace",
    description: "Turning social videos into useful, organized information. Explore this and the other projects in my portfolio.",
    href: "/portfolio#projects",
    detail: "Applied AI · Mobile",
  },
];

export default function Home() {
  useEffect(() => { document.title = "Ronak Sharma — Build & understand"; }, []);

  return (
    <div className="site-home">
      <SiteHeader />
      <main id="site-main" className="home-main" tabIndex={-1}>
        <div className="home-overview">
          <section className="home-intro" aria-labelledby="home-title">
            <p className="home-greeting">Hello, I’m</p>
            <h1 id="home-title">Ronak Sharma<span aria-hidden="true">.</span></h1>
            <p className="home-role">Data Science Engineer <span aria-hidden="true">/</span> AI Builder</p>
            <div className="home-bio">
              <p>I build AI systems and learn by taking ideas apart. My work spans machine learning, deep learning, NLP, and systems.</p>
              <p>This is my corner of the internet for the things I build and the things I want to understand deeply.</p>
            </div>
            <nav className="home-social" aria-label="Connect">
              <a href="https://github.com/Ronaknowal" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span><span className="sr-only"> (opens in a new tab)</span></a>
              <a href="https://www.linkedin.com/in/ronak-sharma-a6455a1b5/" target="_blank" rel="noopener noreferrer">LinkedIn <span aria-hidden="true">↗</span><span className="sr-only"> (opens in a new tab)</span></a>
              <Link to="/portfolio#about">More about me <span aria-hidden="true">→</span></Link>
            </nav>
          </section>

          <section id="explore" className="home-destinations" aria-labelledby="directory-title">
            <div className="home-directory-heading"><span aria-hidden="true">~/</span><h2 id="directory-title">Explore this space</h2></div>
            <Link to="/portfolio" className="home-destination home-destination--portfolio">
              <div className="home-destination__title"><h3>portfolio<span aria-hidden="true">/</span></h3><span aria-hidden="true">↗</span></div>
              <p>Projects, experience, and the work behind them.</p>
            </Link>
            <Link to="/learn" className="home-destination home-destination--learn">
              <div className="home-destination__title"><h3>learn<span aria-hidden="true">/</span></h3><span aria-hidden="true">↗</span></div>
              <p>Understand the ideas. Build from scratch. Explore with live labs.</p>
            </Link>
            <nav className="home-learning-shortcuts" aria-label="Start learning">
              <Link to="/learn/paths"><span>Guided paths</span><small>A sequence to follow</small><span aria-hidden="true">→</span></Link>
              <Link to="/learn/projects"><span>Build projects</span><small>The whole system, end to end</small><span aria-hidden="true">→</span></Link>
              <Link to="/learn/catalogue"><span>Find a concept</span><small>Go straight to your question</small><span aria-hidden="true">→</span></Link>
            </nav>
            <Link to="/articles" className="home-destination home-destination--articles">
              <div className="home-destination__title"><h3>articles<span aria-hidden="true">/</span></h3><span aria-hidden="true">↗</span></div>
              <p>A space for ideas, observations, and longer reads.</p>
            </Link>
          </section>
        </div>

        <aside className="home-principle" aria-label="Approach to learning">
          <span className="site-eyebrow">The thread through it all</span>
          <p>Understand it. <span>Build it.</span> Test its limits.</p>
        </aside>

        <section className="home-highlights" aria-labelledby="highlights-title">
          <div className="home-highlights__intro">
            <p className="site-eyebrow">Selected entries</p>
            <h2 id="highlights-title">On the workbench</h2>
            <p>A few things to open, explore, and pull apart.</p>
          </div>
          <div className="home-highlights__list">
            {highlights.map(item => (
              <Link className="home-highlight" key={item.href} to={item.href}>
                <div><span className="site-eyebrow">{item.type}</span><h3>{item.title}</h3><p>{item.description}</p><span className="home-highlight__detail">{item.detail}</span></div>
                <span className="home-highlight__arrow" aria-hidden="true">↗</span>
              </Link>
            ))}
          </div>
        </section>
      </main>
      <footer className="home-footer">
        <p>Ronak Sharma <span aria-hidden="true">/</span> Always learning.</p>
        <a href="#site-main">Back to top <span aria-hidden="true">↑</span></a>
      </footer>
    </div>
  );
}
