import { useState } from "react";
import "./plot-output.css";

export default function PlotOutput({ example, alt, children }) {
  const [expanded, setExpanded] = useState(false);
  const src = "/learn-assets/plots/" + example.artifact;
  return <figure className="lesson-plot">
    <button className="lesson-plot__toggle" type="button" aria-pressed={expanded} onClick={() => setExpanded(value => !value)}>{expanded ? "Fit chart" : "Enlarge chart"}</button>
    <div className={"lesson-plot__scroll" + (expanded ? " is-expanded" : "")} tabIndex={0} role="region" aria-label={alt}>
      <img src={src} alt={alt} loading="lazy" />
    </div>
    <figcaption><strong>Rendered Python output.</strong> {children} <a href={src} target="_blank" rel="noreferrer">Open full-size chart</a>.</figcaption>
  </figure>;
}
