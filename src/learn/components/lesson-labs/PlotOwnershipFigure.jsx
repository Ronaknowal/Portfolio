import './scientific-concept-visuals.css';

export function PlotOwnershipDiagram() {
  return <figure className="sci-visual sci-plot-ownership" aria-label="Figure contains an Axes; the Axes contains Axis objects and plotted Artists">
    <figcaption><strong>Three similarly named objects, three different jobs</strong><span>Containment diagram — not a plot of measurements.</span></figcaption>
    <div className="sci-figure-boundary"><strong>fig · Figure</strong><code>fig.savefig("report.svg")</code><div className="sci-axes-boundary"><strong>ax · Axes</strong><span>One panel inside the canvas</span><div className="sci-plot-parts"><div className="sci-axis-part"><strong>ax.yaxis · Axis</strong><span>y scale, ticks, label</span></div><div className="sci-artist-part"><span aria-hidden="true">○ ─── ○ ─── ○</span><strong>Line2D Artist</strong><code>ax.plot(...)</code><small>Also here: text, legend and other Artists</small></div><div className="sci-xaxis-part"><strong>ax.xaxis · Axis</strong><span>x scale, ticks, label</span></div></div></div></div>
    <p className="sci-caption">Change the x ticks on the panel with <code>ax.set_xticks(...)</code>; its x Axis manages those ticks. Add a line to that panel with <code>ax.plot(...)</code>. Save the complete canvas with <code>fig.savefig(...)</code>. Multiple panels can live inside one Figure. Figure, Axes and Axis are themselves Artists too; “Artist” is the broad drawable-object family, not a separate peer outside the hierarchy.</p>
  </figure>;
}
