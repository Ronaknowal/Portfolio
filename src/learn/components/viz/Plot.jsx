import { useLayoutEffect, useRef, useState } from "react";
import { colors, fonts } from "../../styles";
import "./plot.css";

const palette = [colors.gold, colors.green, "#c084fc", "#60a5fa"];
const valuesPerPage = 50;

function endpointLabels(minimum, maximum) {
  const compact = (value) => {
    if (value === 0) return "0";
    const rounded = Number(value.toPrecision(5));
    return Math.abs(rounded) >= 1e5 || Math.abs(rounded) < 1e-3
      ? rounded.toExponential().replace("e+", "e")
      : String(rounded);
  };
  const labels = [compact(minimum), compact(maximum)];
  // Close limits can need more digits than an axis can carry. Keep their full
  // values in flowing text instead of rounding them to the same tick.
  const needsExactRange = minimum !== maximum && labels[0] === labels[1];
  return {
    labels: needsExactRange ? ["min", "max"] : labels,
    needsExactRange,
    rounded: labels.some((value, index) => Number(value) !== [minimum, maximum][index]),
  };
}

// series: Array<{ name: string, color?: string, points: Array<[x, y]> }>
export default function Plot({ series, width = 480, height = 240, xLabel, yLabel, label }) {
  const containerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(width);
  const [showValues, setShowValues] = useState(false);
  const [valuesPage, setValuesPage] = useState(0);

  useLayoutEffect(() => {
    const container = containerRef.current;
    let previousWidth;
    const updateWidth = (measuredWidth) => {
      const nextWidth = Math.floor(measuredWidth);
      if (nextWidth > 0 && nextWidth !== previousWidth) {
        previousWidth = nextWidth;
        setContainerWidth(nextWidth);
      }
    };
    updateWidth(container.getBoundingClientRect().width);
    // Recompute coordinates rather than shrinking fixed-size SVG text on phones.
    const observer = new ResizeObserver(([entry]) => updateWidth(entry.contentRect.width));
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const allPoints = series.flatMap((item) => item.points);
  const xs = allPoints.map(([x]) => x);
  const ys = allPoints.map(([, y]) => y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const xTicks = endpointLabels(xMin, xMax);
  const yTicks = endpointLabels(yMin, yMax);
  const pad = {
    t: 18,
    r: 12,
    b: 30,
    l: Math.max(42, Math.max(...yTicks.labels.map((value) => value.length)) * 7.5 + 12),
  };
  const plotWidth = containerWidth - pad.l - pad.r;
  const plotHeight = height - pad.t - pad.b;
  const xTicksTooWide = xMin !== xMax && xTicks.labels.join("").length * 7.5 + 12 > plotWidth;
  const horizontalLabels = xTicksTooWide ? ["min", "max"] : xTicks.labels;
  const sx = (x) => pad.l + ((x - xMin) / xRange) * plotWidth;
  const sy = (y) => pad.t + plotHeight - ((y - yMin) / yRange) * plotHeight;
  const tickStyle = { fill: "#adb5b0", fontFamily: fonts.mono, fontSize: 12 };
  const pageCount = Math.max(1, Math.ceil(allPoints.length / valuesPerPage));
  const currentPage = Math.min(valuesPage, pageCount - 1);
  const firstValue = currentPage * valuesPerPage;
  const lastValue = Math.min(firstValue + valuesPerPage, allPoints.length);
  let precedingPoints = 0;
  const visibleValues = showValues ? series.flatMap((item, index) => {
    const seriesStart = precedingPoints;
    precedingPoints += item.points.length;
    return item.points.slice(Math.max(0, firstValue - seriesStart), Math.max(0, lastValue - seriesStart))
      .map(([x, y], pointIndex) => ({ key: `${index}-${pointIndex}`, name: item.name || `Series ${index + 1}`, x, y }));
  }) : [];

  return (
    <figure className="lesson-plot" ref={containerRef} style={{ maxWidth: width }}>
      {label && <figcaption className="lesson-plot-caption">{label}</figcaption>}
      {series.length > 1 && (
        <ul className="lesson-plot-legend" aria-label="Plotted series">
          {series.map((item, index) => (
            <li key={index}>
              <span aria-hidden="true" style={{ background: item.color || palette[index % palette.length] }} />
              {item.name}
            </li>
          ))}
        </ul>
      )}
      {yLabel && <div className="lesson-plot-axis-label">Vertical axis: {yLabel}</div>}
      <svg
        viewBox={`0 0 ${containerWidth} ${height}`}
        width={containerWidth}
        height={height}
        role="img"
        aria-label={`${label || "Line plot"}. Horizontal axis: ${xLabel || "x"}; vertical axis: ${yLabel || "y"}. Exact coordinates are available below.`}
      >
        <line x1={pad.l} y1={pad.t + plotHeight} x2={pad.l + plotWidth} y2={pad.t + plotHeight} stroke="#626b65" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + plotHeight} stroke="#626b65" />
        <text x={pad.l} y={pad.t + plotHeight + 20} style={tickStyle} textAnchor="start">{horizontalLabels[0]}</text>
        {xMin !== xMax && <text x={pad.l + plotWidth} y={pad.t + plotHeight + 20} style={tickStyle} textAnchor="end">{horizontalLabels[1]}</text>}
        <text x={pad.l - 8} y={pad.t + plotHeight + 4} style={tickStyle} textAnchor="end">{yTicks.labels[0]}</text>
        {yMin !== yMax && <text x={pad.l - 8} y={pad.t + 4} style={tickStyle} textAnchor="end">{yTicks.labels[1]}</text>}
        {series.map((item, index) => {
          const color = item.color || palette[index % palette.length];
          const path = item.points.map(([x, y], pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${sx(x)} ${sy(y)}`).join(" ");
          return (
            <g key={index}>
              <path d={path} stroke={color} fill="none" strokeWidth={1.5} />
              {item.points.map(([x, y], pointIndex) => (
                <circle key={pointIndex} cx={sx(x)} cy={sy(y)} r={2} fill={color} />
              ))}
            </g>
          );
        })}
      </svg>
      {xLabel && <div className="lesson-plot-axis-label">Horizontal axis: {xLabel}</div>}
      {(xTicks.needsExactRange || xTicksTooWide || yTicks.needsExactRange) && (
        <div className="lesson-plot-range">
          {(xTicks.needsExactRange || xTicksTooWide) && <div>Horizontal limits: min {xMin}; max {xMax}.</div>}
          {yTicks.needsExactRange && <div>Vertical limits: min {yMin}; max {yMax}.</div>}
        </div>
      )}
      <details className="lesson-plot-values" onToggle={(event) => { setShowValues(event.currentTarget.open); setValuesPage(0); }}>
        <summary>Exact plotted values{(xTicks.rounded || yTicks.rounded) && " (axis ticks rounded)"}</summary>
        {showValues && <>
          <table>
            <thead><tr><th scope="col">Series</th><th scope="col">{xLabel || "x"}</th><th scope="col">{yLabel || "y"}</th></tr></thead>
            <tbody>{visibleValues.map(({ key, name, x, y }) => <tr key={key}><th scope="row">{name}</th><td>{x}</td><td>{y}</td></tr>)}</tbody>
          </table>
          {pageCount > 1 && <div className="lesson-plot-value-pages">
            <button type="button" disabled={currentPage === 0} onClick={() => setValuesPage(currentPage - 1)}>Previous values</button>
            <span aria-live="polite">{firstValue + 1}–{lastValue} of {allPoints.length}</span>
            <button type="button" disabled={currentPage === pageCount - 1} onClick={() => setValuesPage(currentPage + 1)}>Next values</button>
          </div>}
        </>}
      </details>
    </figure>
  );
}
