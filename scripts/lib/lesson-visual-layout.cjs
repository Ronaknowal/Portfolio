// Browser-side geometry triage. Findings require interpretation: a grid crossing
// a label or a label deliberately inside a node need not be a layout defect.
function inspectLessonVisualLayout(rootSelector = 'main.reader-content') {
  const root = document.querySelector(rootSelector);
  if (!root) throw new Error('Lesson content is not mounted');
  const visible = element => {
    const box = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return box.width > 0 && box.height > 0 && style.visibility !== 'hidden' && style.display !== 'none' && Number(style.opacity) !== 0;
  };
  const boxOf = element => {
    const box = element.getBBox(), matrix = element.getScreenCTM();
    const points = [[box.x, box.y], [box.x + box.width, box.y], [box.x, box.y + box.height], [box.x + box.width, box.y + box.height]]
      .map(([x, y]) => new DOMPoint(x, y).matrixTransform(matrix));
    return { left: Math.min(...points.map(p => p.x)), right: Math.max(...points.map(p => p.x)), top: Math.min(...points.map(p => p.y)), bottom: Math.max(...points.map(p => p.y)) };
  };
  const overlap = (a, b, allowance = 2) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > allowance && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > allowance;
  const segmentEnters = (a, b, box) => {
    // Clip a line segment to the inner glyph rectangle, allowing a 2px gap.
    const left = box.left + 2, right = box.right - 2, top = box.top + 2, bottom = box.bottom - 2;
    let low = 0, high = 1;
    const dx = b.x - a.x, dy = b.y - a.y;
    for (const [p, q] of [[-dx, a.x - left], [dx, right - a.x], [-dy, a.y - top], [dy, bottom - a.y]]) {
      if (Math.abs(p) < 1e-9) { if (q < 0) return false; }
      else if (p < 0) low = Math.max(low, q / p);
      else high = Math.min(high, q / p);
      if (low > high) return false;
    }
    return true;
  };
  return [...root.querySelectorAll('svg')].flatMap((svg, svgIndex) => {
    if (!visible(svg)) return [];
    const labels = [...svg.querySelectorAll('text')].filter(visible).map(element => ({
      element, text: element.textContent.trim().slice(0, 140), box: boxOf(element),
    })).filter(label => label.text);
    if (!labels.length) return [];
    const viewport = svg.getBoundingClientRect(), issues = [];
    for (let index = 0; index < labels.length; index += 1) {
      const label = labels[index];
      if (label.box.left < viewport.left - 2 || label.box.right > viewport.right + 2 || label.box.top < viewport.top - 2 || label.box.bottom > viewport.bottom + 2) {
        issues.push({ type: 'label-outside-svg', label: label.text });
      }
      for (const other of labels.slice(index + 1)) {
        if (overlap(label.box, other.box)) issues.push({ type: 'overlapping-labels', labels: [label.text, other.text] });
      }
      for (const line of svg.querySelectorAll('line')) {
        // Grid/axis crossings carry no foreground meaning. Stems, edges and
        // arrows do: flag those for inspection, without changing the graphic.
        if (/grid|axis|guide/i.test(line.getAttribute('class') || '') || line.closest('g[class*="grid"], g[class*="axis"], g[class*="guide"]')) continue;
        const style = getComputedStyle(line);
        if (style.stroke === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) continue;
        const matrix = line.getScreenCTM();
        const a = new DOMPoint(line.x1.baseVal.value, line.y1.baseVal.value).matrixTransform(matrix);
        const b = new DOMPoint(line.x2.baseVal.value, line.y2.baseVal.value).matrixTransform(matrix);
        if (segmentEnters(a, b, label.box)) issues.push({ type: 'line-through-label', label: label.text, lineClass: line.getAttribute('class') || '' });
      }
    }
    const matrix = svg.getScreenCTM();
    const scaleX = Math.hypot(matrix.a, matrix.b), scaleY = Math.hypot(matrix.c, matrix.d);
    if (Math.abs(scaleX / scaleY - 1) > .02) issues.push({ type: 'unequal-svg-axis-scaling', ratio: scaleX / scaleY });
    return [{ svgIndex, description: (svg.getAttribute('aria-label') || svg.closest('figure')?.querySelector('figcaption')?.textContent || '').trim().slice(0, 180), labelCount: labels.length, issues }];
  });
}

module.exports = { inspectLessonVisualLayout };
