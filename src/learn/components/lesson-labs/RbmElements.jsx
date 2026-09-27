import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { useEffect, useRef, useState } from 'react';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralTable, formatNeural as f } from './NeuralLessonElements.jsx';
export const rbmAssetBase = '/learn-code/boltzmann-machines-restricted-boltzmann-machines-rbm/';
export const stateName = v => v.join('');
export const vector = v => '[' + v.map(x => f(x, 5)).join(', ') + ']';
export function RbmParameters({
  model,
  onChange,
  compact = false
}) {
  const fields = [['w', 0, 'Interaction W₁'], ['w', 1, 'Interaction W₂'], ['a', 0, 'Visible bias a₁'], ['a', 1, 'Visible bias a₂'], ['b', 0, 'Hidden bias b']];
  return <fieldset><legend>Current three-switch model</legend><div className="neural-controls">{fields.filter(([kind]) => !compact || kind !== 'a').map(([kind, i, label]) => <NeuralNumber key={label} label={label} value={kind === 'w' ? model.w[i][0] : model[kind][i]} min={-6} max={6} range={false} onChange={value => onChange({
        ...model,
        [kind]: model[kind].map((x, j) => i === j ? kind === 'w' ? [value] : value : x)
      })} />)}</div></fieldset>;
}
export function RbmFigure({
  title,
  description,
  width = 420,
  height,
  children
}) {
  return <figure className="rbm-figure"><figcaption>{title}</figcaption><div className="rbm-diagram-scroll" tabIndex={0} role="region" aria-label={title}><svg width={width} style={{
        minWidth: width,
        width: '100%',
        height: 'auto'
      }} viewBox={'0 0 ' + width + ' ' + height} role="img" aria-label={description || title}>{children}</svg></div>{description && <p>{description}</p>}</figure>;
}
export function ProbabilityBars({
  title,
  rows,
  baselineLabel = 'the original model'
}) {
  return <figure className="rbm-probabilities"><figcaption>{title}</figcaption>{rows.map(({
      label,
      current,
      baseline
    }) => <div className="rbm-probability-row" key={label}><span>{label}</span><div className="rbm-probability-track"><i style={{
          width: current * 100 + '%'
        }} />{baseline !== undefined && <b style={{
          left: baseline * 100 + '%'
        }} />}</div><strong>{f(current, 6)}</strong></div>)}<p>Scale 0–1; amber length is probability. {rows.some(row => row.baseline !== undefined) && <>White marker: {baselineLabel}.</>}</p></figure>;
}
export function PixelTile({
  title,
  values,
  mask = null,
  selected = -1,
  onSelect = null,
  signed = false,
  extent = 1
}) {
  return <figure className="rbm-pixel-figure"><figcaption>{title}</figcaption><div className={'rbm-pixels ' + (onSelect ? 'rbm-pixels-interactive' : '')} role={onSelect ? 'group' : undefined} aria-label={title}>{values.map((value, i) => {
        const shade = signed ? Math.min(1, Math.abs(value) / extent) : value;
        const background = signed ? value < 0 ? 'rgba(155,187,224,' + shade + ')' : 'rgba(230,184,84,' + shade + ')' : 'rgb(' + [1, 2, 3].map(() => Math.round(25 + 230 * shade)).join(',') + ')';
        const label = 'Pixel ' + i + ', row ' + Math.floor(i / 8) + ', column ' + i % 8 + ': ' + f(value, 6) + (mask && !mask[i] ? ', missing' : '');
        const props = {
          style: {
            background
          },
          className: (mask && !mask[i] ? 'rbm-missing ' : '') + (i === selected ? 'rbm-selected' : ''),
          'aria-label': label,
          title: label
        };
        return onSelect ? <button key={i} {...props} onClick={() => onSelect(i)} aria-pressed={i === selected} /> : <span key={i} {...props} />;
      })}</div>{mask && <small>Hatching marks missing evidence, independently of pixel value.</small>}</figure>;
}
export function useRbmAsset(file) {
  const host = useRef(null),
    [visible, setVisible] = useState(false),
    [attempt, setAttempt] = useState(0),
    [state, setState] = useState({
      file: null,
      data: null,
      error: null
    });
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting)) {
        setVisible(true);
        observer.disconnect();
      }
    }, {
      rootMargin: '200px'
    });
    if (host.current) observer.observe(host.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible) return;
    const control = new AbortController();
    setState({
      file,
      data: null,
      error: null
    });
    fetch(rbmAssetBase + file, {
      signal: control.signal
    }).then(r => {
      if (!r.ok) throw Error('download');
      return r.json();
    }).then(data => {
      if (!control.signal.aborted) setState({
        file,
        data,
        error: null
      });
    }).catch(() => {
      if (!control.signal.aborted) setState({
        file,
        data: null,
        error: 'This saved RBM asset could not be loaded. Retry to request it again.'
      });
    });
    return () => control.abort();
  }, [file, visible, attempt]);
  return {
    host,
    data: state.file === file ? state.data : null,
    error: state.file === file ? state.error : null,
    retry: () => setAttempt(n => n + 1)
  };
}
export function RbmResource({
  resource,
  children
}) {
  return <div ref={resource.host}>{resource.error ? <p role="alert">{resource.error} <button onClick={resource.retry}>Retry model download</button></p> : resource.data ? children(resource.data) : <p role="status">Loading this saved RBM asset…</p>}</div>;
}
export function SwitchGraph({
  model,
  visible,
  hidden,
  general = false
}) {
  const positions = [[95, 180], [325, 180]],
    active = visible;
  return <RbmFigure title={general ? 'General BM: a direct visible interaction' : 'RBM: the hidden switch joins two visible switches'} width={420} height={245} description="Undirected edges are energy interactions. Selecting a switch changes the inspected state; editing an interaction changes the distribution.">
    {general ? <line x1="125" y1="180" x2="295" y2="180" stroke="#e6b854" strokeWidth={active.every(Boolean) ? 4 : 1.5} /> : positions.map(([x, y], i) => <g key={i}><line x1="210" y1="65" x2={x} y2={y - 25} stroke={active[i] && hidden ? '#e6b854' : '#888'} strokeWidth={active[i] && hidden ? 4 : 1.5} /><text x={(x + 210) / 2 + (i ? 20 : -20)} y="118" textAnchor="middle" fill="#eee" fontSize="13">W{i + 1}={f(model.w[i][0], 3)}</text></g>)}
    {!general && <g><circle cx="210" cy="45" r="25" fill={hidden ? '#e6b854' : '#222'} stroke="#ccc" /><text x="210" y="50" textAnchor="middle" fill={hidden ? '#111' : '#eee'} fontSize="16">h={hidden}</text></g>}
    {positions.map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r="30" fill={active[i] ? '#e6b854' : '#222'} stroke="#ddd" /><text x={x} y={y + 5} textAnchor="middle" fill={active[i] ? '#111' : '#eee'} fontSize="15">v{i + 1}={active[i]}</text></g>)}
    <text x="210" y="235" textAnchor="middle" fill="#ccc" fontSize="13">{general ? 'direct edge: J = ln 3' : 'No within-visible or within-hidden edge'}</text>
  </RbmFigure>;
}
export function RbmProgram({
  file
}) {
  return <>
    <RemoteCodeBlock source={rbmAssetBase + file} language="python" filename={file} title={"Read the complete " + (file) + " program"} />
    <p><a href={rbmAssetBase + file} download>Download {file}</a></p>
  </>;
}
export function RbmStateTable({
  distribution
}) {
  return <NeuralTable caption="Every joint event; marginalization adds matching visible rows" headers={['Visible', 'Hidden', 'Energy', 'Joint probability']} rows={distribution.joint.map(row => [stateName(row.v), row.h, f(row.energy, 6), f(row.probability, 8)])} />;
}
