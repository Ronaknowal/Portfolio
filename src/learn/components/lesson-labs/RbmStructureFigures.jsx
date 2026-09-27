import { useId } from 'react';
import { RbmFigure } from './RbmElements.jsx';
export function RbmFamilyFigure() {
  const id = useId().replaceAll(':', '');
  return <div className="rbm-two">{['RBM', 'Deep belief network', 'Deep Boltzmann machine'].map((name, kind) => <RbmFigure key={name} title={name} width={310} height={300} description={kind === 0 ? 'One undirected visible/hidden pair.' : kind === 1 ? 'Only the top pair is undirected. Lower arrows specify conditional generative directions.' : 'All adjacent-layer interactions are undirected; hidden inference no longer separates like one RBM.'}>
    <defs><marker id={id + kind} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#e6b854" /></marker></defs>
    {(kind === 0 ? [0] : [0, 1]).flatMap(layer => (layer === 0 ? [105, 205] : [75, 155, 235]).flatMap(from => [75, 155, 235].map(to => <line key={layer + '-' + from + '-' + to} x1={from} y1={50 + layer * 95 + 14} x2={to} y2={145 + layer * 95 - 14} stroke="#e6b854" opacity=".8" markerEnd={kind === 1 && layer === 1 ? 'url(#' + id + kind + ')' : undefined} />)))}
    {(kind === 0 ? [0, 1] : [0, 1, 2]).map(layer => <g key={layer}>{(layer === 0 ? [105, 205] : [75, 155, 235]).map(x => <circle key={x} cx={x} cy={50 + layer * 95} r="14" fill={layer === (kind === 0 ? 1 : 2) ? '#777' : '#242424'} stroke="#ddd" />)}<text x="155" y={26 + layer * 95} textAnchor="middle" fill="#eee" fontSize="13">{layer === (kind === 0 ? 1 : 2) ? 'visible variables' : 'hidden layer ' + (kind === 0 ? 1 : 2 - layer)}</text></g>)}
  </RbmFigure>)}</div>;
}
export function RbmAisFigure() {
  return <RbmFigure title="AIS bridges two distributions through intermediate ones" width={600} height={250} description="This is a method schematic, not a measured estimate. Each step contributes a ratio of unnormalized densities at the current state, then applies a transition preserving the new intermediate distribution. Average complete importance weights to estimate the normalizer ratio.">
    {[0, 1, 2, 3].map(i => <g key={i}><rect x={10 + i * 150} y="42" width="132" height="54" rx="4" fill="#222" stroke="#e6b854" /><text x={76 + i * 150} y="65" textAnchor="middle" fill="#eee" fontSize="14">{['tractable base', 'bridge 1', 'bridge 2', 'target RBM'][i]}</text><text x={76 + i * 150} y="85" textAnchor="middle" fill="#ccc" fontSize="12">{i === 0 ? 'known Z₀' : i === 3 ? 'unknown Z₃' : 'intermediate density'}</text>{i < 3 && <><path d={'M' + (142 + i * 150) + ' 69 H' + (158 + i * 150)} stroke="#e6b854" /><text x={150 + i * 150} y="116" textAnchor="middle" fill="#eee" fontSize="12">× ratio</text></>}{i > 0 && <><path d={'M' + (40 + i * 150) + ' 99 V156 H' + (112 + i * 150) + ' V99'} fill="none" stroke="#aaa" /><text x={76 + i * 150} y="180" textAnchor="middle" fill="#ccc" fontSize="12">invariant transition</text></>}</g>)}
    <text x="300" y="220" textAnchor="middle" fill="#eee" fontSize="14">A ratio estimate and its logarithm have different bias properties.</text>
  </RbmFigure>;
}
