import { useId, useMemo, useState } from "react";
import { colors, fonts } from "../styles";
import TokenStream from "./viz/TokenStream";
import { BPE_MERGE_LIMIT, DEFAULT_BPE_CORPUS, learnedBpeVocabulary, mergeBpePair, nextBpePair, prepareBpeCorpus } from "../data/bpe-trainer-model.js";

export default function BPETrainer() {
  const corpusId = useId();
  const [corpus, setCorpus] = useState(DEFAULT_BPE_CORPUS);
  const prepared = useMemo(() => prepareBpeCorpus(corpus), [corpus]);
  const [state, setState] = useState(() => prepareBpeCorpus(DEFAULT_BPE_CORPUS).state);
  const [merges, setMerges] = useState([]);
  const vocab = useMemo(() => learnedBpeVocabulary(prepared.alphabet || [], merges), [prepared, merges]);
  const nextBest = useMemo(() => nextBpePair(state), [state]);
  const canStep = !prepared.error && nextBest && merges.length < BPE_MERGE_LIMIT;

  function doStep() {
    if (!canStep) return;
    const { next, merged } = mergeBpePair(state, nextBest.pair);
    setState(next);
    setMerges(previous => [...previous, { pair: nextBest.pair, merged, count: nextBest.count }]);
  }

  function reset(newCorpus = corpus) {
    setCorpus(newCorpus);
    setState(prepareBpeCorpus(newCorpus).state || {});
    setMerges([]);
  }

  return (
    <div data-bpe-trainer="true" style={{
      border: `1px solid ${colors.border}`,
      borderRadius: 6,
      padding: 16,
      margin: "20px 0",
      background: colors.cardBg,
    }}>
      <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textSecondary, marginBottom: 10, letterSpacing: 1 }}>
        INTERACTIVE · BPE TRAINER
      </div>

      <p id={`${corpusId}-help`} style={{ color: colors.textSecondary, fontSize: 13, lineHeight: 1.6, marginBottom: 12 }}>
        Edit the corpus to restart its segmentation immediately, then merge one pair at a time.
        This character-level teaching model splits words on whitespace and appends a word-end marker.
        Frequency ties follow the displayed word order. Each step merges non-overlapping occurrences from left to right.
      </p>
      <label htmlFor={corpusId} style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textSecondary, display: "block", marginBottom: 4 }}>
        Corpus
      </label>
      <textarea
        id={corpusId}
        aria-describedby={`${corpusId}-help ${corpusId}-status`}
        aria-invalid={Boolean(prepared.error)}
        value={corpus}
        onChange={(e) => reset(e.target.value)}
        rows={2}
        style={{
          width: "100%",
          fontFamily: fonts.mono,
          fontSize: 12,
          background: "rgba(0,0,0,0.4)",
          color: colors.textSecondary,
          border: `1px solid ${colors.border}`,
          borderRadius: 4,
          padding: 8,
          marginBottom: 10,
          resize: "vertical",
        }}
      />

      <p id={`${corpusId}-status`} role="status" style={{ fontSize: 13, color: colors.textSecondary, marginBottom: 12 }}>
        {prepared.error || (merges.length >= BPE_MERGE_LIMIT ? 'Reached the 64-step teaching limit. Restart merges or edit the corpus.' : !nextBest ? 'Every word is a single symbol. No adjacent pair remains.' : `Next pair occurs ${nextBest.count} times, counting word frequency.`)}
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
        <button onClick={doStep} disabled={!canStep} style={btn(!canStep ? colors.textDark : colors.gold)}>
          Train Step {nextBest ? `(${nextBest.pair.join(" + ")})` : ""}
        </button>
        <button onClick={() => reset(corpus)} disabled={Boolean(prepared.error)} style={btn(colors.gold)}>Restart merges</button>
        <button onClick={() => reset(DEFAULT_BPE_CORPUS)} style={btn(colors.gold)}>Restore example</button>
      </div>

      {!prepared.error && <>
      <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textSecondary, marginBottom: 6 }}>
        Learned vocabulary ({vocab.length} tokens): original symbols plus all learned merges
      </div>
      <TokenStream tokens={vocab} />

      <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textSecondary, marginTop: 14, marginBottom: 6 }}>
        Merges so far ({merges.length})
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {merges.length === 0 && (
          <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textSecondary }}>No merges yet.</div>
        )}
        {merges.map((m, i) => (
          <div key={i} style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textSecondary }}>
            {String(i + 1).padStart(2, "0")}. <span style={{ color: colors.gold }}>{m.pair.join(" + ")}</span> → <span style={{ color: colors.green }}>{m.merged}</span> (count {m.count})
          </div>
        ))}
      </div>

      <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textSecondary, marginTop: 14, marginBottom: 6 }}>
        Current word segmentations
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {Object.entries(state).map(([w, { symbols, count }]) => (
          <div key={w} style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, minWidth: 0 }}>
            <span style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textSecondary, minWidth: 80, overflowWrap: "anywhere", maxWidth: "100%" }}>
              {w} <span style={{ color: colors.textSecondary }}>(×{count})</span>
            </span>
            <TokenStream tokens={symbols} />
          </div>
        ))}
      </div>
      </>}
    </div>
  );
}

function btn(color) {
  return {
    fontFamily: fonts.mono,
    fontSize: 11,
    padding: "6px 12px",
    background: "transparent",
    color,
    border: `1px solid ${color}55`,
    borderRadius: 3,
    cursor: color === colors.textDark ? "default" : "pointer",
  };
}
