import { useMemo, useState } from "react";
import { colors, fonts } from "../styles";
import TokenStream from "./viz/TokenStream";

const DEFAULT_CORPUS = "low low low low low lower lower newest newest newest newest newest newest widest widest widest";

function tokenizeCorpus(corpus) {
  const words = corpus.trim().split(/\s+/);
  const freq = {};
  for (const w of words) freq[w] = (freq[w] || 0) + 1;
  return freq;
}

function initSymbols(freq) {
  const result = {};
  for (const [w, count] of Object.entries(freq)) {
    result[w] = { symbols: [...w, "</w>"], count };
  }
  return result;
}

function countPairs(state) {
  const pairs = {};
  for (const { symbols, count } of Object.values(state)) {
    for (let i = 0; i < symbols.length - 1; i++) {
      const key = `${symbols[i]} ${symbols[i + 1]}`;
      pairs[key] = (pairs[key] || 0) + count;
    }
  }
  return pairs;
}

function bestPair(pairs) {
  let best = null, bestCount = 0;
  for (const [k, v] of Object.entries(pairs)) {
    if (v > bestCount) { best = k; bestCount = v; }
  }
  return best ? { pair: best.split(" "), count: bestCount } : null;
}

function mergePair(state, [a, b]) {
  const merged = a + b;
  const next = {};
  for (const [w, { symbols, count }] of Object.entries(state)) {
    const out = [];
    let i = 0;
    while (i < symbols.length) {
      if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
        out.push(merged);
        i += 2;
      } else {
        out.push(symbols[i]);
        i += 1;
      }
    }
    next[w] = { symbols: out, count };
  }
  return { next, merged };
}

export default function BPETrainer() {
  const [corpus, setCorpus] = useState(DEFAULT_CORPUS);
  const [state, setState] = useState(() => initSymbols(tokenizeCorpus(DEFAULT_CORPUS)));
  const [merges, setMerges] = useState([]);

  const vocab = useMemo(() => {
    const s = new Set();
    for (const { symbols } of Object.values(state)) symbols.forEach((x) => s.add(x));
    return Array.from(s).sort();
  }, [state]);

  const pairs = useMemo(() => countPairs(state), [state]);
  const nextBest = bestPair(pairs);

  function doStep() {
    if (!nextBest) return;
    const { next, merged } = mergePair(state, nextBest.pair);
    setState(next);
    setMerges((m) => [...m, { pair: nextBest.pair, merged, count: nextBest.count }]);
  }

  function reset(newCorpus = corpus) {
    setState(initSymbols(tokenizeCorpus(newCorpus)));
    setMerges([]);
  }

  return (
    <div style={{
      border: `1px solid ${colors.border}`,
      borderRadius: 6,
      padding: 16,
      margin: "20px 0",
      background: colors.cardBg,
    }}>
      <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginBottom: 10, letterSpacing: 1 }}>
        INTERACTIVE · BPE TRAINER
      </div>

      <label style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textMuted, display: "block", marginBottom: 4 }}>
        Corpus
      </label>
      <textarea
        value={corpus}
        onChange={(e) => setCorpus(e.target.value)}
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

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button onClick={doStep} disabled={!nextBest} style={btn(!nextBest ? colors.textDark : colors.gold)}>
          Train Step {nextBest ? `(${nextBest.pair.join(" + ")})` : ""}
        </button>
        <button onClick={() => reset(corpus)} style={btn(colors.textMuted)}>Reset</button>
      </div>

      <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textMuted, marginBottom: 6 }}>
        Vocabulary ({vocab.length} tokens)
      </div>
      <TokenStream tokens={vocab} />

      <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textMuted, marginTop: 14, marginBottom: 6 }}>
        Merges so far ({merges.length})
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {merges.length === 0 && (
          <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textDim }}>No merges yet.</div>
        )}
        {merges.map((m, i) => (
          <div key={i} style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textSecondary }}>
            {String(i + 1).padStart(2, "0")}. <span style={{ color: colors.gold }}>{m.pair.join(" + ")}</span> → <span style={{ color: colors.green }}>{m.merged}</span> (count {m.count})
          </div>
        ))}
      </div>

      <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textMuted, marginTop: 14, marginBottom: 6 }}>
        Current word segmentations
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {Object.entries(state).map(([w, { symbols, count }]) => (
          <div key={w} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textMuted, minWidth: 80 }}>
              {w} <span style={{ color: colors.textDark }}>(×{count})</span>
            </span>
            <TokenStream tokens={symbols} />
          </div>
        ))}
      </div>
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
