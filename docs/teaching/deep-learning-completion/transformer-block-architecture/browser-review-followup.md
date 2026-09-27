# Transformer rendered follow-up

Continuation of `browser-review-initial.md`, preserving the actual earlier controls and outputs.

On the18:53 production build, changing normalizer offset5 changes mean4to9, leaves variance7.5 fixed, and changes mean-square23.5to88.5. The new denominator view shows LayerNorm2.73861 unchanged and RMSNorm4.84768to9.40744. Offset0 is null for both; scale0 produces zero outputs and finite .00316 denominators. Centered vectors and denominator effects are now visible rather than hidden in prose.

The new feed-forward write workspace starts with input`[1,2,-1,-2]`, responses`[2,4,3]`, writes`[2,4,1.5,-1.5]`. Editing only the second position's first input from1to2 changes only its write to`[1,0,1,-1]`. Changing the shared down-projection row1 column1 to2 changes first write to`[4,4,1.5,-1.5]`and second to`[2,0,1,-1]`. The distinction between position independence and shared weights is therefore observable. SwiGLU default up responses`[2,4,3]`, gates`[1,-2,-1]`, output`[1.46212,-.95362,-.40341,.40341]`; gate matrix row1 column1 set0 zeros the first gate and first write. Reset works.

New static representations show communication masks, cross-attention checkpoint shape, checkpoint ownership and parallelism. At320px the page client/scroll width is305; nominal diagrams preserve720communication,300bypass,440write,290mask,600checkpoint and340plot widths inside local scroll regions. There are no KaTeX error nodes. The six-layer fitted-feature dump is now behind a disclosure, leaving the first-pass route readable.

Desktop finding: three encoder/decoder/encoder-decoder columns were~218px wide and clipped the290px diagrams. Author changed this to auto-fit min320px tracks, yielding at most two on the reader desktop and one when narrower. Independent reviewer assessed this CSS-only change. Final rendered verification follows on the refreshed build.
