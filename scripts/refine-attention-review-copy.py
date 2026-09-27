from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def replace(path, before, after):
    target = ROOT / path
    text = target.read_text(encoding='utf-8')
    assert before in text, (path, before)
    target.write_text(text.replace(before, after), encoding='utf-8', newline='\n')

rwkv = 'src/learn/components/lesson-labs/RwkvMechanismFigures.jsx'
replace(rwkv, '[false, true].map(store => <div className=', '[false, true].map(store => <div key={String(store)} className=')
replace(rwkv, '<svg key={String(store)} className=', '<svg className=')
replace(rwkv, 'a3by3local score triangle', 'a 3 by 3 local score triangle')
replace(rwkv, 'Actual45pointLibras validation trajectory', 'Actual 45 point Libras validation trajectory')
replace(rwkv, '45by2coordinates project to45by16, pass two memory blocks, mean to16features and project to15logits.', '45 by 2 coordinates project to 45 by 16, pass through two memory blocks, average to 16 features, and project to 15 logits.')
replace(rwkv, 'values 2,8,-1, summing to4.25', 'values 2, 8, and minus 1, summing to 4.25')

for name, css, scope in [('RwkvMemoryLabs', 'rwkv-memory', 'rwkv'), ('SelfAttentionLabs', 'self-attention-labs', 'self-attention')]:
    component = f'src/learn/components/lesson-labs/{name}.jsx'
    replace(component, 'NeuralPlot,', 'NeuralPlot as BaseNeuralPlot,')
    target = ROOT / component
    code = target.read_text(encoding='utf-8')
    # Topic-local wrapper preserves the shared component's behavior and earlier lessons.
    last_import = code.rfind('\nimport ')
    end = code.index('\n', last_import + 1)
    code = code[:end] + f'\nfunction NeuralPlot(props) {{\n  return <div className="{scope}-plot-scroll" role="region" aria-label={{`${{props.title}}; scroll horizontally if needed`}} tabIndex={{0}}><BaseNeuralPlot {{...props}} /></div>;\n}}\n' + code[end:]
    target.write_text(code, encoding='utf-8', newline='\n')
    with (ROOT / f'src/learn/components/lesson-labs/{css}.css').open('a', encoding='utf-8') as stream:
        stream.write(f'\n.{scope}-plot-scroll {{ min-width: 0; max-width: 100%; overflow-x: auto; }}\n.{scope}-plot-scroll > .neural-plot {{ min-width: 340px; }}\n')

lesson = 'docs/teaching/drafts/self-attention-multi-head-attention/lesson.md'
replace(lesson, "The horizontal axis is donor point number; the vertical axis is receiver point number.", "In the selected-row strip, donor point number runs horizontally and bar height is its weight. The receiver and head selectors choose which row is displayed.")
replace(lesson, "Reversing both axes of the original heatmap reproduces the reversed sequence's attention map up to rounding.", "The full attention matrix would permute both its receiver and donor axes under reversal, up to rounding. In the displayed row strip, selecting the same physical receiver in its new position reveals the corresponding reordered donor weights.")
spec = 'docs/teaching/drafts/self-attention-multi-head-attention/visual-specifications.md'
replace(spec, 'Initial task selects zero-based receiver 1 and asks the learner to mark legal donors for a next-token causal model. Start with blank/visible mask choices in the challenge; offer the already-explained causal pattern only as an optional closed hint.', 'The initial live view selects zero-based receiver 1 and the useful causal mask [True, True, False, False]. Arbitrary row checkboxes remain editable; a restore-causal action makes the explained pattern easy to recover.')
replace(spec, 'The current computed result is visible.: “Will the outputs remain equal after changing the head partition?”', 'Changing the head partition immediately displays the recomputed one-head and two-head outputs beside each other.')
replace(spec, 'The shape view lets the learner enter B,L,d and a valid divisor H (small diagram bounds B<=4,L<=8,d<=24), with clear validation for nondivisibility. Show formula counts and a count of actual displayed channels; use labeled brackets rather than hundreds of boxes. The computed counts update immediately when dimensions change.', 'The manuscript retains the static general shape table and the exact two-position channel comparison. The implemented storage calculator separately exposes sequence length and head count at fixed width; the construction lab exposes one or two heads at width two. No general B,L,d,H shape editor is claimed.')
replace(spec, 'Ask the learner to construct values that make outputs agree despite different distributions, with the The current computed result is visible.', 'The learner can edit values to make outputs agree despite different distributions while both computed mixtures remain visible.')
replace(spec, 'This is a genuine unsolved entity-editing task, not a single preset switch to an already-equal answer.', 'The independent values and weights remain directly editable, so the equality can be explored beyond a preset.')
replace(spec, 'Phase two verifies the three-donor contrast/equality construction, the two- Compute the comparison from the complete current inputs.', 'Implementation checks verify the three-donor contrast and equality, the two-donor equal-value control, and recomputation from complete current inputs.')
replace(spec, 'Reset clears all states.', 'Reset restores the original values and weights, with their computed outputs visible.')
