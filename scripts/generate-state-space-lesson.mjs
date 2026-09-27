// Rebuild the current teaching revision; historical manuscripts remain intact.
import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'state-space-models-s4-mamba-mamba-2';
let manuscript = fs.readFileSync(`docs/teaching/concept-intuition/${id}/lesson.md`, 'utf8').replaceAll('\r\n', '\n');
manuscript = manuscript.replace(/^\* /gm, '- ');
manuscript = manuscript.replace(/<details><summary>(.*?)<\/summary>(.*?)<\/details>/gs, (_, title, content) => `<details>\n<summary>${title}</summary>\n\n${content.trim()}\n\n</details>`);
const equations = [
  ['hₜ = 0.8 hₜ₋₁ + 0.2 uₜ.', 'h_t=0.8h_{t-1}+0.2u_t.'],
  ['hₜ = Āhₜ₋₁ + B̄uₜ,  \nyₜ = Chₜ + Duₜ.', 'h_t=\\bar A h_{t-1}+\\bar B u_t,\\qquad y_t=Ch_t+Du_t.'],
  ['dh(t)/dt = Ah(t) + Bu(t),  \ny(t) = Ch(t) + Du(t).', '\\frac{dh(t)}{dt}=Ah(t)+Bu(t),\\qquad y(t)=Ch(t)+Du(t).'],
  ['Ā = exp(ΔA),  \nB̄ = integral from 0 to Δ of exp(sA)B ds.', '\\bar A=e^{\\Delta A},\\qquad \\bar B=\\int_0^\\Delta e^{sA}B\\,ds.'],
  ['Ā=e^(−Δ),  B̄=1−e^(−Δ).', '\\bar A=e^{-\\Delta},\\qquad \\bar B=1-e^{-\\Delta}.'],
  ['dh/dt=u,  so hₜ=hₜ₋₁+Δuₜ.', '\\frac{dh}{dt}=u\\quad\\Longrightarrow\\quad h_t=h_{t-1}+\\Delta u_t.'],
  ['exp(Δ [[A,B],[0,0]]) = [[Ā,B̄],[0,I]].', '\\exp\\left(\\Delta\\begin{bmatrix}A&B\\\\0&0\\end{bmatrix}\\right)=\\begin{bmatrix}\\bar A&\\bar B\\\\0&I\\end{bmatrix}.'],
  ['Ā = (I−ΔA/2)⁻¹(I+ΔA/2),  \nB̄ = (I−ΔA/2)⁻¹ ΔB.', '\\bar A=(I-\\Delta A/2)^{-1}(I+\\Delta A/2),\\quad \\bar B=(I-\\Delta A/2)^{-1}\\Delta B.'],
  ['Kₗ = C Āˡ B̄,  for l=0,1,2,...', 'K_\\ell=C\\bar A^\\ell\\bar B,\\qquad \\ell=0,1,2,\\ldots'],
  ['yₜ = sum over j=0…t of Kₜ₋ⱼ uⱼ + Duₜ.', 'y_t=\\sum_{j=0}^t K_{t-j}u_j+Du_t.'],
  ['y(t)=C exp(At)h(0) + integral from 0 to t of C exp(A(t−s))B u(s) ds + Du(t).', 'y(t)=Ce^{At}h(0)+\\int_0^t Ce^{A(t-s)}Bu(s)\\,ds+Du(t).'],
  ['A = [[−.2,−2],[2,−.2]].', 'A=\\begin{bmatrix}-.2&-2\\\\2&-.2\\end{bmatrix}.'],
  ['2 Re(cₙ hₙ)', '2\\operatorname{Re}(c_nh_n)'],
  ['dc/dt = −A₊c/t + B₊f(t)/t,', '\\frac{dc}{dt}=-\\frac{A_+c}{t}+\\frac{B_+f(t)}{t},'],
  ['K_T(z)=C {I−(zĀ)^T} (I−zĀ)⁻¹ B̄', 'K_T(z)=C\\{I-(z\\bar A)^T\\}(I-z\\bar A)^{-1}\\bar B'],
  ['(sI−A)⁻¹ = R₀−R₀p(1+q*R₀p)⁻¹q*R₀.', '(sI-A)^{-1}=R_0-R_0p(1+q^*R_0p)^{-1}q^*R_0.'],
  ['hₜ=(1−gₜ)hₜ₋₁+gₜuₜ,  0≤gₜ≤1.', 'h_t=(1-g_t)h_{t-1}+g_tu_t,\\qquad 0\\le g_t\\le1.'],
  ['exp(−Δₜ)=1−sigmoid(zₜ),  \nB̄ₜ=1−exp(−Δₜ)=sigmoid(zₜ).', 'e^{-\\Delta_t}=1-\\sigma(z_t),\\quad \\bar B_t=1-e^{-\\Delta_t}=\\sigma(z_t).'],
  ['hₜ,d,n = exp(Δₜ,d A_d,n) hₜ₋₁,d,n + Δₜ,d Bₜ,n uₜ,d,  \nyₜ,d = sum over n of Cₜ,n hₜ,d,n + D_d uₜ,d.', 'h_{t,d,n}=e^{\\Delta_{t,d}A_{d,n}}h_{t-1,d,n}+\\Delta_{t,d}B_{t,n}u_{t,d},\\qquad y_{t,d}=\\sum_n C_{t,n}h_{t,d,n}+D_du_{t,d}.'],
  ['(a₂,b₂) after (a₁,b₁) = (a₂⊙a₁, a₂⊙b₁+b₂).', '(a_2,b_2)\\circ(a_1,b_1)=(a_2\\odot a_1,\\ a_2\\odot b_1+b_2).'],
  ['Sₜ = aₜ Sₜ₋₁ + bₜvₜᵀ,  \nyₜ = cₜᵀSₜ.', 'S_t=a_tS_{t-1}+b_tv_t^\\top,\\qquad y_t=c_t^\\top S_t.'],
  ['yᵢ = sum over j≤i of (cᵢᵀbⱼ) Lᵢⱼ vⱼ,', 'y_i=\\sum_{j\\le i}(c_i^\\top b_j)L_{ij}v_j,'],
  ['Y = ((CBᵀ) ⊙ L)V.', 'Y=((CB^\\top)\\odot L)V.'],
  ['z ← z + W_out GELU(TemporalMixer(LayerNorm(z))).', 'z\\leftarrow z+\\operatorname{Affine}_{out}\\left(\\operatorname{GELU}\\left(\\operatorname{Mixer}(\\operatorname{LayerNorm}(z))\\right)\\right).'],
  ['hₜ=αₜhₜ₋₁+βₜBₜ₋₁xₜ₋₁+γₜBₜxₜ,', 'h_t=\\alpha_t h_{t-1}+\\beta_tB_{t-1}x_{t-1}+\\gamma_tB_tx_t,'],
  ['R(θ)=[[cos θ,−sin θ],[sin θ,cos θ]].', 'R(\\theta)=\\begin{bmatrix}\\cos\\theta&-\\sin\\theta\\\\\\sin\\theta&\\cos\\theta\\end{bmatrix}.'],
];
for (const [plain, latex] of equations) {
  if (!manuscript.includes(plain)) throw new Error(`Missing equation ${plain}`);
  manuscript = manuscript.replace(plain, `\\[\n${latex}\n\\]`);
}
const rendered = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  assetBase: `/learn-code/${id}/`,
  replacements: [
    ['**Inline figure: retain and write', '<RetainWriteFigure />'],
    ['**Inline figure: overlapping impulse trails', '<ImpulseTrailsFigure />'],
    ['**Inline figure: several memory clocks', '<MemoryRatesFigure />'],
    ['**Inline figure: marked event versus fixed age', '<MarkedMemoryFigure />'],
    ['**Inline figure: one matrix memory update', '<MatrixWriteFigure />'],
    ['**Inline figure: four paths', '<StatePathsFigure />'],
    ['**Inline figure: continuous decay', '<SamplingFigure />'],
    ['**Inline figure: an impulse ledger.', '<ImpulseFigure />'],
    ['**Inline figure: a shrinking spiral', '<OscillatorFigure />'],
    ['**Inline figure: history, basis', '<PolynomialFigure />'],
    ['**Inline figure: annotated write', '<StateSpaceSelectionLab />'],
    ['**Inline figure: a shape-aware', '<MambaBlockFigure />'],
    ['**Inline figure: signed influence', '<SSDFigure />'],
    ['**Inline figure: a block matrix', '<Prose>The exact local and incoming contributions above form the four-stage computation. The workshop below lets you alter chunk boundaries independently from its coefficients.</Prose>'],
    ['**Inline figure: three recorded paths', '<RealTrajectoriesFigure />'],
    ['**Inline figure: from one measured path', '<TrainingPipelineFigure />'],
    ['**Inline figure: endpoint interpolation', '<MambaThreeFigure />'],
  ],
  additions: [
    ['The comparison must include', '<StateSpaceSystemLab />'],
    ['This is **diagonal plus low rank**', '<DplrFigure />'],
    ['An LTI operator can copy', '<FixedDelayFigure />'],
    ['Setting a₂=0 erases', '<StateSpaceSSDLab />'],
    ['The saved result file includes', '<LearningEvidenceFigure />'],
    ['The per-step mixer is causal', '<StateSpaceTrajectoryLab />'],
    ['A smaller retained state', '<CacheCountsFigure />'],
    ['It writes mechanism-results.json', '<StateSpaceProgram file="state_space_mechanisms.py" title="Read the complete recurrence, discretization, convolution and SSD program" />'],
    ['The second part of the program', '<StateSpaceProgram file="state_space_library_bridge.py" title="Read the ordinary Mamba scan and complete-block training route" />'],
  ],
});
// Its paragraph begins with the surrounding file instructions, so place the full program before the selective excerpt.
rendered.jsx = rendered.jsx.replace('<Prose>{"The core selective step in that complete file is:"}</Prose>', '<StateSpaceProgram file="trajectory_state_models.py" title="Read the complete trainable diagonal and selective classifiers" />\n<Prose>{"The core selective step in that complete file is:"}</Prose>');
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Complete prepared manuscript, with implemented figures and live investigations.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { StateSpaceSystemLab, StateSpaceSelectionLab, StateSpaceSSDLab, StateSpaceTrajectoryLab, StateSpaceProgram } from '../../components/lesson-labs/StateSpaceLabs.jsx';
import { StatePathsFigure, SamplingFigure, ImpulseFigure, MemoryRatesFigure, OscillatorFigure, PolynomialFigure, DplrFigure, FixedDelayFigure, MambaBlockFigure, SSDFigure, RealTrajectoriesFigure, TrainingPipelineFigure, LearningEvidenceFigure, CacheCountsFigure, MambaThreeFigure } from '../../components/lesson-labs/StateSpaceFigures.jsx';
import { RetainWriteFigure, ImpulseTrailsFigure, MarkedMemoryFigure, MatrixWriteFigure } from '../../components/lesson-labs/StateSpaceIntuition.jsx';
export default {
  title: 'State Space Models: S4 and the Mamba Family',
  readTime: '~95 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson state-space-lesson"><LessonIntro prerequisites="Basic recurrent updates and matrix products. Sampling, complex modes and the attention connection are explained locally." sections={${JSON.stringify(rendered.sections)}}>How can a few changing numbers preserve useful information from a long stream? Build the memory step by step, then train it on real movement.</LessonIntro>
${rendered.jsx}
  </div>,
};
`);
console.log(`State-space active manuscript: ${rendered.sections.length} sections; complete mechanisms and retained investigations.`);
