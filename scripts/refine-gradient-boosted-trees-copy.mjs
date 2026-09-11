import fs from 'node:fs';
import assert from 'node:assert/strict';

const path = 'src/learn/data/topics/gradient-boosted-trees-xgboost-lightgbm-catboost.jsx';
let source = fs.readFileSync(path, 'utf8');
const replacements = [
  [String.raw`\sum_i(r_i-c)^2=\sum_i(r_i-\bar r)^2+n(c-\bar r)^2.`, String.raw`\begin{aligned}\sum_i(r_i-c)^2={}&\sum_i(r_i-\bar r)^2\\&+n(c-\bar r)^2.\end{aligned}`],
  [String.raw`\begin{gathered}r_i^{(m)}=-\left.\frac{\partial L(y_i,F)}{\partial F}\right|_{F=F_{m-1}(x_i)},\\\rho_m\in\arg\min_\rho\sum_i L\!\left(y_i,F_{m-1}(x_i)+\rho h_m(x_i)\right).\end{gathered}`, String.raw`\begin{gathered}f_i=F_{m-1}(x_i),\quad h_i=h_m(x_i),\\r_i^{(m)}=-\left.\frac{\partial L(y_i,F)}{\partial F}\right|_{F=f_i},\\\rho_m\in\arg\min_\rho\sum_i L(y_i,f_i+\rho h_i).\end{gathered}`],
  [String.raw`L(y_i,F_i+w)\approx L(y_i,F_i)+g_iw+\tfrac12h_iw^2.`, String.raw`\begin{aligned}L(y_i,F_i+w)\approx{}&L(y_i,F_i)\\&+g_iw+\tfrac12h_iw^2.\end{aligned}`],
  [String.raw`\widetilde J=\sum_{\ell=1}^{T}\left[G_\ell w_\ell+\tfrac12(H_\ell+\lambda)w_\ell^2+\alpha|w_\ell|\right]+\gamma T.`, String.raw`\begin{gathered}\widetilde J=\sum_{\ell=1}^{T}C_\ell+\gamma T,\\C_\ell=G_\ell w_\ell+\tfrac12(H_\ell+\lambda)w_\ell^2\\+\alpha|w_\ell|.\end{gathered}`],
  [String.raw`\begin{gathered}Gw+\tfrac12(H+\lambda)w^2\\=\tfrac12(H+\lambda)\left(w+\frac{G}{H+\lambda}\right)^2-\frac{G^2}{2(H+\lambda)}.\end{gathered}`, String.raw`\begin{aligned}Gw+\tfrac12(H+\lambda)w^2\\={}&\tfrac12(H+\lambda)\left(w+\frac{G}{H+\lambda}\right)^2\\&-\frac{G^2}{2(H+\lambda)}.\end{aligned}`],
  [String.raw`w^*=-\frac{S_\alpha(G)}{H+\lambda},\qquad Q(G,H)=\frac{S_\alpha(G)^2}{2(H+\lambda)}.`, String.raw`\begin{gathered}w^*=-\frac{S_\alpha(G)}{H+\lambda},\\Q(G,H)=\frac{S_\alpha(G)^2}{2(H+\lambda)}.\end{gathered}`],
  [String.raw`\begin{aligned}\operatorname{Gain}_{\mathrm{net}}={}&Q(G_L,H_L)+Q(G_R,H_R)\\&-Q(G_L+G_R,H_L+H_R)-\gamma.\end{aligned}`, String.raw`\begin{gathered}Q_L=Q(G_L,H_L),\quad Q_R=Q(G_R,H_R),\\Q_P=Q(G_L+G_R,H_L+H_R),\\\operatorname{Gain}_{\mathrm{net}}=Q_L+Q_R-Q_P-\gamma.\end{gathered}`],
  [String.raw`\text{illustrative dense work}\;\sim\;\text{binning setup}+\sum_{m=1}^{M}\sum_{v\text{ visited}}O(dn_v+dB).`, String.raw`\begin{gathered}\text{Illustrative dense work:}\\\text{binning setup}\\+\sum_{m=1}^{M}\sum_{v\text{ visited}}O(dn_v+dB).\end{gathered}`],
  [String.raw`\widehat G=\sum_{i\in A}g_i+\frac{r}{s}\sum_{i\in B}g_i,\qquad\mathbb E[\widehat G\mid g]=G.`, String.raw`\begin{gathered}\widehat G=\sum_{i\in A}g_i+\frac{r}{s}\sum_{i\in B}g_i,\\\mathbb E[\widehat G\mid g]=G.\end{gathered}`],
  ['Predictions become .5,.5,3.5,3.5; MSE is 2.25', 'Predictions become 1.5,1.5,2.5,2.5; MSE is 2.25'],
];
for (const [before, after] of replacements) {
  assert.equal(source.split(before).length, 2, before);
  source = source.replace(before, after);
}
// These explicit typography edits apply only to prose, not identifiers or programs.
for (const [before, after] of [
  ['XGBoost3.4.1', 'XGBoost 3.4.1'], ['LightGBM4.7.0', 'LightGBM 4.7.0'],
  ['CatBoost1.2.10', 'CatBoost 1.2.10'], ['scikit-learn1.9.1', 'scikit-learn 1.9.1'],
  ['NumPy2.3.5', 'NumPy 2.3.5'], ['noise20', 'noise 20'], ['seed42', 'seed 42'],
  ['September2026', 'September 2026'], ['checked11September 2026', 'checked 11 September 2026'],
  ['Feature0', 'Feature 0'], ['count235', 'count 235'], ['gain3083143', 'gain 3083143'],
  ['count157', 'count 157'], ['entry156', 'entry 156'], ['is197', 'is 197'], ['contains198', 'contains 198'],
  ['has198', 'has 198'], ['stores200', 'stores 200'], ['all200', 'all 200'], ['RMSE33.', 'RMSE 33.'], ['baseline126.', 'baseline 126.'],
  ['a100×100', 'a 100×100'], ['accuracy.5012', 'accuracy .5012'], ['of.99', 'of .99'], ['A.5', 'A .5'],
  ['score−4', 'score −4'], ['lambda0', 'lambda 0'], ['alpha0', 'alpha 0'], ['rate1', 'rate 1'], ['rate.1', 'rate .1'],
  ['keep20%', 'keep 20%'], ['sample10%', 'sample 10%'], ['is.1', 'is .1'], ['is10', 'is 10'], ['value8', 'value 8'],
  ['were10%', 'were 10%'], ['is12.5%', 'is 12.5%'], ['remaining80%', 'remaining 80%'], ['stores18', 'stores 18'],
  ['index12', 'index 12'], ['only12', 'only 12'], ['weight9', 'weight 9'], ['not.1', 'not .1'],
  ['from20 to50', 'from 20 to 50'], ['depth3', 'depth 3'], ['strength1', 'strength 1'], ['is1.', 'is 1.'],
  ['rounds1–5', 'rounds 1–5'], ['is3 with', 'is 3 with'], ['round5', 'round 5'], ['Round2', 'Round 2'], ['Rounds3,4,5', 'Rounds 3, 4, 5'],
  ['section4', 'section 4'], ['section7', 'section 7'], ['sections2–4', 'sections 2–4'], ['Algorithm2', 'Algorithm 2'], ['sections3–4', 'sections 3–4'], ['sections3–5', 'sections 3–5'],
]) source = source.replaceAll(before, after);
fs.writeFileSync(path, source);
console.log('Applied explicit equation line breaks, one worked-answer repair and prose spacing.');
