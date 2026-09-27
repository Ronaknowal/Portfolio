import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { transform } from 'esbuild';

const baseline = JSON.parse(fs.readFileSync('docs/teaching/evidence/concept-intuition-baseline.json', 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);
const topics = {
  'sequence-to-sequence-encoder-decoder': {
    files:['src/learn/components/lesson-labs/Seq2SeqIntuitionFigures.jsx','src/learn/components/lesson-labs/seq2seq-intuition.css','docs/teaching/drafts/sequence-to-sequence-encoder-decoder/lesson.md','scripts/generate-seq2seq-lesson.mjs'],
    preserved:['src/learn/data/seq2seq-models.js','src/learn/components/lesson-labs/Seq2SeqLabs.jsx','public/learn-code/sequence-to-sequence-encoder-decoder/calculated-inputs.json','public/learn-code/sequence-to-sequence-encoder-decoder/seed-one-inference.json'],
    check() {
      near((2+18)/8,2.5);near((1+3)/2,2);near((1/4)/(1/12),3);
      assert.deepEqual([1,2,3].map(i=>3-i+i),[3,3,3]);assert.deepEqual([1,2,3].map(i=>3-(4-i)+i),[1,3,5]);
      const joint=[[.25,.25],[.25,.25]];joint.forEach(row=>near(row[1]/(row[0]+row[1]),.5));
      const h=1e-5,z=Math.log(.4/.6),sig=v=>1/(1+Math.exp(-v));assert.ok(Math.abs((sig(z+h)-sig(z-h))/(2*h)-.24)<1e-8);
      near(-1.1/((5+5)/6),-.66);assert.ok(-.66>-.8 && -1<-.8);
      return ['Token versus sequence weighting', 'Source-reversal dependency path counts', 'Replacement-prefix joint and conditional masses', 'Exact two-answer expected reward derivative', 'Length-normalized stopping-bound counterexample'];
    },
  },
  'rnns-lstms-grus': {
    files: ['src/learn/components/lesson-labs/RecurrentIntuitionFigures.jsx','src/learn/components/lesson-labs/recurrent-intuition.css','docs/teaching/drafts/rnns-lstms-grus/lesson.md','scripts/render-recurrent-lesson.mjs'],
    preserved: ['src/learn/data/recurrent-models.js','src/learn/components/lesson-labs/RecurrentLabs.jsx','public/learn-assets/rnns-lstms-grus/pen-models.json','public/learn-assets/rnns-lstms-grus/calculated-inputs.json'],
    check() {
      for(const a of [-1,-.4,0,.8,1])for(const b of [-1,0,1])for(const z of [0,.1,.7,1])assert.ok(z*a+(1-z)*b>=-1&&z*a+(1-z)*b<=1);
      const multiply=(m,v)=>m.map(r=>r.reduce((s,a,i)=>s+a*v[i],0)),A=[[0,2],[0,0]],B=[[0,0],[2,0]];
      assert.deepEqual(multiply(B,multiply(A,[0,1])),[0,4]);assert.deepEqual(multiply(A,multiply(A,[0,1])),[0,0]);
      const direct=.746494**2,total=direct+.458119*.318697;assert.ok(Math.abs(direct-.557253)<1e-6&&Math.abs(total-.703254)<1e-6);
      near(4*32*(2+12+2)+12*32,2432);near(4*32*(2+32+2),4608);assert.equal(10*(12+1),130);
      const seq=['A','B','C','D'];assert.deepEqual(seq.slice(3).reverse(),['D']);assert.deepEqual(seq.slice(0).reverse(),['D','C','B','A']);
      return ['GRU convex-state range invariant', 'Alternating zero-eigenvalue matrices preserve an amplifying direction', 'Two-state chain-rule path sum', 'Projected LSTM parameter dimensions', 'Bidirectional output versus endpoint dependencies'];
    },
  },
  'capsule-networks': {
    files: ['src/learn/components/lesson-labs/CapsuleIntuitionFigures.jsx', 'src/learn/components/lesson-labs/capsule-intuition.css', 'docs/teaching/drafts/capsule-networks/lesson.md', 'scripts/generate-capsule-lesson.mjs'],
    preserved: ['src/learn/data/capsule-models.js', 'src/learn/components/lesson-labs/CapsuleLabs.jsx', 'public/learn-assets/capsule-networks/frozen-model.f32', 'public/learn-assets/capsule-networks/calculated-inputs.json'],
    check() {
      const radial=10/26**2,tangent=5/26; near(tangent/radial,13); near(.6**2+.8**2,1); near(.6*(-.8)+.8*.6,0);
      const sig=x=>1/(1+Math.exp(-x)),h=1e-5,derivative=((1+h)*sig(1+h)-(1-h)*sig(1-h))/(2*h); assert.ok(Math.abs(derivative-(sig(1)+sig(1)*(1-sig(1))))<1e-8);
      near(sig(-2*Math.log(.5)),.8); near(sig(-2*Math.log(2)),.2);
      near(2/(1+1),1); near(18/10,1.8); near(1/10,.1);
      const spread=(a,b)=>Math.max(0,.2-(a-b))**2; near(spread(.6,.3),0); near(spread(.9,.8),.01); near(spread(.7,.4),spread(.6,.3));
      return ['Orthogonal squash perturbations and sensitivity ratio', 'Full versus detached scalar product derivative', 'Gaussian coding-cost activation contrast', 'Gaussian prior precision arithmetic', 'Relative spread-loss boundary and translation'];
    },
  },
  'convnext-modern-cnn-designs': {
    files: ['src/learn/components/lesson-labs/ConvNeXtIntuitionFigures.jsx', 'src/learn/components/lesson-labs/convnext-intuition.css', 'docs/teaching/drafts/convnext-modern-cnn-designs/lesson.md', 'scripts/generate-convnext-lesson.mjs'],
    preserved: ['src/learn/data/convnext-models.js', 'src/learn/components/lesson-labs/ConvNeXtLabs.jsx', 'public/learn-assets/convnext-modern-cnn-designs/calculated-inputs.json', 'public/learn-assets/convnext-modern-cnn-designs/saved-models.json'],
    check() {
      const normalize = r => { const m=(r[0]+r[1])/2,s=Math.sqrt(r.reduce((a,v)=>a+(v-m)**2,0)/2+1e-6);return r.map(v=>(v-m)/s); };
      const rows=[[1,3],[9,5]], pooled=normalize([5,4]), first=rows.map(normalize);
      assert.ok(pooled[0]>.99999); assert.ok(Math.abs((first[0][0]+first[1][0])/2)<1e-6);
      near(5/(8.5+1e-6)*25,14.705880622837574); near(12/(8.5+1e-6)*144,203.2940937301066);
      normalize([1,3]).forEach((v,i)=>near(v,normalize([11,13])[i]));
      assert.equal(3136**2,9834496); assert.equal((3136/196)**2,256);
      return ['Pooling and channel normalization order counterexample', 'Identity-initialized GRN parameter gradients', 'Patch-target brightness removal', 'Global position-pair scaling across resolutions'];
    },
  },
  'depthwise-separable-dilated-convolutions': {
    files: ['src/learn/components/lesson-labs/DepthwiseIntuitionFigures.jsx', 'src/learn/components/lesson-labs/depthwise-intuition.css', 'docs/teaching/drafts/depthwise-separable-dilated-convolutions/lesson.md', 'scripts/generate-depthwise-convolution-lesson.mjs'],
    preserved: ['src/learn/data/depthwise-convolution-models.js', 'src/learn/components/lesson-labs/DepthwiseConvolutionLabs.jsx', 'public/learn-code/depthwise-separable-dilated-convolutions/convolution-factorization.py', 'public/learn-code/depthwise-separable-dilated-convolutions/digit-inference.json'],
    check() {
      assert.equal(Math.floor((8+2-(1+2*(3-1)))/2)+1,3);
      const support = d => [...new Set([-d,0,d].flatMap(c=>[-1,0,1].map(v=>c+v)))].sort((a,b)=>a-b);
      assert.deepEqual(support(3),[-4,-3,-2,-1,0,1,2,3,4]); assert.ok(!support(4).includes(-2)&&!support(4).includes(2));
      [-2,0,3].forEach(x=>near(5+4*((2*x+1)-3)/2,4*x+1));
      [-2,2].forEach(x=>near(Math.max(x,0)-Math.max(-x,0),x));
      assert.deepEqual([[100,0],[0,10]].map(([a,b])=>[4*a,3*b]),[[400,0],[0,30]]);
      assert.deepEqual([[100,0],[0,10]].map(([a])=>[4*a,0]),[[400,0],[0,0]]);
      return ['Stencil-placement output count', 'Discrete shifted-interval support and holes', 'Fixed-statistic affine BatchNorm folding', 'Expansion preserves sign before narrow clipping', 'Same discarded kernel produces input-dependent error'];
    },
  },
  'end-to-end-supervised-learning-error-analysis': {
    files: ['src/learn/components/lesson-labs/EndToEndIntuitionFigures.jsx', 'src/learn/components/lesson-labs/endtoend-intuition.css', 'src/learn/components/lesson-labs/EndToEndLabs.jsx'],
    preserved: ['src/learn/data/endtoend-models.js', 'src/learn/data/endtoend-examples.js', 'src/learn/components/lesson-labs/EndToEndFigures.jsx'],
    check() {
      near((12-10)/2,1); near((12000-10)/2,5995);
      const n=10,z=1.96,p=n/(n+z*z); near((1-p)**2,z*z*p*(1-p)/n); near(p,.7224598312333834);
      const reference=[0,1,0,1],candidate=[0,0,1,0], delta=reference.map((v,i)=>v-candidate[i]);
      near(delta.reduce((s,v)=>s+v,0)/4,.25); near([0,1,1,3].reduce((s,i)=>s+delta[i],0)/4,.75);
      const pairs=[[.6,.6],[.6,.4],[.4,.6],[.4,.4]]; near(pairs.reduce((s,r)=>s+Math.max(...r),0)/4,.55);
      pairs[0].forEach((_,j)=>near(pairs.reduce((s,r)=>s+r[j],0)/4,.5));
      near(10*(1-.8),2); near(1-2/10,.8);
      return ['Unit/schema mismatch through a frozen scaler', 'Perfect-sample Wilson endpoint satisfies score inversion', 'Paired row resampling preserves repair differences', 'Exact four-outcome selection optimism', 'Expected deferral-cost boundary'];
    },
  },
  'time-series-validation-forecasting-baselines': {
    files: ['src/learn/components/lesson-labs/TimeSeriesIntuitionFigures.jsx', 'src/learn/components/lesson-labs/timeseries-intuition.css', 'src/learn/components/lesson-labs/TimeSeriesLabs.jsx'],
    preserved: ['src/learn/data/timeseries-models.js', 'src/learn/data/timeseries-examples.js', 'src/learn/components/lesson-labs/TimeSeriesFigures.jsx'],
    check() {
      near((0+4)/2,2); near(Math.sqrt((0+0+16+16)/4),Math.sqrt(8));
      [.8,1.2].forEach((a,j)=>[1,2,3].forEach((h,i)=>near(2*a**h,[[1.6,1.28,1.024],[2.4,2.88,3.456]][j][i])));
      assert.deepEqual([1,3,7].map(h=>20-h),[19,17,13]); assert.equal(14+7,21);
      const matrices = [Array.from({length:10},(_,r)=>Array.from({length:7},(_,d)=>r!==d)),Array.from({length:10},(_,r)=>Array.from({length:7},()=>r!==0))];
      matrices.forEach((matrix,i)=>{for(let d=0;d<7;d++) assert.equal(matrix.filter(row=>row[d]).length,9); assert.equal(matrix.filter(row=>row.every(Boolean)).length,[3,9][i]);});
      near((1+9)/2,5); near(Math.exp((Math.log(1)+Math.log(9))/2),3);
      return ['Pooled versus averaged horizon RMSE', 'Recursive perturbation shrinkage and amplification', 'Direct and complete-joint label maturity', 'Equal daily coverage with distinct whole-path coverage', 'Arithmetic versus geometric mean after transformation'];
    },
  },
  'ml-problem-formulation-baselines-data-leakage': {
    files: ['src/learn/components/lesson-labs/FormulationIntuitionFigures.jsx', 'src/learn/components/lesson-labs/formulation-intuition.css', 'src/learn/components/lesson-labs/FormulationLabs.jsx'],
    preserved: ['src/learn/data/formulation-models.js', 'src/learn/data/formulation-examples.js', 'src/learn/components/lesson-labs/FormulationFigures.jsx'],
    check() {
      assert.ok(10+7<=20 && 18+7>20 && 19<=18+7 && 19<=20);
      near(40/100,.4); near(60/100,.6); assert.ok(30/100>20/100);
      const y=[1,2,12];
      assert.deepEqual([2,5].map(c=>y.reduce((s,v)=>s+Math.abs(v-c),0)),[11,14]);
      assert.deepEqual([2,5].map(c=>y.reduce((s,v)=>s+(v-c)**2,0)),[101,74]);
      [.6,.4,.2].forEach((p,i)=>near(p*12-3,[4.2,1.8,-.6][i]));
      near(20/1000,.02); near((20+900)/1000,.92);
      return ['Known-positive versus mature-negative label eligibility', 'Proxy and desired-outcome rates can move oppositely', 'Mean and median loss totals', 'Capacity-constrained expected action benefit', 'Selective-label population extremes'];
    },
  },
  'rademacher-complexity-generalization-bounds': {
    files: ['src/learn/components/lesson-labs/RademacherIntuitionFigures.jsx', 'src/learn/components/lesson-labs/rademacher-intuition.css'],
    preserved: ['src/learn/data/rademacher-models.js', 'src/learn/data/rademacher-examples.js', 'src/learn/components/lesson-labs/RademacherLabs.jsx', 'src/learn/components/lesson-labs/RademacherFigures.jsx'],
    check() {
      near(.5+(.75-.5)+(.625-.75)+(.68-.625), .68);
      [.5,.25,.125].forEach((step,i)=>{ near(1/step+1,[3,5,9][i]); assert.ok(Math.abs([.5,.75,.625][i]-.68)<=step/2); });
      const c=2,n=100,r=c*c/n; near(c*Math.sqrt(r/n),r);
      [-2,0,2].forEach(x=>near(3*Math.max(2*x,0),.3*Math.max(20*x,0)));
      const kl = row => row.reduce((s,p)=>s+(p ? p*Math.log(3*p) : 0),0);
      near(kl([.5,0,.5]),Math.log(1.5)); near(kl([0,1,0]),Math.log(3));
      return ['Nested prediction-grid cover radii and telescoping reconstruction', 'Illustrative local-envelope fixed point', 'Positive ReLU rescaling preserves scores', 'Equal-mean distributions retain different KL costs'];
    },
  },
  'calibration-conformal-prediction': {
    files: ['src/learn/components/lesson-labs/CalibrationIntuitionFigures.jsx', 'src/learn/components/lesson-labs/calibration-intuition.css'],
    preserved: ['src/learn/data/calibration-models.js', 'src/learn/data/calibration-examples.js', 'src/learn/components/lesson-labs/CalibrationLabs.jsx', 'src/learn/components/lesson-labs/CalibrationFigures.jsx'],
    check() {
      const pinball = (y, q, tau) => Math.max(tau * (y-q), (tau-1) * (y-q));
      near(pinball(2, 1, .8), .8); near(pinball(2, 3, .8), .2);
      near(.2 * .8 - .8 * .2, 0);
      assert.equal(Math.max(10-12,12-20), -2); assert.equal(Math.max(10-11,11-20), -1);
      assert.deepEqual([.6,.3,.1].map((p,i)=>1-p <= [.3,.8,.95][i]), [false,true,true]);
      assert.deepEqual([[10,1],[14,2],[8,1]].map(([m,r])=>[m-r,m+r]), [[9,11],[12,16],[7,9]]);
      near(4 / (4+1), .8); assert.ok(4/(4+6)<.8);
      return ['Pinball slope and asymmetric costs', 'CQR maximum as endpoint intersection', 'Class-specific candidate inclusion', 'Paired jackknife-plus endpoint construction', 'Test-weight atom and finite weighted quantile mass'];
    },
  },
  'pac-learning-vc-dimension': {
    files: ['src/learn/components/lesson-labs/PacIntuitionFigures.jsx', 'src/learn/components/lesson-labs/pac-intuition.css', 'src/learn/components/lesson-labs/PacLabs.jsx'],
    preserved: ['src/learn/data/pac-models.js', 'src/learn/data/pac-examples.js', 'src/learn/components/lesson-labs/PacFigures.jsx'],
    check() {
      const patterns = ['000','001','010','011','100','110','111'];
      const prefixes = [...new Set(patterns.map(p => p.slice(0,2)))];
      assert.equal(prefixes.length, 4);
      assert.equal(prefixes.filter(p => patterns.filter(s => s.startsWith(p)).length === 2).length, 3);
      near([0,1,1,2].reduce((a,b)=>a+b)/4/4, .25);
      near(.025+.0125+.00625+.00625, .05);
      assert.ok(.55 > .5 && .55 < .5+.1 && .65 >= .5+.1 && .35 <= .5-.1);
      return ['Sauer prefix/extra-extension enumeration', 'Unseen-label completion expected risk', 'Countable-family confidence allocation', 'Pseudo-threshold versus positive/negative margin witness'];
    },
  },
  'dbscan-density-based-clustering': {
    files: ['src/learn/components/lesson-labs/DbscanIntuitionFigures.jsx', 'src/learn/components/lesson-labs/dbscan-intuition.css'],
    preserved: ['src/learn/data/dbscan-models.js', 'src/learn/data/dbscan-examples.js', 'src/learn/components/lesson-labs/DbscanLabs.jsx', 'src/learn/components/lesson-labs/DbscanFigures.jsx'],
    check() {
      near(Math.min(Math.max(.75, 1.75), Math.max(.75, 1)), 1);
      near(Math.max(.75, 1.25, 1), 1.25);
      near((.8 - .6) / .8, .25);
      near((3 - 2) + (5 - 2) + (5 - 2), 3 * 1 + 2 * 2);
      const original = [-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4];
      const count = (array, value) => array.filter(x => Math.abs(x - value) <= 1).length;
      assert.equal(count(original, 0), 3);
      assert.equal(count([...original, 0], 0), 4);
      assert.equal(count([...original, 0], 4), 1);
      assert.ok(count([...original, 0], -1) >= 4 && count([...original, 0], 1) >= 4);
      return ['OPTICS candidate minimum versus endpoint maximum', 'Xi local relative steepness illustration', 'Persistence sum equals count-by-density area', 'Duplicate new row changes border into core and preserves isolated J'];
    },
  },
  'clustering-evaluation-validation-silhouette-ari-nmi': {
    files: ['src/learn/components/lesson-labs/ClusteringEvaluationIntuitionFigures.jsx', 'src/learn/components/lesson-labs/clustering-evaluation-intuition.css'],
    preserved: ['src/learn/data/clustering-evaluation-models.js', 'src/learn/data/clustering-evaluation-examples.js', 'src/learn/components/lesson-labs/ClusteringEvaluationLabs.jsx', 'src/learn/components/lesson-labs/ClusteringEvaluationFigures.jsx'],
    check() {
      near((2 / 3 + 2 / 3) / 7, 4 / 21);
      near(5 / 2, 2.5);
      const cellExpectation = 16 / 70 * (-1 / 8) + 16 / 70 * (3 / 8 * Math.log2(3 / 2)) + .5 / 70;
      const enumerationMean = [0, 1, 2, 3, 4].reduce((sum, r) => sum + [1, 16, 36, 16, 1][r] / 70 * [r, 4 - r, 4 - r, r].reduce((mi, count) => mi + (count ? count / 8 * Math.log2(count / 2) : 0), 0), 0);
      near(4 * cellExpectation, enumerationMean);
      assert.equal(2 * 2 / 8, .5);
      near(2 / 3, .6666666666666666);
      const original = [1, 3, 2], tree = [1, 2, 2];
      const centered = array => array.map(v => v - array.reduce((a, b) => a + b) / array.length);
      const a = centered(original), b = centered(tree);
      near(a.reduce((s, v, i) => s + v * b[i], 0) / Math.sqrt(a.reduce((s, v) => s + v * v, 0) * b.reduce((s, v) => s + v * v, 0)), Math.sqrt(3) / 2);
      near((5.5 - 1) / 5.5, 9 / 11);
      return ['DB and Dunn contrasting distance summaries', 'Cellwise expected MI matches exact enumeration mean', 'One-to-one matching versus purity counts', 'Conditional co-assignment denominator', 'Cophenetic distance correlation', 'Subset versus focal-row silhouette'];
    },
  },
  'pca-dimensionality-reduction': {
    files: ['src/learn/components/lesson-labs/PcaIntuitionFigures.jsx', 'src/learn/components/lesson-labs/pca-intuition.css'],
    preserved: ['src/learn/data/pca-models.js', 'src/learn/data/pca-examples.js', 'src/learn/components/lesson-labs/PcaLabs.jsx', 'src/learn/components/lesson-labs/PcaFigures.jsx'],
    check() {
      near(.5 * 6 + .5 * (2 / 3), 10 / 3);
      const scores = [[-3, -1], [-3, 1], [3, -1], [3, 1]].map(row => row.map(value => value / Math.SQRT2));
      const whitened = scores.map(([a, b]) => [a / Math.sqrt(6), b / Math.sqrt(2 / 3)]);
      [0, 1].forEach(j => near(whitened.reduce((sum, row) => sum + row[j] ** 2, 0) / 3, 1));
      near(Math.hypot(...whitened[0].map((v, j) => v - whitened[1][j])), Math.sqrt(3));
      near(Math.hypot(...whitened[0].map((v, j) => v - whitened[2][j])), Math.sqrt(3));
      const diagonalProjection = ([a, b]) => [(a + b) / 2, (a + b) / 2];
      assert.deepEqual(diagonalProjection([3, 1]), [2, 2]);
      assert.deepEqual(diagonalProjection([3, 3]), [3, 3]);
      const multiplyShape = ([a, b], [c, d]) => { assert.equal(b, c); return [a, d]; };
      assert.deepEqual(multiplyShape([1000, 100], [100, 15]), [1000, 15]);
      assert.deepEqual(multiplyShape([15, 1000], [1000, 100]), [15, 100]);
      return ['Eigenbasis angle recovers earlier horizontal variance', 'Whitened sample variances and changed pair distances', 'Perpendicular versus parallel noise projection', 'Randomized sketch multiplication shapes'];
    },
  },
  'k-means-hierarchical-clustering': {
    files: ['src/learn/components/lesson-labs/KMeansStructureFigures.jsx', 'src/learn/components/lesson-labs/k-means-structure.css'],
    preserved: ['src/learn/data/k-means-hierarchical-models.js', 'src/learn/data/k-means-hierarchical-examples.js', 'src/learn/components/lesson-labs/KMeansHierarchicalLabs.jsx', 'src/learn/components/lesson-labs/KMeansHierarchicalFigures.jsx'],
    check() {
      assert.deepEqual([1 - 0, 2 - 1, 6 - 2], [1, 1, 4]);
      let mean = 0;
      [0, 6, 9].forEach((value, index) => { mean += (value - mean) / (index + 1); });
      near(mean, 5);
      near(4 - 2 ** 2 / 2 + 2 * (1 - 4) ** 2, (0 - 4) ** 2 + (2 - 4) ** 2);
      near((2 + 6 + 6 + 20) / 4, 1.5 ** 2 + 2.5 ** 2);
      near(Math.hypot(.5, .5), Math.SQRT1_2);
      return ['Single-link retained edge lengths and endpoint separation', 'Incremental mean arrival trace', 'BIRCH restricted-scatter identity', 'Direct mapped versus kernel mean distance', 'Spherical normalization contrast'];
    },
  },
  'survival-analysis-cox-regression-kaplan-meier-hazard-models': {
    files: ['src/learn/components/lesson-labs/SurvivalIntuitionFigures.jsx', 'src/learn/components/lesson-labs/survival-intuition.css'],
    preserved: ['src/learn/data/survival-models.js', 'src/learn/data/survival-examples.js', 'src/learn/components/lesson-labs/SurvivalLabs.jsx', 'src/learn/components/lesson-labs/SurvivalFigures.jsx'],
    check() {
      near((1 / 2) / (1 / 56), 28);
      near(1 - Math.exp(-.2) ** 3, .4511883639059736);
      near(2 / 6.5 ** 2, 8 / 169);
      near(2 / (6.5 * (6.5 - 3 / 2)), 4 / 65);
      near(Math.exp(-.1 * (10 / 2)), Math.exp(-.05 * 10));
      near(.9 * .8 * .25, .18);
      near(.9 * .8, .72);
      near(.2 + .5, 1 - .3);
      return ['Greenwood risk-set contribution ratio', 'Rate-to-three-day survival product', 'Breslow and Efron denominator arithmetic', 'AFT time-argument identity', 'Person-period likelihood prefixes', 'Cause-specific versus subdistribution mass'];
    },
  },
  'multi-label-multi-output-learning': {
    files: ['src/learn/components/lesson-labs/MultioutputCompressionFigure.jsx', 'src/learn/components/lesson-labs/multioutput-compression.css'],
    preserved: ['src/learn/data/multioutput-models.js', 'src/learn/data/multioutput-examples.js', 'src/learn/components/lesson-labs/MultioutputLabs.jsx', 'src/learn/components/lesson-labs/MultioutputFigures.jsx'],
    check() {
      [.8 - 1, .3 - 0, .4 - 1].map(value => 2 * value).forEach((value, index) => near(value, [-.4, .6, -1.2][index]));
      near(.25 + .4, .65);
      near(9 * .1 / (9 * .1 + .9), .5);
      const commonEnergy = 200 * .5;
      const rareEnergy = 2 * .99 ** 2 + 198 * .01 ** 2;
      near(commonEnergy / (commonEnergy + rareEnergy), .9805844283192783);
      assert.equal(99 * 2 + 2, 200);
      return ['Feature-times-error matrix gradient row', 'Summing both probability-tree histories', 'Class-weighted loss balance', 'Exact rare/common projection counts and energy'];
    },
  },
  'recommender-systems-collaborative-filtering-matrix-factorization': {
    files: ['src/learn/components/lesson-labs/RecommenderIntuitionFigures.jsx', 'src/learn/components/lesson-labs/recommender-intuition.css'],
    preserved: ['src/learn/data/recommender-models.js', 'src/learn/data/recommender-examples.js', 'src/learn/components/lesson-labs/RecommenderLabs.jsx', 'src/learn/components/lesson-labs/RecommenderFigures.jsx'],
    check() {
      const cost = p => (4 - p) ** 2 + (2 - 2 * p) ** 2 + p ** 2;
      near(cost(4 / 3), 28 / 3);
      for (const p of [0, 1, 2, 3]) assert.ok(cost(p) > cost(4 / 3));
      const q = [[1, 0], [0, 1], [1, 1]];
      const gram = [0, 1].map(a => [0, 1].map(b => q.reduce((sum, row, i) => sum + [5, 1, 3][i] * row[a] * row[b], a === b ? 1 : 0)));
      assert.deepEqual(gram, [[9, 3], [3, 5]]);
      near(.8 * .4 * (.5 / .8) + .2 * .8 * (.5 / .2), .6);
      near(18 / Math.sqrt(340), .9761870601839528);
      near(1 / (1 + Math.exp(-2)), .8807970779778823);
      return ['Scalar ALS minimizer and nearby costs', 'Direct weighted outer products equal displayed Gram decomposition', 'IPS contribution cancellation', 'Raw cosine and BPR multiplier arithmetic'];
    },
  },
};

const selected = process.argv.slice(2);
assert.ok(selected.length, 'Pass one or more owned topic IDs; no implicit cross-curriculum run.');
for (const topicId of selected) {
  const config = topics[topicId];
  assert.ok(config, `No check registered for ${topicId}`);
  const sourceFiles = [`src/learn/data/topics/${topicId}.jsx`, ...config.files];
  for (const file of sourceFiles.filter(file => file.endsWith('.jsx'))) {
    await transform(fs.readFileSync(file, 'utf8'), { loader: 'jsx', jsx: 'automatic' });
  }
  const checks = config.check();
  const preservation = config.preserved.map(file => {
    assert.ok(baseline.actualFileHashes[file], `Missing baseline for ${file}`);
    assert.equal(hash(file), baseline.actualFileHashes[file], `Changed retained engine/example ${file}`);
    return file;
  });
  const receipt = {
    topicId, date: new Date().toISOString(), authorChecksPassed: true,
    sourceFiles, sourceHashes: Object.fromEntries(sourceFiles.map(file => [file, hash(file)])),
    actualChecks: ['All changed JSX parsed with installed esbuild', ...checks, 'Unchanged retained runtime/example identities matched pre-task baseline'],
    retainedUnchanged: preservation,
    pending: ['Independent full-reading review', 'Rendered desktop, intermediate and phone visual checks', 'Shared production integration'],
    limits: 'Author arithmetic and parse checks only. No new native fit, browser visit or independent review is implied.',
  };
  fs.writeFileSync(`docs/teaching/concept-intuition/${topicId}/author-checks.json`, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(`${topicId}: author checks passed (${checks.length} arithmetic groups; ${preservation.length} retained sources)`);
}
