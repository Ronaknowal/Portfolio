"""Small Bernoulli RBMs with exact normalization; author evidence, CPU only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parent
SEEDS = (11, 29, 47)
EPOCHS, BATCH, RATE = 300, 64, 0.05


def bits(count):
    return np.array(list(itertools.product((0., 1.), repeat=count)))


def free_energy(v, w, a, b):
    return -v @ a - np.logaddexp(0, v @ w + b).sum(axis=-1)


def hidden_distribution(w, a, b):
    h = bits(len(b))
    logits = a + h @ w.T
    log_mass = h @ b + np.logaddexp(0, logits).sum(axis=1)
    log_z = logsumexp(log_mass)
    return h, np.exp(log_mass-log_z), expit(logits), float(log_z)


def exact_negative(w, a, b):
    h, probability, pv, log_z = hidden_distribution(w, a, b)
    return pv.T @ (probability[:, None]*h), probability @ pv, probability @ h, log_z


def positive(v, w, b):
    ph = expit(v @ w + b)
    return v.T @ ph / len(v), v.mean(axis=0), ph.mean(axis=0)


def gibbs(v, w, a, b, rng):
    h = (rng.random((len(v), len(b))) < expit(v @ w+b)).astype(float)
    return (rng.random(v.shape) < expit(h @ w.T+a)).astype(float)


def conditional_missing(v, observed, w, a, b):
    h = bits(len(b))
    logits = a+h@w.T
    # Observed entries contribute their chosen states; missing entries are summed out.
    log_mass = h@b + logits[:, observed]@v[observed]
    log_mass += np.logaddexp(0, logits[:, ~observed]).sum(axis=1)
    ph = np.exp(log_mass-logsumexp(log_mass))
    result = v.copy()
    result[~observed] = ph @ expit(logits[:, ~observed])
    return result


def metrics(v, w, a, b):
    log_z = hidden_distribution(w, a, b)[3]
    nll = free_energy(v, w, a, b)+log_z
    # Mean -> mean reconstruction is a deterministic diagnostic, NOT a Gibbs draw.
    recon = expit(expit(v@w+b)@w.T+a)
    return dict(nll_nats_per_image=float(nll.mean()),
                mean_reconstruction_mse=float(np.mean((recon-v)**2)),
                per_image_nll=nll.tolist(), log_z=log_z)


def exact_checks():
    v = bits(2)
    w, a, b = np.full((2, 1), np.log(3.)), np.zeros(2), np.zeros(1)
    h, ph, pv, log_z = hidden_distribution(w, a, b)
    p = np.exp(-free_energy(v, w, a, b)-log_z)
    assert np.allclose(p, [.1, .2, .2, .5])
    joint = np.exp(v@a[:, None]+h@b + v@w@h.T)
    assert abs(joint.sum()-np.exp(log_z)) < 1e-12
    vh, vm, hm, _ = exact_negative(w, a, b)
    pos = positive(v[3:], w, b)
    grad = [pos[0]-vh, pos[1]-vm, pos[2]-hm]
    # Independent scalar central differences, every parameter in the tiny model.
    defects = []
    def objective(ww, aa, bb):
        return float(-free_energy(v[3], ww, aa, bb)-hidden_distribution(ww, aa, bb)[3])
    for group, theta in enumerate((w, a, b)):
        for index in np.ndindex(theta.shape):
            plus, minus = [x.copy() for x in (w,a,b)], [x.copy() for x in (w,a,b)]
            plus[group][index] += 1e-5
            minus[group][index] -= 1e-5
            defects.append(abs((objective(*plus)-objective(*minus))/2e-5-grad[group][index]))
    # Enumerate all alternating Gibbs paths, not sampled transition frequencies.
    visible_given_hidden = np.prod(pv[:, None, :]**v[None, :, :]
                                  *(1-pv[:, None, :])**(1-v[None, :, :]), axis=2)
    posterior = expit(v@w+b).ravel()
    transition = np.c_[1-posterior, posterior]@visible_given_hidden
    assert np.max(abs(p@transition-p)) < 1e-12
    q = np.array([0.,0.,0.,1.])
    traces = []
    for step in range(11):
        negative_vh = v.T@(q[:,None]*expit(v@w+b))
        traces.append(dict(step=step, distribution=q.tolist(),
                           weight_gradient=(pos[0]-negative_vh).ravel().tolist(),
                           total_variation=float(abs(q-p).sum()/2)))
        q = q@transition
    updated = [x+.1*g for x,g in zip((w,a,b),grad)]
    # Distribution ratios are invariant to adding a constant to all energies.
    shifted = np.exp(-free_energy(v,w,a,b)-100-logsumexp(-free_energy(v,w,a,b)-100))
    assert np.max(abs(shifted-p)) < 1e-12
    missing = conditional_missing(np.array([1.,0.]), np.array([True,False]), w,a,b)
    assert abs(missing[1]-5/7) < 1e-12
    # Mean-visible replacement differs from averaging the nonlinear hidden response.
    exact_hidden_after_one = traces[1]['distribution']@posterior
    mean_visible = np.array([.725,.725])
    mean_hidden = float(expit(mean_visible@w+b)[0])
    sharp_w, sharp_a, sharp_b = np.full((2,1),20.), np.full(2,-10.), np.array([-20.])
    sharp = metrics(v[3:],sharp_w,sharp_a,sharp_b)
    independent = metrics(v[3:],np.zeros((2,1)),np.full(2,np.log(9.)),np.zeros(1))
    return dict(visible_states=v.tolist(), hidden_states=h.tolist(), joint_mass=joint.tolist(),
                visible_probability=p.tolist(), hidden_probability=ph.tolist(),
                log_z=log_z, z=float(np.exp(log_z)), model_vh=vh.tolist(),
                model_v=vm.tolist(), model_h=hm.tolist(), gradient=[x.tolist() for x in grad],
                finite_difference_max_error=max(defects), transition=transition.tolist(), traces=traces,
                updated_parameters=[x.tolist() for x in updated],
                original_log_probability=objective(w,a,b), updated_log_probability=objective(*updated),
                conditional_second_given_first=missing[1],
                expected_hidden_after_one=exact_hidden_after_one, mean_replacement_hidden=mean_hidden,
                reconstruction_counterexample=dict(sharp_correlated=sharp,independent=independent))


def investigation_checks(output):
    w, a, b = np.full((2,1),np.log(3.)), np.zeros(2), np.zeros(1)
    changed_a = np.array([np.log(2.),0.])
    changed_probability = np.exp(-free_energy(bits(2),w,changed_a,b)
                                 -hidden_distribution(w,changed_a,b)[3])
    pos = positive(np.array([[1.,0.]]),w,b)
    neg = exact_negative(w,a,b)
    changed_gradients = [(p-n).tolist() for p,n in zip(pos,neg)]
    hidden_on = float(expit(2*np.log(3.)+np.log(2.)))
    after_one_11 = (1-hidden_on)*.25+hidden_on*.75**2
    zero_input = np.array([[0.,0.]])
    sharp = metrics(zero_input,np.full((2,1),20.),np.full(2,-10.),np.array([-20.]))
    independent = metrics(zero_input,np.zeros((2,1)),np.full(2,np.log(9.)),np.zeros(1))
    fit = output['fits'][0]
    ww, aa, bb = np.array(fit['weights']), np.array(fit['visible_bias']), np.array(fit['hidden_bias'])
    rows = list(csv.DictReader((ROOT/'digits-400.csv').open(encoding='utf-8')))
    row = next(r for r in rows if int(r['source_id'])==fit['intervention']['source_id'])
    v = np.array([int(row[f'pixel_{j}'])>=8 for j in range(64)],dtype=float)
    observed = np.array([j%8<4 for j in range(64)])
    base = conditional_missing(v,observed,ww,aa,bb)
    changed = v.copy(); changed[18]=1-changed[18]
    edited = conditional_missing(changed,observed,ww,aa,bb)
    return dict(bias_edit_probability=changed_probability.tolist(),observation10_gradients=changed_gradients,
                hidden_bias_ln2_next_probability11=after_one_11,
                input00_comparison=dict(sharp=sharp,independent=independent),
                fresh_completion=dict(source_id=int(row['source_id']),observed_index=18,
                                      original=base.tolist(),edited=edited.tolist(),
                                      max_missing_change=float(np.max(abs(edited[~observed]-base[~observed])))))


def main():
    records = list(csv.DictReader((ROOT/'digits-400.csv').open(encoding='utf-8')))
    retained, seen, dropped = [], {}, []
    for row in records:
        signature = tuple(int(row[f'pixel_{j}']) >= 8 for j in range(64))
        if signature in seen:
            dropped.append(dict(source_id=int(row['source_id']), retained_id=seen[signature]))
        else:
            seen[signature] = int(row['source_id'])
            retained.append((int(row['source_id']), int(row['digit']), signature))
    ids = np.array([r[0] for r in retained])
    labels = np.array([r[1] for r in retained])
    x = np.array([r[2] for r in retained], dtype=float)
    split_rng = np.random.default_rng(91)
    role = dict(fit=[], development=[], assessment=[])
    for digit in range(10):
        rows = split_rng.permutation(np.flatnonzero(labels==digit))
        role['fit'].extend(rows[:-16])
        role['development'].extend(rows[-16:-8])
        role['assessment'].extend(rows[-8:])
    role = {name: np.array(rows) for name,rows in role.items()}
    fit = x[role['fit']]
    mean = (fit.sum(axis=0)+.5)/(len(fit)+1)  # symmetric beta(.5,.5) posterior means
    initial_a = np.log(mean)-np.log1p(-mean)
    baseline = {name: float(-(x[rows]*np.log(mean)+(1-x[rows])*np.log1p(-mean)).sum(1).mean())
                for name,rows in role.items()}
    observed = np.array([j%8 < 4 for j in range(64)])  # fixed left half, all assessment images
    results = []
    for method in ('exact', 'cd1', 'pcd1'):
        for seed in SEEDS:
            w = np.random.default_rng(seed).normal(0,.01,(64,8))
            a, b = initial_a.copy(), np.zeros(8)
            order_rng, sample_rng = np.random.default_rng(seed+1000), np.random.default_rng(seed+2000)
            particles = (sample_rng.random((BATCH,64)) < mean).astype(float)
            history = []
            for epoch in range(1,EPOCHS+1):
                order = order_rng.permutation(len(fit))
                for start in range(0,len(fit),BATCH):
                    data = fit[order[start:start+BATCH]]
                    pos = positive(data,w,b)
                    if method == 'exact':
                        neg = exact_negative(w,a,b)[:3]
                    else:
                        negative_data = gibbs(data if method=='cd1' else particles,w,a,b,sample_rng)
                        neg = positive(negative_data,w,b)
                        if method == 'pcd1':
                            particles = negative_data
                    # Simultaneous ascent using statistics from the OLD parameter state.
                    w += RATE*(pos[0]-neg[0])
                    a += RATE*(pos[1]-neg[1])
                    b += RATE*(pos[2]-neg[2])
                if epoch in (1,10,50,100,300):
                    history.append(dict(epoch=epoch, **{name: metrics(x[rows],w,a,b)
                                                        for name,rows in role.items() if name!='assessment'}))
            evaluated = {name: metrics(x[rows],w,a,b) for name,rows in role.items()}
            test = x[role['assessment']]
            completion = np.array([conditional_missing(v,observed,w,a,b) for v in test])
            h, hp, pv, _ = hidden_distribution(w,a,b)
            draw_rng = np.random.default_rng(seed+3000)
            hidden_draws = draw_rng.choice(len(h),size=16,p=hp)
            exact_samples = (draw_rng.random((16,64)) < pv[hidden_draws]).astype(int)
            # First assessment source only: retain changed-evidence and ignored-missing-value controls.
            input_case = test[0].copy()
            changed = input_case.copy(); changed[27] = 1-changed[27]  # observed column 3
            null = input_case.copy(); null[28] = 1-null[28]  # unobserved column 4
            base = completion[0]
            edited = conditional_missing(changed,observed,w,a,b)
            null_output = conditional_missing(null,observed,w,a,b)
            assert np.max(abs(null_output-base)) < 1e-12
            result = dict(method=method,seed=seed,parameters=584,history=history,metrics=evaluated,
                          weights=w.tolist(),visible_bias=a.tolist(),hidden_bias=b.tolist(),
                          assessment_completion=completion.tolist(),
                          completion_mse=float(np.mean((completion[:,~observed]-test[:,~observed])**2)),
                          completion_correct=int(np.sum((completion[:,~observed]>=.5)==test[:,~observed])),
                          exact_samples=exact_samples.tolist(),hidden_sample_states=h[hidden_draws].tolist(),
                          intervention=dict(source_id=int(ids[role['assessment'][0]]),observed_index=27,
                                            original=base.tolist(), edited=edited.tolist(),
                                            max_missing_change=float(np.max(abs(edited[~observed]-base[~observed]))),
                                            missing_placeholder_null_max=float(np.max(abs(null_output-base)))))
            results.append(result)
            print(method,seed, 'NLL',round(evaluated['assessment']['nll_nats_per_image'],6),
                  'reconstruction',round(evaluated['assessment']['mean_reconstruction_mse'],6),
                  'completion',round(result['completion_mse'],6),flush=True)
    output = dict(protocol=dict(threshold='pixel >= 8',raw_rows=len(records),unique_binary=len(x),
                               dropped_duplicates=dropped,seeds=SEEDS,epochs=EPOCHS,batch=BATCH,rate=RATE,
                               hidden_units=8,split_seed=91,roles={k:ids[v].tolist() for k,v in role.items()},
                               csv_sha256=hashlib.sha256((ROOT/'digits-400.csv').read_bytes()).hexdigest()),
                  baseline=dict(nll_nats_per_image=baseline,pixel_probability=mean.tolist(),
                                completion_mse=float(np.mean((mean[~observed]-x[role['assessment']][:,~observed])**2)),
                                completion_correct=int(np.sum((mean[~observed]>=.5)==x[role['assessment']][:,~observed]))),
                  exact=exact_checks(),fits=results)
    output['investigations'] = investigation_checks(output)
    (ROOT/'calculated-inputs.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
