"""Exact small Rademacher calculations and bounded Monte Carlo, Python 3.12+.

Requires NumPy. Convention: E_sign max_f sum(sign_i*f_i)/n, no absolute value.
Constructed examples, not model benchmarks. Run to write checked-results.json.
"""
from itertools import product
from pathlib import Path
import json
import math
import numpy as np


def signs(n):
    if not 1 <= n <= 12:
        raise ValueError("Exact enumeration supports 1 to 12 observations")
    return np.array(list(product([-1, 1], repeat=n)), dtype=float)


def finite_complexity(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("Need finite hypothesis-by-observation values")
    noise = signs(values.shape[1])
    correlations = noise @ values.T / values.shape[1]
    best = correlations.max(axis=1)
    return {"values": values.tolist(), "signs": noise.astype(int).tolist(),
            "correlations": correlations.tolist(), "maxima": best.tolist(),
            "best_indices": correlations.argmax(axis=1).tolist(),
            "complexity": float(best.mean()),
            "max_after_averaging": float(correlations.mean(axis=0).max())}


def threshold_values(x, both_orientations=False):
    x = np.asarray(x, float)
    if x.ndim != 1 or not len(x) or np.any(~np.isfinite(x)):
        raise ValueError("Need finite scalar inputs")
    # All distinct restrictions of h_t(x)=+1 iff x>=t, including both extremes.
    cuts = [-math.inf] + np.unique(x).tolist() + [math.inf]
    values = np.unique(np.array([np.where(x >= t, 1, -1) for t in cuts]), axis=0)
    return np.unique(np.vstack([values, -values]), axis=0) if both_orientations else values


def linear_complexity(x, radius=1):
    x = np.asarray(x, float)
    if x.ndim != 2 or not len(x) or np.any(~np.isfinite(x)) or not np.isfinite(radius) or radius < 0:
        raise ValueError("Need finite sample matrix and nonnegative radius")
    vectors = signs(len(x)) @ x
    maxima = radius*np.linalg.norm(vectors, axis=1)/len(x)
    return {"x": x.tolist(), "radius": radius, "signed_sums": vectors.tolist(),
            "maxima": maxima.tolist(), "complexity": float(maxima.mean()),
            "energy_upper": float(radius*np.linalg.norm(x)/len(x))}


def monte_carlo_linear(x, radius, draws, seed, eta=.05):
    x = np.asarray(x, float)
    if x.ndim != 2 or not len(x) or np.any(~np.isfinite(x)) or not np.isfinite(radius) or radius < 0:
        raise ValueError("Need finite sample matrix and nonnegative radius")
    if not isinstance(draws, int) or draws < 1 or not 0 < eta < 1:
        raise ValueError("Need positive draws and eta in (0,1)")
    rng = np.random.default_rng(seed)
    noise = rng.choice([-1, 1], size=(draws, len(x)))
    maxima = radius*np.linalg.norm(noise @ x, axis=1)/len(x)
    ceiling = radius*np.linalg.norm(x, axis=1).mean()
    correction = ceiling*math.sqrt(math.log(1/eta)/(2*draws))
    return {"draws": draws, "seed": seed, "eta": eta,
            "estimate": float(maxima.mean()), "per_draw_upper": float(ceiling),
            "one_sided_correction": correction,
            "upper": min(float(ceiling), float(maxima.mean())+correction),
            "sample_standard_error": float(maxima.std(ddof=1)/math.sqrt(draws)) if draws>1 else None}


def kernel_complexity(gram, radius=1):
    gram = np.asarray(gram, float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not len(gram) or np.any(~np.isfinite(gram)):
        raise ValueError("Need a square Gram matrix")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Need a finite nonnegative radius")
    if not np.allclose(gram, gram.T) or np.linalg.eigvalsh(gram).min() < -1e-10:
        raise ValueError("Kernel Gram matrix must be symmetric positive semidefinite")
    noise = signs(len(gram))
    quadratic = np.einsum('ij,jk,ik->i', noise, gram, noise)
    maxima = radius*np.sqrt(np.maximum(quadratic, 0))/len(gram)
    return {"gram": gram.tolist(), "complexity": float(maxima.mean()),
            "trace_upper": float(radius*math.sqrt(np.trace(gram))/len(gram))}


def ramp(margins, rho):
    if not np.isfinite(rho) or rho <= 0 or np.any(~np.isfinite(margins)):
        raise ValueError("Margin threshold must be positive")
    return np.clip(1-np.asarray(margins)/rho, 0, 1)


def main():
    thresholds = threshold_values([-1, 0, 1])
    constant = np.array([[-1]*3, [1]*3])
    y = np.array([1, -1, 1])
    base = finite_complexity(thresholds)
    loss = finite_complexity((1-thresholds*y)/2)
    assert np.isclose(loss['complexity'], base['complexity']/2)
    assert finite_complexity(np.array([[1]*3]))['complexity'] == 0
    assert np.isclose(finite_complexity(np.vstack([thresholds, thresholds[0]]))['complexity'], base['complexity'])
    assert np.isclose(finite_complexity(thresholds+7)['complexity'], base['complexity'])
    assert np.isclose(finite_complexity(2*thresholds)['complexity'], 2*base['complexity'])
    mixed = .25*thresholds[0]+.75*thresholds[-1]
    assert np.isclose(finite_complexity(np.vstack([thresholds,mixed]))['complexity'],base['complexity'])
    parallel = linear_complexity([[1, 0], [1, 0]])
    orthogonal = linear_complexity([[1, 0], [0, 1]])
    rotated = linear_complexity([[0, 1], [-1, 0]])
    assert np.isclose(rotated['complexity'], orthogonal['complexity'])
    assert np.isclose(linear_complexity([[-1,0],[0,1]])['complexity'],orthogonal['complexity'])
    assert np.isclose(kernel_complexity([[1,-.9],[-.9,1]])['complexity'],kernel_complexity([[1,.9],[.9,1]])['complexity'])
    margin_inputs=np.array([-.2,.1,.4,1.2])
    margin_energy=float(np.linalg.norm(margin_inputs)/4)
    scalar_margin_fixture={'x':margin_inputs.tolist(),'labels':[1,1,1,1],'w':1,'budget':1,
        'energy':margin_energy,'default_ramp':ramp(margin_inputs,.5).tolist(),
        'smaller_rho_ramp':ramp(margin_inputs,.25).tolist(),
        'edited_first_ramp':ramp([.2,.1,.4,1.2],.5).tolist(),
        'default_complexity_addend':2*margin_energy/.5,
        'smaller_rho_complexity_addend':2*margin_energy/.25,
        'scaled_ramp':ramp(3*margin_inputs,1.5).tolist()}
    population_functions=np.array(list(product([0,1],repeat=2)),float)
    samples=list(product([0,1],repeat=2))
    sample_means=[population_functions[:,sample].mean(axis=1) for sample in samples]
    population_means=population_functions.mean(axis=1)
    expected_gap=float(np.mean([(population_means-mean).max() for mean in sample_means]))
    expected_rad=float(np.mean([finite_complexity(population_functions[:,sample])['complexity'] for sample in samples]))
    expected_ghost=float(np.mean([(other-own).max() for own in sample_means for other in sample_means]))
    assert expected_gap<=expected_ghost<=2*expected_rad
    result = {
        'convention': 'E_sign max_f sum(sign_i*f_i)/n; no abs',
        'singleton': finite_complexity([[1, 1, 1]]),
        'constants': finite_complexity(constant), 'thresholds': base,
        'two_orientations': finite_complexity(threshold_values([-1,0,1], True)),
        'all_labels': finite_complexity(signs(3)), 'classification_loss': loss,
        'abs_changes_singleton': float(np.abs(signs(3).sum(axis=1)/3).mean()),
        'geometry': {'parallel':parallel,'orthogonal':orthogonal,'rotated':rotated,
                     'doubled_budget':linear_complexity([[1,0],[0,1]],2),
                     'zero_sample':linear_complexity([[0]]),
                     'added_nonzero':linear_complexity([[0],[1]])},
        'kernels': {str(r):kernel_complexity([[1,r],[r,1]]) for r in [0,.9,1]},
        'monte_carlo': [monte_carlo_linear([[1,0],[1,0]],1,t,131) for t in [16,64,256,1024]],
        'massart': {'M16_n100':math.sqrt(2*math.log(16)/100),
                    'threshold_M4_n3':math.sqrt(2*math.log(4)/3)},
        'margins': {'values':[-.2,.1,.4,1.2],'rho':.5,
                    'ramp':ramp([-.2,.1,.4,1.2],.5).tolist(),
                    'ramp_mean':float(ramp([-.2,.1,.4,1.2],.5).mean()),
                    'scaled_ramp':ramp(3*np.array([-.2,.1,.4,1.2]),1.5).tolist()},
        'scalar_margin_fixture':scalar_margin_fixture,
        'finite_population_proof':{'population':[0,1],'probabilities':[.5,.5],
            'loss_functions':population_functions.tolist(),'samples':samples,
            'expected_largest_gap':expected_gap,'expected_ghost_supremum':expected_ghost,
            'expected_rademacher':expected_rad,'twice_expected_rademacher':2*expected_rad},
        'practice': {'thresholds_n2':finite_complexity(threshold_values([2,5])),
                     'linear_3_4':linear_complexity([[3,0],[0,4]],2),
                     'massart_M8_n200':math.sqrt(2*math.log(8)/200),
                     'ramp':ramp([-.1,.2,.8],.4).tolist()},
    }
    Path(__file__).with_name('checked-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k]['complexity'] for k in ['singleton','constants','thresholds','two_orientations','all_labels','classification_loss']},indent=2))


if __name__ == '__main__':
    main()
