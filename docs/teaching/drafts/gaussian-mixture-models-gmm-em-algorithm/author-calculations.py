"""Bounded phase-one arithmetic and fixture evidence; no website implementation.

Run from this directory with Python, NumPy, SciPy and scikit-learn installed.
The checked iris.csv is an offline input; this script never downloads data.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import sklearn
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def expectation(x, weights, means, variances):
    log_joint = np.log(weights) - 0.5 * (
        np.log(2 * np.pi * variances)
        + (x[:, None] - means) ** 2 / variances
    )
    log_density = logsumexp(log_joint, axis=1)
    return np.exp(log_joint - log_density[:, None]), log_density


def em_trace(x, initial_means, initial_variances, steps=20, floor=0.05):
    weights = np.array([0.5, 0.5])
    means = np.array(initial_means, dtype=float)
    variances = np.array(initial_variances, dtype=float)
    trace = []
    for iteration in range(steps + 1):
        responsibilities, log_density = expectation(x, weights, means, variances)
        trace.append(dict(iteration=iteration, weights=weights.tolist(),
                          means=means.tolist(), variances=variances.tolist(),
                          responsibilities=responsibilities.tolist(),
                          log_likelihood=float(log_density.sum())))
        effective_counts = responsibilities.sum(axis=0)
        weights = effective_counts / len(x)
        means = (responsibilities.T @ x) / effective_counts
        scatter = (responsibilities * (x[:, None] - means) ** 2).sum(axis=0)
        variances = np.maximum(scatter / effective_counts, floor)
    return trace


def main():
    x = np.array([-2., -1., 1., 2.])
    report = dict(environment=dict(python=__import__('sys').version.split()[0],
                                  numpy=np.__version__, scipy=scipy.__version__,
                                  sklearn=sklearn.__version__))
    report['em'] = em_trace(x, [-1, 1], [1, 1])
    report['em_null'] = em_trace(x, [0, 0], [2.5, 2.5], steps=2)
    report['em_variations'] = {
        'edited_point': em_trace(np.array([-2., -1., 1., 3.]), [-1, 1], [1, 1], steps=1),
        'asymmetric_initialization': em_trace(x, [-2, -1], [1, 1]),
        'floor_active': em_trace(np.array([-2., -2., 2., 2.]), [-2, 2], [1, 1], floor=0.25, steps=2),
    }
    report['responsibility'] = []
    for weights, means, variances, point in [
        ([.5,.5],[-2,2],[1,1],0), ([.5,.5],[-2,2],[1,1],2),
        ([.5,.5],[-2,2],[1,1],8), ([.2,.8],[-2,2],[1,1],0),
        ([.8,.2],[-2,2],[1,1],0), ([.5,.5],[-2,2],[4,1],0),
        ([.2,.8],[0,0],[1,1],3), ([.2,.8],[0,0],[1,1],0),
    ]:
        r,l = expectation(np.array([point]), np.array(weights), np.array(means), np.array(variances))
        report['responsibility'].append(dict(weights=weights, means=means,
            variances=variances, x=point, responsibility=r[0].tolist(),
            density=float(np.exp(l[0])), log_density=float(l[0])))
    report['underflow'] = dict(log_joint=[-1000,-1001],
        log_density=float(logsumexp([-1000,-1001])),
        responsibility=np.exp(np.array([-1000,-1001])-logsumexp([-1000,-1001])).tolist())
    report['collapse'] = []
    for sigma in [1, .1, .01, .001]:
        _, logs = expectation(x, np.array([.5,.5]), np.array([-2.,0.]), np.array([sigma**2,4.]))
        report['collapse'].append(dict(sigma=sigma, log_likelihood=float(logs.sum())))
    report['geometry'] = []
    for rho in [.75, 0, -.75]:
        covariance = np.array([[1,rho],[rho,1]])
        for point in [[1,1],[1,-1],[0,0]]:
            p = np.array(point)
            report['geometry'].append(dict(rho=rho, x=point,
                mahalanobis_squared=float(p @ np.linalg.solve(covariance,p)),
                density=float(multivariate_normal.pdf(p,cov=covariance))))
    with (HERE/'iris.csv').open(newline='',encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    raw = np.array([[float(row['sepal_length_cm']),float(row['sepal_width_cm'])] for row in rows])
    labels = np.array([row['species'] for row in rows])
    order = np.random.default_rng(16).permutation(len(raw))
    train, validation, test = order[:90], order[90:120], order[120:]
    scaler = StandardScaler().fit(raw[train])
    transformed = scaler.transform(raw)
    candidates = []
    fitted = {}
    for kind in ['full','tied','diag','spherical']:
        for components in range(1,5):
            model = GaussianMixture(n_components=components,covariance_type=kind,
                n_init=5,random_state=16,reg_covar=1e-4,tol=1e-6,max_iter=500).fit(transformed[train])
            fitted[kind,components] = model
            candidates.append(dict(covariance_type=kind,components=components,
                train_mean_log_density=float(model.score(transformed[train])),
                validation_mean_log_density=float(model.score(transformed[validation])),
                aic=float(model.aic(transformed[train])),bic=float(model.bic(transformed[train])),
                converged=bool(model.converged_),iterations=int(model.n_iter_),
                weights=model.weights_.tolist(),means=model.means_.tolist(),covariances=model.covariances_.tolist()))
    best = max(candidates,key=lambda item:item['validation_mean_log_density'])
    model = fitted[best['covariance_type'],best['components']]
    baseline = fitted['full',1]
    report['iris'] = dict(csv_sha256=hashlib.sha256((HERE/'iris.csv').read_bytes()).hexdigest(),
        train_ids=(train+1).tolist(),validation_ids=(validation+1).tolist(),test_ids=(test+1).tolist(),
        scale_mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),candidates=candidates,
        selected=[best['covariance_type'],best['components']],
        test_mean_log_density=float(model.score(transformed[test])),
        baseline_test_mean_log_density=float(baseline.score(transformed[test])),
        test_ari=float(adjusted_rand_score(labels[test],model.predict(transformed[test]))),
        test_ids_in_order=(test+1).tolist(),test_labels=labels[test].tolist(),
        test_component_labels=model.predict(transformed[test]).tolist(),
        test_log_density=model.score_samples(transformed[test]).tolist(),
        test_responsibilities=model.predict_proba(transformed[test]).tolist())
    report['bayesian_sensitivity'] = []
    for concentration in [.01,1,10]:
        model = BayesianGaussianMixture(n_components=6,covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=concentration,
            n_init=3,random_state=16,reg_covar=1e-4,tol=1e-6,max_iter=1000).fit(transformed[train])
        report['bayesian_sensitivity'].append(dict(concentration=concentration,
            weights=model.weights_.tolist(), active_above_point01=int((model.weights_>.01).sum()),
            converged=bool(model.converged_)))
    for trace in [report['em'],report['em_null'],*report['em_variations'].values()]:
        assert min(np.diff([step['log_likelihood'] for step in trace])) >= -1e-10
        for step in trace:
            assert np.allclose(np.sum(step['responsibilities'],axis=1),1,atol=1e-14)
    assert all(item['converged'] for item in candidates)
    assert np.isclose(report['em'][1]['means'][0],-(2*np.tanh(2)+np.tanh(1))/2)
    report['author_checks'] = 'Analytic first-step mean; row normalization; bounded-EM nondecrease including null/edit/floor cases; all 16 real-data candidates converged. Phase-two UI/native integration deferred.'
    (HERE/'checked-results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(em_first=report['em'][1],em_last=report['em'][-1],
        iris_candidates=[{k:r[k] for k in ['covariance_type','components','validation_mean_log_density','bic']} for r in candidates],
        selected=report['iris']['selected'],test=report['iris']['test_mean_log_density'],
        baseline_test=report['iris']['baseline_test_mean_log_density'],ari=report['iris']['test_ari'],
        bayesian=report['bayesian_sensitivity']),indent=2))


def supplement():
    """Check added proof/practice quantities, reusing the completed fit grid."""
    path = HERE/'checked-results.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    x = np.array([-2.,-1.,1.,2.])
    old, new = report['em'][0], report['em'][1]
    q = np.array(old['responsibilities'])
    def bound(state):
        log_joint = np.log(state['weights']) - 0.5 * (
            np.log(2*np.pi*np.array(state['variances']))
            + (x[:,None]-state['means'])**2 / state['variances'])
        expected = float((q*log_joint).sum())
        entropy = float(-(q*np.log(q)).sum())
        return dict(q_function=expected,entropy=entropy,elbo=expected+entropy)
    report['bound'] = dict(old=bound(old),new_with_old_q=bound(new),
                          new_objective=new['log_likelihood'])
    assert np.isclose(report['bound']['old']['elbo'],old['log_likelihood'])
    assert old['log_likelihood'] <= report['bound']['new_with_old_q']['elbo'] <= new['log_likelihood']
    report['small_variance'] = [dict(variance=v,
        responsibility_at_one=float(1/(1+np.exp(-3/(2*v)))),responsibility_at_midpoint=.5)
        for v in [4,1,.25]]
    report['practice'] = dict(weighted_mstep=dict(count=1.3,weight=13/30,mean=1,variance=28/13),
        unequal_weight_boundary=float(1-np.log(3)/2),
        criterion=dict(aic=[310,308],bic=[300+5*np.log(100),286+11*np.log(100)]),
        conditional_at_one=dict(weights=[.5,.5],means=[.5,3.5],variances=[.75,.75],mean=2,variance=3),
        conditional_at_zero=dict(second_weight=float(1/(1+np.exp(2))),mean=float(3/(1+np.exp(2)))))
    report['ellipse_mass'] = float(1-np.exp(-.5))
    report['practice']['changed_small_variance'] = [
        dict(x=.5,variance=v,responsibility=float(1/(1+np.exp(-6/(2*v)))))
        for v in [1,.25]]
    report['practice']['changed_geometry'] = [
        dict(rho=rho,mahalanobis_squared=[
            float(np.array(point) @ np.linalg.solve([[1,rho],[rho,1]],point))
            for point in [[2,1],[2,-1]]]) for rho in [.5,0,-.5]]
    report['covariance_gallery'] = [
        dict(covariance=cov,determinant=float(np.linalg.det(cov)),eigenvalues=np.linalg.eigvalsh(cov).tolist())
        for cov in [[[2,.75],[.75,1]],[[.5,-.25],[-.25,1.5]],
                    [[1,.5],[.5,1]],[[2,0],[0,1]],[[.5,0],[0,1.5]],[[1.5,0],[0,1.5]],[[1,0],[0,1]]]]
    assert all(row['determinant'] > 0 for row in report['covariance_gallery'])
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['bound','small_variance','practice','ellipse_mass']},indent=2))


if __name__ == '__main__':
    if '--supplement' in __import__('sys').argv:
        supplement()
    else:
        main()
        supplement()
