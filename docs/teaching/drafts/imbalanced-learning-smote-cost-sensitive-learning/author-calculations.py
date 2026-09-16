"""Bounded author evidence: five predeclared Yeast fits, no reserve predictions."""
from pathlib import Path
import json
import platform
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
PENALTY = 0.01
FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 12

def fit_logistic(features, labels, sample_weight=None):
    design = np.column_stack([np.ones(len(labels)), features])
    weight = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    weight = weight / weight.sum()
    def objective(parameters):
        margin = design @ parameters
        loss = np.dot(weight, np.logaddexp(0, margin)-labels*margin)
        loss += PENALTY * np.dot(parameters[1:], parameters[1:]) / 2
        gradient = design.T @ (weight*(expit(margin)-labels))
        gradient[1:] += PENALTY*parameters[1:]
        return loss, gradient
    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter':1000, 'gtol':1e-9, 'ftol':1e-13})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, {'iterations':int(result.nit), 'objective':float(result.fun),
                      'gradient_inf':float(np.max(np.abs(result.jac))), 'success':bool(result.success)}

def minority_interpolation(minority, count, k, seed):
    minority = np.asarray(minority, dtype=float)
    if minority.ndim != 2 or not np.isfinite(minority).all():
        raise ValueError('Use a finite minority feature matrix.')
    if not 1 <= k < len(minority) or count < 0:
        raise ValueError('Require 1 <= k < minority count, and nonnegative output count.')
    distance = np.sum((minority[:,None,:]-minority[None,:,:])**2, axis=2)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind='stable')[:,:k]
    rng = np.random.default_rng(seed)
    anchor = rng.integers(len(minority), size=count)
    neighbor = neighbors[anchor, rng.integers(k,size=count)]
    fraction = rng.random(count)
    generated = minority[anchor]+fraction[:,None]*(minority[neighbor]-minority[anchor])
    return generated, {'anchor':anchor.tolist(),'neighbor':neighbor.tolist(),'fraction':fraction.tolist()}

def threshold_counts(labels, scores, threshold):
    selected = np.asarray(scores) >= threshold
    labels = np.asarray(labels)
    tp = int(np.sum(selected & (labels==1)))
    fp = int(np.sum(selected & (labels==0)))
    fn = int(np.sum(~selected & (labels==1)))
    tn = int(np.sum(~selected & (labels==0)))
    return {'threshold':float(threshold), 'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'cost':FALSE_POSITIVE_COST*fp+FALSE_NEGATIVE_COST*fn,
            'alerts':tp+fp}

def select_threshold(labels, scores):
    candidates = np.r_[np.inf, np.sort(np.unique(scores))[::-1]]
    records = [threshold_counts(labels,scores,t) for t in candidates]
    # Higher threshold wins a cost tie: candidates are descending.
    best = min(range(len(records)),key=lambda j:records[j]['cost'])
    return candidates[best], records

def constructed_fixtures():
    easy_probability, hard_probability = .9, .2
    easy_ce, hard_ce = -np.log(easy_probability), -np.log(hard_probability)
    return {
        'precision_counterexample':[threshold_counts([0,1,1],[.9,.8,.7],t) for t in [.7,.8,.9]],
        'weighted_p_point_one':9*.1/(9*.1+.9),
        'smote_anchor':[0,0], 'smote_neighbor':[2,0], 'fraction':.5,'synthetic':[1,0],
        'prior_corrected_q_point_five':(.01/.99)/((.01/.99)+1),
        'expansion_99_to_1':198/100,
        'weighted_single_step':{'intercept_gradient':-.25,'coefficient_gradient':-.75,
                                'intercept_after':.1,'coefficient_after':.3,
                                'scores_after':expit(np.array([.1,.7])).tolist()},
        'population_shift':{'initial_precision':80/179,'shifted_precision':8/(8+99.9)},
        'ap_distinct_example':5/6,
        'focal_loss_mass':{'gamma':2,'easy_count':10000,'hard_count':10,
                           'easy_ce_total':float(10000*easy_ce),'hard_ce_total':float(10*hard_ce),
                           'easy_focal_total':float(10000*(1-easy_probability)**2*easy_ce),
                           'hard_focal_total':float(10*(1-hard_probability)**2*hard_ce)},
        'prior_q_point_eight':4/103,
        'changed_practice':{'tied_threshold':threshold_counts([0,1,0,1],[.95,.8,.8,.4],.8),
                            'cost_cutoff':2/9,'select_risk':2*.8,'skip_risk':7*.2,
                            'weighted_optimum':4*.2/(4*.2+.8),'inverse_probability':.5/(4*.5+.5),
                            'smote_point':[2,2.5],'prior_probability':4/53,
                            'oversampled_rows':1920,'undersampled_rows':80}
    }

def calculate():
    raw = np.loadtxt(ROOT/'yeast.data',dtype=str)
    all_features = raw[:,1:9].astype(float)
    columns = [0,1,2,3,6,7]
    features = all_features[:,columns]
    labels = (raw[:,-1]=='ME2').astype(int)
    identifiers, first_ids, counts = np.unique(raw[:,0],return_index=True,return_counts=True)
    for identifier in identifiers[counts>1]:
        if len(np.unique(raw[raw[:,0]==identifier,1:],axis=0)) != 1:
            raise ValueError('Repeated protein ID has conflicting input or label; resolve before splitting.')
    kept_ids=np.sort(first_ids)
    removed_ids=np.setdiff1d(np.arange(len(labels)),kept_ids)
    development, reserve = train_test_split(kept_ids,train_size=1000,
                                            stratify=labels[kept_ids],random_state=61)
    fitting, remaining = train_test_split(development,train_size=600,
                                         stratify=labels[development],random_state=62)
    tuning, inspection = train_test_split(remaining,train_size=200,
                                         stratify=labels[remaining],random_state=63)
    scaler = StandardScaler().fit(features[fitting])
    fit_features = scaler.transform(features[fitting])
    tune_features = scaler.transform(features[tuning])
    inspection_features = scaler.transform(features[inspection])
    fit_labels = labels[fitting]
    minority_ids = np.flatnonzero(fit_labels==1)
    majority_ids = np.flatnonzero(fit_labels==0)
    n_positive,n_negative=len(minority_ids),len(majority_ids)
    balanced_weight = np.where(fit_labels==1,len(fitting)/(2*n_positive),len(fitting)/(2*n_negative))
    duplicate_ids=np.random.default_rng(64).choice(minority_ids,size=n_negative-n_positive,replace=True)
    under_ids=np.r_[np.random.default_rng(65).choice(majority_ids,size=n_positive,replace=False),minority_ids]
    synthetic, synthesis = minority_interpolation(fit_features[minority_ids],n_negative-n_positive,3,66)
    methods=[('original',fit_features,fit_labels,None),
             ('balanced_weight',fit_features,fit_labels,balanced_weight),
             ('random_over',np.vstack([fit_features,fit_features[duplicate_ids]]),np.r_[fit_labels,np.ones(len(duplicate_ids))],None),
             ('random_under',fit_features[under_ids],fit_labels[under_ids],None),
             ('smote',np.vstack([fit_features,synthetic]),np.r_[fit_labels,np.ones(len(synthetic))],None)]
    records=[]
    for name,x,y,weight in methods:
        parameters, convergence = fit_logistic(x,y,weight)
        tune_scores=expit(parameters[0]+tune_features@parameters[1:])
        inspection_scores=expit(parameters[0]+inspection_features@parameters[1:])
        threshold, sweep=select_threshold(labels[tuning],tune_scores)
        ranked=np.argsort(-inspection_scores,kind='stable')
        records.append({'name':name,'fit_rows':len(y),'fit_class_counts':np.bincount(y.astype(int)).tolist(),
                        'parameters':parameters.tolist(),'convergence':convergence,
                        'tuning_scores':tune_scores.tolist(),'inspection_scores':inspection_scores.tolist(),
                        'tuning_investigation_fixture':{str(t):threshold_counts(labels[tuning],tune_scores,t) for t in [.5,.15]},
                        'chosen_threshold':float(threshold),'tuning_sweep':sweep,
                        'inspection_default':threshold_counts(labels[inspection],inspection_scores,.5),
                        'inspection_tuned':threshold_counts(labels[inspection],inspection_scores,threshold),
                        'average_precision':float(average_precision_score(labels[inspection],inspection_scores)),
                        'roc_auc':float(roc_auc_score(labels[inspection],inspection_scores)),
                        'brier_score':float(brier_score_loss(labels[inspection],inspection_scores)),
                        'top_ten_source_ids':inspection[ranked[:10]].tolist(),
                        'top_ten_positives':int(labels[inspection[ranked[:10]]].sum())})
    # Named input scenarios declared after inference inspection, not used for method selection.
    # Preserve actual positive/negative examples for error analysis; no reserve evaluation.
    first_positive=int(inspection[np.flatnonzero(labels[inspection]==1)[0]])
    first_negative=int(inspection[np.flatnonzero(labels[inspection]==0)[0]])
    fixture=constructed_fixtures()
    return {'versions':{'python':platform.python_version(),'numpy':np.__version__,
                        'scipy':scipy.__version__,'sklearn':sklearn.__version__},
            'data':{'source_rows':len(labels),'study_rows':len(kept_ids),'positive':int(labels[kept_ids].sum()),'binary_positive':'ME2',
                    'kept_source_ids':kept_ids.tolist(),'removed_exact_duplicate_source_ids':removed_ids.tolist(),
                    'duplicate_protein_ids':identifiers[counts>1].tolist(),
                    'columns':columns,'names':['mcg','gvh','alm','mit','vac','nuc'],
                    'class_counts':dict(zip(*[a.tolist() for a in np.unique(raw[:,-1],return_counts=True)]))},
            'split':{'development_ids':development.tolist(),'reserve_ids':reserve.tolist(),'fitting_ids':fitting.tolist(),
                     'tuning_ids':tuning.tolist(),'inspection_ids':inspection.tolist(),
                     'positive_counts':{name:int(labels[ids].sum()) for name,ids in [('fitting',fitting),('tuning',tuning),('inspection',inspection),('reserve',reserve)]},
                     'inspection_labels':labels[inspection].tolist(),'tuning_labels':labels[tuning].tolist()},
            'scaler':{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()},
            'resampling':{'duplicated_source_ids':fitting[duplicate_ids].tolist(),
                          'undersampled_source_ids':fitting[under_ids].tolist(),
                          'minority_source_ids':fitting[minority_ids].tolist(),
                          'synthesis':synthesis,'first_ten_synthetic_scaled':synthetic[:10].tolist(),
                          'first_ten_synthetic_source_scale':scaler.inverse_transform(synthetic[:10]).tolist()},
            'declared_penalty':PENALTY,'costs':{'fp':FALSE_POSITIVE_COST,'fn':FALSE_NEGATIVE_COST},
            'methods':records,'inspection_case_ids':[first_positive,first_negative],
            'constructed':fixture,'fits':5,'reserve_scored':False}

if __name__=='__main__':
    result=calculate()
    # JSON has no nonstandard Infinity. The no-alert threshold is named instead.
    def clean(value):
        if isinstance(value,float) and not np.isfinite(value):
            return 'above-all-scores'
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list): return [clean(v) for v in value]
        return value
    (ROOT/'calculated-inputs.json').write_text(json.dumps(clean(result),indent=2),encoding='utf-8')
    print(json.dumps({'counts':result['split']['positive_counts'],
                      'methods':[{k:r[k] for k in ['name','fit_rows','chosen_threshold','inspection_default','inspection_tuned',
                                                  'average_precision','roc_auc','brier_score','top_ten_positives','convergence']}
                                 for r in result['methods']]},indent=2))
