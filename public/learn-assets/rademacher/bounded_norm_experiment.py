"""Offline constrained logistic models, margin bounds and measured validation.

Python 3.12+, NumPy, SciPy, scikit-learn. Run beside banknote-subset.csv and
complexity_calculations.py. Writes experiment-results.json. No network required.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from complexity_calculations import ramp, monte_carlo_linear


def fit_ball(x, y, radius):
    """Minimize mean logistic margin loss subject to ||w||_2 <= radius."""
    def objective(w):
        margin = y*(x@w)
        return float(np.logaddexp(0,-margin).mean()), -(x.T@(y*expit(-margin)))/len(y)
    constraint = {'type':'ineq','fun':lambda w:radius**2-w@w,
                  'jac':lambda w:-2*w}
    result = minimize(objective,np.zeros(x.shape[1]),jac=True,method='SLSQP',
                      constraints=[constraint],options={'maxiter':500,'ftol':1e-12})
    if not result.success or np.linalg.norm(result.x)>radius+1e-7:
        raise RuntimeError(result.message)
    # Numerical solvers may exceed a hard radius by roundoff. Return a feasible
    # predictor, then report the objective and stationarity of that predictor.
    w=result.x.copy()
    if np.linalg.norm(w)>=radius:
        w *= radius*(1-1e-12)/np.linalg.norm(w)
    _,gradient=objective(w)
    multiplier=max(0.,float(-gradient@w/(2*(w@w)))) if w@w else 0.
    residual=np.linalg.norm(gradient+2*multiplier*w)
    if residual>1e-5:
        raise RuntimeError('KKT stationarity residual exceeds author tolerance')
    return w,{'iterations':int(result.nit),'objective':objective(w)[0],
             'norm':float(np.linalg.norm(w)), 'stationarity_residual':float(residual),
             'multiplier':multiplier}


def evaluate(x,y,w):
    score=x@w
    predicted=np.where(score>=0,1,-1)
    return {'errors':int((predicted!=y).sum()),'n':len(y),
            'error_rate':float((predicted!=y).mean()),
            'log_loss':float(np.logaddexp(0,-y*score).mean()),
            'margins':(y*score).tolist(),'predictions':predicted.tolist()}


def main():
    folder=Path(__file__).parent
    rows=list(csv.DictReader((folder/'banknote-subset.csv').open(encoding='utf-8')))
    features=['variance','skewness','curtosis','entropy']
    # Inspect the retained CSV schema rather than infer class semantics.
    if 'curtosis' not in rows[0]:
        features=['variance','skewness','kurtosis','entropy']
    x=np.array([[float(r[c]) for c in features] for r in rows])
    label_key='label' if 'label' in rows[0] else 'class'
    y=np.array([2*int(r[label_key])-1 for r in rows])
    role_indices={'representation':np.arange(80),'fit':np.arange(80,320),
                  'validation':np.arange(320,400),'assessment':np.arange(400,480)}
    scaler=StandardScaler().fit(x[role_indices['representation']])
    # Frozen representation defined on every future input; clip then add bias.
    mapped=np.column_stack([np.clip(scaler.transform(x)/3,-1,1),np.ones(len(x))])
    fit=role_indices['fit']; val=role_indices['validation']; test=role_indices['assessment']
    radii=[.25,.5,1,2,4]
    margins=[.5,1]
    delta=.05
    comparisons=len(radii)*len(margins)
    confidence=3*math.sqrt(math.log(2*comparisons/delta)/(2*len(fit)))
    energy=float(np.linalg.norm(mapped[fit])/len(fit))
    result={'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
            'data_roles':{k:[int(rows[i]['source_row']) for i in indices] for k,indices in role_indices.items()},
            'representation':{'features':features,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                              'clip_standard_deviations':3,'bias_coordinate':1,
                              'global_row_norm_upper':math.sqrt(5)},
            'radii':radii,'margin_thresholds':margins,'delta':delta,'comparisons':comparisons,
            'energy_factor':energy,'confidence_addend':confidence,'models':[]}
    for radius in radii:
        w,solver=fit_ball(mapped[fit],y[fit],radius)
        training=evaluate(mapped[fit],y[fit],w)
        validation=evaluate(mapped[val],y[val],w)
        bounds=[]
        for rho in margins:
            empirical_ramp=float(ramp(training['margins'],rho).mean())
            complexity_addend=2*radius*energy/rho
            bound=empirical_ramp+complexity_addend+confidence
            bounds.append({'rho':rho,'empirical_ramp':empirical_ramp,
                           'complexity_addend':complexity_addend,
                           'raw_upper':bound,'clipped_upper':min(1,bound)})
        result['models'].append({'radius':radius,'weights':w.tolist(),'solver':solver,
                                 'fit':training,'validation':validation,'bounds':bounds})
    # Selection precedes all assessment calculations; ties use validation log loss,
    # then smaller radius. No assessment-dependent parameter or model choice.
    chosen=min(range(len(radii)),key=lambda i:(result['models'][i]['validation']['errors'],
                  result['models'][i]['validation']['log_loss'],radii[i]))
    result['selection']={'rule':'validation errors, then log loss, then smaller radius',
                         'chosen_radius':radii[chosen]}
    for model in result['models']:
        model['assessment']=evaluate(mapped[test],y[test],np.array(model['weights']))
    prior_sign=1 if y[fit].sum()>=0 else -1
    result['majority_baseline']={'class_sign':prior_sign,
            'validation_errors':int((y[val]!=prior_sign).sum()),
            'assessment_errors':int((y[test]!=prior_sign).sum())}
    # This measures only sign-draw uncertainty for the fixed input matrix.
    result['monte_carlo_unit_ball']=monte_carlo_linear(mapped[fit],1,2048,131,.05)
    result['assessment_labels']=y[test].tolist()
    result['assessment_source_rows']=[int(rows[i]['source_row']) for i in test]
    (folder/'experiment-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'energy_factor':energy,'confidence':confidence,'selected':radii[chosen],
       'rows':[{'B':m['radius'],'train_errors':m['fit']['errors'],'validation_errors':m['validation']['errors'],
                'assessment_errors':m['assessment']['errors'],'bounds':m['bounds']} for m in result['models']]},indent=2))


if __name__=='__main__':
    main()
