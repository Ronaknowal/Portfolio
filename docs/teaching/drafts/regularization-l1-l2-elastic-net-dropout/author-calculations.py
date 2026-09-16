"""Bounded content evidence, not a production implementation or test suite."""
from pathlib import Path
import itertools
import json
import warnings
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HERE = Path(__file__).parent

def soft(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)

def coordinate_fit(X, y, strength, ratio, tolerance=1e-10, max_sweeps=10000):
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, p = X.shape
    mean_x, mean_y = X.mean(axis=0), y.mean()
    Z, target = X - mean_x, y - mean_y
    curvature = np.mean(Z * Z, axis=0)
    weight = np.zeros(p)
    residual = target.copy()
    history = []
    for sweep in range(max_sweeps):
        for j in range(p):
            partial = residual + Z[:, j] * weight[j]
            correlation = float(Z[:, j] @ partial / n)
            denominator = curvature[j] + strength * (1-ratio)
            new_weight = float(soft(correlation, strength * ratio) / denominator) if denominator > 0 else 0.0
            weight[j] = new_weight
            residual = partial - Z[:, j] * new_weight
        smooth_gradient = -(Z.T @ residual) / n + strength * (1-ratio) * weight
        violation = np.where(weight != 0, np.abs(smooth_gradient + strength * ratio * np.sign(weight)), np.maximum(np.abs(smooth_gradient) - strength * ratio, 0))
        objective = float(np.mean(residual**2)/2 + strength*(ratio*np.abs(weight).sum()+(1-ratio)*(weight @ weight)/2))
        history.append({'sweep':sweep+1,'weights':weight.tolist(),'objective':objective,'kkt_residual':float(np.max(violation))})
        if np.max(violation) <= tolerance:
            return {'weights':weight.tolist(),'intercept':float(mean_y - mean_x @ weight),'history':history,'converged':True}
    return {'weights':weight.tolist(),'intercept':float(mean_y - mean_x @ weight),'history':history,'converged':False}

def analytic_inputs():
    X=np.array([[1,1],[1,-1],[-1,1],[-1,-1]],float)
    y=X @ np.array([3,.4])
    base={str(r):coordinate_fit(X,y,1,r) for r in [0,.5,1]}
    changed=y.copy(); changed[0]+=4
    translated=y+7
    duplicate=np.array([[-1,-1],[1,1]],float)
    duplicate_y=np.array([-2,2],float)
    masks=[]
    x=np.array([2,1.]); weight=np.array([1.,-1.]); keep=.5; target=1.
    for mask in itertools.product([0,1], repeat=2):
        prediction=float((x*np.array(mask)/keep)@weight)
        masks.append({'mask':list(mask),'probability':.25,'prediction':prediction,'half_squared_loss':(target-prediction)**2/2})
    scalar_paths=[{'z':z,'ridge':z/2,'lasso':float(soft(z,1)),'elastic_net':float(soft(z,.5)/1.5)} for z in np.linspace(-4,4,33)]
    aic_bic=[]
    for name,log_likelihood,k in [('small',-150,3),('large',-146,5)]:
        aic_bic.append({'model':name,'log_likelihood':log_likelihood,'parameters':k,'aic':-2*log_likelihood+2*k,'bic':-2*log_likelihood+k*np.log(100)})
    difference=np.array([[-1,1,0],[0,-1,1.]])
    return {'orthogonal_X':X.tolist(),'orthogonal_y':y.tolist(),'orthogonal_base':base,'changed_first_target':coordinate_fit(X,changed,1,1),'target_shift_null':coordinate_fit(X,translated,1,1),'duplicate':{str(r):coordinate_fit(duplicate,duplicate_y,1,r) for r in [0,.5,1]},'scalar_paths':scalar_paths,'dropout_masks':masks,'dropout_mean':float(np.mean([m['prediction'] for m in masks])),'dropout_expected_half_loss':float(np.mean([m['half_squared_loss'] for m in masks])),'criteria':aic_bic,'factor_penalty':[{'strength':l,'optimal_product':max(1-2*l,0),'balanced_magnitude':np.sqrt(max(1-2*l,0))} for l in [0,.1,.25,.5,1]],'smoothness':{'input':[0,2,0],'difference_matrix':difference.tolist(),'solution':np.linalg.solve(np.eye(3)+difference.T@difference,np.array([0,2,0])).tolist(),'ridge_solution':[0,1,0]},'early_stopping':{'eigenvalues':[1,4],'step_size':.1,'steps':1,'fit_factors':[.1,.4],'matching_ridge_strengths':[9,6]}}

def make_model(family, strength, n_fit):
    if family=='ridge':
        estimator=Ridge(alpha=n_fit*strength, solver='svd')
    elif family=='lasso':
        estimator=Lasso(alpha=strength, max_iter=50000, tol=1e-8)
    elif family=='elastic_net':
        estimator=ElasticNet(alpha=strength,l1_ratio=.5,max_iter=50000,tol=1e-8)
    elif family=='ols':
        estimator=LinearRegression()
    else:
        raise ValueError(family)
    return make_pipeline(PolynomialFeatures(degree=2,include_bias=False),StandardScaler(),estimator)

def real_inputs():
    data=np.loadtxt(HERE/'airfoil-self-noise.dat')
    development,reserved=train_test_split(np.arange(len(data)),train_size=1200,random_state=41)
    X,y=data[development,:5],data[development,5]
    splits=list(KFold(n_splits=3,shuffle=True,random_state=202).split(X))
    strengths=[.001,.01,.1,1,10,100]
    rows=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for family in ['ridge','lasso','elastic_net']:
            for strength in strengths:
                fits=[]
                for train,valid in splits:
                    model=make_model(family,strength,len(train)).fit(X[train],y[train])
                    pred=model.predict(X[valid])
                    fits.append({'training_mse':float(np.mean((model.predict(X[train])-y[train])**2)),'validation_mse':float(np.mean((pred-y[valid])**2)),'coefficients':model[-1].coef_.tolist(),'intercept':float(model[-1].intercept_),'nonzero':int(np.count_nonzero(model[-1].coef_)),'validation_predictions':pred.tolist(),'iterations':int(getattr(model[-1],'n_iter_',0) or 0)})
                rows.append({'family':family,'strength':strength,'mean_validation_mse':float(np.mean([f['validation_mse'] for f in fits])),'fits':fits})
        baselines=[]
        for train,valid in splits:
            ols=make_model('ols',0,len(train)).fit(X[train],y[train])
            baselines.append({'mean_prediction':float(y[train].mean()),'mean_mse':float(np.mean((y[valid]-y[train].mean())**2)),'ols_mse':float(np.mean((ols.predict(X[valid])-y[valid])**2))})
        selected=[]
        for family in ['ridge','lasso','elastic_net']:
            best=min([r for r in rows if r['family']==family],key=lambda r:r['mean_validation_mse'])
            fitted=make_model(family,best['strength'],len(X)).fit(X,y)
            if family=='ridge':
                changed_X=X[0].copy(); changed_X[0]+=500
                inference_fixture={'row_id':int(development[0]),'raw_X':X[0].tolist(),'observed_target':float(y[0]),'base_prediction':float(fitted.predict(X[:1])[0]),'changed_X':changed_X.tolist(),'changed_prediction':float(fitted.predict(changed_X[None,:])[0])}
            names=fitted[0].get_feature_names_out(['frequency_hz','attack_degrees','chord_m','speed_mps','displacement_m']).tolist()
            selected.append({'family':family,'strength':best['strength'],'selection_mse':best['mean_validation_mse'],'coefficients':fitted[-1].coef_.tolist(),'intercept':float(fitted[-1].intercept_),'nonzero':int(np.count_nonzero(fitted[-1].coef_)),'scale_mean':fitted[1].mean_.tolist(),'scale_scale':fitted[1].scale_.tolist(),'development_row0_prediction':float(fitted.predict(X[:1])[0])})
    return {'development_indices':development.tolist(),'reserved_indices':reserved.tolist(),'folds':[{'train_indices':development[t].tolist(),'validation_indices':development[v].tolist()} for t,v in splits],'feature_names':names,'candidate_results':rows,'baselines':baselines,'selected':selected,'inference_fixture':inference_fixture,'warnings':[str(w.message) for w in caught],'reserved_predictions_computed':False}

if __name__=='__main__':
    result={'constructed':analytic_inputs(),'airfoil':real_inputs()}
    (HERE/'calculated-inputs.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'selected':result['airfoil']['selected'],'baselines':result['airfoil']['baselines'],'warnings':result['airfoil']['warnings'],'constructed':result['constructed']},indent=2))
