"""Content-phase Gaussian conditioning and fixed real-data comparison."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import norm
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared

HERE=Path(__file__).parent

def posterior(x,y,test,kernel,noise):
    covariance=kernel(x,x)+noise*np.eye(len(x))
    lower=np.linalg.cholesky(covariance)
    alpha=solve_triangular(lower.T,solve_triangular(lower,y,lower=True),lower=False)
    cross=kernel(x,test)
    v=solve_triangular(lower,cross,lower=True)
    mean=cross.T@alpha
    cov=kernel(test,test)-v.T@v
    log_marginal=-.5*y@alpha-np.log(np.diag(lower)).sum()-.5*len(x)*np.log(2*np.pi)
    return mean,cov,float(log_marginal)

def rbf(x,z,length=1.):
    return np.exp(-.5*((np.asarray(x)[:,None]-np.asarray(z)[None,:])/length)**2)

def tiny():
    x=np.array([0.,2.]); y=np.array([1.,-1.]); test=np.array([0.,1.,2.,4.])
    result={}
    for length in [.3,1.,3.]:
        mu,cov,lml=posterior(x,y,test,lambda a,b:rbf(a,b,length),.25)
        result[str(length)]={'mean':mu.tolist(),'latent_variance':np.diag(cov).tolist(),'observation_variance':(np.diag(cov)+.25).tolist(),'log_marginal':lml}
    base=posterior(x,y,test,rbf,.25)
    changed=posterior(x,np.array([3.,-2.]),test,rbf,.25)
    result['changed_y']={'mean':changed[0].tolist(),'same_covariance':bool(np.allclose(base[1],changed[1]))}
    result['zero_cross']={'prior_variance':1.,'covariance_with_observation':0.,'posterior_variance':1.}
    result['single_observation']={'noise_variance':.25,'observation':2.,'rho':.5,'mean':.8,'latent_variance':.8,'observation_variance':1.05}
    # Variance reduction from a candidate noisy observation uses current posterior covariance.
    target=np.array([1.]); candidates=np.array([1.,4.])
    allpoints=np.r_[target,candidates]
    _,cov,_=posterior(x,y,allpoints,rbf,.25)
    result['design_reductions']=[float(cov[0,i]**2/(cov[i,i]+.25)) for i in [1,2]]
    result['ei']=[{'mean':m,'sd':s,'value':float((1-m)*norm.cdf((1-m)/s)+s*norm.pdf((1-m)/s))} for m,s in [(.8,.1),(1.,.5),(1.2,.1),(1.,.2)]]
    result['single_intervals']={'latent': [.8-1.96*np.sqrt(.8),.8+1.96*np.sqrt(.8)],'observation':[.8-1.96*np.sqrt(1.05),.8+1.96*np.sqrt(1.05)]}
    rng=np.random.default_rng(23); grid=np.linspace(-1,4,81); z=rng.normal(size=(len(grid),3))
    result['prior_samples']={str(length): {'x':grid.tolist(),'y':(np.linalg.cholesky(rbf(grid,grid,length)+1e-10*np.eye(len(grid)))@z).T.tolist()} for length in [.3,1.,3.]}
    assert np.allclose(base[1],changed[1])
    for v in result.values():
        if isinstance(v,dict) and 'latent_variance' in v: assert np.min(v['latent_variance'])>=0
    return result

def real():
    data=np.genfromtxt(HERE/'mauna-loa-monthly.csv',delimiter=',',names=True)
    x=((data['year']-1990)+(data['month']-.5)/12).reshape(-1,1)
    y=data['co2_ppm']; train=np.arange(72); dev=np.arange(72,96); test=np.arange(96,120)
    candidates={'rbf':4.*RBF(1.,(.2,20.)), 'trend_periodic':DotProduct(1.,sigma_0_bounds='fixed')+4.*ExpSineSquared(1.,1.,length_scale_bounds=(.2,5.),periodicity_bounds='fixed')+RBF(3.,(.5,20.))}
    result={'train_rows':72,'development_rows':24,'test_rows':24,'noise_variance_ppm2':.09,'development':{}}
    def fit(kernel,indices):
        center=y[indices].mean()
        gp=GaussianProcessRegressor(kernel=kernel,alpha=.09,normalize_y=False,n_restarts_optimizer=1,random_state=23)
        gp.fit(x[indices],y[indices]-center)
        return gp,center
    def report(gp,center,indices):
        mean,sd=gp.predict(x[indices],return_std=True);mean+=center; obs_sd=np.sqrt(sd**2+.09)
        return {'mae':float(np.abs(y[indices]-mean).mean()),'rmse':float(np.sqrt(np.mean((y[indices]-mean)**2))),'covered':int((np.abs(y[indices]-mean)<=1.959963984540054*obs_sd).sum()),'x':x[indices,0].tolist(),'actual':y[indices].tolist(),'mean':mean.tolist(),'latent_sd':sd.tolist(),'observation_sd':obs_sd.tolist(),'kernel':str(gp.kernel_),'fitted_log_theta':gp.kernel_.theta.tolist(),'log_marginal':float(gp.log_marginal_likelihood_value_)}
    frozen_kernels={}
    for name,kernel in candidates.items():
        gp,center=fit(kernel,train); result['development'][name]=report(gp,center,dev)
        frozen_kernels[name]=clone(gp.kernel_)
    result['explorer']={}
    for cutoff in [72,84,96]:
        result['explorer'][str(cutoff)]={}
        for name,kernel in frozen_kernels.items():
            center=y[:cutoff].mean()
            gp=GaussianProcessRegressor(kernel=clone(kernel),alpha=.09,optimizer=None,normalize_y=False)
            gp.fit(x[:cutoff],y[:cutoff]-center)
            full=report(gp,center,np.arange(cutoff,cutoff+24))
            prefix=report(gp,center,np.arange(cutoff,cutoff+6))
            assert np.allclose(full['mean'][:6],prefix['mean'])
            assert np.allclose(full['latent_sd'][:6],prefix['latent_sd'])
            result['explorer'][str(cutoff)][name]=full
    chosen=min(result['development'],key=lambda name:result['development'][name]['mae'])
    gp,center=fit(clone(candidates[chosen]),np.arange(96)); result['selected']=chosen;result['final_test']=report(gp,center,test)
    result['final_center']=float(center)
    seasonal=y[84+np.arange(24)%12]
    result['seasonal_naive']={'mae':float(np.abs(y[test]-seasonal).mean()),'predictions':seasonal.tolist()}
    return result

if __name__=='__main__':
    result={'tiny':tiny(),'real':real()}
    (HERE/'checked-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'tiny':{k:v for k,v in result['tiny'].items() if k!='prior_samples'},'real':{'selected':result['real']['selected'],'dev':{name:{k:v for k,v in row.items() if k in ['mae','rmse','covered','kernel','log_marginal']} for name,row in result['real']['development'].items()},'test':{k:v for k,v in result['real']['final_test'].items() if k in ['mae','rmse','covered','kernel','log_marginal']},'baseline_mae':result['real']['seasonal_naive']['mae']}},indent=2))
