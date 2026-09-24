"""Bounded NMF author calculations and offline visual inputs, not phase-two review."""
from pathlib import Path
import json
import platform
import numpy as np
import sklearn
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

def loss(X, W, H):
    return float(np.sum((X - W @ H) ** 2) / 2)

def update(X, W, H):
    H = H * (W.T @ X) / ((W.T @ W) @ H)
    W = W * (X @ H.T) / (W @ (H @ H.T))
    return W, H

def main():
    X = np.array([[2., 1., 3.], [1., 2., 3.], [3., 3., 6.]])
    W1 = np.array([[2., 1.], [1., 2.], [3., 3.]])
    H1 = np.array([[1., 0., 1.], [0., 1., 1.]])
    W2 = np.array([[1.5, .5], [.5, 1.5], [2., 2.]])
    H2 = np.array([[1.25, .25, 1.5], [.25, 1.25, 1.5]])
    W = np.array([[1., .5], [.5, 1.], [1., 1.]])
    H = np.array([[1., .2, .8], [.2, 1., .8]])
    trace = [{'iteration': 0, 'W': W.tolist(), 'H': H.tolist(), 'loss': loss(X,W,H)}]
    for step in range(1, 41):
        W,H = update(X,W,H)
        trace.append({'iteration': step, 'W': W.tolist(), 'H': H.tolist(), 'loss': loss(X,W,H)})
    altered = X.copy(); altered[0, 1] = 2
    alteredW, alteredH = update(altered,np.array(trace[0]['W']),np.array(trace[0]['H']))
    data = np.loadtxt(ROOT / 'digits-300.csv', delimiter=',',skiprows=1)
    images = data[:, 2:] / 16
    train, rest = train_test_split(np.arange(len(images)),test_size=.4,random_state=19,stratify=data[:,1])
    validation, test = train_test_split(rest,test_size=.5,random_state=19,stratify=data[rest,1])
    runs=[]; visual={}
    for k in (1,4,8,16):
        for seed in (7,19):
            nmf=NMF(n_components=k,init='random',solver='cd',random_state=seed,max_iter=2000,tol=1e-5)
            coefficients=nmf.fit_transform(images[train]); dictionary=nmf.components_
            validation_coefficients=nmf.transform(images[validation])
            runs.append({'k':k,'seed':seed,'iterations':nmf.n_iter_, 'train_mse':float(np.mean((images[train]-coefficients@dictionary)**2)),'validation_mse':float(np.mean((images[validation]-validation_coefficients@dictionary)**2))})
            if k==8 and seed==19:
                test_coefficients=nmf.transform(images[test])
                visual={'H':dictionary.tolist(),'W_test':test_coefficients.tolist(),'reconstructed_test':(test_coefficients@dictionary).tolist(),'test_mse':float(np.mean((images[test]-test_coefficients@dictionary)**2))}
    pca=PCA(n_components=8,svd_solver='full').fit(images[train])
    pca_reconstruction=pca.inverse_transform(pca.transform(images[test]))
    mean=np.repeat(images[train].mean(axis=0,keepdims=True),len(test),axis=0)
    output={'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'ambiguity_products':[ (W1@H1).tolist(),(W2@H2).tolist() ],'trace':trace,'altered_one_step':{'W':alteredW.tolist(),'H':alteredH.tolist(),'loss':loss(altered,alteredW,alteredH)},'loss_comparison':{str(x):{'frobenius_half':2.,'kl':float(x*np.log(x/(x+2))-x+x+2),'is':float(x/(x+2)-np.log(x/(x+2))-1)} for x in (2,20)},'splits':{'train':train.tolist(),'validation':validation.tolist(),'test':test.tolist()},'runs':runs,'visual':visual,'pca_test_mse':float(np.mean((images[test]-pca_reconstruction)**2)),'mean_test_mse':float(np.mean((images[test]-mean)**2)),'practice':{'loss':loss(np.array([[3.,2.,5.]]),np.array([[3.,2.]]),H1)}}
    (ROOT/'calculated-inputs.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:output[key] for key in ('environment','runs','pca_test_mse','mean_test_mse','loss_comparison')},indent=2))
    print('NMF8 test MSE',visual['test_mse'])
    print('trace loss',[trace[i]['loss'] for i in (0,1,2,10,40)])
    print('first update',trace[1])

if __name__ == '__main__':
    main()
