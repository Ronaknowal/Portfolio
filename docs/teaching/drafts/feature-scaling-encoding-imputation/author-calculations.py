"""Small content calculations and actual offline pipeline comparison."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent

def cross_fit(categories, target, folds, smoothing=2.):
    output = np.empty(len(target))
    records = []
    for held in np.unique(folds):
        train = folds != held
        prior = target[train].mean()
        for row in np.flatnonzero(~train):
            matching = train & (categories == categories[row])
            output[row] = (target[matching].sum() + smoothing * prior) / (matching.sum() + smoothing)
        records.append({'held_fold':int(held),'prior':float(prior),'encoded_rows':np.flatnonzero(~train).tolist()})
    return output, records

def main():
    data = pd.read_csv(ROOT/'penguins.csv')
    numeric=['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']
    categorical=['sex']
    X, y = data[numeric+categorical], data['species']
    train,test = train_test_split(np.arange(len(data)),test_size=.25,random_state=20,stratify=y)
    results=[]; scaled=None
    for name, scaler in [('raw','passthrough'),('standard',StandardScaler()),('minmax',MinMaxScaler()),('robust',RobustScaler())]:
        num=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',scaler)])
        cat=Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='not_recorded',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
        pre=ColumnTransformer([('numeric',num,numeric),('category',cat,categorical)])
        model=Pipeline([('prepare',pre),('classify',KNeighborsClassifier(n_neighbors=5))])
        model.fit(X.iloc[train],y.iloc[train]); predictions=model.predict(X.iloc[test])
        results.append({'method':name,'accuracy':float(accuracy_score(y.iloc[test],predictions)),'correct':int(np.sum(y.iloc[test].to_numpy()==predictions)),'confusion':confusion_matrix(y.iloc[test],predictions,labels=model.classes_).tolist()})
        if name=='standard':
            preparation=model.named_steps['prepare']
            scaled={'feature_names':preparation.get_feature_names_out().tolist(),'train_medians':preparation.named_transformers_['numeric'].named_steps['impute'].statistics_.tolist(),'mean':preparation.named_transformers_['numeric'].named_steps['scale'].mean_.tolist(),'scale':preparation.named_transformers_['numeric'].named_steps['scale'].scale_.tolist(),'test_source_rows':test.tolist(),'raw_test':X.iloc[test].astype(object).where(pd.notna(X.iloc[test]),None).to_dict(orient='records'),'transformed_test':preparation.transform(X.iloc[test]).tolist(),'predictions':predictions.tolist(),'truth':y.iloc[test].tolist()}
    baseline=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train),1)),y.iloc[train])
    baseline_accuracy=accuracy_score(y.iloc[test],baseline.predict(np.zeros((len(test),1))))
    values=np.array([1.,2.,3.,4.,100.])[:,None]
    scale_examples={name:{'values':scaler.fit_transform(values).ravel().tolist(),'new_150':float(scaler.transform([[150.]])[0,0])} for name,scaler in [('standard',StandardScaler()),('minmax',MinMaxScaler()),('robust',RobustScaler())]}
    categories=np.array(['A','A','B','B','C','C']); target=np.array([1.,1.,0.,0.,1.,0.]); folds=np.array([0,1,0,1,0,1])
    encoding,record=cross_fit(categories,target,folds)
    changed=target.copy();changed[0]=0.;changed_encoding,_=cross_fit(categories,changed,folds)
    donors=np.array([[1.,10.,100.],[3.,np.nan,300.],[np.nan,14.,500.]])
    query=np.array([[2.,12.,np.nan]])
    imputed=KNNImputer(n_neighbors=2).fit(donors).transform(query)
    output={'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'data_sha256':hashlib.sha256((ROOT/'penguins.csv').read_bytes()).hexdigest(),'missing_counts':data.isna().sum().to_dict(),'splits':{'train':train.tolist(),'test':test.tolist()},'results':results,'baseline_accuracy':float(baseline_accuracy),'scaled_inspection':scaled,'scale_fixture':scale_examples,'target_encoding':{'categories':categories.tolist(),'target':target.tolist(),'folds':folds.tolist(),'encoded':encoding.tolist(),'fold_records':record,'changed_target':changed.tolist(),'changed_encoded':changed_encoding.tolist()},'knn_imputed':imputed.tolist()}
    (ROOT/'calculated-inputs.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:output[key] for key in ['environment','data_sha256','missing_counts','results','baseline_accuracy','scale_fixture','target_encoding','knn_imputed']},indent=2))

if __name__=='__main__':
    main()
