"""Two offline experiments with fixed data roles and predeclared comparisons.

Run from any folder: python uncertainty_experiments.py
Keep both CSV files and calibration_calculations.py beside this program.
The probability mapping, conformal threshold and final assessment use distinct
data. Test outcomes are reported for every declared method, never used to tune.
"""
from pathlib import Path
import csv
import json
import numpy as np
import scipy
import sklearn
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from calibration_calculations import conformal_threshold, conformal_rank, reliability


DIRECTORY = Path(__file__).resolve().parent


def load_rows(filename, features, target):
    with (DIRECTORY/filename).open(encoding="utf-8",newline="") as source:
        rows = list(csv.DictReader(source))
    ids = np.array([int(row["source_row"]) for row in rows])
    assert len(set(ids)) == len(ids)
    x = np.array([[float(row[name]) for name in features] for row in rows])
    y = np.array([float(row[target]) for row in rows])
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return rows, ids, x, y


def classification_experiment():
    rows, ids, x, y = load_rows("banknote-subset.csv",["variance","skewness","curtosis","entropy"],"class")
    y = y.astype(int)
    # First 320 legacy 'pool' rows are split without inspecting their outcomes.
    train, probability_calibration = np.arange(240), np.arange(240,320)
    conformal, test = np.arange(320,400), np.arange(400,480)
    groups = [train,probability_calibration,conformal,test]
    assert len(set(np.concatenate(groups))) == 480
    base = make_pipeline(StandardScaler(),SVC(C=1.0,kernel="rbf"))
    base.fit(x[train],y[train])
    models = {}
    for method in ["sigmoid","isotonic","temperature"]:
        models[method] = CalibratedClassifierCV(FrozenEstimator(base),method=method)
        models[method].fit(x[probability_calibration],y[probability_calibration])
        np.testing.assert_array_equal(models[method].classes_,[0,1])
    alpha = .1
    output = {"primary_predeclared_method":"sigmoid","alpha":alpha,
              "base_parameters":{"model":"StandardScaler + SVC","C":1,"kernel":"rbf"},
              "data_roles":{name:ids[index].tolist() for name,index in zip(
                  ["train","probability_calibration","conformal","test"],groups)},
              "test_labels":y[test].tolist(),"methods":{}}
    for name in ["raw_sigmoid_score","sigmoid","isotonic","temperature","prior_constant"]:
        if name == "raw_sigmoid_score":
            pc, pt = expit(base.decision_function(x[conformal])), expit(base.decision_function(x[test]))
        elif name == "prior_constant":
            pc,pt = np.full(len(conformal),y[train].mean()),np.full(len(test),y[train].mean())
        else:
            pc = models[name].predict_proba(x[conformal])[:,1]
            pt = models[name].predict_proba(x[test])[:,1]
        probabilities = np.column_stack([1-pt,pt])
        true_class_probability = np.where(y[conformal]==1,pc,1-pc)
        scores = 1-true_class_probability
        q = conformal_threshold(scores,alpha)
        # Reuse the actual score comparison: algebraically equivalent double
        # subtraction can exclude equality by one floating-point rounding unit.
        sets = (1-probabilities) <= q
        covered = sets[np.arange(len(test)),y[test]]
        sizes = sets.sum(axis=1)
        output["methods"][name] = {
            "calibration_scores":scores.tolist(),"rank":conformal_rank(len(conformal),alpha),"q":q,
            "test_probabilities_class1":pt.tolist(),"test_sets":sets.tolist(),
            "correct":int(((pt>=.5)==y[test]).sum()),"brier":float(brier_score_loss(y[test],pt)),
            "log_loss":float(log_loss(y[test],probabilities)),"auc":float(roc_auc_score(y[test],pt)),
            "reliability":reliability(pt,y[test],[0,.2,.4,.6,.8,1]),
            "covered":int(covered.sum()),"test_n":len(test),"mean_set_size":float(sizes.mean()),
            "size_counts":{str(k):int((sizes==k).sum()) for k in range(3)},
            "class_coverage":{str(k):{"covered":int(covered[y[test]==k].sum()),"n":int((y[test]==k).sum())} for k in [0,1]}}
    return output


def regression_experiment():
    names=["frequency_hz","attack_angle_deg","chord_length_m","free_stream_velocity_m_s","displacement_thickness_m"]
    rows,ids,x,y=load_rows("airfoil-subset.csv",names,"scaled_sound_pressure_db")
    train=np.array([i for i,row in enumerate(rows) if row["split"]=="train"])
    cal=np.array([i for i,row in enumerate(rows) if row["split"]=="conformal"])
    test=np.array([i for i,row in enumerate(rows) if row["split"]=="test"])
    assert [len(train),len(cal),len(test)]==[240,120,120]
    point=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(x[train],y[train])
    low=GradientBoostingRegressor(loss="quantile",alpha=.05,n_estimators=80,max_depth=2,random_state=67).fit(x[train],y[train])
    high=GradientBoostingRegressor(loss="quantile",alpha=.95,n_estimators=80,max_depth=2,random_state=67).fit(x[train],y[train])
    # A fixed pointwise rearrangement resolves crossing before computing scores.
    lcal,hcal=low.predict(x[cal]),high.predict(x[cal])
    ltest,htest=low.predict(x[test]),high.predict(x[test])
    crossed_cal,crossed_test=int((lcal>hcal).sum()),int((ltest>htest).sum())
    lcal,hcal=np.minimum(lcal,hcal),np.maximum(lcal,hcal)
    ltest,htest=np.minimum(ltest,htest),np.maximum(ltest,htest)
    alpha=.1
    prediction=point.predict(x[test])
    abs_scores=np.abs(y[cal]-point.predict(x[cal]))
    cqr_scores=np.maximum(lcal-y[cal],y[cal]-hcal)
    q_abs,q_cqr=conformal_threshold(abs_scores,alpha),conformal_threshold(cqr_scores,alpha)
    constant=float(y[train].mean())
    q_constant=conformal_threshold(np.abs(y[cal]-constant),alpha)
    intervals={"constant":(np.full(len(test),constant-q_constant),np.full(len(test),constant+q_constant)),
               "ridge_absolute":(prediction-q_abs,prediction+q_abs),
               "raw_quantiles":(ltest,htest),"cqr":(ltest-q_cqr,htest+q_cqr)}
    output={"alpha":alpha,"target_unit":"dB","features":names,
            "parameters":{"ridge_alpha":1,"quantile_levels":[.05,.95],"trees":80,"max_depth":2,"seed":67},
            "data_roles":{name:ids[idx].tolist() for name,idx in [("train",train),("conformal",cal),("test",test)]},
            "rank":conformal_rank(len(cal),alpha),"q_absolute":q_abs,"q_cqr":q_cqr,"q_constant":q_constant,
            "absolute_scores":abs_scores.tolist(),"cqr_scores":cqr_scores.tolist(),
            "test_y":y[test].tolist(),"test_frequency_hz":x[test,0].tolist(),
            "point_predictions":prediction.tolist(),"point_mae":float(mean_absolute_error(y[test],prediction)),
            "constant_mae":float(mean_absolute_error(y[test],np.full(len(test),constant))),
            "quantile_crossings":{"cal":crossed_cal,"test":crossed_test},"methods":{}}
    for name,(lower,upper) in intervals.items():
        covered=(lower<=y[test])&(y[test]<=upper)
        widths=np.maximum(upper-lower,0)
        output["methods"][name]={"lower":lower.tolist(),"upper":upper.tolist(),"covered":int(covered.sum()),
            "n":len(test),"mean_width":float(widths.mean()),"width_quantiles":np.quantile(widths,[0,.25,.5,.75,1]).tolist(),
            "empty_count":int((lower>upper).sum()),
            "frequency_groups":{label:{"covered":int(covered[mask].sum()),"n":int(mask.sum()),"mean_width":float(widths[mask].mean())}
                for label,mask in [("below_2000_hz",x[test,0]<2000),("at_least_2000_hz",x[test,0]>=2000)]}}
    return output


if __name__ == "__main__":
    result={"versions":{"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
            "classification":classification_experiment(),"regression":regression_experiment()}
    (DIRECTORY/"experiment-results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Banknote: method, correct/80, Brier, covered/80, mean set size")
    for name,row in result["classification"]["methods"].items():
        print(name,row["correct"],round(row["brier"],6),row["covered"],row["mean_set_size"])
    print("Airfoil: method, covered/120, mean width dB")
    for name,row in result["regression"]["methods"].items():
        print(name,row["covered"],round(row["mean_width"],6))
