"""Author examples and a fixed-budget, label-oracle benchmark; no web implementation."""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def entropy(probability):
    p = np.asarray(probability, dtype=float)
    return -np.sum(np.where(p > 0, p*np.log(np.maximum(p, 1e-300)), 0), axis=-1)


def benchmark(budget=30, evaluate_test=True, strategies=None):
    data=np.genfromtxt(HERE/"banknote-subset.csv",delimiter=",",names=True,dtype=None,encoding="utf-8")
    raw=np.column_stack([data[name] for name in ["variance","skewness","curtosis","entropy"]])
    truth=data["class"];pool=np.arange(320);dev=np.arange(320,400);test=np.arange(400,480)
    x=StandardScaler().fit(raw[pool]).transform(raw)
    strategies=["random","entropy","committee","farthest"] if strategies is None else list(strategies)
    if not 0 <= budget <= 314 or not strategies or not set(strategies) <= {"random","entropy","committee","farthest"}:
        raise ValueError("Use a budget from 0 to 314 and recognized query strategies.")
    seeds=[7,19,31,43,59];traces={name:[] for name in strategies};final_models={}
    for seed in seeds:
        seed_rng=np.random.default_rng(seed)
        initial=np.concatenate([seed_rng.choice(np.flatnonzero(truth[pool]==c),3,replace=False)
                                for c in [0,1]])
        for strategy in strategies:
            labeled=np.full(320,-1);labeled[initial]=truth[initial]
            rng=np.random.default_rng(seed+10000)
            scores=[];queries=[]
            for acquired in range(budget+1):
                known=np.flatnonzero(labeled>=0)
                model=LogisticRegression(C=1,max_iter=500).fit(x[known],labeled[known])
                scores.append(int(np.sum(model.predict(x[dev])==truth[dev])))
                if acquired==budget:
                    final_models[(strategy,seed)]=model
                    break
                remaining=np.flatnonzero(labeled<0)
                p=model.predict_proba(x[remaining])
                if strategy=="random":
                    local=int(rng.integers(len(remaining)));score=None
                elif strategy=="entropy":
                    value=entropy(p);local=int(np.argmax(value));score=float(value[local])
                elif strategy=="farthest":
                    value=np.linalg.norm(x[remaining,None,:]-x[None,known,:],axis=2).min(axis=1)
                    local=int(np.argmax(value));score=float(value[local])
                else:
                    members=[]
                    for _ in range(3):
                        # Stratified bootstrap retains both classes and each class's observed count.
                        bootstrap=np.concatenate([rng.choice(known[labeled[known]==c],
                                                             np.sum(labeled[known]==c),replace=True)
                                                  for c in [0,1]])
                        member=LogisticRegression(C=1,max_iter=500).fit(x[bootstrap],labeled[bootstrap])
                        members.append(member.predict_proba(x[remaining]))
                    members=np.stack(members)
                    value=entropy(members.mean(axis=0))-entropy(members).mean(axis=0)
                    local=int(np.argmax(value));score=float(value[local])
                row=int(remaining[local])
                # This is the oracle boundary: only the selected target is revealed.
                label=int(truth[row]);labeled[row]=label
                queries.append({"pool_row":row,"source_row":int(data["source_row"][row]),
                                "label":label,"score":score,"model_probability":p[local].tolist()})
            assert len(set(v["pool_row"] for v in queries))==budget
            assert np.sum(labeled>=0)==6+budget
            traces[strategy].append({"seed":seed,"initial_pool_rows":initial.tolist(),
                                     "development_correct":scores,"queries":queries})
    mean_final={name:float(np.mean([v["development_correct"][-1] for v in traces[name]]))
                for name in strategies}
    selected=max(mean_final,key=mean_final.get)
    final_test=[]
    for seed in seeds if evaluate_test else []:
        prediction=final_models[(selected,seed)].predict(x[test])
        final_test.append({"seed":seed,"correct":int(np.sum(prediction==truth[test])),
                           "predictions":prediction.tolist()})
    return {"strategies":strategies,"run_seeds":seeds,"traces":traces,
            "mean_final_development_correct":mean_final,"selected":selected,"final_test":final_test}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare label acquisition under a fixed budget.")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--strategies", nargs="+", choices=["random", "entropy", "committee", "farthest"], default=["random", "entropy", "committee", "farthest"])
    args = parser.parse_args()
    result = benchmark(args.budget, not args.development_only, args.strategies)
    (HERE / "active-learning-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    milestones = sorted(set(q for q in [0, 5, 10, 20, args.budget] if q <= args.budget))
    print("new-query checkpoints", milestones)
    for strategy in result["strategies"]:
        mean = [float(np.mean([run["development_correct"][q] for run in result["traces"][strategy]])) for q in milestones]
        print(strategy, mean)
    print("selected", result["selected"])
    print("test correct out of 80", [run["correct"] for run in result["final_test"]])
