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


def small_examples():
    multiclass = np.array([[.5,.5,0.], [.45,.3,.25], [.6,.2,.2]])
    uncertainty = {"probabilities":multiclass.tolist(),
                   "least_confident":(1-multiclass.max(axis=1)).tolist(),
                   "margin":(np.sort(multiclass,axis=1)[:,-1]-np.sort(multiclass,axis=1)[:,-2]).tolist(),
                   "entropy_nats":entropy(multiclass).tolist()}
    committees = {
        "shared_ambiguity":np.array([[.5,.5],[.5,.5]]),
        "opposing_confident":np.array([[.95,.05],[.05,.95]]),
        "confident_agreement":np.array([[.95,.05],[.95,.05]]),
    }
    committee = {}
    for name,p in committees.items():
        predictive = float(entropy(p.mean(axis=0)))
        expected = float(entropy(p).mean())
        committee[name] = {"members":p.tolist(),"predictive_entropy":predictive,
                           "expected_entropy":expected,"information":predictive-expected}
    assert committee["shared_ambiguity"]["information"] == 0
    assert committee["confident_agreement"]["information"] == 0

    def threshold_trace(queries, true_threshold=5.5):
        remaining = np.arange(.5,8,1.)
        result=[]
        for x in queries:
            before=remaining.copy()
            y=int(x>=true_threshold)
            remaining=remaining[(x>=remaining)==y]
            result.append({"query":x,"label":y,"before":before.tolist(),"after":remaining.tolist()})
        return result
    threshold = {"observed":[[-1,0],[9,1]],"query_pool":list(range(9)),
                 "bisection":threshold_trace([4,6,5]),
                 "edge_scan":threshold_trace([1,2,3,4,5,6]),
                 "outside_disagreement":threshold_trace([0])}
    assert threshold["bisection"][-1]["after"] == [5.5]
    assert len(threshold["outside_disagreement"][0]["after"]) == 8

    def farthest(points, anchors, count):
        points=np.asarray(points,dtype=float)
        anchors=np.asarray(anchors,dtype=float)
        distance=np.linalg.norm(points[:,None,:]-anchors[None,:,:],axis=2).min(axis=1)
        available=np.ones(len(points),dtype=bool)
        selected=[];trace=[]
        for _ in range(min(count,len(points))):
            index=int(np.argmax(np.where(available,distance,-np.inf)))
            selected.append(index);available[index]=False
            distance=np.minimum(distance,np.linalg.norm(points-points[index],axis=1))
            trace.append({"index":index,"radius":float(distance.max()),"distances":distance.tolist()})
        return {"indices":selected,"trace":trace}
    points=[[0,1],[.1,1],[0,4],[4,0]]
    diversity = {
        "points":points,"anchors":[[0,0]],"probability_class1":[.5,.52,.7,.72],
        "entropy_scores":entropy(np.array([[.5,.5],[.48,.52],[.3,.7],[.28,.72]])).tolist(),
        "top_entropy_indices":[0,1],"farthest":farthest(points,[[0,0]],2),
        "changed_point":farthest([[0,1],[.1,1],[0,2],[4,0]],[[0,0]],2),
        "all_coincident":farthest([[0,0],[0,0],[0,0]],[[0,0]],2),
    }
    assert diversity["all_coincident"]["indices"] == [0,1]

    loss=np.array([0.,1.,0.,1.]);q=np.array([.1,.4,.1,.4])
    weighted=loss/(4*q)
    audit={"loss":loss.tolist(),"query_probability":q.tolist(),"population_risk":float(loss.mean()),
           "ordinary_query_mean":float(q@loss),"importance_outcomes":weighted.tolist(),
           "importance_expectation":float(q@weighted)}
    p=np.array([.6,.3,.1]);z=np.array([2.,-1.])
    blocks=(p-np.array([1.,0.,0.]))[:,None]*z
    supplementary={
        "expected_remaining_8":[(a*a+(8-a)**2)/8 for a in [4,1,0]],
        "exercise_remaining_4":[(a*a+(4-a)**2)/4 for a in [1,2,4]],
        "contradiction_at_8_label0":np.arange(.5,8,1.)[(8>=np.arange(.5,8,1.))==0].tolist(),
        "binary_entropy_point2":float(entropy([.2,.8])),
        "gradient_actual":[.99*2,.01*2],"gradient_expected":2*.99*.01*2,
        "cost_ratios":[.12/1,.18/4],"expected_risks":[.5*.1+.5*.3,.9*.15+.1*.2],
        "gp_reduction":.5**2/(1+.25),"gp_zero_covariance":0./(1+.25),
        "temperature_order":{},
    }
    for temperature in [0.5,1.,2.]:
        probability=1/(1+np.exp(-np.array([-4.,-1.,.5])/temperature))
        score=entropy(np.column_stack([1-probability,probability]))
        supplementary["temperature_order"][str(temperature)]={"p":probability.tolist(),"order":np.argsort(-score).tolist()}
    assert all(v["order"]==[2,1,0] for v in supplementary["temperature_order"].values())
    return {"supplementary":supplementary,"uncertainty":uncertainty,"committee":committee,"threshold":threshold,
            "diversity":diversity,"sampling_bias":audit,
            "badge":{"p":p.tolist(),"z":z.tolist(),"predicted_class":0,
                     "gradient_blocks":blocks.tolist(),"norm":float(np.linalg.norm(blocks))}}


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


if __name__=="__main__":
    result={"examples":small_examples(),"banknotes":benchmark()}
    (HERE/"checked-results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    b=result["banknotes"]
    print(json.dumps({"examples":result["examples"],
                      "development":{name:{"milestones":[float(np.mean([v["development_correct"][q] for v in b["traces"][name]]))
                                                       for q in [0,5,10,20,30]],
                                           "final":[v["development_correct"][-1] for v in b["traces"][name]]}
                                     for name in b["strategies"]},
                      "selected":b["selected"],"test":[v["correct"] for v in b["final_test"]]},indent=2))
