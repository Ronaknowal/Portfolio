"""Content evidence: bounded Wine fits and exact feature-attribution fixtures."""
from pathlib import Path
from functools import partial
from itertools import product
import json
import math
import platform
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
NAMES = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
         'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
         'color_intensity', 'hue', 'od280_od315', 'proline']

def shapley(values, dimension):
    result = np.zeros(dimension)
    for j in range(dimension):
        for mask in range(1 << dimension):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(dimension-size-1) / math.factorial(dimension)
            result[j] += weight * (values[mask | (1 << j)] - values[mask])
    return result

def background_game(predict, instance, background):
    values = []
    for mask in range(1 << len(instance)):
        hybrid = background.copy()
        kept = [j for j in range(len(instance)) if mask & (1 << j)]
        hybrid[:, kept] = instance[kept]
        values.append(float(predict(hybrid).mean()))
    return np.array(values)

def mutual_information(counts):
    p = np.asarray(counts, dtype=float)
    p /= p.sum()
    independent = p.sum(axis=1, keepdims=True) * p.sum(axis=0, keepdims=True)
    nonzero = p > 0
    return float((p[nonzero] * np.log2(p[nonzero] / independent[nonzero])).sum())

def subset_world(labels):
    x = np.array(list(product([0, 1], repeat=2)))
    y = np.array(labels)
    records = []
    for mask in range(4):
        columns = [j for j in range(2) if mask & (1 << j)]
        predictions = []
        for row in x:
            matching = np.all(x[:, columns] == row[columns], axis=1)
            count = np.bincount(y[matching], minlength=2)
            predictions.append(int(np.argmax(count)))
        records.append({'mask': mask, 'predictions': predictions, 'accuracy': float(np.mean(predictions == y))})
    return records

def hand_examples():
    nonlinear = lambda v: v[:, 0] + v[:, 1] + v[:, 0]*v[:, 1]
    x = np.array([2., 3.])
    backgrounds = [np.array([[0., 0.]]), np.array([[0., 0.], [1., 1.]])]
    games = [background_game(nonlinear, x, b) for b in backgrounds]
    rows = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
    target = rows[:, 0]
    donor = np.array([2, 3, 0, 1])
    reliance = []
    for label, coefficients in [('first_only', [1., 0.]), ('second_only', [0., 1.]), ('average', [.5, .5])]:
        base = float(np.mean((rows@coefficients-target)**2))
        changes = []
        for group in [[0], [1], [0, 1]]:
            changed = rows.copy()
            changed[:, group] = rows[donor][:, group]
            changes.append({'group': group, 'prediction': (changed@coefficients).tolist(),
                            'mse_increase': float(np.mean((changed@coefficients-target)**2)-base)})
        reliance.append({'model': label, 'coefficient': coefficients, 'baseline_mse': base, 'changes': changes})
    return {'practice': {'information_bits': mutual_information([[2,0],[0,6]]),
                         'group_unanimity_individual': [1/3]*3, 'grouped_unanimity': [.5,.5],
                         'four_player_individual': [.25]*4, 'null_family_probability': 1-.95**100,
                         'sigmoid_margin_point_five': float(1/(1+np.exp(-.5)))},
            'mi': {'noisy_copy_bits': mutual_information([[3,1],[1,3]]),
                   'independent_bits': mutual_information([[1,1],[1,1]]),
                   'xor_joint_bits': mutual_information([[1,0],[0,1],[0,1],[1,0]])},
            'subset_xor': subset_world([0,1,1,0]), 'subset_first_feature': subset_world([0,0,1,1]),
            'permutation': {'x': rows.tolist(), 'target': target.tolist(), 'donor': donor.tolist(), 'models': reliance},
            'nonlinear_shapley': [{'background': b.tolist(), 'coalitions': g.tolist(), 'phi': shapley(g,2).tolist()}
                                 for b,g in zip(backgrounds,games)],
            'dependent_shapley': {'conditional': {'coalitions':[.5,1,1,1], 'phi':[.25,.25]},
                                  'replacement': {'coalitions':[.5,1,.5,1], 'phi':[.5,0]}}}

def wine_study():
    raw = np.loadtxt(ROOT/'wine.data', delimiter=',')
    x, y = raw[:,1:], raw[:,0].astype(int)
    development, reserved = train_test_split(np.arange(len(y)), train_size=138, stratify=y, random_state=51)
    fitting, inspection = train_test_split(development, train_size=100, stratify=y[development], random_state=52)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=53).split(x[fitting],y[fitting]))
    score_mi = partial(mutual_info_classif, discrete_features=False, n_neighbors=3, random_state=54)
    def pipeline(k):
        return make_pipeline(SelectKBest(score_mi,k=k), DecisionTreeClassifier(max_depth=3,min_samples_leaf=5,random_state=55))
    candidates = []
    for k in [3,6,13]:
        runs = []
        for local_fit,local_validation in folds:
            fit_ids,val_ids = fitting[local_fit],fitting[local_validation]
            model = pipeline(k).fit(x[fit_ids],y[fit_ids])
            pred = model.predict(x[val_ids])
            runs.append({'fit_ids':fit_ids.tolist(),'validation_ids':val_ids.tolist(),
                         'selected':np.flatnonzero(model[0].get_support()).tolist(),
                         'mi_nats':model[0].scores_.tolist(),'predictions':pred.tolist(),
                         'correct':int(np.sum(pred==y[val_ids])),'accuracy':float(np.mean(pred==y[val_ids]))})
        candidates.append({'k':k,'folds':runs,'mean_accuracy':float(np.mean([r['accuracy'] for r in runs]))})
    winner = max(candidates,key=lambda r:r['mean_accuracy']) # predeclared ties prefer fewer columns
    chosen = pipeline(winner['k']).fit(x[fitting],y[fitting])
    chosen_pred = chosen.predict(x[inspection])
    # Four raw measurements declared before fitting, for transparent 16-coalition explanations.
    explanatory_columns = [0,1,6,12]
    small_x = x[:,explanatory_columns]
    small_model = DecisionTreeClassifier(max_depth=3,min_samples_leaf=5,random_state=55).fit(small_x[fitting],y[fitting])
    small_pred = small_model.predict(small_x[inspection])
    baseline_class = int(np.argmax(np.bincount(y[fitting])))
    donor_orders = [np.random.default_rng(56+r).permutation(len(inspection)) for r in range(20)]
    perm = []
    baseline_accuracy = float(np.mean(small_pred == y[inspection]))
    for j in range(4):
        repeats = []
        for donor in donor_orders:
            changed = small_x[inspection].copy()
            changed[:,j] = changed[donor,j]
            pred = small_model.predict(changed)
            repeats.append({'accuracy':float(np.mean(pred==y[inspection])), 'correct':int(np.sum(pred==y[inspection]))})
        drops = [baseline_accuracy-r['accuracy'] for r in repeats]
        perm.append({'feature':NAMES[explanatory_columns[j]], 'drops':drops,
                     'mean':float(np.mean(drops)), 'std_population':float(np.std(drops)), 'repeats':repeats})
    background_ids = np.sort(fitting) # whole fitting reference; no inspection or reserve rows
    background = small_x[background_ids]
    output_index = int(np.flatnonzero(small_model.classes_==1)[0])
    predict = lambda values: small_model.predict_proba(values)[:,output_index]
    explained_ids = inspection[:12]
    explanations = []
    for row_id in explained_ids:
        values = background_game(predict,small_x[row_id],background)
        phi = shapley(values,4)
        explanations.append({'source_id':int(row_id),'input':small_x[row_id].tolist(),
                             'actual_class':int(y[row_id]), 'coalitions':values.tolist(),
                             'baseline':float(values[0]),'prediction':float(values[-1]), 'phi':phi.tolist(),
                             'efficiency_error':float(values[0]+phi.sum()-values[-1])})
    alternative_ids = np.sort(fitting)[-12:]
    first = small_x[explained_ids[0]]
    alternative_game = background_game(predict,first,small_x[alternative_ids])
    changed_inference = []
    for name,index,value in [('alcohol_contrast',0,13.5),('malic_acid_null',1,4.1)]:
        changed = first.copy()
        changed[index] = value
        game = background_game(predict,changed,background)
        changed_inference.append({'name':name,'input':changed.tolist(),'coalitions':game.tolist(),
                                  'phi':shapley(game,4).tolist(),'prediction':float(game[-1])})
    tree = small_model.tree_
    return {'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
            'feature_names':NAMES,'development_ids':development.tolist(),'reserved_ids':reserved.tolist(),
            'fitting_ids':fitting.tolist(),'inspection_ids':inspection.tolist(),'candidates':candidates,
            'selected':{'k':winner['k'],'columns':np.flatnonzero(chosen[0].get_support()).tolist(),
                        'inspection_predictions':chosen_pred.tolist(),'correct':int(np.sum(chosen_pred==y[inspection])),
                        'accuracy':float(np.mean(chosen_pred==y[inspection]))},
            'four_feature_model':{'columns':explanatory_columns,'inspection_predictions':small_pred.tolist(),
                                 'accuracy':baseline_accuracy,'correct':int(np.sum(small_pred==y[inspection])),
                                 'confusion_matrix':confusion_matrix(y[inspection],small_pred,labels=[1,2,3]).tolist(),
                                 'classes':small_model.classes_.tolist(), 'mdi':small_model.feature_importances_.tolist(),
                                 'tree':{'children_left':tree.children_left.tolist(),'children_right':tree.children_right.tolist(),
                                         'feature':tree.feature.tolist(),'threshold':tree.threshold.tolist(),
                                         'value':tree.value[:,0,:].tolist(),'n_node_samples':tree.n_node_samples.tolist(),
                                         'impurity':tree.impurity.tolist()}},
            'majority_class_baseline':{'class':baseline_class,'correct':int(np.sum(y[inspection]==baseline_class)),
                                   'accuracy':float(np.mean(y[inspection]==baseline_class))},
            'permutation':perm,'donor_orders':[p.tolist() for p in donor_orders],
            'background_ids':background_ids.tolist(),'explained':explanations,'changed_inference':changed_inference,
            'alternative_reference':{'background_ids':alternative_ids.tolist(),'coalitions':alternative_game.tolist(),
                                     'phi':shapley(alternative_game,4).tolist()},
            'fits':{'selection_cv':9,'selected_refit':1,'four_feature_refit':1,'total':11},
            'reserved_scored':False}

if __name__ == '__main__':
    result = {'constructed':hand_examples(),'wine':wine_study()}
    (ROOT/'calculated-inputs.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    wine = result['wine']
    print(json.dumps({'candidate_scores':[(c['k'],c['mean_accuracy']) for c in wine['candidates']],
                      'selected':wine['selected'],'four_accuracy':wine['four_feature_model']['accuracy'],
                      'mdi':wine['four_feature_model']['mdi'],
                      'permutation':[(r['feature'],r['mean'],r['std_population']) for r in wine['permutation']],
                      'first_explanation':wine['explained'][0],
                      'alternative':wine['alternative_reference'],'constructed':result['constructed']},indent=2))
