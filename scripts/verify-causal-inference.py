"""Independent exact probability and ancestor-moralization checks for the causal lesson."""
import contextlib
import datetime
from fractions import Fraction as F
import io
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/causal-inference-verification'
OUT.mkdir(parents=True, exist_ok=True)
node_source = r'''
import * as model from './src/learn/data/causal-inference-models.js';
import {causalInferenceExamples as examples} from './src/learn/data/causal-inference-examples.js';
const graphs = [];
const nodes = ['A','B','C','D'];
const possible = nodes.flatMap((source,i) => nodes.slice(i+1).map(target => [source,target]));
for(let mask=0;mask<64;mask++) {
  const graph={nodes,edges:possible.filter((_,index)=>mask & (1<<index))};
  for(let first=0;first<4;first++)for(let second=first+1;second<4;second++){
    const rest=nodes.filter((_,index)=>index!==first&&index!==second);
    for(let selection=0;selection<4;selection++){
      const conditioned=rest.filter((_,index)=>selection&(1<<index));
      graphs.push({graph,start:nodes[first],end:nodes[second],conditioned,
        separated:model.traceCausalPaths(graph,nodes[first],nodes[second],conditioned).separated});
    }
  }
}
const offers=[];for(const highShare of [0,.1,.3,.5,.7,1])for(const lowAssignment of [0,.05,.2,.5,.8,.95,1])for(const highAssignment of [0,.05,.2,.5,.8,.95,1]) {
  const input={highShare,lowAssignment,highAssignment};offers.push({input,result:model.offerPopulation(input)});
}
const frontdoor=[];for(const mediatorLow of [0,.1,.2,.5,.8,.9,1])for(const mediatorHigh of [0,.1,.2,.5,.8,.9,1])for(const directEffect of [0,.1]) {
  const input={mediatorLow,mediatorHigh,directEffect};frontdoor.push({input,result:model.frontdoorPopulation(input)});
}
const counterfactual=[];for(let step=0;step<=10;step++)for(const observedTreatment of [0,1])for(const observedOutcome of [0,1])for(const intervention of [0,1]){
  const input={overlap:step/40,observedTreatment,observedOutcome,intervention};counterfactual.push({input,result:model.counterfactualResponseTypes(input)});
}
const estimators=[];for(const highShare of [0,.3,.7,1])for(const assignment of [[.2,.8],[.1,.3],[.9,.6]])for(const estimatedAssignment of [assignment,[.5,.5]])for(const estimatedOutcomes of [[[.1,.2],[.3,.4]],[[.15,.15],[.2,.2]]]){
  const input={highShare,assignment,estimatedAssignment,estimatedOutcomes};estimators.push({input,result:model.augmentedEffectExpectation(input)});
}
const rules=Object.entries(model.CAUSAL_RULE_PRESETS).map(([key,preset])=>({key,preset,result:model.inspectDoRule(preset)}));
const invalid=[];
for(const [label,callback] of [
  ['cyclic',()=>model.traceCausalPaths({nodes:['A','B'],edges:[['A','B'],['B','A']]},'A','B')],
  ['duplicate',()=>model.validateCausalGraph({nodes:['A','B'],edges:[['A','B'],['A','B']]})],
  ['unknown-node',()=>model.traceCausalPaths({nodes:['A','B'],edges:[]},'A','C')],
  ['condition-endpoint',()=>model.traceCausalPaths({nodes:['A','B'],edges:[]},'A','B',['A'])],
  ['bad-rule',()=>model.inspectDoRule({...model.CAUSAL_RULE_PRESETS.observation,rule:7})],
  ['negative-population',()=>model.offerPopulation({highShare:-.1})],
  ['nan-assignment',()=>model.offerPopulation({lowAssignment:NaN})],
  ['negative-mediator',()=>model.frontdoorPopulation({mediatorLow:-.2})],
  ['direct-out-of-range',()=>model.frontdoorPopulation({directEffect:.2})],
  ['coupling-out-of-range',()=>model.counterfactualResponseTypes({overlap:.3})],
  ['unknown-treatment',()=>model.counterfactualResponseTypes({intervention:2})],
  ['zero-estimated-propensity',()=>model.augmentedEffectExpectation({estimatedAssignment:[0,.5]})]
]) { let failed=false;try{callback()}catch(error){if(!(error instanceof RangeError))throw error;failed=true;}if(!failed)throw Error(label);invalid.push(label);}
console.log(JSON.stringify({examples,graphs,offers,frontdoor,counterfactual,estimators,rules,
  ambiguity:model.latentCausalAmbiguity(),invalid,tiny:model.formatCausalNumber(1e-12)}));
'''
payload = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', node_source],
                                             cwd=ROOT, text=True, encoding='utf-8'))
(OUT / 'model-fixtures.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
maximum_error = 0.0


def fraction(value):
    return F(str(value))


def close(actual, reference, label='probability', tolerance=3e-12):
    global maximum_error
    if reference is None:
        assert actual is None, (label, actual)
        return
    assert actual is not None and math.isfinite(actual), (label, actual, reference)
    error = abs(actual - float(reference))
    maximum_error = max(maximum_error, error)
    assert error <= tolerance, (label, actual, reference, error)


def moral_separated(graph, start, end, conditioned):
    # Restrict to ancestors of query/conditioning nodes, marry each child's parents,
    # forget arrow directions, remove conditioned nodes, then test connectivity.
    ancestors = set([start, end] + conditioned)
    while True:
        expanded = ancestors | {source for source, target in graph['edges'] if target in ancestors}
        if expanded == ancestors:
            break
        ancestors = expanded
    neighbors = {node: set() for node in ancestors}
    for child in ancestors:
        parents = [source for source, target in graph['edges'] if target == child and source in ancestors]
        for parent in parents:
            neighbors[parent].add(child)
            neighbors[child].add(parent)
        for first, second in itertools.combinations(parents, 2):
            neighbors[first].add(second)
            neighbors[second].add(first)
    reached, frontier = set(), [start]
    while frontier:
        node = frontier.pop()
        if node in reached or node in conditioned:
            continue
        reached.add(node)
        frontier.extend(neighbors[node] - reached)
    return end not in reached


for fixture in payload['graphs']:
    assert fixture['separated'] == moral_separated(fixture['graph'], fixture['start'], fixture['end'], fixture['conditioned']), fixture
for fixture in payload['rules']:
    preset, result = fixture['preset'], fixture['result']
    for node in preset['Z']:
        assert result['valid'] == moral_separated(result['transformed'], preset['Y'], node, preset['X'] + preset['W'])
    assert result['valid'] == ('Fails' not in fixture['key'])
    if fixture['key'] == 'deletionFails':
        assert result['eligibleActions'] == [] and result['removed'] == []

for fixture in payload['offers']:
    inp, result = fixture['input'], fixture['result']
    high, low_e, high_e = [fraction(inp[key]) for key in ['highShare', 'lowAssignment', 'highAssignment']]
    shares, e = [1-high, high], [low_e, high_e]
    mu = [[F(1,10), F(1,5)], [F(3,10), F(2,5)]]
    close(sum(row['mass'] for row in result['rows']), 1)
    for x in [0, 1]:
        masses = [shares[z] * (e[z] if x else 1-e[z]) for z in [0, 1]]
        total = sum(masses)
        observational = None if total == 0 else sum(masses[z]*mu[z][x] for z in [0,1])/total
        causal = sum(shares[z]*mu[z][x] for z in [0,1])
        unsupported = any(shares[z] > 0 and masses[z] == 0 for z in [0,1])
        close(result['observedRisks'][x], observational)
        close(result['interventionRisks'][x], causal)
        close(result['adjustedRisks'][x], None if unsupported else causal)
        share_key = 'treatedHighShare' if x else 'untreatedHighShare'
        close(result[share_key], None if total == 0 else masses[1]/total, share_key)
    close(result['causalDifference'], F(1,10))
    assert result['overlap'] == all(shares[z] == 0 or 0 < e[z] < 1 for z in [0,1])

for fixture in payload['frontdoor']:
    inp, result = fixture['input'], fixture['result']
    mediator = [fraction(inp['mediatorLow']), fraction(inp['mediatorHigh'])]
    direct = fraction(inp['directEffect'])
    posterior_u = [F(1,5), F(4,5)]
    inner = []
    for m in [0,1]:
        terms = []
        for x in [0,1]:
            supported = mediator[x] > 0 if m else mediator[x] < 1
            value = F(1,10) + F(1,2)*m + F(1,5)*posterior_u[x] + direct*x if supported else None
            close(result['outcomeTable'][m][x], value)
            terms.append(value)
        inner.append(None if None in terms else sum(terms)/2)
        close(result['mediatorRisks'][m], inner[-1])
    for x in [0,1]:
        weights = [1-mediator[x], mediator[x]]
        unsupported = any(weights[m] > 0 and inner[m] is None for m in [0,1])
        proposed = None if unsupported else sum(weights[m]*(inner[m] or 0) for m in [0,1])
        truth = F(1,5)+F(1,2)*mediator[x]+direct*x
        observed = F(1,10)+F(1,2)*mediator[x]+F(1,5)*posterior_u[x]+direct*x
        close(result['frontdoorRisks'][x], proposed)
        close(result['interventionRisks'][x], truth)
        close(result['observedRisks'][x], observed)
        if direct == 0 and proposed is not None:
            assert proposed == truth

for fixture in payload['counterfactual']:
    inp, result = fixture['input'], fixture['result']
    r = fraction(inp['overlap'])
    types = [((0,0),r), ((0,1),F(3,4)-r), ((1,0),F(1,4)-r), ((1,1),r)]
    compatible = [(pair,mass) for pair,mass in types if pair[inp['observedTreatment']] == inp['observedOutcome']]
    mass = sum(weight for _,weight in compatible)
    risk = sum(pair[inp['intervention']]*weight for pair,weight in compatible)/mass
    close(result['counterfactualRisk'], risk)
    close(result['compatibleMass'], mass)
    close(sum(row['posterior'] for row in result['types']), 1)
    if inp['intervention'] == inp['observedTreatment']:
        close(result['counterfactualRisk'], inp['observedOutcome'])
    close(result['populationEffect'], F(1,2))

for fixture in payload['estimators']:
    inp, result = fixture['input'], fixture['result']
    e = list(map(fraction, inp['assignment']))
    g = list(map(fraction, inp['estimatedAssignment']))
    m = [list(map(fraction, row)) for row in inp['estimatedOutcomes']]
    high = fraction(inp['highShare'])
    weights = [1-high,high]
    mu = [[F(1,10),F(1,5)],[F(3,10),F(2,5)]]
    # Integrate treatment conditional on Z analytically, unlike JS's joint-outcome enumeration.
    augmented = sum(weights[z]*(m[z][1]+e[z]/g[z]*(mu[z][1]-m[z][1])
                                -m[z][0]-(1-e[z])/(1-g[z])*(mu[z][0]-m[z][0])) for z in [0,1])
    weighted = sum(weights[z]*(e[z]/g[z]*mu[z][1]-(1-e[z])/(1-g[z])*mu[z][0]) for z in [0,1])
    close(result['augmented'], augmented)
    close(result['inverseWeighted'], weighted)
    if e == g or m == mu:
        close(result['augmented'], F(1,10))

assert payload['ambiguity'][0]['observed'] == payload['ambiguity'][1]['observed']
close(payload['ambiguity'][0]['effect'], F(7,10))
close(payload['ambiguity'][1]['effect'], -F(1,10))
assert payload['tiny'] != '0'

programs, namespaces = [], {}
for key, example in payload['examples'].items():
    path = OUT / f'program-{key}.py'
    path.write_text(example['code'], encoding='utf-8')
    output = subprocess.check_output([sys.executable, '-X', 'utf8', '-I', str(path)], text=True, encoding='utf-8').strip()
    assert output == example['expected'], (key, output, example['expected'])
    programs.append({'key':key,'stdout':output})
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example['code'], str(path), 'exec'), namespace)
    namespaces[key] = namespace
original = json.loads((ROOT/'docs/teaching/evidence/causal-inference-original-content.json').read_text(encoding='utf-8'))
assert payload['examples']['originalOffer']['code'].strip() == original['originalProgram'].strip()
assert payload['examples']['originalOffer']['expected'] == original['expected']

# Changed-data tasks: calculate independent answers, including an actual program function.
practice = {}
risks = [F(3,10)*F(1,10)+F(7,10)*F(2,5), F(3,10)*F(3,10)+F(7,10)*F(9,20)]
assert risks == [F(31,100),F(81,200)] and risks[1]-risks[0] == F(19,200)
raw = F(135,1000)/F(38,100)-F(230,1000)/F(62,100)
practice['changed_population'] = {'risks':list(map(str,risks)),'ate':str(risks[1]-risks[0]),'raw':float(raw)}
selection = namespaces['selection']
selection['rows'] = [(x, y, int(x or y),
                      (F(1,4) if x else F(3,4)) * (F(3,4) if y else F(1,4)))
                     for x, y in itertools.product([0,1], repeat=2)]
selected = [selection['risk'](x, True) for x in [0,1]]
unselected = [selection['risk'](x) for x in [0,1]]
assert selected == [F(1),F(3,4)] and unselected == [F(3,4),F(3,4)]
practice['collider'] = {'selected_y_risks':list(map(str,selected)),
                       'difference':str(selected[1]-selected[0]),
                       'actualLearnerFunction':True}
education = {'nodes':['S','E','T','Y'],'edges':[['S','E'],['S','Y'],['E','T'],['T','Y']]}
backdoor_graph = {**education,'edges':[edge for edge in education['edges'] if edge[0] != 'T']}
assert moral_separated(backdoor_graph,'T','Y',['E'])
assert moral_separated(backdoor_graph,'T','Y',['S'])
practice['education_adjustment'] = 'Both E and S block the sole backdoor path.'
changed_hidden = [[F(1,5),F(2,5)],[F(3,5),F(4,5)]]
hidden_risks = [sum(changed_hidden[u][x] for u in [0,1])/2 for x in [0,1]]
assert hidden_risks == [F(2,5),F(3,5)] and hidden_risks[1]-hidden_risks[0] == F(1,5)
for x,y in itertools.product([0,1],repeat=2):
    close(payload['ambiguity'][0]['observed'][2*x+y]['mass'], F(1,2)*(changed_hidden[x][x] if y else 1-changed_hidden[x][x]))
practice['changed_hidden_mechanism'] = {'do0':str(hidden_risks[0]),'do1':str(hidden_risks[1])}
natural_rows = [(y,y,F(3,10) if y else F(7,10)) for y in [0,1]]
intervened_rows = [(y,1,F(3,10) if y else F(7,10)) for y in [0,1]]
conditional_risk = namespaces['ruleThree']['risk_given_w_one']
natural_risk, forced_risk = conditional_risk(natural_rows), conditional_risk(intervened_rows)
assert natural_risk == 1 and forced_risk == F(3,10)
practice['rule_three_selection'] = {'natural':str(natural_risk),'intervened':str(forced_risk),'actualLearnerFunction':True}
frontdoor_program = namespaces['frontdoor']
frontdoor_program['mediator'] = [F(1,5),F(3,5)]
changed_output = io.StringIO()
with contextlib.redirect_stdout(changed_output):
    frontdoor_program['compare'](F(0))
    frontdoor_program['compare'](F(1,10))
assert "formula: ['3/10', '1/2'] | true do: ['3/10', '1/2']" in changed_output.getvalue()
assert "formula: ['7/20', '11/20'] | true do: ['3/10', '3/5']" in changed_output.getvalue()
practice['changed_frontdoor'] = {'actualLearnerFunction':True,'stdout':changed_output.getvalue().strip()}
assert (F(3,4)-F(1,5))/F(3,4) == F(11,15)
assert F(1,5)/F(1,4) == F(4,5)
practice['counterfactual'] = {'treated_success_failure_without':'11/15','untreated_success_success_with':'4/5'}
arm = lambda e,g,mu,m: m+e/g*(mu-m)
assert arm(F(1,4),F(1,2),F(4,5),F(2,5)) == F(3,5)
assert arm(F(1,4),F(1,4),F(4,5),F(2,5)) == F(4,5)
assert arm(F(1,4),F(1,2),F(4,5),F(4,5)) == F(4,5)
practice['augmented_arm'] = ['3/5','4/5','4/5']
assert F(2,5)*F(1,2)+F(3,10)*F(1,5) == F(13,50)
assert (F(1,5)+F(1,50))/F(2,5) == F(11,20)
practice['instrument'] = {'first_stage':.4,'assignment_effect':.2,'local_effect':.5,'population_ate':.26,'exclusion_failure_ratio':.55}
# Reuse the learner's real augmented-score function with changed nuisance models.
changed_score = namespaces['estimators']['expected_score']([F(1,3),F(2,3)],[[F(1,10),F(1,5)],[F(3,10),F(2,5)]])
assert changed_score == F(1,10)
practice['actual_program_changed_nuisance'] = str(changed_score)

record = {'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'python':sys.version.split()[0], 'graphSeparationCases':len(payload['graphs']),
          'ruleCases':len(payload['rules']), 'offerCases':len(payload['offers']),
          'frontdoorCases':len(payload['frontdoor']), 'counterfactualCases':len(payload['counterfactual']),
          'estimatorCases':len(payload['estimators']), 'invalidCases':payload['invalid'],
          'maximumAbsoluteError':maximum_error, 'programs':programs, 'preservedOriginalPrograms':1,
          'practice':practice, 'allPassed':True}
(OUT/'native-results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in record.items() if key not in ['practice','programs']},indent=2))
