"""Bounded complementary review: category refinement, local IB curvature and finite InfoNCE."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/mutual-information-independent-review'
OUT.mkdir(parents=True, exist_ok=True)


def decimal(value):
    if isinstance(value, F):
        return D(value.numerator) / D(value.denominator)
    return D(str(value))


def entropy(values):
    return -sum((decimal(p) * decimal(p).ln() for p in values if p), D(0)) / D(2).ln()


def information(table):
    row = [sum(values) for values in table]
    column = [sum(values[j] for values in table) for j in range(len(table[0]))]
    return entropy(row) + entropy(column) - entropy(list(itertools.chain.from_iterable(table)))


def close(actual, expected, tolerance=2e-12):
    assert abs(actual-float(expected)) <= tolerance, (actual, str(expected))


with localcontext() as context:
    context.prec = 70
    tables = []
    weights = [[2,0,5,1],[0,0,0,0],[3,7,0,2]]
    total = sum(map(sum,weights))
    original = [[F(value,total) for value in row] for row in weights]
    tables.append(original)
    # Splitting each X category by an independent auxiliary label preserves MI with Y.
    for split in [F(1,7),F(2,5),F(9,10)]:
        tables.append([[value*weight for value in row] for row in original for weight in [split,1-split]])
    tables.append([list(reversed(row)) for row in reversed(original)])
    tables.append([list(column) for column in zip(*original)])
    for table in tables:
        assert abs(information(table)-information(original)) < D('1e-65')

    perturbations = []
    for label_error in [D('.1'),D('.2'),D('.3')]:
        critical_beta = 1/(1-2*label_error)**2
        for multiplier in [D('.8'),D('1.2')]:
            beta = critical_beta*multiplier
            delta = D('.0001')
            rate = 1-entropy([D('.5')-delta,D('.5')+delta])
            predictive_delta = (1-2*label_error)*delta
            relevance = 1-entropy([D('.5')-predictive_delta,D('.5')+predictive_delta])
            objective = rate-beta*relevance
            quadratic = 2/D(2).ln()*(1-beta*(1-2*label_error)**2)
            assert abs(objective/delta**2-quadratic) < D('5e-8')
            assert (objective > 0) == (multiplier < 1)
            perturbations.append({'labelError':float(label_error),'beta':float(beta),
                                  'noise':float(D('.5')-delta),'objective':float(objective),
                                  'quadraticCoefficient':float(quadratic)})

    input_payload = {'tables':[[list(map(float,row)) for row in table] for table in tables],
                     'perturbations':perturbations}
    node = r'''
import {readFileSync} from 'node:fs';
import {finiteInformation,bottleneckRepresentation,bottleneckIterations} from './src/learn/data/mutual-information-models.js';
const input=JSON.parse(readFileSync(0,'utf8'));
console.log(JSON.stringify({tables:input.tables.map(finiteInformation),
 perturbations:input.perturbations.map(p=>bottleneckRepresentation('noisy',p.noise,p.labelError,p.beta)),
 stationary:bottleneckIterations(3,40,'symmetric',.1).current,
 better:bottleneckIterations(3,40,'signal',.1).current}));
'''
    models = json.loads(subprocess.check_output(['node','--input-type=module','-e',node],
                                               input=json.dumps(input_payload),text=True,encoding='utf-8',cwd=ROOT))
    for table,result in zip(tables,models['tables']):
        close(result['mi'],information(table))
        assert all((row is None) == (sum(table[i]) == 0) for i,row in enumerate(result['conditional']))
    for reference,result in zip(perturbations,models['perturbations']):
        close(result['objective'],reference['objective'],2e-15)
    assert abs(models['stationary']['objective']) < 1e-14
    assert models['better']['objective'] < -.6

    # Free-energy gap for changed, nonuniform three-input/three-label laws and arbitrary encoders.
    free_energy = []
    px = [F(1,6),F(1,3),F(1,2)]
    labels = [[F(1,2),F(1,3),F(1,6)],[F(1,4),F(1,4),F(1,2)],[F(1,10),F(7,10),F(1,5)]]
    reference = [F(1,5),F(4,5)]
    decoder = [[F(1,3)]*3,[F(1,2),F(1,3),F(1,6)]]

    def kl(left,right):
        return sum((decimal(p)*(decimal(p).ln()-decimal(q).ln()) for p,q in zip(left,right) if p),D(0))/D(2).ln()

    for encoder_one in [[F(1,7),F(2,5),F(3,4)],[F(1,2)]*3,[F(1),F(0),F(1,3)]]:
        encoder = [[1-p,p] for p in encoder_one]
        pxz = [[px[x]*encoder[x][z] for z in [0,1]] for x in range(3)]
        pzy = [[sum(px[x]*encoder[x][z]*labels[x][y] for x in range(3)) for y in range(3)] for z in [0,1]]
        pz = list(map(sum,pzy))
        true_decoder = [[p/pz[z] for p in row] for z,row in enumerate(pzy)]
        pxy = [[px[x]*value for value in row] for x,row in enumerate(labels)]
        for beta in [D(0),D('.7'),D(3)]:
            original_free = sum(decimal(px[x])*kl(encoder[x],reference) for x in range(3))
            original_free += beta*sum(decimal(pxz[x][z])*kl(labels[x],decoder[z]) for x in range(3) for z in [0,1])
            objective = information(pxz)-beta*information(pzy)
            gap = kl(pz,reference)+beta*sum(decimal(pz[z])*kl(true_decoder[z],decoder[z]) for z in [0,1])
            discrepancy = original_free-objective-beta*information(pxy)-gap
            assert abs(discrepancy) < D('1e-65')
            free_energy.append({'beta':float(beta),'identityError':str(discrepancy),'nonnegativeGap':str(gap)})

    # Enumerate the expectation, not a sampled loss, under the stated candidate law.
    joint = [[F(2,5),F(1,10)],[F(1,10),F(2,5)]]
    target_marginal = [F(1,2),F(1,2)]
    contrastive = []
    for candidates in [2,3,4]:
        for scores in [[[1,1],[1,1]],[[4,1],[1,4]],[[1,9],[9,1]],[[3,2],[7,1]]]:
            loss = D(0)
            for x,positive in itertools.product([0,1],repeat=2):
                for negatives in itertools.product([0,1],repeat=candidates-1):
                    mass = joint[x][positive]
                    for negative in negatives:
                        mass *= target_marginal[negative]
                    probability = D(scores[x][positive])/D(scores[x][positive]+sum(scores[x][y] for y in negatives))
                    loss -= decimal(mass)*probability.ln()/D(2).ln()
            lower = D(candidates).ln()/D(2).ln()-loss
            assert lower <= information(joint)+D('1e-65') and lower <= D(candidates).ln()/D(2).ln()
            contrastive.append({'K':candidates,'scores':scores,'expectedLossBits':float(loss),'lowerBound':float(lower)})
    # Independence does not rescue wrong negative sampling: deterministic-negative construction.
    # Positive Y is fair, negative Y is always zero, score(1)=99 and score(0)=1.
    wrong_loss = -(D('.5')*D('.5').ln()+D('.5')*D('.99').ln())/D(2).ln()
    wrong_lower = 1-wrong_loss
    assert wrong_lower > D('.49')  # Yet population context/target MI is exactly zero.

paths=['src/learn/data/topics/mutual-information-information-bottleneck.jsx',
       'src/learn/data/mutual-information-models.js',
       'src/learn/data/mutual-information-examples.js',
       'src/learn/components/lesson-labs/MutualInformationLabs.jsx']
record={'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sourceFiles':[{'path':path,'sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest()} for path in paths],
        'rectangularAndRefinedLaws':len(tables),'mutualInformationBits':float(information(original)),
        'ibLocalCurvatureCases':perturbations,'changedNonuniformFreeEnergyCases':free_energy,
        'contrastiveExpectedLossCases':contrastive,'wrongNegativeSamplingLowerBound':float(wrong_lower),
        'allPassed':True}
(OUT/'results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(ROOT/'docs/teaching/evidence/mutual-information-independent-review.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'checkedAt':record['checkedAt'],'rectangular':len(tables),'ibCurvature':len(perturbations),
                  'freeEnergy':len(free_energy),'contrastive':len(contrastive),'allPassed':True},indent=2))
