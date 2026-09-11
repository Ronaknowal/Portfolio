"""Execute complete displayed programs and save their actual stdout."""
import contextlib
import ast
import black
import io
import json
import re
from pathlib import Path

EXAMPLES = {}


def add(name, title, question, code, interpretation):
    original_ast = ast.dump(ast.parse(code))
    code = black.format_str(code, mode=black.Mode(line_length=78))
    assert ast.dump(ast.parse(code)) == original_ast
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, '<random-variables-' + name + '>', 'exec'), {'__name__': '__main__'})
    def readable(text):
        return re.sub(r'([A-Za-z]{2,})(?=[0-9−])', r'\1 ', text)
    EXAMPLES[name] = dict(title=readable(title), question=readable(question), language='python', code=code.strip() + '\n', expected=output.getvalue().rstrip(), interpretation=readable(interpretation))


add('mapping', 'Collect the probability of every preimage',
    'If the two head probabilities differ, are the three possible head counts equally likely?', '''
from fractions import Fraction as Q
from itertools import product

def pushforward(outcomes, value_of):
    result = {}
    for outcome, mass in outcomes:
        value = value_of(outcome)
        result[value] = result.get(value, Q()) + mass
    return result

p, q = Q(1, 2), Q(1, 4)
outcomes = [(bits, (p if bits[0] else 1-p) * (q if bits[1] else 1-q))
            for bits in product([0, 1], repeat=2)]
law = pushforward(outcomes, sum)
print('head-count law:', {x: str(mass) for x, mass in sorted(law.items())})
print('total:', sum(law.values()))
print('P(X <= 1):', sum(mass for x, mass in law.items() if x <= 1))
print('expected heads:', sum(x * mass for x, mass in law.items()))
''', 'The two outcomes with one head contribute to the same mass. The distribution is induced by a fixed function; it is not a second random experiment.')

add('moments', 'Compare a prediction with the whole distribution',
    'Does squaring the mean produce the expected squared value? Which constant minimizes squared loss?', '''
from fractions import Fraction as Q

def expectation(values, masses):
    if len(values) != len(masses) or not values or any(p < 0 for p in masses) or sum(masses) != 1:
        raise ValueError('Use matching finite values and normalized nonnegative masses.')
    return sum((x*p for x, p in zip(values, masses)), Q())

values, masses = [-2, 0, 3], [Q(1,4), Q(1,2), Q(1,4)]
mu = expectation(values, masses)
second = expectation([x*x for x in values], masses)
variance = expectation([(x-mu)**2 for x in values], masses)
print('mean / second moment / squared mean:', mu, second, mu*mu)
print('variance:', variance)
for c in [Q(0), mu, Q(1)]:
    loss = expectation([(x-c)**2 for x in values], masses)
    assert loss == variance + (c-mu)**2
    print('prediction / loss:', c, loss)
print('after Y=1000X+7: mean / variance:', 1000*mu+7, 1000**2*variance)
''', 'The minimum loss is the population variance, reached at the mean. A change from seconds to milliseconds multiplies variance by one million, while an offset changes only the mean.')

add('indicators', 'Count successes even when trials depend on each other',
    'When two of four objects are marked and two are drawn without replacement, what changes relative to replacement?', '''
from fractions import Fraction as Q
from itertools import permutations, product

def describe(draws):
    pairs = [(int(a < 2), int(b < 2)) for a, b in draws]
    weight = Q(1, len(pairs))
    mean = sum((a+b)*weight for a,b in pairs)
    variance = sum((a+b-mean)**2*weight for a,b in pairs)
    covariance = sum(a*b*weight for a,b in pairs) - Q(1,4)
    return mean, variance, covariance

print('replacement:', describe(list(product(range(4), repeat=2))))
print('without replacement:', describe(list(permutations(range(4), 2))))
''', 'Both count means are one because each indicator still has mean one half. The negative covariance without replacement reduces the count variance; linearity alone never required independence.')

add('joint', 'Keep the joint law, not just its marginals',
    'Can matching, opposite and independent pairings share the same individual distributions?', '''
from fractions import Fraction as Q
from itertools import product

def moments(rows):
    mx = sum(x*p for x,y,p in rows)
    my = sum(y*p for x,y,p in rows)
    vx = sum((x-mx)**2*p for x,y,p in rows)
    vy = sum((y-my)**2*p for x,y,p in rows)
    covariance = sum((x-mx)*(y-my)*p for x,y,p in rows)
    return mx, my, vx, vy, covariance

laws = {
    'matching': [(x,x,Q(1,3)) for x in [-1,0,1]],
    'opposite': [(x,-x,Q(1,3)) for x in [-1,0,1]],
    'independent': [(x,y,Q(1,9)) for x,y in product([-1,0,1], repeat=2)],
    'nonlinear': [(x,x*x,Q(1,3)) for x in [-1,0,1]],
}
for name, rows in laws.items():
    print(name, 'means, variances, covariance:', tuple(map(str, moments(rows))))
print('nonlinear P(X=0,Y=1):', Q(0))
print('product of its marginals:', Q(1,3)*Q(2,3))
''', 'The first three laws have the same means and variances but different covariance. The last has zero covariance even though Y is determined by X; a joint cell with mass zero versus marginal product 2/9 is a direct independence counterexample.')

add('noise', 'Trace common error through an average and a difference',
    'Which part of the noise survives averaging two readings, and which part cancels in their difference?', '''
from fractions import Fraction as Q
from itertools import product

def noise_law(common=Q(2), local=Q(1)):
    return [(10+common*s+local*e, 20+common*s+local*f)
            for s,e,f in product([-1,1], repeat=3)]

def mean(values):
    return sum(values, Q()) / len(values)

def variance(values):
    mu = mean(values)
    return mean([(x-mu)**2 for x in values])

rows = noise_law()
a, b = [row[0] for row in rows], [row[1] for row in rows]
covariance = mean([(x-mean(a))*(y-mean(b)) for x,y in rows])
print('means:', mean(a), mean(b))
print('variances / covariance:', variance(a), variance(b), covariance)
print('average mean / variance:', mean([(x+y)/2 for x,y in rows]), variance([(x+y)/2 for x,y in rows]))
print('difference mean / variance:', mean([y-x for x,y in rows]), variance([y-x for x,y in rows]))
for common in [Q(0), Q(3)]:
    changed = noise_law(common, Q(1))
    print('changed common / difference variance:', common, variance([y-x for x,y in changed]))
''', 'These are exact synthetic populations in mV, not measured device performance. The average targets a different baseline from either original reading. Differencing eliminates the common term algebraically, but it does not eliminate independent local error.')

add('matrix', 'Propagate a covariance matrix and reject an impossible one',
    'What covariance does the pair (average, difference) have? Why can [[1,2],[2,1]] not be a covariance matrix?', '''
from fractions import Fraction as Q

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def multiply(a, b):
    if not a or not b or len(a[0]) != len(b):
        raise ValueError('Incompatible nonempty matrices.')
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))), Q())
             for j in range(len(b[0]))] for i in range(len(a))]

sigma = [[Q(5),Q(4)], [Q(4),Q(5)]]
transform = [[Q(1,2),Q(1,2)], [Q(-1),Q(1)]]
changed = multiply(multiply(transform, sigma), transpose(transform))
print('covariance of (average,difference):', [[str(x) for x in row] for row in changed])
bad = [[Q(1),Q(2)], [Q(2),Q(1)]]
direction = [[Q(1),Q(-1)]]
print('impossible variance:', multiply(multiply(direction,bad),transpose(direction))[0][0])
''', 'A covariance matrix represents variances of every linear combination. A negative value for one such variance refutes the proposed matrix. The transformed off-diagonal zero is a lack of linear covariance, not a general proof of independence.')

add('conditioning', 'Separate within-group uncertainty from changing group means',
    'How can both groups have covariance −1 while their pooled covariance is positive?', '''
from fractions import Fraction as Q
from itertools import product

def moments(rows):
    total = sum(p for x,y,p in rows)
    if total == 0:
        return None
    mx = sum(x*p for x,y,p in rows)/total
    my = sum(y*p for x,y,p in rows)/total
    vy = sum((y-my)**2*p for x,y,p in rows)/total
    cov = sum((x-mx)*(y-my)*p for x,y,p in rows)/total
    return mx,my,vy,cov

def decomposition(positive_probability):
    rows = [(g,g+u,g-u,(positive_probability if g==2 else 1-positive_probability)/2)
            for g,u in product([-2,2],[-1,1])]
    overall = moments([(x,y,p) for g,x,y,p in rows])
    groups = [(sum(p for h,x,y,p in rows if h==g),
               moments([(x,y,p) for h,x,y,p in rows if h==g])) for g in [-2,2]]
    within_v = sum(weight*m[2] for weight,m in groups if m is not None)
    between_v = sum(weight*(m[1]-overall[1])**2 for weight,m in groups if m is not None)
    within_c = sum(weight*m[3] for weight,m in groups if m is not None)
    between_c = sum(weight*(m[0]-overall[0])*(m[1]-overall[1]) for weight,m in groups if m is not None)
    assert overall[2] == within_v+between_v
    assert overall[3] == within_c+between_c
    return overall, (within_v,between_v), (within_c,between_c), groups

for p in [Q(1,2),Q(1,4),Q(0)]:
    overall, variances, covariances, groups = decomposition(p)
    print('positive group probability:', p)
    print('Y variance = within + between:', overall[2], tuple(map(str,variances)))
    print('covariance = within + between:', overall[3], tuple(map(str,covariances)))
    print('undefined groups:', sum(m is None for weight,m in groups))
''', 'A group with zero probability has no conditional law identified by division. The positive pooled covariance is a between-group effect in this specified population; these calculations alone do not identify a causal effect.')

add('prediction', 'Distinguish the best line from the best informed prediction',
    'If X is symmetric and Y=X², can a straight-line predictor use X effectively?', '''
from fractions import Fraction as Q

rows = [(Q(x),Q(x*x),Q(1,3)) for x in [-1,0,1]]
mx = sum(x*p for x,y,p in rows)
my = sum(y*p for x,y,p in rows)
variance_x = sum((x-mx)**2*p for x,y,p in rows)
covariance = sum((x-mx)*(y-my)*p for x,y,p in rows)
slope = covariance/variance_x
intercept = my-slope*mx
linear_risk = sum((y-(intercept+slope*x))**2*p for x,y,p in rows)
conditional_risk = sum((y-x*x)**2*p for x,y,p in rows)
print('best affine intercept / slope:', intercept, slope)
print('affine risk:', linear_risk)
print('conditional-mean risk:', conditional_risk)
''', 'The best affine predictor is constant because covariance is zero. The full conditional mean X² predicts Y perfectly in this model. A restriction to straight lines is an extra modeling decision.')

add('continuous', 'Fold a continuous interval without losing a branch',
    'For U uniform on [−1,1], which U values make 0.25≤U²≤0.81? What is their total probability?', '''
from math import sqrt
from fractions import Fraction as Q

def squared_uniform_interval(a, b):
    if not 0 <= a <= b <= 1:
        raise ValueError('Use 0 <= a <= b <= 1.')
    return [(-sqrt(b),-sqrt(a)), (sqrt(a),sqrt(b))], sqrt(b)-sqrt(a)

intervals, mass = squared_uniform_interval(.25,.81)
print('two preimage intervals:', intervals)
print('interval probability:', round(mass,12))
# Direct LOTUS: E[U^(2k)] = (1/2)*integral(-1,1) u^(2k) du.
def moment_y(k):
    if not isinstance(k,int) or k < 0:
        raise ValueError('Use a nonnegative integer moment.')
    return Q(1,2*k+1)

print('mean / variance:', moment_y(1), moment_y(2)-moment_y(1)**2)
print('zero-width probability:', squared_uniform_interval(.25,.25)[1])
''', 'The two input intervals each have length0.4 and uniform density1/2, giving total probability0.4. The transformed density diverges near zero but its point mass remains zero.')

add('sample_mean', 'Compare independent readings with copied readings',
    'Does writing one observed bit eight times reduce uncertainty about its underlying mean?', '''
from fractions import Fraction as Q
from math import comb

def mean_law(n,p):
    if not isinstance(n,int) or not 1 <= n <= 30 or not 0 <= p <= 1:
        raise ValueError('Use 1 <= integer n <= 30 and a probability p.')
    return [(Q(k,n),Q(comb(n,k))*p**k*(1-p)**(n-k)) for k in range(n+1)]

for n in [1,4,8]:
    p = Q(1,4)
    rows = mean_law(n,p)
    assert sum(mass for value,mass in rows) == 1
    variance = sum((value-p)**2*mass for value,mass in rows)
    print('n / independent variance / copied variance:', n,variance,p*(1-p))
print('n=4 exact law:', [(str(x),str(p)) for x,p in mean_law(4,Q(1,4))])
''', 'The distribution of the independent average narrows as n grows. Copies retain the original bit distribution. These are exact distributions over repeated datasets, not a claim about the appearance of one finite dataset.')

add('sample_covariance', 'Check what the n−1 correction actually corrects',
    'Why is the average squared distance to the sample mean too small on average under iid sampling?', '''
from fractions import Fraction as Q
from itertools import product

def sample_covariance(xs,ys,ddof=1):
    n=len(xs)
    if n != len(ys) or n <= ddof or ddof not in (0,1):
        raise ValueError('Matching paired samples need n > ddof, ddof 0 or1.')
    mx,my=sum(xs,Q())/n,sum(ys,Q())/n
    return sum(((x-mx)*(y-my) for x,y in zip(xs,ys)),Q())/(n-ddof)

xs=list(map(Q,[1,2,4]))
print('one dataset: divide by n / n-1:', sample_covariance(xs,xs,0),sample_covariance(xs,xs,1))
independent=list(product([Q(0),Q(1)],repeat=3))
for ddof in [0,1]:
    expected=sum(sample_covariance(row,row,ddof) for row in independent)/len(independent)
    print('iid expected sample variance, ddof',ddof,':',expected)
copied=[(Q(0),)*3,(Q(1),)*3]
print('copied expected corrected variance:',sum(sample_covariance(row,row) for row in copied)/2)
''', 'Dividing by n−1 is unbiased for the stated iid model with finite second moments. Every copied dataset has sample variance zero, even though the marginal population variance is1/4; the denominator cannot repair dependence.')

add('stable', 'Avoid subtracting two nearly equal large moments',
    'Can shifting every reading by a trillion change its true variance?', '''
from fractions import Fraction as Q

def welford(values):
    n=0
    mean=0.0
    m2=0.0
    for value in values:
        n+=1
        delta=value-mean
        mean+=delta/n
        m2+=delta*(value-mean)
    if not n:
        raise ValueError('At least one value is required.')
    return mean,m2/n

values=[10**12-1,10**12,10**12+1]
exact_mean=sum(map(Q,values))/3
exact_variance=sum((Q(x)-exact_mean)**2 for x in values)/3
mean,variance=welford(values)
print('mean:',mean)
print('exact variance:',exact_variance)
print('Welford variance:',round(variance,12))
print('population normalization n:',len(values))
''', 'Centering avoids the particular cancellation in E[X²]−E[X]². This finite calculation does not make floating-point arithmetic exact or establish safe behavior for arbitrarily large inputs. Population normalization is explicit; use n−1 only for the corresponding sample-estimator question.')

add('tails', 'Check the integrals before naming a moment',
    'Can a positive random variable have a finite mean but an infinite second moment?', '''
from fractions import Fraction as Q

# Pareto density alpha*x^(-alpha-1), x>=1.
# Integral alpha*x^(k-alpha-1) converges exactly when k<alpha.
def pareto_moment(alpha,k):
    if alpha <= 0 or k < 0:
        raise ValueError('Use alpha>0 and k>=0.')
    return alpha/(alpha-k) if k < alpha else None

for alpha in [Q(1),Q(3,2),Q(3)]:
    mean,second=pareto_moment(alpha,1),pareto_moment(alpha,2)
    variance=second-mean*mean if second is not None else None
    print('alpha / mean / second / variance:',alpha,mean,second,variance)
''', 'None marks a divergent nonnegative moment, not a missing numerical integration result. At alpha=3/2 the mean is3 but the second moment diverges. A finite sample can still return finite numbers, which do not establish finite population moments.')

add('dice', 'Solve a changed joint-variable problem',
    'For two independent fair dice, let S be their sum and D their difference. Does zero covariance make S and D independent?', '''
from fractions import Fraction as Q
from itertools import product

rows=[(a+b,a-b,Q(1,36)) for a,b in product(range(1,7),repeat=2)]
ms=sum(s*p for s,d,p in rows);md=sum(d*p for s,d,p in rows)
vs=sum((s-ms)**2*p for s,d,p in rows)
vd=sum((d-md)**2*p for s,d,p in rows)
cov=sum((s-ms)*(d-md)*p for s,d,p in rows)
joint=sum(p for s,d,p in rows if s==2 and d==0)
product_mass=sum(p for s,d,p in rows if s==2)*sum(p for s,d,p in rows if d==0)
print('means:',ms,md)
print('variances / covariance:',vs,vd,cov)
print('P(S=2,D=0) versus product:',joint,product_mass)
''', 'The zero covariance follows from equal die variances and cancellation. The joint cell differs from the product of marginals, so these dice-derived variables are dependent. This is an independently solvable transfer of the pairing principle.')

path = Path('src/learn/data/random-variables-examples.js')
path.write_text('// Complete programs; outputs captured by the topic builder.\nexport const randomVariableExamples = ' + json.dumps(EXAMPLES, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
print(json.dumps({'programsExecuted':len(EXAMPLES),'path':str(path)}))
