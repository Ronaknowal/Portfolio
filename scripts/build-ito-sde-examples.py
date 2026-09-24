"""Author complete topic-owned programs and record their actual Python output."""
import json
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'scratch/ito-sde-verification/programs'
OUT.mkdir(parents=True,exist_ok=True)
examples = {}


def add(key,title,question,code,interpretation):
    code = textwrap.dedent(code).strip()+"\n"
    file = OUT/(key+'.py')
    file.write_text(code,encoding='utf-8')
    result = subprocess.run([sys.executable,str(file)],capture_output=True,
                            encoding='utf-8',check=True,timeout=30)
    assert not result.stderr,(key,result.stderr)
    examples[key]={'title':title,'question':question,'language':'python','code':code,
                   'expected':result.stdout.rstrip('\n'),'interpretation':interpretation}


add('units','Scale one noisy update and then change the clock',
    'A state has drift 0.3 units/second and diffusion 0.8 units/sqrt(second). How does the uncertainty change when the step is quartered?',r'''
    import math

    drift, diffusion, innovation = 0.3, 0.8, -0.5
    for step in (1.0, 0.25, 0.0625):
        systematic = drift*step
        noise = diffusion*math.sqrt(step)*innovation
        variance = diffusion**2*step
        print(f"h={step:.4f}: drift={systematic:.4f}, noise={noise:.4f}, variance={variance:.4f}")
    step_seconds = 0.25
    step_milliseconds = 1000*step_seconds
    drift_per_ms = drift/1000
    diffusion_per_sqrt_ms = diffusion/math.sqrt(1000)
    seconds = drift*step_seconds + diffusion*math.sqrt(step_seconds)*innovation
    milliseconds = (drift_per_ms*step_milliseconds
                    + diffusion_per_sqrt_ms*math.sqrt(step_milliseconds)*innovation)
    print(f"same update in seconds: {seconds:.6f}")
    print(f"same update in milliseconds: {milliseconds:.6f}")
    ''','The innovation is held fixed for this scaling comparison, not redrawn. Four independent quarter-step noise contributions have the same total variance as one full-step contribution.')

add('integrals','Calculate three sums on exactly the same observations',
    'For increments 1/2, −1/4, 3/4 and −1/2, is the sum of left-point contributions half the squared endpoint?',r'''
    from fractions import Fraction as F

    def sums(increments):
        value = left = right = symmetric = quadratic = F(0)
        for increment in increments:
            after = value+increment
            left += value*increment
            right += after*increment
            symmetric += (value+after)*increment/2
            quadratic += increment**2
            value = after
        return value,left,right,symmetric,quadratic

    increments = [F(1,2),F(-1,4),F(3,4),F(-1,2)]
    terminal,left,right,symmetric,quadratic = sums(increments)
    for name,value in [('endpoint',terminal),('left',left),('right',right),
                       ('symmetric',symmetric),('sum of squares',quadratic)]:
        print(name,str(value))
    print('left identity:',2*left == terminal**2-quadratic)
    print('right minus left:',str(right-left))
    print('Brownian Ito target for T=1:',str((terminal**2-1)/2))
    ''','These finite algebraic identities hold for any signed increments. Calling the deterministic list a Brownian sample is unnecessary; the stochastic limit is a separate statement about the random sum of squares.')

add('isometry','Check a genuinely adapted random coefficient',
    'Let the coefficient on the second interval be 0.7 plus the first Brownian increment. Does the integral still have mean zero, and what is its second moment?',r'''
    import math
    import numpy as np

    def finite_isometry(first_time,second_time,coefficient):
        if not (1/8192 <= first_time <= 4 and 1/8192 <= second_time <= 4 and math.isfinite(coefficient) and abs(coefficient) <= 3):
            raise ValueError('Use intervals in [1/8192,4] and a finite coefficient in [-3,3].')
        points,weights = np.polynomial.hermite.hermgauss(5)
        points *= math.sqrt(2)
        weights /= math.sqrt(math.pi)
        mean = second = energy = 0.0
        for i,z1 in enumerate(points):
            for j,z2 in enumerate(points):
                weight = weights[i]*weights[j]
                first = math.sqrt(first_time)*z1
                later = math.sqrt(second_time)*z2
                integral = coefficient*first+(coefficient+first)*later
                mean += weight*integral
                second += weight*integral**2
                energy += weight*(coefficient**2*first_time
                                  +(coefficient+first)**2*second_time)
        return mean,second,energy

    mean,second,energy = finite_isometry(0.25,0.75,0.7)
    print(f'mean (rounded): {0.0 if abs(mean)<1e-12 else mean:.6f}')
    print(f'integral second moment: {second:.6f}')
    print(f'expected coefficient energy: {energy:.6f}')
    print(f'analytic value: {0.7**2+0.25*0.75:.6f}')
    ''','Five-point Gaussian quadrature integrates these low-degree polynomial moments exactly apart from floating-point error. The second coefficient may depend on the first increment, but cannot inspect the second one.')

add('growth','Solve the model before summarizing samples',
    'For X0=1, μ=0.2, σ=1 and T=2, do the mean and median tell the same growth story?',r'''
    import math
    from statistics import NormalDist

    def growth_statistics(initial,mu,sigma,time):
        if not (0.1 <= initial <= 3 and -0.5 <= mu <= 1 and (sigma == 0 or 0.05 <= sigma <= 1.2) and (time == 0 or 0.0625 <= time <= 4)):
            raise ValueError('Use the declared finite teaching ranges; exact zero noise/time is supported.')
        center = math.log(initial)+(mu-sigma**2/2)*time
        spread = sigma*math.sqrt(time)
        mean = initial*math.exp(mu*time)
        median = math.exp(center)
        variance = mean**2*math.expm1(sigma**2*time)
        quantiles = [math.exp(center+spread*NormalDist().inv_cdf(p))
                     for p in (0.05,0.95)]
        return mean,median,variance,quantiles

    mean,median,variance,quantiles = growth_statistics(1,0.2,1,2)
    print(f'mean={mean:.6f}; median={median:.6f}; variance={variance:.6f}')
    print('pointwise 5th/95th percentiles:',[round(q,6) for q in quantiles])
    print(f'almost-sure long-run log rate: {0.2-1/2:.3f}')
    print(f'second-moment exponential rate: {2*0.2+1:.3f}')
    ''','The model mean grows while the median shrinks. Under the Brownian long-time law, the negative log rate makes almost every path tend to zero, even though rare large outcomes sustain a growing mean. These are model statements, not financial predictions.')

add('ou','Compute OU uncertainty and its Brownian coupling',
    'What is the law after 0.75 time units, and why does the exact one-step OU noise need more than a reused normalized Brownian draw?',r'''
    import math
    from decimal import Decimal,localcontext

    def ou_moments(theta,target,eta,mean0,variance0,time):
        if not ((theta == 0 or 0.05 <= theta <= 3) and (eta == 0 or 0.05 <= eta <= 1.2) and 0 <= variance0 <= 4 and (time == 0 or 1/8192 <= time <= 4) and abs(target) <= 2 and abs(mean0) <= 2):
            raise ValueError('Use the declared finite OU teaching ranges; exact zero rates/time are supported.')
        attenuation = math.exp(-theta*time)
        added = eta**2*time if theta == 0 else eta**2*(-math.expm1(-2*theta*time))/(2*theta)
        return target+(mean0-target)*attenuation, variance0*attenuation**2+added

    def coupled_noise(theta,step):
        if not ((theta == 0 or 0.05 <= theta <= 3) and 1/8192 <= step <= 4):
            raise ValueError('Use a nonnegative reversion rate and positive step.')
        with localcontext() as context:
            context.prec = 70
            rate,h = Decimal(str(theta)),Decimal(str(step))
            covariance = h if rate == 0 else (1-(-rate*h).exp())/rate
            variance = h if rate == 0 else (1-(-2*rate*h).exp())/(2*rate)
            residual = variance-covariance**2/h
        return float(covariance),float(variance),float(residual)

    mean,variance = ou_moments(1.3,-0.2,0.6,0.8,0.4,0.75)
    print(f'mean={mean:.6f}; variance={variance:.6f}')
    covariance,variance,residual = coupled_noise(1.3,0.25)
    print(f'Cov(deltaW,J)={covariance:.6f}; Var(J)={variance:.6f}')
    print(f'Var(J | deltaW)={residual:.9f}')
    z1,z2 = 0.4,-0.7
    delta_w = math.sqrt(0.25)*z1
    weighted_noise = covariance/0.25*delta_w+math.sqrt(residual)*z2
    print(f'deltaW={delta_w:.6f}; coupled J={weighted_noise:.6f}')
    ''','J is the exponentially weighted Brownian increment. The extra independent normal component restores the correct joint covariance with deltaW. A correct marginal transition alone is not proof of a chosen pathwise coupling.')

add('conventions','Convert the drift and preserve the model',
    'Compare dX=0.1X dt+0.8X dW with the same written equation interpreted as Stratonovich. Which Itô drift reproduces the second model?',r'''
    import math

    initial,w,time = 1.0,0.3,2.0
    written_drift,sigma = 0.1,0.8
    for name,ito_drift in [('written Ito',written_drift),
                            ('written Stratonovich',written_drift+sigma**2/2),
                            ('converted Ito',0.42)]:
        log_drift = ito_drift-sigma**2/2
        value = initial*math.exp(log_drift*time+sigma*w)
        mean = initial*math.exp(ito_drift*time)
        print(f'{name}: log drift={log_drift:.3f}, selected value={value:.6f}, mean={mean:.6f}')
    ''','The last two laws agree, including the same-noise selected state. Copying the written drift unchanged across conventions gives the first, different model. The selected W_T value is a conditional illustration, not a mean.')

add('covariance','Keep the mixed derivative when noises are correlated',
    'Let X=W1 and Y=0.6W1+0.8W2, with independent Brownian drivers. What happens to E[XY] and Var(Y−X)?',r'''
    import numpy as np

    def covariance_rate(mixing):
        mixing=np.asarray(mixing,dtype=float)
        if (mixing.ndim != 2 or not all(1 <= size <= 4 for size in mixing.shape)
                or not np.isfinite(mixing).all() or np.max(np.abs(mixing)) > 3):
            raise ValueError('Use a finite matrix of noise coefficients.')
        return mixing@mixing.T

    mixing=np.array([[1.0,0.0],[0.6,0.8]])
    rate=covariance_rate(mixing)
    time=1.5
    difference=np.array([-1.0,1.0])
    print('covariance rate:',rate.tolist())
    print(f'E[X_T Y_T]={time*rate[0,1]:.6f}')
    print(f'Var(Y_T-X_T)={time*difference@rate@difference:.6f}')
    print('For f(x,y)=xy, the two symmetric Hessian entries cancel the one-half factor.')
    ''','The drivers are independent; the state coordinates are correlated because they share W1. The covariance matrix, not the number of variable names, determines the product correction.')

archive=json.loads((ROOT/'docs/teaching/evidence/ito-calculus-original-content.json').read_text(encoding='utf-8'))
add('original','Run the original Euler-Maruyama experiment',
    'Why can this one path end below one even though the exact model mean at time one is above one?',
    archive['blocks'][0]['code'],
    'This preserves the original program and output. Its discrete approximation and one realized path are distinct from the exact model expectation. Later comparisons hold the Brownian driver fixed when changing the numerical grid.')
examples['original']['code']=archive['blocks'][0]['code']
assert examples['original']['expected']==archive['blocks'][1]['code']

add('coupling','Use one Brownian grid for every solver resolution',
    'Will the exact terminal GBM value change when the same 64 fine increments are grouped into four coarse steps?',r'''
    import math
    import random

    def solve_from_increments(increments,horizon,mu,sigma,initial):
        if (not 1 <= len(increments) <= 512 or not 0.0625 <= horizon <= 4
                or not 0.1 <= initial <= 3 or not -0.5 <= mu <= 1 or not 0 <= sigma <= 1.2
                or not all(math.isfinite(dw) and abs(dw) <= 20 for dw in increments)):
            raise ValueError('Use increments, positive time and positive start.')
        step=horizon/len(increments)
        euler=milstein=initial
        for dw in increments:
            euler *= 1+mu*step+sigma*dw
            milstein *= 1+mu*step+sigma*dw+sigma**2/2*(dw**2-step)
        exact=initial*math.exp((mu-sigma**2/2)*horizon+sigma*math.fsum(increments))
        if not all(math.isfinite(value) for value in (euler,milstein,exact)) or exact == 0:
            raise ArithmeticError('The result is outside representable arithmetic.')
        return euler,milstein,exact

    rng=random.Random(17)
    fine=[rng.gauss(0,1)/8 for _ in range(64)]
    for group in (16,4,1):
        coarse=[math.fsum(fine[start:start+group]) for start in range(0,64,group)]
        euler,milstein,exact=solve_from_increments(coarse,1,0.4,0.6,1)
        print(f'{len(coarse)} steps: EM={euler:.6f}, Milstein={milstein:.6f}, exact={exact:.6f}')
    print('Every exact endpoint uses the same summed Brownian increments.')
    print('negative EM step:',solve_from_increments([-2],1,0.4,1,1)[0])
    ''','Grouping changes numerical resolution while preserving the noise over each coarse interval. Both numerical methods can have errors; their errors need not improve monotonically on each individual path. No positivity clamp is used.')

add('errors','Calculate strong error without Monte Carlo noise',
    'For the declared GBM and shared noise, what are the terminal RMS error and the bias of the first moment at each resolution?',r'''
    from decimal import Decimal,localcontext

    def exact_errors(mu,sigma,time,steps,method='euler'):
        if (type(steps) is not int or not 1 <= steps <= 1024 or method not in ('euler','milstein')
                or not -0.5 <= mu <= 1 or not (sigma == 0 or 0.05 <= sigma <= 1.2) or not 0.25 <= time <= 2):
            raise ValueError('Use a supported method and integer step count.')
        with localcontext() as context:
            context.prec=90
            m,s,t=map(lambda x:Decimal(str(x)),(mu,sigma,time))
            h=t/steps
            v=s*s*h
            extra=v*v/2 if method=='milstein' else Decimal(0)
            second_exact=((2*m+s*s)*t).exp()
            second_numeric=((1+m*h)**2+v+extra)**steps
            cross=(m*t).exp()*(1+m*h+v+extra)**steps
            squared=second_exact+second_numeric-2*cross
            bias=(1+m*h)**steps-(m*t).exp()
            if squared < 0:
                raise ArithmeticError('Insufficient working precision.')
            return float(squared.sqrt()),float(bias)

    for steps in (4,16,64,256):
        for method in ('euler','milstein'):
            rms,bias=exact_errors(0.4,0.6,1,steps,method)
            print(f'{method}, n={steps}: RMS={rms:.8f}, mean bias={bias:.8f}')
    rms,bias=exact_errors(0,0.05,0.25,512,'milstein')
    print(f'small-error case: RMS={rms:.12e}, mean bias={bias:.1f}')
    ''','These are finite-grid analytic moments for this GBM with X0=1, calculated at high precision; no sample was used. A Monte Carlo estimate fluctuates around these targets. The mean bias is shared by EM and scalar Milstein here, although their path errors differ.')

add('generator','Recover a moment equation from the generator',
    'For an OU state with current mean 0.4 and variance 0.3, θ=1.2, target zero and η=0.8, what is the rate of change of its second moment?',r'''
    def ou_second_rate(theta,target,eta,mean,variance):
        if not (0 <= theta <= 3 and 0 <= eta <= 1.2 and abs(target) <= 2 and abs(mean) <= 2 and 0 <= variance <= 4):
            raise ValueError('Use the declared bounded finite teaching parameters.')
        second=variance+mean**2
        generator_expectation=-2*theta*(second-target*mean)+eta**2
        mean_rate=theta*(target-mean)
        variance_rate=eta**2-2*theta*variance
        moment_rate=variance_rate+2*mean*mean_rate
        return generator_expectation,moment_rate

    generator,moments=ou_second_rate(1.2,0,0.8,0.4,0.3)
    print(f'from E[L(x^2)]: {generator:.6f}')
    print(f'from variance plus mean squared: {moments:.6f}')
    print('The density can change while a selected moment decreases.')
    ''','For f(x)=x², the generator is 2x times the drift plus η². Taking expectation and separately differentiating variance plus mean² produce the same result. This is an exact moment identity, not a solved numerical density PDE.')

add('tilt','Change probabilities through a normalized weight',
    'With T=0.7 and c=0.4, what terminal Brownian mean results from weighting paths by exp(c W_T−c²T/2)?',r'''
    import math
    import numpy as np

    time,shift=0.7,0.4
    nodes,weights=np.polynomial.hermite.hermgauss(48)
    terminal=math.sqrt(2*time)*nodes
    base=weights/math.sqrt(math.pi)
    likelihood=np.exp(shift*terminal-shift**2*time/2)
    weighted=base*likelihood
    total=float(np.sum(weighted))
    mean=float(np.sum(weighted*terminal))
    variance=float(np.sum(weighted*(terminal-mean)**2))
    print(f'total weight: {total:.9f}')
    print(f'new mean: {mean:.9f}; target: {shift*time:.9f}')
    print(f'new variance: {variance:.9f}; target: {time:.9f}')
    print('This finite-horizon constant shift is normalized by the Gaussian moment formula.')
    ''','The coordinate W_T has not been replaced by a new observed value; its probability law has been reweighted. Gaussian quadrature checks the derived identity. A general random drift change needs a true-martingale condition, not just a formally written exponential.')

add('time_transform','Cancel a drift with a time-dependent transformation',
    'For f(t,x)=x³−3tx, does averaging the transformed future Brownian state recover the transformed present state?',r'''
    import math
    import numpy as np

    def conditional_transform(time,state,step):
        if not (0 <= time <= 4 and abs(state) <= 3 and 1/8192 <= step <= 4):
            raise ValueError('Use finite bounded time, state and positive step.')
        nodes,weights=np.polynomial.hermite.hermgauss(5)
        later=state+math.sqrt(2*step)*nodes
        future=later**3-3*(time+step)*later
        expectation=float(weights@future)/math.sqrt(math.pi)
        current=state**3-3*time*state
        return current,expectation

    current,future=conditional_transform(0.4,-0.7,0.3)
    print(f'current transformed value: {current:.9f}')
    print(f'conditional future mean: {future:.9f}')
    print('Ito drift: f_t + f_xx/2 = -3x + 3x = 0')
    print('diffusion coefficient: f_x = 3x^2 - 3t')
    print('For W0=0: E[(W_T^3-3*T*W_T)^2] = 6*T^3')
    ''','The time derivative is essential: W³ alone has drift 3W. This low-degree Gaussian quadrature is exact apart from floating-point error. The displayed finite-horizon second moment supplies the integrability check for a true martingale.')

target=ROOT/'src/learn/data/ito-sde-examples.js'
target.write_text('// Complete executed Python examples; original program preserved exactly.\n'
                  +'export const itoSdeExamples = '+json.dumps(examples,indent=2,ensure_ascii=False)+';\n',encoding='utf-8')
print(f'Wrote {len(examples)} complete programs; all exact stdout captured.')
