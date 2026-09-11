// Complete executed Python examples; original program preserved exactly.
export const itoSdeExamples = {
  "units": {
    "title": "Scale one noisy update and then change the clock",
    "question": "A state has drift 0.3 units/second and diffusion 0.8 units/sqrt(second). How does the uncertainty change when the step is quartered?",
    "language": "python",
    "code": "import math\n\ndrift, diffusion, innovation = 0.3, 0.8, -0.5\nfor step in (1.0, 0.25, 0.0625):\n    systematic = drift*step\n    noise = diffusion*math.sqrt(step)*innovation\n    variance = diffusion**2*step\n    print(f\"h={step:.4f}: drift={systematic:.4f}, noise={noise:.4f}, variance={variance:.4f}\")\nstep_seconds = 0.25\nstep_milliseconds = 1000*step_seconds\ndrift_per_ms = drift/1000\ndiffusion_per_sqrt_ms = diffusion/math.sqrt(1000)\nseconds = drift*step_seconds + diffusion*math.sqrt(step_seconds)*innovation\nmilliseconds = (drift_per_ms*step_milliseconds\n                + diffusion_per_sqrt_ms*math.sqrt(step_milliseconds)*innovation)\nprint(f\"same update in seconds: {seconds:.6f}\")\nprint(f\"same update in milliseconds: {milliseconds:.6f}\")\n",
    "expected": "h=1.0000: drift=0.3000, noise=-0.4000, variance=0.6400\nh=0.2500: drift=0.0750, noise=-0.2000, variance=0.1600\nh=0.0625: drift=0.0187, noise=-0.1000, variance=0.0400\nsame update in seconds: -0.125000\nsame update in milliseconds: -0.125000",
    "interpretation": "The innovation is held fixed for this scaling comparison, not redrawn. Four independent quarter-step noise contributions have the same total variance as one full-step contribution."
  },
  "integrals": {
    "title": "Calculate three sums on exactly the same observations",
    "question": "For increments 1/2, −1/4, 3/4 and −1/2, is the sum of left-point contributions half the squared endpoint?",
    "language": "python",
    "code": "from fractions import Fraction as F\n\ndef sums(increments):\n    value = left = right = symmetric = quadratic = F(0)\n    for increment in increments:\n        after = value+increment\n        left += value*increment\n        right += after*increment\n        symmetric += (value+after)*increment/2\n        quadratic += increment**2\n        value = after\n    return value,left,right,symmetric,quadratic\n\nincrements = [F(1,2),F(-1,4),F(3,4),F(-1,2)]\nterminal,left,right,symmetric,quadratic = sums(increments)\nfor name,value in [('endpoint',terminal),('left',left),('right',right),\n                   ('symmetric',symmetric),('sum of squares',quadratic)]:\n    print(name,str(value))\nprint('left identity:',2*left == terminal**2-quadratic)\nprint('right minus left:',str(right-left))\nprint('Brownian Ito target for T=1:',str((terminal**2-1)/2))\n",
    "expected": "endpoint 1/2\nleft -7/16\nright 11/16\nsymmetric 1/8\nsum of squares 9/8\nleft identity: True\nright minus left: 9/8\nBrownian Ito target for T=1: -3/8",
    "interpretation": "These finite algebraic identities hold for any signed increments. Calling the deterministic list a Brownian sample is unnecessary; the stochastic limit is a separate statement about the random sum of squares."
  },
  "isometry": {
    "title": "Check a genuinely adapted random coefficient",
    "question": "Let the coefficient on the second interval be 0.7 plus the first Brownian increment. Does the integral still have mean zero, and what is its second moment?",
    "language": "python",
    "code": "import math\nimport numpy as np\n\ndef finite_isometry(first_time,second_time,coefficient):\n    if not (1/8192 <= first_time <= 4 and 1/8192 <= second_time <= 4 and math.isfinite(coefficient) and abs(coefficient) <= 3):\n        raise ValueError('Use intervals in [1/8192,4] and a finite coefficient in [-3,3].')\n    points,weights = np.polynomial.hermite.hermgauss(5)\n    points *= math.sqrt(2)\n    weights /= math.sqrt(math.pi)\n    mean = second = energy = 0.0\n    for i,z1 in enumerate(points):\n        for j,z2 in enumerate(points):\n            weight = weights[i]*weights[j]\n            first = math.sqrt(first_time)*z1\n            later = math.sqrt(second_time)*z2\n            integral = coefficient*first+(coefficient+first)*later\n            mean += weight*integral\n            second += weight*integral**2\n            energy += weight*(coefficient**2*first_time\n                              +(coefficient+first)**2*second_time)\n    return mean,second,energy\n\nmean,second,energy = finite_isometry(0.25,0.75,0.7)\nprint(f'mean (rounded): {0.0 if abs(mean)<1e-12 else mean:.6f}')\nprint(f'integral second moment: {second:.6f}')\nprint(f'expected coefficient energy: {energy:.6f}')\nprint(f'analytic value: {0.7**2+0.25*0.75:.6f}')\n",
    "expected": "mean (rounded): 0.000000\nintegral second moment: 0.677500\nexpected coefficient energy: 0.677500\nanalytic value: 0.677500",
    "interpretation": "Five-point Gaussian quadrature integrates these low-degree polynomial moments exactly apart from floating-point error. The second coefficient may depend on the first increment, but cannot inspect the second one."
  },
  "growth": {
    "title": "Solve the model before summarizing samples",
    "question": "For X0=1, μ=0.2, σ=1 and T=2, do the mean and median tell the same growth story?",
    "language": "python",
    "code": "import math\nfrom statistics import NormalDist\n\ndef growth_statistics(initial,mu,sigma,time):\n    if not (0.1 <= initial <= 3 and -0.5 <= mu <= 1 and (sigma == 0 or 0.05 <= sigma <= 1.2) and (time == 0 or 0.0625 <= time <= 4)):\n        raise ValueError('Use the declared finite teaching ranges; exact zero noise/time is supported.')\n    center = math.log(initial)+(mu-sigma**2/2)*time\n    spread = sigma*math.sqrt(time)\n    mean = initial*math.exp(mu*time)\n    median = math.exp(center)\n    variance = mean**2*math.expm1(sigma**2*time)\n    quantiles = [math.exp(center+spread*NormalDist().inv_cdf(p))\n                 for p in (0.05,0.95)]\n    return mean,median,variance,quantiles\n\nmean,median,variance,quantiles = growth_statistics(1,0.2,1,2)\nprint(f'mean={mean:.6f}; median={median:.6f}; variance={variance:.6f}')\nprint('pointwise 5th/95th percentiles:',[round(q,6) for q in quantiles])\nprint(f'almost-sure long-run log rate: {0.2-1/2:.3f}')\nprint(f'second-moment exponential rate: {2*0.2+1:.3f}')\n",
    "expected": "mean=1.491825; median=0.548812; variance=14.219106\npointwise 5th/95th percentiles: [0.053602, 5.619116]\nalmost-sure long-run log rate: -0.300\nsecond-moment exponential rate: 1.400",
    "interpretation": "The model mean grows while the median shrinks. Under the Brownian long-time law, the negative log rate makes almost every path tend to zero, even though rare large outcomes sustain a growing mean. These are model statements, not financial predictions."
  },
  "ou": {
    "title": "Compute OU uncertainty and its Brownian coupling",
    "question": "What is the law after 0.75 time units, and why does the exact one-step OU noise need more than a reused normalized Brownian draw?",
    "language": "python",
    "code": "import math\nfrom decimal import Decimal,localcontext\n\ndef ou_moments(theta,target,eta,mean0,variance0,time):\n    if not ((theta == 0 or 0.05 <= theta <= 3) and (eta == 0 or 0.05 <= eta <= 1.2) and 0 <= variance0 <= 4 and (time == 0 or 1/8192 <= time <= 4) and abs(target) <= 2 and abs(mean0) <= 2):\n        raise ValueError('Use the declared finite OU teaching ranges; exact zero rates/time are supported.')\n    attenuation = math.exp(-theta*time)\n    added = eta**2*time if theta == 0 else eta**2*(-math.expm1(-2*theta*time))/(2*theta)\n    return target+(mean0-target)*attenuation, variance0*attenuation**2+added\n\ndef coupled_noise(theta,step):\n    if not ((theta == 0 or 0.05 <= theta <= 3) and 1/8192 <= step <= 4):\n        raise ValueError('Use a nonnegative reversion rate and positive step.')\n    with localcontext() as context:\n        context.prec = 70\n        rate,h = Decimal(str(theta)),Decimal(str(step))\n        covariance = h if rate == 0 else (1-(-rate*h).exp())/rate\n        variance = h if rate == 0 else (1-(-2*rate*h).exp())/(2*rate)\n        residual = variance-covariance**2/h\n    return float(covariance),float(variance),float(residual)\n\nmean,variance = ou_moments(1.3,-0.2,0.6,0.8,0.4,0.75)\nprint(f'mean={mean:.6f}; variance={variance:.6f}')\ncovariance,variance,residual = coupled_noise(1.3,0.25)\nprint(f'Cov(deltaW,J)={covariance:.6f}; Var(J)={variance:.6f}')\nprint(f'Var(J | deltaW)={residual:.9f}')\nz1,z2 = 0.4,-0.7\ndelta_w = math.sqrt(0.25)*z1\nweighted_noise = covariance/0.25*delta_w+math.sqrt(residual)*z2\nprint(f'deltaW={delta_w:.6f}; coupled J={weighted_noise:.6f}')\n",
    "expected": "mean=0.177192; variance=0.175672\nCov(deltaW,J)=0.213440; Var(J)=0.183829\nVar(J | deltaW)=0.001601164\ndeltaW=0.200000; coupled J=0.142742",
    "interpretation": "J is the exponentially weighted Brownian increment. The extra independent normal component restores the correct joint covariance with deltaW. A correct marginal transition alone is not proof of a chosen pathwise coupling."
  },
  "conventions": {
    "title": "Convert the drift and preserve the model",
    "question": "Compare dX=0.1X dt+0.8X dW with the same written equation interpreted as Stratonovich. Which Itô drift reproduces the second model?",
    "language": "python",
    "code": "import math\n\ninitial,w,time = 1.0,0.3,2.0\nwritten_drift,sigma = 0.1,0.8\nfor name,ito_drift in [('written Ito',written_drift),\n                        ('written Stratonovich',written_drift+sigma**2/2),\n                        ('converted Ito',0.42)]:\n    log_drift = ito_drift-sigma**2/2\n    value = initial*math.exp(log_drift*time+sigma*w)\n    mean = initial*math.exp(ito_drift*time)\n    print(f'{name}: log drift={log_drift:.3f}, selected value={value:.6f}, mean={mean:.6f}')\n",
    "expected": "written Ito: log drift=-0.220, selected value=0.818731, mean=1.221403\nwritten Stratonovich: log drift=0.100, selected value=1.552707, mean=2.316367\nconverted Ito: log drift=0.100, selected value=1.552707, mean=2.316367",
    "interpretation": "The last two laws agree, including the same-noise selected state. Copying the written drift unchanged across conventions gives the first, different model. The selected W_T value is a conditional illustration, not a mean."
  },
  "covariance": {
    "title": "Keep the mixed derivative when noises are correlated",
    "question": "Let X=W1 and Y=0.6W1+0.8W2, with independent Brownian drivers. What happens to E[XY] and Var(Y−X)?",
    "language": "python",
    "code": "import numpy as np\n\ndef covariance_rate(mixing):\n    mixing=np.asarray(mixing,dtype=float)\n    if (mixing.ndim != 2 or not all(1 <= size <= 4 for size in mixing.shape)\n            or not np.isfinite(mixing).all() or np.max(np.abs(mixing)) > 3):\n        raise ValueError('Use a finite matrix of noise coefficients.')\n    return mixing@mixing.T\n\nmixing=np.array([[1.0,0.0],[0.6,0.8]])\nrate=covariance_rate(mixing)\ntime=1.5\ndifference=np.array([-1.0,1.0])\nprint('covariance rate:',rate.tolist())\nprint(f'E[X_T Y_T]={time*rate[0,1]:.6f}')\nprint(f'Var(Y_T-X_T)={time*difference@rate@difference:.6f}')\nprint('For f(x,y)=xy, the two symmetric Hessian entries cancel the one-half factor.')\n",
    "expected": "covariance rate: [[1.0, 0.6], [0.6, 1.0]]\nE[X_T Y_T]=0.900000\nVar(Y_T-X_T)=1.200000\nFor f(x,y)=xy, the two symmetric Hessian entries cancel the one-half factor.",
    "interpretation": "The drivers are independent; the state coordinates are correlated because they share W1. The covariance matrix, not the number of variable names, determines the product correction."
  },
  "original": {
    "title": "Run the original Euler-Maruyama experiment",
    "question": "Why can this one path end below one even though the exact model mean at time one is above one?",
    "language": "python",
    "code": "import math\nimport random\n\nrng = random.Random(5)\nx, mu, sigma, dt = 1.0, 0.4, 0.3, 0.25\npath = [x]\n\nfor _ in range(4):\n    z = rng.gauss(0, 1)\n    x += mu * x * dt + sigma * x * math.sqrt(dt) * z\n    path.append(round(x, 3))\n\nprint(path)\nprint(round(math.exp(mu), 3))  # E[X_1] for X_0 = 1 in this GBM model",
    "expected": "[1.0, 0.923, 0.856, 1.028, 0.777]\n1.492",
    "interpretation": "This preserves the original program and output. Its discrete approximation and one realized path are distinct from the exact model expectation. Later comparisons hold the Brownian driver fixed when changing the numerical grid."
  },
  "coupling": {
    "title": "Use one Brownian grid for every solver resolution",
    "question": "Will the exact terminal GBM value change when the same 64 fine increments are grouped into four coarse steps?",
    "language": "python",
    "code": "import math\nimport random\n\ndef solve_from_increments(increments,horizon,mu,sigma,initial):\n    if (not 1 <= len(increments) <= 512 or not 0.0625 <= horizon <= 4\n            or not 0.1 <= initial <= 3 or not -0.5 <= mu <= 1 or not 0 <= sigma <= 1.2\n            or not all(math.isfinite(dw) and abs(dw) <= 20 for dw in increments)):\n        raise ValueError('Use increments, positive time and positive start.')\n    step=horizon/len(increments)\n    euler=milstein=initial\n    for dw in increments:\n        euler *= 1+mu*step+sigma*dw\n        milstein *= 1+mu*step+sigma*dw+sigma**2/2*(dw**2-step)\n    exact=initial*math.exp((mu-sigma**2/2)*horizon+sigma*math.fsum(increments))\n    if not all(math.isfinite(value) for value in (euler,milstein,exact)) or exact == 0:\n        raise ArithmeticError('The result is outside representable arithmetic.')\n    return euler,milstein,exact\n\nrng=random.Random(17)\nfine=[rng.gauss(0,1)/8 for _ in range(64)]\nfor group in (16,4,1):\n    coarse=[math.fsum(fine[start:start+group]) for start in range(0,64,group)]\n    euler,milstein,exact=solve_from_increments(coarse,1,0.4,0.6,1)\n    print(f'{len(coarse)} steps: EM={euler:.6f}, Milstein={milstein:.6f}, exact={exact:.6f}')\nprint('Every exact endpoint uses the same summed Brownian increments.')\nprint('negative EM step:',solve_from_increments([-2],1,0.4,1,1)[0])\n",
    "expected": "4 steps: EM=1.190663, Milstein=1.092875, exact=1.078167\n16 steps: EM=1.087403, Milstein=1.086825, exact=1.078167\n64 steps: EM=0.969409, Milstein=1.085464, exact=1.078167\nEvery exact endpoint uses the same summed Brownian increments.\nnegative EM step: -0.6000000000000001",
    "interpretation": "Grouping changes numerical resolution while preserving the noise over each coarse interval. Both numerical methods can have errors; their errors need not improve monotonically on each individual path. No positivity clamp is used."
  },
  "errors": {
    "title": "Calculate strong error without Monte Carlo noise",
    "question": "For the declared GBM and shared noise, what are the terminal RMS error and the bias of the first moment at each resolution?",
    "language": "python",
    "code": "from decimal import Decimal,localcontext\n\ndef exact_errors(mu,sigma,time,steps,method='euler'):\n    if (type(steps) is not int or not 1 <= steps <= 1024 or method not in ('euler','milstein')\n            or not -0.5 <= mu <= 1 or not (sigma == 0 or 0.05 <= sigma <= 1.2) or not 0.25 <= time <= 2):\n        raise ValueError('Use a supported method and integer step count.')\n    with localcontext() as context:\n        context.prec=90\n        m,s,t=map(lambda x:Decimal(str(x)),(mu,sigma,time))\n        h=t/steps\n        v=s*s*h\n        extra=v*v/2 if method=='milstein' else Decimal(0)\n        second_exact=((2*m+s*s)*t).exp()\n        second_numeric=((1+m*h)**2+v+extra)**steps\n        cross=(m*t).exp()*(1+m*h+v+extra)**steps\n        squared=second_exact+second_numeric-2*cross\n        bias=(1+m*h)**steps-(m*t).exp()\n        if squared < 0:\n            raise ArithmeticError('Insufficient working precision.')\n        return float(squared.sqrt()),float(bias)\n\nfor steps in (4,16,64,256):\n    for method in ('euler','milstein'):\n        rms,bias=exact_errors(0.4,0.6,1,steps,method)\n        print(f'{method}, n={steps}: RMS={rms:.8f}, mean bias={bias:.8f}')\nrms,bias=exact_errors(0,0.05,0.25,512,'milstein')\nprint(f'small-error case: RMS={rms:.12e}, mean bias={bias:.1f}')\n",
    "expected": "euler, n=4: RMS=0.25060757, mean bias=-0.02772470\nmilstein, n=4: RMS=0.12920177, mean bias=-0.02772470\neuler, n=16: RMS=0.11793833, mean bias=-0.00731908\nmilstein, n=16: RMS=0.03634495, mean bias=-0.00731908\neuler, n=64: RMS=0.05742640, mean bias=-0.00185589\nmilstein, n=64: RMS=0.00938284, mean bias=-0.00185589\neuler, n=256: RMS=0.02849217, mean bias=-0.00046564\nmilstein, n=256: RMS=0.00236506, mean bias=-0.00046564\nsmall-error case: RMS=1.246263736118e-08, mean bias=0.0",
    "interpretation": "These are finite-grid analytic moments for this GBM with X0=1, calculated at high precision; no sample was used. A Monte Carlo estimate fluctuates around these targets. The mean bias is shared by EM and scalar Milstein here, although their path errors differ."
  },
  "generator": {
    "title": "Recover a moment equation from the generator",
    "question": "For an OU state with current mean 0.4 and variance 0.3, θ=1.2, target zero and η=0.8, what is the rate of change of its second moment?",
    "language": "python",
    "code": "def ou_second_rate(theta,target,eta,mean,variance):\n    if not (0 <= theta <= 3 and 0 <= eta <= 1.2 and abs(target) <= 2 and abs(mean) <= 2 and 0 <= variance <= 4):\n        raise ValueError('Use the declared bounded finite teaching parameters.')\n    second=variance+mean**2\n    generator_expectation=-2*theta*(second-target*mean)+eta**2\n    mean_rate=theta*(target-mean)\n    variance_rate=eta**2-2*theta*variance\n    moment_rate=variance_rate+2*mean*mean_rate\n    return generator_expectation,moment_rate\n\ngenerator,moments=ou_second_rate(1.2,0,0.8,0.4,0.3)\nprint(f'from E[L(x^2)]: {generator:.6f}')\nprint(f'from variance plus mean squared: {moments:.6f}')\nprint('The density can change while a selected moment decreases.')\n",
    "expected": "from E[L(x^2)]: -0.464000\nfrom variance plus mean squared: -0.464000\nThe density can change while a selected moment decreases.",
    "interpretation": "For f(x)=x², the generator is 2x times the drift plus η². Taking expectation and separately differentiating variance plus mean² produce the same result. This is an exact moment identity, not a solved numerical density PDE."
  },
  "tilt": {
    "title": "Change probabilities through a normalized weight",
    "question": "With T=0.7 and c=0.4, what terminal Brownian mean results from weighting paths by exp(c W_T−c²T/2)?",
    "language": "python",
    "code": "import math\nimport numpy as np\n\ntime,shift=0.7,0.4\nnodes,weights=np.polynomial.hermite.hermgauss(48)\nterminal=math.sqrt(2*time)*nodes\nbase=weights/math.sqrt(math.pi)\nlikelihood=np.exp(shift*terminal-shift**2*time/2)\nweighted=base*likelihood\ntotal=float(np.sum(weighted))\nmean=float(np.sum(weighted*terminal))\nvariance=float(np.sum(weighted*(terminal-mean)**2))\nprint(f'total weight: {total:.9f}')\nprint(f'new mean: {mean:.9f}; target: {shift*time:.9f}')\nprint(f'new variance: {variance:.9f}; target: {time:.9f}')\nprint('This finite-horizon constant shift is normalized by the Gaussian moment formula.')\n",
    "expected": "total weight: 1.000000000\nnew mean: 0.280000000; target: 0.280000000\nnew variance: 0.700000000; target: 0.700000000\nThis finite-horizon constant shift is normalized by the Gaussian moment formula.",
    "interpretation": "The coordinate W_T has not been replaced by a new observed value; its probability law has been reweighted. Gaussian quadrature checks the derived identity. A general random drift change needs a true-martingale condition, not just a formally written exponential."
  },
  "time_transform": {
    "title": "Cancel a drift with a time-dependent transformation",
    "question": "For f(t,x)=x³−3tx, does averaging the transformed future Brownian state recover the transformed present state?",
    "language": "python",
    "code": "import math\nimport numpy as np\n\ndef conditional_transform(time,state,step):\n    if not (0 <= time <= 4 and abs(state) <= 3 and 1/8192 <= step <= 4):\n        raise ValueError('Use finite bounded time, state and positive step.')\n    nodes,weights=np.polynomial.hermite.hermgauss(5)\n    later=state+math.sqrt(2*step)*nodes\n    future=later**3-3*(time+step)*later\n    expectation=float(weights@future)/math.sqrt(math.pi)\n    current=state**3-3*time*state\n    return current,expectation\n\ncurrent,future=conditional_transform(0.4,-0.7,0.3)\nprint(f'current transformed value: {current:.9f}')\nprint(f'conditional future mean: {future:.9f}')\nprint('Ito drift: f_t + f_xx/2 = -3x + 3x = 0')\nprint('diffusion coefficient: f_x = 3x^2 - 3t')\nprint('For W0=0: E[(W_T^3-3*T*W_T)^2] = 6*T^3')\n",
    "expected": "current transformed value: 0.497000000\nconditional future mean: 0.497000000\nIto drift: f_t + f_xx/2 = -3x + 3x = 0\ndiffusion coefficient: f_x = 3x^2 - 3t\nFor W0=0: E[(W_T^3-3*T*W_T)^2] = 6*T^3",
    "interpretation": "The time derivative is essential: W³ alone has drift 3W. This low-degree Gaussian quadrature is exact apart from floating-point error. The displayed finite-horizon second moment supplies the integrability check for a true martingale."
  }
};
