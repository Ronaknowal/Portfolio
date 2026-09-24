// Complete independently runnable programs; expected text is actual Python stdout.
export const pdeExamples = {
  "balance": {
    "title": "Check the complete control-volume balance",
    "question": "The conductivity changes with x. Which term disappears if you incorrectly replace the flux derivative by k times the second temperature derivative?",
    "interpretation": "The local and interval residuals are exactly zero as symbolic polynomials. A negative source is cooling. The check uses capacity C=1 and the stated dimensionless teaching field, not measured material data.",
    "environment": "Python 3.12; SymPy",
    "language": "python",
    "code": "import sympy as sp\n\nx, t = sp.symbols('x t', real=True)\nu = 1 + x + 2*x**2 + t*(3-x)\nk = 1+x\nflux = -k*sp.diff(u,x)\nsource = sp.diff(u,t)+sp.diff(flux,x)\nleft, right = sp.Rational(1,4), sp.Rational(3,4)\naccumulation = sp.integrate(sp.diff(u,t),(x,left,right))\nnet_inflow = flux.subs(x,left)-flux.subs(x,right)\nsource_total = sp.integrate(source,(x,left,right))\nprint('source:', sp.expand(source))\nprint('missing term:', sp.diff(k,x)*sp.diff(u,x))\nprint('accumulation:', accumulation)\nprint('balance residual:', sp.simplify(accumulation-net_inflow-source_total))\nprint('outward normals:', -1, 1)\n",
    "expected": "source: t - 9*x - 2\nmissing term: -t + 4*x + 1\naccumulation: 5/4\nbalance residual: 0\noutward normals: -1 1"
  },
  "transport": {
    "title": "Trace each observation to its actual data",
    "question": "At time 0.5, which of the positions 0.25 and 0.75 can depend on the left inflow history?",
    "interpretation": "A backward characteristic either reaches the initial line or the inflow boundary first. The last identity checks the total density balance; no independently prescribed outflow is used.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "import math\n\ndef trace(x,t,curvature=2):\n    if not all(math.isfinite(v) for v in (x,t,curvature)) or not (0<=x<=1 and 0<=t<=1):\n        raise ValueError('Use finite x,t in [0,1]')\n    if x>=t:\n        return 'initial', x-t, 1+x-t\n    data_time=t-x\n    return 'inflow', data_time, 1-data_time+curvature*data_time**2\n\nfor x in [.25,.75]:\n    origin, datum, value=trace(x,.5)\n    print(f'x={x}: {origin}, coordinate={datum}, u={value}')\nt, curvature=.5,2\nmass=1.5-t+curvature*t**3/3\nderivative=-1+curvature*t*t\nincoming=1-t+curvature*t*t\noutgoing=2-t\nprint('mass:', round(mass,6))\nprint('balance residual:', derivative-(incoming-outgoing))\n",
    "expected": "x=0.25: inflow, coordinate=0.25, u=0.875\nx=0.75: initial, coordinate=0.25, u=1.25\nmass: 1.083333\nbalance residual: 0.0"
  },
  "modes": {
    "title": "Check the endpoint-selected eigenfunctions",
    "question": "Why does mode zero belong to insulated ends but not two zero-value ends? What changes with one of each?",
    "interpretation": "The Dirichlet n=0 sine is the zero function and is excluded. The mixed family here has X(0)=0 and X'(1)=0; its frequency is n+1/2. Rounded residuals corroborate endpoint equations derived in the text.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "import math\n\ndef mode(boundary,n,x):\n    if n<0 or int(n)!=n or (boundary=='dirichlet' and n==0):\n        raise ValueError('Use a permitted mode index')\n    if boundary not in ('dirichlet','neumann','mixed'):\n        raise ValueError('Unknown boundary')\n    frequency=n+.5 if boundary=='mixed' else n\n    w=math.pi*frequency\n    if boundary=='neumann':\n        return math.cos(w*x),-w*math.sin(w*x),w*w\n    return math.sin(w*x),w*math.cos(w*x),w*w\n\nfor boundary,n in [('dirichlet',1),('neumann',0),('neumann',1),('mixed',0)]:\n    a,da,eigenvalue=mode(boundary,n,0)\n    b,db,_=mode(boundary,n,1)\n    residuals=(da,db) if boundary=='neumann' else (a,db) if boundary=='mixed' else (a,b)\n    print(boundary,n,'lambda/pi^2=',round(eigenvalue/math.pi**2,6),\n          'endpoint residuals=',[round(v,12) for v in residuals])\n",
    "expected": "dirichlet 1 lambda/pi^2= 1.0 endpoint residuals= [0.0, 0.0]\nneumann 0 lambda/pi^2= 0.0 endpoint residuals= [-0.0, -0.0]\nneumann 1 lambda/pi^2= 1.0 endpoint residuals= [-0.0, -0.0]\nmixed 0 lambda/pi^2= 0.25 endpoint residuals= [0.0, 0.0]"
  },
  "heat": {
    "title": "Compare the same initial profile under two boundaries",
    "question": "Both rods begin with sin²(pi*x). Which one keeps mean temperature 0.5, and why can their center values be similar at very early times?",
    "interpretation": "The insulated solution retains a constant mode. The Dirichlet sum loses heat at the ends. The final number is the natural logarithm of a positive analytic truncation bound, not a bound on floating-point error.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "import math\n\ndef coefficient(n):\n    return -8/(math.pi*n*(n*n-4)) if n%2 else 0.0\n\ndef heat(x,t,boundary):\n    if not (0<=x<=1 and 0<=t<=.5) or boundary not in ('dirichlet','neumann'):\n        raise ValueError('Use the stated unit-interval problem')\n    if x in (0,1) and (t==0 or boundary=='dirichlet'):\n        return 0.0\n    if t==0:\n        return math.sin(math.pi*x)**2\n    if boundary=='neumann':\n        return .5-.5*math.exp(-4*math.pi**2*t)*math.cos(2*math.pi*x)\n    return math.fsum(coefficient(n)*math.exp(-n*n*math.pi**2*t)*math.sin(n*math.pi*x)\n                     for n in range(1,64,2))\n\nfor t in [.01,.05,.2]:\n    mean=math.fsum(2*coefficient(n)/(n*math.pi)*math.exp(-n*n*math.pi**2*t)\n                   for n in range(1,64,2))\n    print(f'theta={t}: centers=',[round(heat(.5,t,b),6) for b in ('dirichlet','neumann')],\n          'means=',[round(mean,6),.5])\nlog_bound=math.log(36/(5*math.pi*63**2))-64**2*math.pi**2*.05\nprint('log analytic tail bound:',round(log_bound,6))\n",
    "expected": "theta=0.01: centers= [0.836894, 0.836913] means= [0.474506, 0.5]\ntheta=0.05: centers= [0.520207, 0.569456] means= [0.329476, 0.5]\ntheta=0.2: centers= [0.117911, 0.500186] means= [0.075065, 0.5]\nlog analytic tail bound: -2028.7519"
  },
  "energy": {
    "title": "Separate heat amount from squared-temperature energy",
    "question": "Does a conserved mean force the whole profile to stay unchanged? Check the insulated solution and its squared norm.",
    "interpretation": "The mean stays fixed while spatial variation decays. The nonpositive derivative of the squared norm follows from integration by parts; it is not a statement that the physical thermal energy equals the squared temperature.",
    "environment": "Python 3.12; SymPy",
    "language": "python",
    "code": "import sympy as sp\n\nx,t=sp.symbols('x t',real=True)\nu=sp.Rational(1,2)-sp.exp(-4*sp.pi**2*t)*sp.cos(2*sp.pi*x)/2\nmean=sp.integrate(u,(x,0,1))\nsquare=sp.integrate(u*u,(x,0,1))\ndissipation=sp.integrate(sp.diff(u,x)**2,(x,0,1))\nprint('mean:',mean)\nprint('squared norm:',sp.simplify(square))\nprint('energy identity residual:',sp.simplify(sp.diff(square,t)+2*dissipation))\nprint('initial squared norm:',sp.simplify(square.subs(t,0)))\n",
    "expected": "mean: 1/2\nsquared norm: 1/4 + exp(-8*pi**2*t)/8\nenergy identity residual: 0\ninitial squared norm: 3/8"
  },
  "kernel": {
    "title": "Check the diffusion kernel as a normalized spreading profile",
    "question": "As time increases, what happens to the peak, total area and spatial variance of the same unit heat input?",
    "interpretation": "The infinite-domain quadrature checks area and second moment. These finite computations support the stated Gaussian identities; the text separately proves the initial-data limit and explains that a plotted window is not compact support.",
    "environment": "Python 3.12; SciPy",
    "language": "python",
    "code": "import math\nfrom scipy.integrate import quad\n\ndef kernel(x,t,alpha):\n    if t<=0 or alpha<=0:\n        raise ValueError('Positive time and diffusivity required')\n    return math.exp(-x*x/(4*alpha*t))/math.sqrt(4*math.pi*alpha*t)\n\nalpha=.5\nfor t in [.1,.5,1]:\n    mass=quad(lambda x:kernel(x,t,alpha),-math.inf,math.inf)[0]\n    variance=quad(lambda x:x*x*kernel(x,t,alpha),-math.inf,math.inf)[0]\n    print('t=',t,'peak=',round(kernel(0,t,alpha),6),\n          'area=',round(mass,6),'variance=',round(variance,6))\n",
    "expected": "t= 0.1 peak= 1.261566 area= 1.0 variance= 0.1\nt= 0.5 peak= 0.56419 area= 1.0 variance= 0.5\nt= 1 peak= 0.398942 area= 1.0 variance= 1.0"
  },
  "wave": {
    "title": "Use both initial displacement and initial velocity",
    "question": "At x=0.4,t=0.6, compare zero initial velocity with a velocity equal to half the initial bump. Which term changes?",
    "interpretation": "The velocity contributes an integral over the dependence interval clipped to the bump support. Positive endpoint-distance polynomial terms evaluate it without subtracting almost equal primitives. The plotting boundary is not a physical wall.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "import math\n\ndef bump(x):\n    return (1-x*x)**3 if abs(x)<1 else 0.0\n\ndef shifted_bump(center, offset):\n    right, left = (1-center)-offset, (1+center)+offset\n    return (right*left)**3 if right>0 and left>0 else 0.0\n\ndef bump_integral(center, half_width):\n    low, high = max(-half_width, -1-center), min(half_width, 1-center)\n    if high<=low:\n        return 0.0\n    right0, right1 = max(0.0,(1-center)-low), max(0.0,(1-center)-high)\n    left0, left1 = max(0.0,(1+center)+low), max(0.0,(1+center)+high)\n    # Two nonnegative linear factors, each cubed, give degree six.\n    # Integrate their Bernstein expansion: integral s^r(1-s)^(6-r)\n    # over [0,1] is 1/(7*comb(6,r)). Every term is nonnegative.\n    terms = []\n    for i in range(4):\n        for j in range(4):\n            coefficient = math.comb(3,i)*math.comb(3,j)/(7*math.comb(6,i+j))\n            terms.append(coefficient*right0**(3-i)*right1**i\n                         *left0**(3-j)*left1**j)\n    return (high-low)*math.fsum(terms)\n\ndef wave(x,t,velocity=.5,c=1):\n    if t<0 or c<=0 or not all(math.isfinite(v) for v in (x,t,velocity,c)):\n        raise ValueError('Finite inputs, t>=0 and c>0 required')\n    distance=c*t\n    if not math.isfinite(distance):\n        raise ValueError('The characteristic offset c*t must remain finite')\n    shape=(shifted_bump(x,-distance)+shifted_bump(x,distance))/2\n    moving=velocity*bump_integral(x,distance)/(2*c)\n    return shape,moving,shape+moving\n\nfor velocity in [0,.5]:\n    print('velocity factor:',velocity,'parts:',[round(v,6) for v in wave(.4,.6,velocity)])\nprint('initial value:',round(wave(.4,0)[2],6),'expected:',round(bump(.4),6))\nprint('outside dependence of the bump:',wave(3,1)[2])\n",
    "expected": "velocity factor: 0 parts: [0.442368, 0.0, 0.442368]\nvelocity factor: 0.5 parts: [0.442368, 0.162333, 0.604701]\ninitial value: 0.592704 expected: 0.592704\noutside dependence of the bump: 0.0"
  },
  "standing": {
    "title": "Verify the standing-wave equation, data and energy",
    "question": "Two spatial modes have different temporal phases. Does that change the total energy or merely its kinetic/strain split?",
    "interpretation": "The exact residuals check the PDE and fixed endpoints. The initial velocity is not zero, even though the second displacement coefficient vanishes at time zero. Energy is constant for this undamped, homogeneous-boundary model.",
    "environment": "Python 3.12; SymPy",
    "language": "python",
    "code": "import sympy as sp\n\nx,t=sp.symbols('x t',real=True)\nu=sp.sin(sp.pi*x)*sp.cos(sp.pi*t)+sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*t)/4\nenergy=sp.integrate((sp.diff(u,t)**2+sp.diff(u,x)**2)/2,(x,0,1))\nprint('PDE residual:',sp.simplify(sp.diff(u,t,2)-sp.diff(u,x,2)))\nprint('endpoints:',u.subs(x,0),u.subs(x,1))\nprint('initial displacement:',u.subs(t,0))\nprint('initial velocity:',sp.diff(u,t).subs(t,0))\nprint('energy:',sp.simplify(energy))\n",
    "expected": "PDE residual: 0\nendpoints: 0 0\ninitial displacement: sin(pi*x)\ninitial velocity: pi*sin(2*pi*x)/2\nenergy: 5*pi**2/16"
  },
  "poisson": {
    "title": "Test compatibility before selecting a Neumann mean",
    "question": "Can choosing a different mean repair a uniform source with zero outward flux at both ends?",
    "interpretation": "Compatibility is checked with exact fractions. A mean selects the additive constant only after the source and boundary fluxes balance. In the compatible example the zero-mean solution has both negative and positive values; this is a potential or temperature departure, not a nonnegative probability.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "from fractions import Fraction as F\n\ndef neumann(a,b,left,right,mean=0):\n    a,b,left,right,mean=map(F,(a,b,left,right,mean))\n    if left+right!=a+b/2:\n        raise ValueError('Source and outward boundary fluxes are incompatible')\n    return [mean-left/2+a/6+b/24,left,-a/2,-b/6]\n\ndef evaluate(coefficients,x):\n    return sum(value*F(x)**power for power,value in enumerate(coefficients))\n\nfor fluxes in [(0,0),(1,1)]:\n    try:\n        coefficients=neumann(2,0,*fluxes)\n        print('fluxes:',fluxes,'coefficients:',[str(v) for v in coefficients])\n        print('center:',evaluate(coefficients,F(1,2)))\n    except ValueError as error:\n        print('fluxes:',fluxes,str(error))\n",
    "expected": "fluxes: (0, 0) Source and outward boundary fluxes are incompatible\nfluxes: (1, 1) coefficients: ['-1/6', '1', '-1', '0']\ncenter: 1/12"
  },
  "harmonic": {
    "title": "Continue boundary oscillations into a square",
    "question": "Keep the top-boundary amplitude fixed. Which boundary pattern reaches farther into the interior: one oscillation or five?",
    "interpretation": "This is the exact separated square solution evaluated numerically. Frequency changes penetration while the common color/amplitude scale stays fixed. A vanishing discrete residual is not needed to define this analytical field.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "import math\n\ndef harmonic(x,y,n):\n    if not (0<=x<=1 and 0<=y<=1) or n not in (1,3,5):\n        raise ValueError('Use the stated square and frequency set')\n    return math.sin(n*math.pi*x)*math.sinh(n*math.pi*y)/math.sinh(n*math.pi)\n\nfor n in [1,3,5]:\n    attenuation=math.sinh(n*math.pi/2)/math.sinh(n*math.pi)\n    print('n=',n,'half-depth amplitude ratio=',round(attenuation,9),\n          'center value=',round(harmonic(.5,.5,n),9))\nprint('top n=3 at x=.25:',round(harmonic(.25,1,3),6))\n",
    "expected": "n= 1 half-depth amplitude ratio= 0.199268408 center value= 0.199268408\nn= 3 half-depth amplitude ratio= 0.008982566 center value= -0.008982566\nn= 5 half-depth amplitude ratio= 0.000388203 center value= 0.000388203\ntop n=3 at x=.25: 0.707107"
  },
  "weak": {
    "title": "Check a point source through probes rather than a classical second derivative",
    "question": "Move the point source to a=1/3. What does integrating G prime times v prime return for two different probes?",
    "interpretation": "The two smooth pieces produce the point evaluation exactly. The slope jump is minus one; after the minus sign in -u double-prime it represents a positive unit source. The value at the kink is finite, not an infinite spike.",
    "environment": "Python 3.12; SymPy",
    "language": "python",
    "code": "import sympy as sp\n\nx=sp.symbols('x',real=True)\na=sp.Rational(1,3)\nfor v in [x*(1-x),x*x*(1-x)]:\n    pairing=sp.integrate((1-a)*sp.diff(v,x),(x,0,a))+sp.integrate(-a*sp.diff(v,x),(x,a,1))\n    print('probe:',v,'pairing:',pairing,'point value:',v.subs(x,a))\nprint('slopes:',1-a,-a,'jump:',-a-(1-a))\nprint('gradient energy:',sp.simplify(a*(1-a)**2+(1-a)*a*a))\n",
    "expected": "probe: x*(1 - x) pairing: 2/9 point value: 2/9\nprobe: x**2*(1 - x) pairing: 2/27 point value: 2/27\nslopes: 2/3 -1/3 jump: -1\ngradient energy: 2/9"
  },
  "burgers": {
    "title": "Conservation alone does not select a shock",
    "question": "Reverse the two states. Both jumps satisfy the same speed formula, but do they have the same sign of entropy production?",
    "interpretation": "The expanding jump satisfies Rankine-Hugoniot but has positive quadratic-entropy production. For these convex Burgers Riemann data the admissible alternative is a rarefaction. This calculation is not a general entropy-solution theorem for other fluxes.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "from fractions import Fraction as F\n\ndef jump(left,right):\n    left,right=F(left),F(right)\n    speed=(left+right)/2\n    balance=speed*(right-left)-(right*right-left*left)/2\n    entropy=(right**3-left**3)/3-speed*(right*right-left*left)/2\n    return speed,balance,entropy\n\ndef rarefaction(x,t,left=0,right=2):\n    if t<=0 or left>=right:\n        raise ValueError('Positive time and increasing states required')\n    return min(right,max(left,x/t))\n\nfor states in [(2,0),(0,2)]:\n    print('states:',states,'speed, balance, entropy:',[str(v) for v in jump(*states)])\nprint('rarefaction at t=.5:',[rarefaction(x,.5) for x in [-1,.25,.5,2]])\n",
    "expected": "states: (2, 0) speed, balance, entropy: ['1', '0', '-2/3']\nstates: (0, 2) speed, balance, entropy: ['1', '0', '2/3']\nrarefaction at t=.5: [0, 0.5, 1.0, 2]"
  },
  "inverse": {
    "title": "Measure what forward diffusion hides",
    "question": "Each initial mode has maximum magnitude one. How small is its observation after the same positive time?",
    "interpretation": "The natural-log attenuation stays meaningful even when a wider experiment would underflow. The mode sequence proves inverse discontinuity; these three finite samples only illustrate that argument. An exact noiseless nonzero observation is different from a rounded or noisy one.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "import math\n\ndef attenuation(n,time):\n    if int(n)!=n or n<1 or not (0<time<=1):\n        raise ValueError('Use positive mode and observation time')\n    log_value=-n*n*math.pi**2*time\n    return log_value,math.exp(log_value)\n\nfor n in [1,6,12]:\n    log_value,value=attenuation(n,.03)\n    print('n=',n,'log attenuation=',round(log_value,6),'amplitude=',format(value,'.6g'))\n",
    "expected": "n= 1 log attenuation= -0.296088 amplitude= 0.743722\nn= 6 log attenuation= -10.659173 amplitude= 2.34844e-05\nn= 12 log attenuation= -42.636691 amplitude= 3.04173e-19"
  },
  "periodic": {
    "title": "Compute penetration depth and phase delay",
    "question": "At one penetration depth, what fraction of the surface amplitude remains? What changes when the forcing frequency is multiplied by four?",
    "interpretation": "The depth scale halves when angular frequency quadruples. A phase lag of one radian is a time lag of 1/omega, not one second. The formula is an ideal periodic steady solution on a homogeneous half-line; arbitrary startup adds a transient.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "import math\n\ndef penetration(alpha,omega):\n    if alpha<=0 or omega<=0 or not all(math.isfinite(v) for v in (alpha,omega)):\n        raise ValueError('Positive finite diffusivity and angular frequency required')\n    return math.sqrt(2*alpha/omega)\n\nalpha,omega=.5,2\ndepth=penetration(alpha,omega)\nprint('penetration depth:',round(depth,6))\nprint('one-depth amplitude ratio:',round(math.exp(-1),6))\nprint('one-depth phase lag:',1,'radian; time lag:',1/omega)\nprint('four-times-frequency depth ratio:',penetration(alpha,4*omega)/depth)\n",
    "expected": "penetration depth: 0.707107\none-depth amplitude ratio: 0.367879\none-depth phase lag: 1 radian; time lag: 0.5\nfour-times-frequency depth ratio: 0.5"
  },
  "rod": {
    "title": "Finish a physical diffusion model with a uniform criterion",
    "question": "How long until every point is within 0.05 K of the steady profile? How does doubling the rod length change that time when the excess mode amplitude is unchanged?",
    "interpretation": "The supremum is known analytically because the sine maximum is one. The steady temperature scales as length squared when conductivity and source are held fixed; only changing the decay rate would miss that physical change.",
    "environment": "Python 3.12 standard library",
    "language": "python",
    "code": "import math\n\ndef rod(x,t,length=1,amplitude=.2):\n    if length<=0 or not (0<=x<=length) or t<0:\n        raise ValueError('Use x within a positive-length rod and nonnegative time')\n    xi=x/length\n    return length*length*xi*(1-xi)+amplitude*math.exp(-.25*math.pi**2*t/length**2)*math.sin(math.pi*xi)\n\ndef settle(length=1,amplitude=.2,tolerance=.05):\n    if length<=0 or amplitude<0 or tolerance<=0:\n        raise ValueError('Positive length/tolerance and nonnegative excess required')\n    return max(0,math.log(amplitude/tolerance))*(length*length)/(.25*math.pi**2) if amplitude else 0.0\n\nfor length in [1,2]:\n    time=settle(length)\n    print('length:',length,'settling seconds:',round(time,6),\n          'steady center:',length*length/4,'center at settling:',round(rod(length/2,time,length),6))\nprint('already inside tolerance:',settle(amplitude=.02))\n",
    "expected": "length: 1 settling seconds: 0.561844 steady center: 0.25 center at settling: 0.3\nlength: 2 settling seconds: 2.247376 steady center: 1.0 center at settling: 1.05\nalready inside tolerance: 0.0"
  }
};
