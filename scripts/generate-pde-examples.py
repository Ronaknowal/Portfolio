"""Execute complete standalone PDE lesson programs; save their actual stdout."""
import json
from pathlib import Path
import subprocess
import sys
import textwrap

examples = {}


def add(name, title, question, interpretation, source, environment='Python 3.12 standard library'):
    code = textwrap.dedent(source).strip() + '\n'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, encoding='utf-8', check=True)
    examples[name] = {'title':title,'question':question,'interpretation':interpretation,
        'environment':environment,'language':'python','code':code,'expected':result.stdout.rstrip()}


add('balance','Check the complete control-volume balance',
    'The conductivity changes with x. Which term disappears if you incorrectly replace the flux derivative by k times the second temperature derivative?',
    'The local and interval residuals are exactly zero as symbolic polynomials. A negative source is cooling. The check uses capacity C=1 and the stated dimensionless teaching field, not measured material data.',r'''
    import sympy as sp

    x, t = sp.symbols('x t', real=True)
    u = 1 + x + 2*x**2 + t*(3-x)
    k = 1+x
    flux = -k*sp.diff(u,x)
    source = sp.diff(u,t)+sp.diff(flux,x)
    left, right = sp.Rational(1,4), sp.Rational(3,4)
    accumulation = sp.integrate(sp.diff(u,t),(x,left,right))
    net_inflow = flux.subs(x,left)-flux.subs(x,right)
    source_total = sp.integrate(source,(x,left,right))
    print('source:', sp.expand(source))
    print('missing term:', sp.diff(k,x)*sp.diff(u,x))
    print('accumulation:', accumulation)
    print('balance residual:', sp.simplify(accumulation-net_inflow-source_total))
    print('outward normals:', -1, 1)
    ''','Python 3.12; SymPy')

add('transport','Trace each observation to its actual data',
    'At time 0.5, which of the positions 0.25 and 0.75 can depend on the left inflow history?',
    'A backward characteristic either reaches the initial line or the inflow boundary first. The last identity checks the total density balance; no independently prescribed outflow is used.',r'''
    import math

    def trace(x,t,curvature=2):
        if not all(math.isfinite(v) for v in (x,t,curvature)) or not (0<=x<=1 and 0<=t<=1):
            raise ValueError('Use finite x,t in [0,1]')
        if x>=t:
            return 'initial', x-t, 1+x-t
        data_time=t-x
        return 'inflow', data_time, 1-data_time+curvature*data_time**2

    for x in [.25,.75]:
        origin, datum, value=trace(x,.5)
        print(f'x={x}: {origin}, coordinate={datum}, u={value}')
    t, curvature=.5,2
    mass=1.5-t+curvature*t**3/3
    derivative=-1+curvature*t*t
    incoming=1-t+curvature*t*t
    outgoing=2-t
    print('mass:', round(mass,6))
    print('balance residual:', derivative-(incoming-outgoing))
    ''')

add('modes','Check the endpoint-selected eigenfunctions',
    'Why does mode zero belong to insulated ends but not two zero-value ends? What changes with one of each?',
    'The Dirichlet n=0 sine is the zero function and is excluded. The mixed family here has X(0)=0 and X\'(1)=0; its frequency is n+1/2. Rounded residuals corroborate endpoint equations derived in the text.',r'''
    import math

    def mode(boundary,n,x):
        if n<0 or int(n)!=n or (boundary=='dirichlet' and n==0):
            raise ValueError('Use a permitted mode index')
        if boundary not in ('dirichlet','neumann','mixed'):
            raise ValueError('Unknown boundary')
        frequency=n+.5 if boundary=='mixed' else n
        w=math.pi*frequency
        if boundary=='neumann':
            return math.cos(w*x),-w*math.sin(w*x),w*w
        return math.sin(w*x),w*math.cos(w*x),w*w

    for boundary,n in [('dirichlet',1),('neumann',0),('neumann',1),('mixed',0)]:
        a,da,eigenvalue=mode(boundary,n,0)
        b,db,_=mode(boundary,n,1)
        residuals=(da,db) if boundary=='neumann' else (a,db) if boundary=='mixed' else (a,b)
        print(boundary,n,'lambda/pi^2=',round(eigenvalue/math.pi**2,6),
              'endpoint residuals=',[round(v,12) for v in residuals])
    ''')

add('heat','Compare the same initial profile under two boundaries',
    'Both rods begin with sin²(pi*x). Which one keeps mean temperature 0.5, and why can their center values be similar at very early times?',
    'The insulated solution retains a constant mode. The Dirichlet sum loses heat at the ends. The final number is the natural logarithm of a positive analytic truncation bound, not a bound on floating-point error.',r'''
    import math

    def coefficient(n):
        return -8/(math.pi*n*(n*n-4)) if n%2 else 0.0

    def heat(x,t,boundary):
        if not (0<=x<=1 and 0<=t<=.5) or boundary not in ('dirichlet','neumann'):
            raise ValueError('Use the stated unit-interval problem')
        if x in (0,1) and (t==0 or boundary=='dirichlet'):
            return 0.0
        if t==0:
            return math.sin(math.pi*x)**2
        if boundary=='neumann':
            return .5-.5*math.exp(-4*math.pi**2*t)*math.cos(2*math.pi*x)
        return math.fsum(coefficient(n)*math.exp(-n*n*math.pi**2*t)*math.sin(n*math.pi*x)
                         for n in range(1,64,2))

    for t in [.01,.05,.2]:
        mean=math.fsum(2*coefficient(n)/(n*math.pi)*math.exp(-n*n*math.pi**2*t)
                       for n in range(1,64,2))
        print(f'theta={t}: centers=',[round(heat(.5,t,b),6) for b in ('dirichlet','neumann')],
              'means=',[round(mean,6),.5])
    log_bound=math.log(36/(5*math.pi*63**2))-64**2*math.pi**2*.05
    print('log analytic tail bound:',round(log_bound,6))
    ''')

add('energy','Separate heat amount from squared-temperature energy',
    'Does a conserved mean force the whole profile to stay unchanged? Check the insulated solution and its squared norm.',
    'The mean stays fixed while spatial variation decays. The nonpositive derivative of the squared norm follows from integration by parts; it is not a statement that the physical thermal energy equals the squared temperature.',r'''
    import sympy as sp

    x,t=sp.symbols('x t',real=True)
    u=sp.Rational(1,2)-sp.exp(-4*sp.pi**2*t)*sp.cos(2*sp.pi*x)/2
    mean=sp.integrate(u,(x,0,1))
    square=sp.integrate(u*u,(x,0,1))
    dissipation=sp.integrate(sp.diff(u,x)**2,(x,0,1))
    print('mean:',mean)
    print('squared norm:',sp.simplify(square))
    print('energy identity residual:',sp.simplify(sp.diff(square,t)+2*dissipation))
    print('initial squared norm:',sp.simplify(square.subs(t,0)))
    ''','Python 3.12; SymPy')

add('kernel','Check the diffusion kernel as a normalized spreading profile',
    'As time increases, what happens to the peak, total area and spatial variance of the same unit heat input?',
    'The infinite-domain quadrature checks area and second moment. These finite computations support the stated Gaussian identities; the text separately proves the initial-data limit and explains that a plotted window is not compact support.',r'''
    import math
    from scipy.integrate import quad

    def kernel(x,t,alpha):
        if t<=0 or alpha<=0:
            raise ValueError('Positive time and diffusivity required')
        return math.exp(-x*x/(4*alpha*t))/math.sqrt(4*math.pi*alpha*t)

    alpha=.5
    for t in [.1,.5,1]:
        mass=quad(lambda x:kernel(x,t,alpha),-math.inf,math.inf)[0]
        variance=quad(lambda x:x*x*kernel(x,t,alpha),-math.inf,math.inf)[0]
        print('t=',t,'peak=',round(kernel(0,t,alpha),6),
              'area=',round(mass,6),'variance=',round(variance,6))
    ''','Python 3.12; SciPy')

add('wave','Use both initial displacement and initial velocity',
    'At x=0.4,t=0.6, compare zero initial velocity with a velocity equal to half the initial bump. Which term changes?',
    'The velocity contributes an integral over the dependence interval clipped to the bump support. Positive endpoint-distance polynomial terms evaluate it without subtracting almost equal primitives. The plotting boundary is not a physical wall.',r'''
    import math

    def bump(x):
        return (1-x*x)**3 if abs(x)<1 else 0.0

    def shifted_bump(center, offset):
        right, left = (1-center)-offset, (1+center)+offset
        return (right*left)**3 if right>0 and left>0 else 0.0

    def bump_integral(center, half_width):
        low, high = max(-half_width, -1-center), min(half_width, 1-center)
        if high<=low:
            return 0.0
        right0, right1 = max(0.0,(1-center)-low), max(0.0,(1-center)-high)
        left0, left1 = max(0.0,(1+center)+low), max(0.0,(1+center)+high)
        # Two nonnegative linear factors, each cubed, give degree six.
        # Integrate their Bernstein expansion: integral s^r(1-s)^(6-r)
        # over [0,1] is 1/(7*comb(6,r)). Every term is nonnegative.
        terms = []
        for i in range(4):
            for j in range(4):
                coefficient = math.comb(3,i)*math.comb(3,j)/(7*math.comb(6,i+j))
                terms.append(coefficient*right0**(3-i)*right1**i
                             *left0**(3-j)*left1**j)
        return (high-low)*math.fsum(terms)

    def wave(x,t,velocity=.5,c=1):
        if t<0 or c<=0 or not all(math.isfinite(v) for v in (x,t,velocity,c)):
            raise ValueError('Finite inputs, t>=0 and c>0 required')
        distance=c*t
        if not math.isfinite(distance):
            raise ValueError('The characteristic offset c*t must remain finite')
        shape=(shifted_bump(x,-distance)+shifted_bump(x,distance))/2
        moving=velocity*bump_integral(x,distance)/(2*c)
        return shape,moving,shape+moving

    for velocity in [0,.5]:
        print('velocity factor:',velocity,'parts:',[round(v,6) for v in wave(.4,.6,velocity)])
    print('initial value:',round(wave(.4,0)[2],6),'expected:',round(bump(.4),6))
    print('outside dependence of the bump:',wave(3,1)[2])
    ''')

add('standing','Verify the standing-wave equation, data and energy',
    'Two spatial modes have different temporal phases. Does that change the total energy or merely its kinetic/strain split?',
    'The exact residuals check the PDE and fixed endpoints. The initial velocity is not zero, even though the second displacement coefficient vanishes at time zero. Energy is constant for this undamped, homogeneous-boundary model.',r'''
    import sympy as sp

    x,t=sp.symbols('x t',real=True)
    u=sp.sin(sp.pi*x)*sp.cos(sp.pi*t)+sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*t)/4
    energy=sp.integrate((sp.diff(u,t)**2+sp.diff(u,x)**2)/2,(x,0,1))
    print('PDE residual:',sp.simplify(sp.diff(u,t,2)-sp.diff(u,x,2)))
    print('endpoints:',u.subs(x,0),u.subs(x,1))
    print('initial displacement:',u.subs(t,0))
    print('initial velocity:',sp.diff(u,t).subs(t,0))
    print('energy:',sp.simplify(energy))
    ''','Python 3.12; SymPy')

add('poisson','Test compatibility before selecting a Neumann mean',
    'Can choosing a different mean repair a uniform source with zero outward flux at both ends?',
    'Compatibility is checked with exact fractions. A mean selects the additive constant only after the source and boundary fluxes balance. In the compatible example the zero-mean solution has both negative and positive values; this is a potential or temperature departure, not a nonnegative probability.',r'''
    from fractions import Fraction as F

    def neumann(a,b,left,right,mean=0):
        a,b,left,right,mean=map(F,(a,b,left,right,mean))
        if left+right!=a+b/2:
            raise ValueError('Source and outward boundary fluxes are incompatible')
        return [mean-left/2+a/6+b/24,left,-a/2,-b/6]

    def evaluate(coefficients,x):
        return sum(value*F(x)**power for power,value in enumerate(coefficients))

    for fluxes in [(0,0),(1,1)]:
        try:
            coefficients=neumann(2,0,*fluxes)
            print('fluxes:',fluxes,'coefficients:',[str(v) for v in coefficients])
            print('center:',evaluate(coefficients,F(1,2)))
        except ValueError as error:
            print('fluxes:',fluxes,str(error))
    ''')

add('harmonic','Continue boundary oscillations into a square',
    'Keep the top-boundary amplitude fixed. Which boundary pattern reaches farther into the interior: one oscillation or five?',
    'This is the exact separated square solution evaluated numerically. Frequency changes penetration while the common color/amplitude scale stays fixed. A vanishing discrete residual is not needed to define this analytical field.',r'''
    import math

    def harmonic(x,y,n):
        if not (0<=x<=1 and 0<=y<=1) or n not in (1,3,5):
            raise ValueError('Use the stated square and frequency set')
        return math.sin(n*math.pi*x)*math.sinh(n*math.pi*y)/math.sinh(n*math.pi)

    for n in [1,3,5]:
        attenuation=math.sinh(n*math.pi/2)/math.sinh(n*math.pi)
        print('n=',n,'half-depth amplitude ratio=',round(attenuation,9),
              'center value=',round(harmonic(.5,.5,n),9))
    print('top n=3 at x=.25:',round(harmonic(.25,1,3),6))
    ''')

add('weak','Check a point source through probes rather than a classical second derivative',
    'Move the point source to a=1/3. What does integrating G prime times v prime return for two different probes?',
    'The two smooth pieces produce the point evaluation exactly. The slope jump is minus one; after the minus sign in -u double-prime it represents a positive unit source. The value at the kink is finite, not an infinite spike.',r'''
    import sympy as sp

    x=sp.symbols('x',real=True)
    a=sp.Rational(1,3)
    for v in [x*(1-x),x*x*(1-x)]:
        pairing=sp.integrate((1-a)*sp.diff(v,x),(x,0,a))+sp.integrate(-a*sp.diff(v,x),(x,a,1))
        print('probe:',v,'pairing:',pairing,'point value:',v.subs(x,a))
    print('slopes:',1-a,-a,'jump:',-a-(1-a))
    print('gradient energy:',sp.simplify(a*(1-a)**2+(1-a)*a*a))
    ''','Python 3.12; SymPy')

add('burgers','Conservation alone does not select a shock',
    'Reverse the two states. Both jumps satisfy the same speed formula, but do they have the same sign of entropy production?',
    'The expanding jump satisfies Rankine-Hugoniot but has positive quadratic-entropy production. For these convex Burgers Riemann data the admissible alternative is a rarefaction. This calculation is not a general entropy-solution theorem for other fluxes.',r'''
    from fractions import Fraction as F

    def jump(left,right):
        left,right=F(left),F(right)
        speed=(left+right)/2
        balance=speed*(right-left)-(right*right-left*left)/2
        entropy=(right**3-left**3)/3-speed*(right*right-left*left)/2
        return speed,balance,entropy

    def rarefaction(x,t,left=0,right=2):
        if t<=0 or left>=right:
            raise ValueError('Positive time and increasing states required')
        return min(right,max(left,x/t))

    for states in [(2,0),(0,2)]:
        print('states:',states,'speed, balance, entropy:',[str(v) for v in jump(*states)])
    print('rarefaction at t=.5:',[rarefaction(x,.5) for x in [-1,.25,.5,2]])
    ''')

add('inverse','Measure what forward diffusion hides',
    'Each initial mode has maximum magnitude one. How small is its observation after the same positive time?',
    'The natural-log attenuation stays meaningful even when a wider experiment would underflow. The mode sequence proves inverse discontinuity; these three finite samples only illustrate that argument. An exact noiseless nonzero observation is different from a rounded or noisy one.',r'''
    import math

    def attenuation(n,time):
        if int(n)!=n or n<1 or not (0<time<=1):
            raise ValueError('Use positive mode and observation time')
        log_value=-n*n*math.pi**2*time
        return log_value,math.exp(log_value)

    for n in [1,6,12]:
        log_value,value=attenuation(n,.03)
        print('n=',n,'log attenuation=',round(log_value,6),'amplitude=',format(value,'.6g'))
    ''')

add('periodic','Compute penetration depth and phase delay',
    'At one penetration depth, what fraction of the surface amplitude remains? What changes when the forcing frequency is multiplied by four?',
    'The depth scale halves when angular frequency quadruples. A phase lag of one radian is a time lag of 1/omega, not one second. The formula is an ideal periodic steady solution on a homogeneous half-line; arbitrary startup adds a transient.',r'''
    import math

    def penetration(alpha,omega):
        if alpha<=0 or omega<=0 or not all(math.isfinite(v) for v in (alpha,omega)):
            raise ValueError('Positive finite diffusivity and angular frequency required')
        return math.sqrt(2*alpha/omega)

    alpha,omega=.5,2
    depth=penetration(alpha,omega)
    print('penetration depth:',round(depth,6))
    print('one-depth amplitude ratio:',round(math.exp(-1),6))
    print('one-depth phase lag:',1,'radian; time lag:',1/omega)
    print('four-times-frequency depth ratio:',penetration(alpha,4*omega)/depth)
    ''')

add('rod','Finish a physical diffusion model with a uniform criterion',
    'How long until every point is within 0.05 K of the steady profile? How does doubling the rod length change that time when the excess mode amplitude is unchanged?',
    'The supremum is known analytically because the sine maximum is one. The steady temperature scales as length squared when conductivity and source are held fixed; only changing the decay rate would miss that physical change.',r'''
    import math

    def rod(x,t,length=1,amplitude=.2):
        if length<=0 or not (0<=x<=length) or t<0:
            raise ValueError('Use x within a positive-length rod and nonnegative time')
        xi=x/length
        return length*length*xi*(1-xi)+amplitude*math.exp(-.25*math.pi**2*t/length**2)*math.sin(math.pi*xi)

    def settle(length=1,amplitude=.2,tolerance=.05):
        if length<=0 or amplitude<0 or tolerance<=0:
            raise ValueError('Positive length/tolerance and nonnegative excess required')
        return max(0,math.log(amplitude/tolerance))*(length*length)/(.25*math.pi**2) if amplitude else 0.0

    for length in [1,2]:
        time=settle(length)
        print('length:',length,'settling seconds:',round(time,6),
              'steady center:',length*length/4,'center at settling:',round(rod(length/2,time,length),6))
    print('already inside tolerance:',settle(amplitude=.02))
    ''')

Path('src/learn/data/pde-examples.js').write_text(
    '// Complete independently runnable programs; expected text is actual Python stdout.\n'
    'export const pdeExamples = '+json.dumps(examples,indent=2,ensure_ascii=False)+';\n',encoding='utf-8')
print(f'Executed and saved {len(examples)} complete PDE programs.')
