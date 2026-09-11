"""Independent calculations for the proposed PDE lesson; no production model exists yet."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import mpmath as mp
import numpy as np
from scipy.integrate import quad
import sympy as sp

topic_id = 'partial-differential-equations-conservation-boundary-conditions'
directory = Path('scratch/pde-design')
directory.mkdir(parents=True, exist_ok=True)
archive = Path('docs/teaching/evidence/pde-original-plan.json')
if not archive.exists():
    inventory = subprocess.run(['node', 'scripts/build-curriculum-inventory.mjs', '--topic', topic_id],
                               capture_output=True, text=True, encoding='utf-8', check=True)
    original = json.loads(inventory.stdout)
    assert original['topic']['publicationStatus'] == 'planned'
    source = Path('src/learn/data/curriculum/cross-domain-expansion.js')
    text = source.read_text(encoding='utf-8')
    start = text.index('  plan("Partial Differential Equations, Conservation & Boundary Conditions"')
    end = text.index('  plan("Numerical PDEs:', start)
    original['preservation'] = {
        'archivedAt': datetime.now(timezone.utc).isoformat(),
        'catalogueSource': source.as_posix(),
        'catalogueSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'exactStartingEntry': text[start:end],
        'priorPublishedBody': False,
        'priorCompletePrograms': 0,
    }
    archive.write_text(json.dumps(original, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

checks = Counter()
values = {}
max_error = 0.0

def close(actual, expected, name, tolerance=2e-11):
    global max_error
    error = abs(float(actual) - float(expected)) / max(1.0, abs(float(expected)))
    assert error < tolerance, (name, actual, expected, error)
    max_error = max(max_error, error)
    checks[name] += 1

def identity(expression, name):
    assert sp.simplify(expression) == 0, (name, expression)
    checks[name] += 1

x, t = sp.symbols('x t', real=True)
kappa, alpha, omega = sp.symbols('kappa alpha omega', positive=True)

# Conservation and signs: the local identity is checked by independent exact integration.
u = 1 + x + 2*x*x + t*(3-x)
conductivity = 1+x
flux = -conductivity*sp.diff(u, x)
source = sp.diff(u, t) + sp.diff(flux, x)
identity(sp.integrate(sp.diff(u, t), (x, 0, 1)) -
         (flux.subs(x, 0)-flux.subs(x, 1)+sp.integrate(source, (x, 0, 1))), 'balance')
identity(sp.diff(conductivity*sp.diff(u, x), x)-
         (conductivity*sp.diff(u, x, 2)+sp.diff(conductivity, x)*sp.diff(u, x)), 'variable_conductivity')

# Trace to initial data or inflow. Continuous first derivatives agree on x=t.
first = 1+x-t
boundary = 1-(t-x)+kappa*(t-x)**2
identity(sp.diff(first,t)+sp.diff(first,x), 'transport_pde')
identity(sp.diff(boundary,t)+sp.diff(boundary,x), 'transport_pde')
for derivative in [lambda v:v, lambda v:sp.diff(v,x), lambda v:sp.diff(v,t)]:
    identity((derivative(first)-derivative(boundary)).subs(x,t), 'transport_corner')
mass = sp.integrate(boundary, (x,0,t)) + sp.integrate(first,(x,t,1))
identity(sp.diff(mass,t) - (boundary.subs(x,0)-first.subs(x,1)), 'transport_total')
values['transport'] = {'boundaryRegion': str(boundary.subs({x:sp.Rational(1,4), t:sp.Rational(1,2), kappa:2})),
                       'initialRegion': str(first.subs({x:sp.Rational(3,4), t:sp.Rational(1,4)}))}

# Same initial sin²(pi*x) with Dirichlet/Neumann boundaries.
def coefficient(n):
    return -8/(math.pi*n*(n*n-4)) if n % 2 else 0.0

for n in range(1,40):
    expected = 2*quad(lambda z: math.sin(math.pi*z)**2*math.sin(n*math.pi*z),0,1,
                      epsabs=1e-12,epsrel=1e-12)[0]
    close(coefficient(n), expected, 'sine_coefficient_quadrature')
for theta in [.002,.01,.05,.2,.5]:
    def solution(z):
        return math.fsum(coefficient(n)*math.exp(-n*n*math.pi**2*theta)*math.sin(n*math.pi*z)
                         for n in range(1,64,2))
    mass_series = math.fsum(2*coefficient(n)/(n*math.pi)*math.exp(-n*n*math.pi**2*theta)
                            for n in range(1,64,2))
    energy_series = math.fsum(.5*coefficient(n)**2*math.exp(-2*n*n*math.pi**2*theta)
                              for n in range(1,64,2))
    close(quad(solution,0,1,epsabs=1e-12)[0],mass_series,'heat_mass_independent_integration')
    close(quad(lambda z: solution(z)**2,0,1,epsabs=1e-12)[0],energy_series,'heat_squared_norm')
    with mp.workdps(70):
        for location in [0,.13,.5,.83,1]:
            reference = mp.fsum(-8/(mp.pi*n*(n*n-4))*mp.exp(-n*n*mp.pi**2*theta)*
                                mp.sin(n*mp.pi*location) for n in range(1,200,2))
            close(solution(location),reference,'heat_high_precision_tail')
    assert mass_series < .5
    values.setdefault('heatComparison',[]).append({'theta':theta,'centerDirichlet':solution(.5),
        'centerNeumann':.5+.5*math.exp(-4*math.pi**2*theta),'meanDirichlet':mass_series,'meanNeumann':.5,
        'logAnalyticUniformTailBound63':math.log(36/(5*math.pi*63**2))-64**2*math.pi**2*theta})
u_neumann = sp.Rational(1,2)-sp.exp(-4*sp.pi**2*t)*sp.cos(2*sp.pi*x)/2
identity(sp.diff(u_neumann,t)-sp.diff(u_neumann,x,2),'neumann_exact_pde')
identity(sp.integrate(u_neumann,(x,0,1))-sp.Rational(1,2),'neumann_exact_mass')

# d'Alembert includes a velocity integral, not merely two displaced shapes.
bump = (1-x*x)**3
primitive = sp.integrate(bump,x)
values['compactBumpPrimitive'] = str(primitive)
for order in [0,1,2]:
    for endpoint in [-1,1]:
        identity(sp.diff(bump,x,order).subs(x,endpoint),'bump_C2_join')
def bump_numeric(z):
    return (1-z*z)**3 if abs(z)<1 else 0.0
def primitive_numeric(z):
    z = max(-1,min(1,z))
    return z-z**3+3*z**5/5-z**7/7
for time in [.1,.5,1]:
    for location in [-2,-.7,0,.6,2]:
        actual = (bump_numeric(location-time)+bump_numeric(location+time))/2 + (
                  primitive_numeric(location+time)-primitive_numeric(location-time))/4
        expected = (bump_numeric(location-time)+bump_numeric(location+time))/2 + .25*quad(
            bump_numeric,location-time,location+time,points=[z for z in [-1,1] if location-time<z<location+time],epsabs=1e-12)[0]
        close(actual,expected,'wave_compact_velocity_integral')
standing = sp.sin(sp.pi*x)*sp.cos(sp.pi*t)+sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*t)/4
identity(sp.diff(standing,t,2)-sp.diff(standing,x,2),'wave_standing_pde')
energy = sp.integrate((sp.diff(standing,t)**2+sp.diff(standing,x)**2)/2,(x,0,1))
identity(energy-5*sp.pi**2/16,'wave_standing_energy')
values['standingEnergy'] = str(sp.simplify(energy))

# Poisson and boundary compatibility, including nullspace mean selection.
for a,b in [(2,0),(-1,2),(3,-2)]:
    for left in [-1,0,1]:
        for mean in [-1,0,1]:
            right = sp.Rational(a)+sp.Rational(b,2)-left
            candidate = (left*x-sp.Rational(a,2)*x*x-sp.Rational(b,6)*x**3+
                         sp.Rational(mean)-sp.Rational(left,2)+sp.Rational(a,6)+sp.Rational(b,24))
            identity(-sp.diff(candidate,x,2)-(a+b*x),'poisson_pde')
            identity(sp.diff(candidate,x).subs(x,0)-left,'poisson_left_flux')
            identity(-sp.diff(candidate,x).subs(x,1)-right,'poisson_right_flux')
            identity(sp.integrate(candidate,(x,0,1))-mean,'poisson_mean_selection')

# Green's kink: integration on the two smooth pieces equals point evaluation.
for location in [sp.Rational(1,4),sp.Rational(1,3),sp.Rational(3,5)]:
    for power in range(1,6):
        test = x**power*(1-x)
        pairing = sp.integrate((1-location)*sp.diff(test,x),(x,0,location)) + sp.integrate(
            -location*sp.diff(test,x),(x,location,1))
        identity(pairing-test.subs(x,location),'green_weak_identity')
for n in [1,3,5]:
    y = sp.symbols('y',real=True)
    harmonic = sp.sin(n*sp.pi*x)*sp.sinh(n*sp.pi*y)/sp.sinh(n*sp.pi)
    identity(sp.diff(harmonic,x,2)+sp.diff(harmonic,y,2),'harmonic_2d_pde')
    identity(harmonic.subs(y,1)-sp.sin(n*sp.pi*x),'harmonic_top_boundary')
values['harmonicDepthRatios'] = {str(n):math.sinh(n*math.pi/2)/math.sinh(n*math.pi) for n in [1,3,5]}

# Periodically forced depth is a periodic steady regime, not an arbitrary startup.
delta = sp.sqrt(2*alpha/omega)
periodic = sp.exp(-x/delta)*sp.cos(omega*t-x/delta)
identity(sp.diff(periodic,t)-alpha*sp.diff(periodic,x,2),'periodic_depth_pde')
values['seasonalOneDepth'] = {'amplitudeRatio':math.exp(-1),'phaseLagRadians':1}

# Heat kernel and a fully specified forced physical rod, with a uniform finish criterion.
kernel = sp.exp(-x*x/(4*alpha*t))/sp.sqrt(4*sp.pi*alpha*t)
identity(sp.diff(kernel,t)-alpha*sp.diff(kernel,x,2),'kernel_pde')
for diffusivity,time in [(.2,.1),(.5,1),(2,.25)]:
    density = lambda z: math.exp(-z*z/(4*diffusivity*time))/math.sqrt(4*math.pi*diffusivity*time)
    close(quad(density,-np.inf,np.inf)[0],1,'kernel_mass')
    close(quad(lambda z:z*z*density(z),-np.inf,np.inf)[0],2*diffusivity*time,'kernel_variance')
rod = x*(1-x)+sp.exp(-sp.pi**2*t/4)*sp.sin(sp.pi*x)/5
identity(2*sp.diff(rod,t)-sp.diff(rod,x,2)/2-1,'forced_rod_pde')
outflow = (sp.diff(rod,x).subs(x,0)-sp.diff(rod,x).subs(x,1))/2
identity(2*sp.diff(sp.integrate(rod,(x,0,1)),t)-(1-outflow),'forced_rod_balance')
settling = 4*sp.log(4)/sp.pi**2
identity((rod-x*(1-x)).subs({x:sp.Rational(1,2),t:settling})-sp.Rational(1,20),'uniform_settling_threshold')
values['forcedRod'] = {'capacity':2,'conductivity':.5,'source':1,'diffusivity':.25,
    'steadyCenter':.25,'initialCenter':.45,'uniformTolerance':.05,'settlingSeconds':float(settling)}

# Weak versus entropy-admissible shocks; both satisfy the same jump balance.
for left,right in [(2,0),(0,2),(3,-1),(-1,3)]:
    l,r=sp.Rational(left),sp.Rational(right)
    speed=(l+r)/2
    identity(speed*(r-l)-(r*r-l*l)/2,'rankine_hugoniot')
    production=(r**3-l**3)/3-speed*(r*r-l*l)/2
    assert (production<=0)==(left>=right)
    checks['entropy_orientation']+=1
    values.setdefault('shockPairs',[]).append({'left':left,'right':right,'speed':str(speed),'entropyProduction':str(production)})

result = {'checkedAt':datetime.now(timezone.utc).isoformat(),'topicId':topic_id,
          'status':'design fixtures only; no production implementation or browser claim',
          'checks':dict(checks),'maxScaledError':max_error,'values':values,
          'scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(directory/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
