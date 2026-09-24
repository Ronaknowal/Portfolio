"""Build exact displayed fixtures by executing complete, independently runnable programs."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
archive = json.loads((ROOT/'docs/teaching/evidence/differential-geometry-original-content.json').read_text())
examples = {}

def add(key, title, question, code, explanation):
    code = textwrap.dedent(code).strip()
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, check=True)
    examples[key] = dict(title=title, question=question, code=code, expected=result.stdout.rstrip(), explanation=explanation, language='python')

add('charts', 'One circle point, two coordinate labels',
    'At 225 degrees, do the labels −135 degrees and 225 degrees describe different points?', r'''
    import math

    def labels(degrees):
        if not 0 <= degrees <= 360:
            raise ValueError("Use an angle in [0, 360].")
        alpha = None if degrees == 180 else ((degrees + 180) % 360 - 180)
        beta = None if degrees in (0, 360) else degrees
        return alpha, beta

    for degrees in [0, 135, 180, 225, 360]:
        print(degrees, labels(degrees))
    a, b = (math.radians(value) for value in labels(225))
    print("same point:", math.dist((math.cos(a), math.sin(a)),
                                  (math.cos(b), math.sin(b))) < 1e-12)
    ''', 'A chart excludes one point so its angle can vary continuously in an open interval. The point remains valid when one label is unavailable; the other chart covers it.')

add('metric', 'Turn a differential into the metric gradient',
    'For x=u+v, y=v and f=2x−y, why are the differential coefficients (2,1) but the gradient components (3,−1)?', r'''
    # Install once: python -m pip install numpy
    import numpy as np

    def metric_gradient(shear, vertical_cost):
        if not np.isfinite([shear, vertical_cost]).all() or vertical_cost <= 0:
            raise ValueError("Use finite shear and positive cost.")
        S = np.array([[1., shear], [0., 1.]])
        B = np.diag([1., vertical_cost**2])
        G = S.T @ B @ S
        differential = S.T @ np.array([2., -1.])
        gradient = np.linalg.solve(G, differential)
        return G, differential, gradient, S @ gradient

    G, a, gradient, world = metric_gradient(1., 1.)
    print("metric:", G.tolist())
    print("differential:", a.tolist())
    print("gradient:", gradient.tolist())
    print("world gradient:", world.tolist())
    v = np.array([1., 2.])
    print("df(v) and <grad,v>:", float(a @ v), float(gradient @ G @ v))
    print("cost 2, world:", metric_gradient(1., 2.)[3].tolist())
    ''', 'G converts a tangent vector into its metric pairing. Solving G grad = a preserves the differential. Relabeling alone leaves the world gradient (2,−1); increasing the cost of y-motion changes it to (2,−0.25).')

add('area', 'Calculate a sphere patch and unequal-area bands',
    'Do three bands of equal polar-angle width cover equal areas?', r'''
    import math
    import numpy as np

    def patch(theta, phi, radius):
        if radius <= 0 or not 0 < theta < math.pi:
            raise ValueError("Use R>0 and a chart away from the poles.")
        dtheta = radius * np.array([math.cos(theta)*math.cos(phi),
                                   math.cos(theta)*math.sin(phi), -math.sin(theta)])
        dphi = radius * np.array([-math.sin(theta)*math.sin(phi),
                                 math.sin(theta)*math.cos(phi), 0.])
        J = np.column_stack([dtheta, dphi])
        return J.T @ J, np.linalg.norm(np.cross(dtheta, dphi))

    G, weight = patch(math.pi/3, math.pi/6, 2.)
    print("metric:", np.round(G, 6).tolist())
    print("area weight:", round(float(weight), 6))
    for lower, upper in [(0, 30), (30, 60), (60, 90)]:
        share = (math.cos(math.radians(lower))-math.cos(math.radians(upper)))/2
        print((lower, upper), "fraction of full sphere:", round(share, 6))
    ''', 'At R=2 and θ=π/3, G=diag(4,3), so the area multiplier is √12. A 30-degree band near the pole covers much less area than one adjacent to the equator.')

add('paths', 'Separate chord, arc and fixed-time energy',
    'Between perpendicular unit directions, which path stays on the sphere, and how much longer is the long great-circle route?', r'''
    import math

    def route_values(radius, angle, duration):
        if radius <= 0 or duration <= 0 or not 0 <= angle <= math.pi:
            raise ValueError("Require R,T>0 and a separation in [0, pi].")
        short = radius*angle
        long = radius*(2*math.pi-angle)
        chord = 2*radius*math.sin(angle/2)
        return short, long, chord, short**2/(2*duration)

    short, long, chord, energy = route_values(1., math.pi/2, 2.)
    print("short, long, chord:", *(round(x, 6) for x in (short, long, chord)))
    print("constant-speed energy over T=2:", round(energy, 6))
    print("chord midpoint norm:", round(math.hypot(.5, .5), 6))
    print("antipodal distance:", round(route_values(1., math.pi, 1.)[0], 6))
    ''', 'The chord is shorter because it leaves the constraint. Both great-circle routes are geodesics, but the long one does not minimize endpoint distance. At antipodes the shortest length is π and the joining direction is nonunique.')

original = archive['blocks'][0]['text']
add('original', 'Preserved original: project, then normalize',
    'Starting at (1,0), what survives after removing the radial part of (1,2)?', original,
    'The tangent gradient is (0,2). The step (1,−0.2) leaves the circle; normalization returns approximately (0.981,−0.196). The printed angle is the distance between the separate orthogonal endpoint pair in the program, not the angle of that update.')

add('maps', 'Compare the exact sphere map with a retraction',
    'For a tangent displacement of length 0.8, do exponential and normalized steps travel the same angle?', r'''
    import math
    import numpy as np

    def sphere_exp(x, v):
        x, v = np.asarray(x, float), np.asarray(v, float)
        if x.shape != (3,) or v.shape != (3,) or not np.isfinite([x, v]).all():
            raise ValueError("Use finite three-component inputs.")
        if abs(np.linalg.norm(x)-1) > 1e-10 or abs(x @ v) > 1e-10:
            raise ValueError("Require a unit point and tangent vector.")
        length = np.linalg.norm(v)
        return math.cos(length)*x + np.sinc(length/math.pi)*v

    def sphere_log(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.shape != (3,) or y.shape != (3,) or not np.isfinite([x, y]).all():
            raise ValueError("Use finite three-component inputs.")
        if max(abs(np.linalg.norm(x)-1), abs(np.linalg.norm(y)-1)) > 1e-10:
            raise ValueError("Use unit endpoints.")
        cross = np.cross(x, y)
        sine, cosine = np.linalg.norm(cross), float(x @ y)
        if cosine < 0 and sine < 1e-8:
            raise ValueError("Shortest logarithm unresolved at/near the antipode.")
        if sine == 0:
            return np.zeros(3)
        return math.atan2(sine, cosine)/sine * np.cross(cross, x)

    x, v = np.array([1., 0, 0]), np.array([0., .8, 0])
    y = sphere_exp(x, v)
    retracted = (x+v)/np.linalg.norm(x+v)
    print("Exp:", np.round(y, 6).tolist())
    print("Log(Exp):", np.round(sphere_log(x, y), 6).tolist())
    print("angles:", .8, round(math.atan(.8), 6))
    print("both unit:", np.isclose(np.linalg.norm(y), 1),
          np.isclose(np.linalg.norm(retracted), 1))
    ''', 'Both endpoints are valid. Exp follows the geodesic for the supplied tangent length; normalization follows angle atan(0.8). They agree to first order near a zero step, not at arbitrary finite steps.')

add('connection', 'A straight line with nonzero coordinate acceleration',
    'For the Cartesian path (t,1) at t=1, how can r and θ accelerate while the physical path stays straight?', r'''
    import math

    def polar_terms(time, height):
        if height <= 0:
            raise ValueError("This chart fixture uses height > 0.")
        r = math.hypot(time, height)
        dr, dtheta = time/r, -height/r**2
        ddr, ddtheta = height**2/r**3, 2*height*time/r**4
        return (ddr, -r*dtheta*dtheta), (ddtheta, 2*dr*dtheta/r)

    radial, angular = polar_terms(1., 1.)
    print("radial derivative, correction:", *(round(x, 6) for x in radial))
    print("angular derivative, correction:", *(round(x, 6) for x in angular))
    print("covariant acceleration:", round(sum(radial), 12), round(sum(angular), 12))
    ''', 'The moving basis contributes exactly the opposite terms. Nonzero Christoffel symbols describe this coordinate system even though the plane has zero Riemann curvature.')

add('transport', 'Carry an arrow around a spherical triangle',
    'Does an arrow transported without turning locally return to its starting direction after N→A→B→N?', r'''
    import math
    import numpy as np

    def transport(x, y, v):
        x, y, v = map(lambda a: np.asarray(a, float), (x, y, v))
        if x.shape != (3,) or y.shape != (3,) or v.shape != (3,) or not np.isfinite([x,y,v]).all():
            raise ValueError("Use finite three-component inputs.")
        if max(abs(np.linalg.norm(x)-1), abs(np.linalg.norm(y)-1)) > 1e-10:
            raise ValueError("Require unit endpoints.")
        if abs(x @ v) > 1e-10 or 1+x @ y < 1e-6:
            raise ValueError("Require tangency and a resolved shorter route.")
        return v-(v @ y)/(1+x @ y)*(x+y)

    N, A, B = np.eye(3)[[2, 0, 1]]
    arrow = np.array([1., 0, 0])
    for x, y in [(N, A), (A, B), (B, N)]:
        arrow = transport(x, y, arrow)
        print(np.round(arrow, 6).tolist())
    turn = math.atan2(arrow[1], arrow[0])
    print("turn in degrees:", round(math.degrees(turn), 6))
    print("norm:", round(float(np.linalg.norm(arrow)), 6))
    ''', 'The arrow ends at (0,1,0), rotated by 90 degrees. Length was preserved on every leg. The result concerns this specified loop and its orientation, not merely its endpoint.')

add('curvature', 'Compute connection and curvature from a metric',
    'Can nonzero connection coefficients coexist with zero curvature?', r'''
    # Install once: python -m pip install sympy
    import sympy as s

    u, v, R = s.symbols("u v R", positive=True)
    def curvature_from_metric(a):
        coordinates = [u, v]
        G, inverse = s.diag(1, a*a), s.diag(1, 1/(a*a))
        gamma = [[[s.simplify(sum(inverse[k,l]*(s.diff(G[l,j], coordinates[i])
                   + s.diff(G[l,i], coordinates[j])-s.diff(G[i,j], coordinates[l]))
                   for l in range(2))/2) for j in range(2)] for i in range(2)] for k in range(2)]
        numerator = s.diff(gamma[0][1][1], u)-s.diff(gamma[0][0][1], v)
        numerator += sum(gamma[l][1][1]*gamma[0][0][l]
                         - gamma[l][0][1]*gamma[0][1][l] for l in range(2))
        return gamma[0][1][1], s.simplify(numerator/(a*a))

    for name, a in [("plane", u), ("cylinder", R),
                    ("sphere", R*s.sin(u/R)), ("hyperbolic", R*s.exp(u/R))]:
        connection, K = curvature_from_metric(a)
        print(name, "Gamma_u_vv =", connection, "K =", K)
    ''', 'This program constructs the coefficients from the metric before forming curvature. The polar plane has Γᵘᵥᵥ=−u yet K=0; the declared sphere convention gives positive curvature.')

add('means', 'A mean depends on the loss and the geometry',
    'For directions 0°, 0°, 90°, does averaging coordinates and renormalizing equal minimizing squared angular distances?', r'''
    import math

    angles = [0., 0., math.pi/2]
    extrinsic = math.atan2(sum(math.sin(a) for a in angles),
                          sum(math.cos(a) for a in angles))
    intrinsic = sum(angles)/len(angles)  # This data lies in one short arc.
    def angular_loss(mean, data):
        return sum(math.atan2(math.sin(a-mean), math.cos(a-mean))**2 for a in data)

    print("extrinsic / intrinsic degrees:", round(math.degrees(extrinsic), 6),
          round(math.degrees(intrinsic), 6))
    print("angular losses:", round(angular_loss(extrinsic, angles), 6),
          round(angular_loss(intrinsic, angles), 6))
    antipodal = [0., math.pi]
    print("antipodal midpoint losses:", round(angular_loss(math.pi/2, antipodal), 6),
          round(angular_loss(-math.pi/2, antipodal), 6))
    ''', 'The extrinsic mean minimizes summed squared chord distances when the resultant is nonzero; the intrinsic mean minimizes squared angular distances. Antipodal data here has two intrinsic midpoint minimizers, while its extrinsic resultant is zero.')

add('spd', 'Use matrix functions for a covariance geodesic',
    'What midpoint does the affine-invariant SPD metric choose between diag(1,4) and diag(4,1)?', r'''
    import numpy as np

    def matrix_power_spd(A, power):
        A = np.asarray(A, float)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.isfinite(A).all():
            raise ValueError("Require a finite square matrix.")
        if not np.allclose(A, A.T, rtol=0, atol=1e-12):
            raise ValueError("Require a symmetric matrix.")
        values, Q = np.linalg.eigh(A)
        if np.min(values) <= 0:
            raise ValueError("Require positive eigenvalues.")
        return (Q * values**power) @ Q.T

    def affine_path(A, B, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError("Use a path fraction in [0,1].")
        root, inverse = matrix_power_spd(A, .5), matrix_power_spd(A, -.5)
        relative = inverse @ B @ inverse
        return root @ matrix_power_spd(relative, fraction) @ root

    def affine_distance(A, B):
        inverse = matrix_power_spd(A, -.5)
        relative = inverse @ B @ inverse
        matrix_power_spd(relative, 1)  # Check its positive-definite domain.
        return float(np.linalg.norm(np.log(np.linalg.eigvalsh(relative))))

    A, B = np.diag([1., 4.]), np.diag([4., 1.])
    print("arithmetic midpoint:", ((A+B)/2).tolist())
    print("affine midpoint:", np.round(affine_path(A, B, .5), 6).tolist())
    print("affine distance:", round(affine_distance(A, B), 6))
    ''', 'The affine midpoint is 2I rather than 2.5I. Eigensystem powers are matrix functions, not entrywise powers. These small, well-conditioned float64 routines demonstrate the geometry; production covariance processing also needs a justified numerical conditioning policy.')

add('fisher', 'Flatten the Bernoulli Fisher metric locally',
    'If the Fisher coefficient grows without bound near p=0, must that boundary be infinitely far away?', r'''
    import math

    def fisher_coordinate(p):
        if not 0 < p < 1:
            raise ValueError("The regular Bernoulli manifold has 0<p<1.")
        return 2*math.asin(math.sqrt(p))

    for p in [.01, .25, .5]:
        u = fisher_coordinate(p)
        print(p, "u =", round(u, 6), "distance to boundary limit =", round(u, 6))
    distance = abs(fisher_coordinate(.75)-fisher_coordinate(.25))
    print("distance .25 to .75:", round(distance, 6))
    print("hyperbolic vertical distance 1 to 4:", round(math.log(4), 6))
    ''', 'The coordinate u=2 asin√p makes the line element du². Its interval is (0,π), so the omitted endpoints are a finite distance away. The hyperbolic value concerns a different metric, (dx²+dy²)/y², along a vertical line.')

add('laplacian', 'Apply gradient and metric volume together',
    'What does the sphere’s Laplace–Beltrami operator do to the height function cos θ?', r'''
    import sympy as s

    theta, phi, R = s.symbols("theta phi R", positive=True)
    Ginv = s.diag(1/R**2, 1/(R**2*s.sin(theta)**2))
    volume = R**2*s.sin(theta)  # Chart 0 < theta < pi.
    coordinates = [theta, phi]
    def laplacian(f):
        gradient = Ginv * s.Matrix([s.diff(f, c) for c in coordinates])
        return s.simplify(sum(s.diff(volume*gradient[i], coordinates[i])
                              for i in range(2))/volume)

    print("Delta(cos theta):", laplacian(s.cos(theta)))
    print("Delta(constant):", laplacian(s.Integer(1)))
    ''', 'With Δ=div grad, the height function has eigenvalue −2/R²; the positive-semidefinite convention instead uses −Δ. The chart expression has coordinate singularities at poles, but this smooth function and its final result extend across them.')

add('optimize', 'A complete constrained optimization with an independent reference',
    'Can a decreasing, valid sphere iteration locate the smallest Rayleigh value, and can stationarity alone certify it?', r'''
    import numpy as np

    def minimize_rayleigh(A, initial, tolerance=1e-8, limit=500):
        A, x = np.asarray(A, float), np.asarray(initial, float).copy()
        if A.ndim != 2 or A.shape != (x.size, x.size) or x.ndim != 1:
            raise ValueError("Check the matrix/vector shapes.")
        if not np.isfinite(A).all() or not np.isfinite(x).all() or not np.allclose(A, A.T):
            raise ValueError("Use finite data and a symmetric matrix.")
        if not np.isfinite(tolerance) or not isinstance(limit, int):
            raise ValueError("Use a finite tolerance and integer work limit.")
        if np.linalg.norm(x) == 0 or tolerance <= 0 or not 1 <= limit <= 10000:
            raise ValueError("Use a nonzero initial vector and positive work/tolerance.")
        x /= np.linalg.norm(x)
        history = []
        for iteration in range(limit):
            value = float(x @ A @ x)
            gradient = 2*(A @ x-value*x)
            size = float(np.linalg.norm(gradient))
            history.append((value, size, float(np.linalg.norm(x))))
            if size <= tolerance:
                return x, "stationary", history
            step = 1.
            for trial in range(50):
                candidate = x-step*gradient
                candidate /= np.linalg.norm(candidate)
                candidate_value = float(candidate @ A @ candidate)
                if candidate_value <= value-1e-4*step*size**2:
                    x = candidate
                    break
                step *= .5
            else:
                return x, "line-search limit", history
        return x, "iteration limit", history

    A = np.diag([1., 3., 6.])
    x, status, history = minimize_rayleigh(A, [1., 1., 1.])
    print("status:", status)
    print("value / eigenvalue reference:", round(float(x @ A @ x), 8),
          round(float(np.linalg.eigvalsh(A)[0]), 8))
    print("unit / monotone:", abs(np.linalg.norm(x)-1) < 1e-12,
          all(b[0] <= a[0]+1e-12 for a,b in zip(history, history[1:])))
    residual = np.linalg.norm(2*(A @ x-float(x @ A @ x)*x))
    print("gradient norm:", format(float(residual), ".3e"))
    print("feasibility residual:", format(abs(float(np.linalg.norm(x))-1), ".3e"))
    maximum, maximum_status, _ = minimize_rayleigh(A, [0., 0., 1.])
    print("maximum start:", maximum_status, float(maximum @ A @ maximum))
    # Independent changed task: rotate the eigenspaces and change the spectrum.
    Q = np.array([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]])
    changed = Q @ np.diag([2., 5., 9.]) @ Q.T
    answer, changed_status, _ = minimize_rayleigh(changed, [1., 2., 3.])
    print("changed task:", changed_status, round(float(answer @ changed @ answer), 8))
    changed_residual = np.linalg.norm(2*(changed @ answer-float(answer @ changed @ answer)*answer))
    print("changed gradient norm:", format(float(changed_residual), ".3e"))
    ''', 'The first and changed fixtures reach values 1 and 2, checked against independent eigenvalues. A maximum eigenvector also has zero gradient and immediately reports stationary at value 6. The status deliberately does not promise a global minimum; the Hessian and reference diagnose the difference.')

target = ROOT/'src/learn/data/differential-geometry-examples.js'
target.write_text('// Complete programs executed by scripts/build-differential-geometry-examples.py.\n'
                  + 'export const differentialGeometryExamples = '+json.dumps(examples, ensure_ascii=False, indent=2)+';\n', encoding='utf8')
assert examples['original']['code'] == original
assert examples['original']['expected'] == archive['blocks'][1]['text']
print(json.dumps({key: row['expected'] for key,row in examples.items()}, indent=2))
