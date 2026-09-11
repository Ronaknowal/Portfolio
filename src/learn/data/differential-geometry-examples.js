// Complete programs executed by scripts/build-differential-geometry-examples.py.
export const differentialGeometryExamples = {
  "charts": {
    "title": "One circle point, two coordinate labels",
    "question": "At 225 degrees, do the labels −135 degrees and 225 degrees describe different points?",
    "code": "import math\n\ndef labels(degrees):\n    if not 0 <= degrees <= 360:\n        raise ValueError(\"Use an angle in [0, 360].\")\n    alpha = None if degrees == 180 else ((degrees + 180) % 360 - 180)\n    beta = None if degrees in (0, 360) else degrees\n    return alpha, beta\n\nfor degrees in [0, 135, 180, 225, 360]:\n    print(degrees, labels(degrees))\na, b = (math.radians(value) for value in labels(225))\nprint(\"same point:\", math.dist((math.cos(a), math.sin(a)),\n                              (math.cos(b), math.sin(b))) < 1e-12)",
    "expected": "0 (0, None)\n135 (135, 135)\n180 (None, 180)\n225 (-135, 225)\n360 (0, None)\nsame point: True",
    "explanation": "A chart excludes one point so its angle can vary continuously in an open interval. The point remains valid when one label is unavailable; the other chart covers it.",
    "language": "python"
  },
  "metric": {
    "title": "Turn a differential into the metric gradient",
    "question": "For x=u+v, y=v and f=2x−y, why are the differential coefficients (2,1) but the gradient components (3,−1)?",
    "code": "# Install once: python -m pip install numpy\nimport numpy as np\n\ndef metric_gradient(shear, vertical_cost):\n    if not np.isfinite([shear, vertical_cost]).all() or vertical_cost <= 0:\n        raise ValueError(\"Use finite shear and positive cost.\")\n    S = np.array([[1., shear], [0., 1.]])\n    B = np.diag([1., vertical_cost**2])\n    G = S.T @ B @ S\n    differential = S.T @ np.array([2., -1.])\n    gradient = np.linalg.solve(G, differential)\n    return G, differential, gradient, S @ gradient\n\nG, a, gradient, world = metric_gradient(1., 1.)\nprint(\"metric:\", G.tolist())\nprint(\"differential:\", a.tolist())\nprint(\"gradient:\", gradient.tolist())\nprint(\"world gradient:\", world.tolist())\nv = np.array([1., 2.])\nprint(\"df(v) and <grad,v>:\", float(a @ v), float(gradient @ G @ v))\nprint(\"cost 2, world:\", metric_gradient(1., 2.)[3].tolist())",
    "expected": "metric: [[1.0, 1.0], [1.0, 2.0]]\ndifferential: [2.0, 1.0]\ngradient: [3.0, -1.0]\nworld gradient: [2.0, -1.0]\ndf(v) and <grad,v>: 4.0 4.0\ncost 2, world: [2.0, -0.25]",
    "explanation": "G converts a tangent vector into its metric pairing. Solving G grad = a preserves the differential. Relabeling alone leaves the world gradient (2,−1); increasing the cost of y-motion changes it to (2,−0.25).",
    "language": "python"
  },
  "area": {
    "title": "Calculate a sphere patch and unequal-area bands",
    "question": "Do three bands of equal polar-angle width cover equal areas?",
    "code": "import math\nimport numpy as np\n\ndef patch(theta, phi, radius):\n    if radius <= 0 or not 0 < theta < math.pi:\n        raise ValueError(\"Use R>0 and a chart away from the poles.\")\n    dtheta = radius * np.array([math.cos(theta)*math.cos(phi),\n                               math.cos(theta)*math.sin(phi), -math.sin(theta)])\n    dphi = radius * np.array([-math.sin(theta)*math.sin(phi),\n                             math.sin(theta)*math.cos(phi), 0.])\n    J = np.column_stack([dtheta, dphi])\n    return J.T @ J, np.linalg.norm(np.cross(dtheta, dphi))\n\nG, weight = patch(math.pi/3, math.pi/6, 2.)\nprint(\"metric:\", np.round(G, 6).tolist())\nprint(\"area weight:\", round(float(weight), 6))\nfor lower, upper in [(0, 30), (30, 60), (60, 90)]:\n    share = (math.cos(math.radians(lower))-math.cos(math.radians(upper)))/2\n    print((lower, upper), \"fraction of full sphere:\", round(share, 6))",
    "expected": "metric: [[4.0, -0.0], [-0.0, 3.0]]\narea weight: 3.464102\n(0, 30) fraction of full sphere: 0.066987\n(30, 60) fraction of full sphere: 0.183013\n(60, 90) fraction of full sphere: 0.25",
    "explanation": "At R=2 and θ=π/3, G=diag(4,3), so the area multiplier is √12. A 30-degree band near the pole covers much less area than one adjacent to the equator.",
    "language": "python"
  },
  "paths": {
    "title": "Separate chord, arc and fixed-time energy",
    "question": "Between perpendicular unit directions, which path stays on the sphere, and how much longer is the long great-circle route?",
    "code": "import math\n\ndef route_values(radius, angle, duration):\n    if radius <= 0 or duration <= 0 or not 0 <= angle <= math.pi:\n        raise ValueError(\"Require R,T>0 and a separation in [0, pi].\")\n    short = radius*angle\n    long = radius*(2*math.pi-angle)\n    chord = 2*radius*math.sin(angle/2)\n    return short, long, chord, short**2/(2*duration)\n\nshort, long, chord, energy = route_values(1., math.pi/2, 2.)\nprint(\"short, long, chord:\", *(round(x, 6) for x in (short, long, chord)))\nprint(\"constant-speed energy over T=2:\", round(energy, 6))\nprint(\"chord midpoint norm:\", round(math.hypot(.5, .5), 6))\nprint(\"antipodal distance:\", round(route_values(1., math.pi, 1.)[0], 6))",
    "expected": "short, long, chord: 1.570796 4.712389 1.414214\nconstant-speed energy over T=2: 0.61685\nchord midpoint norm: 0.707107\nantipodal distance: 3.141593",
    "explanation": "The chord is shorter because it leaves the constraint. Both great-circle routes are geodesics, but the long one does not minimize endpoint distance. At antipodes the shortest length is π and the joining direction is nonunique.",
    "language": "python"
  },
  "original": {
    "title": "Preserved original: project, then normalize",
    "question": "Starting at (1,0), what survives after removing the radial part of (1,2)?",
    "code": "import math\n\nx = (1.0, 0.0)       # point on the unit circle\ngradient = (1.0, 2.0) # Euclidean gradient at x\n\ndot = sum(a * b for a, b in zip(x, gradient))\ntangent_gradient = tuple(g - dot * xi for g, xi in zip(gradient, x))\neta = 0.1\ncandidate = tuple(xi - eta * gi for xi, gi in zip(x, tangent_gradient))\nnorm = math.sqrt(sum(value * value for value in candidate))\nnext_x = tuple(value / norm for value in candidate)  # retraction by normalisation\n\nprint(tangent_gradient)\nprint(tuple(round(value, 3) for value in next_x))\nprint(round(math.acos(0), 3))  # distance between orthogonal unit vectors",
    "expected": "(0.0, 2.0)\n(0.981, -0.196)\n1.571",
    "explanation": "The tangent gradient is (0,2). The step (1,−0.2) leaves the circle; normalization returns approximately (0.981,−0.196). The printed angle is the distance between the separate orthogonal endpoint pair in the program, not the angle of that update.",
    "language": "python"
  },
  "maps": {
    "title": "Compare the exact sphere map with a retraction",
    "question": "For a tangent displacement of length 0.8, do exponential and normalized steps travel the same angle?",
    "code": "import math\nimport numpy as np\n\ndef sphere_exp(x, v):\n    x, v = np.asarray(x, float), np.asarray(v, float)\n    if x.shape != (3,) or v.shape != (3,) or not np.isfinite([x, v]).all():\n        raise ValueError(\"Use finite three-component inputs.\")\n    if abs(np.linalg.norm(x)-1) > 1e-10 or abs(x @ v) > 1e-10:\n        raise ValueError(\"Require a unit point and tangent vector.\")\n    length = np.linalg.norm(v)\n    return math.cos(length)*x + np.sinc(length/math.pi)*v\n\ndef sphere_log(x, y):\n    x, y = np.asarray(x, float), np.asarray(y, float)\n    if x.shape != (3,) or y.shape != (3,) or not np.isfinite([x, y]).all():\n        raise ValueError(\"Use finite three-component inputs.\")\n    if max(abs(np.linalg.norm(x)-1), abs(np.linalg.norm(y)-1)) > 1e-10:\n        raise ValueError(\"Use unit endpoints.\")\n    cross = np.cross(x, y)\n    sine, cosine = np.linalg.norm(cross), float(x @ y)\n    if cosine < 0 and sine < 1e-8:\n        raise ValueError(\"Shortest logarithm unresolved at/near the antipode.\")\n    if sine == 0:\n        return np.zeros(3)\n    return math.atan2(sine, cosine)/sine * np.cross(cross, x)\n\nx, v = np.array([1., 0, 0]), np.array([0., .8, 0])\ny = sphere_exp(x, v)\nretracted = (x+v)/np.linalg.norm(x+v)\nprint(\"Exp:\", np.round(y, 6).tolist())\nprint(\"Log(Exp):\", np.round(sphere_log(x, y), 6).tolist())\nprint(\"angles:\", .8, round(math.atan(.8), 6))\nprint(\"both unit:\", np.isclose(np.linalg.norm(y), 1),\n      np.isclose(np.linalg.norm(retracted), 1))",
    "expected": "Exp: [0.696707, 0.717356, 0.0]\nLog(Exp): [0.0, 0.8, 0.0]\nangles: 0.8 0.674741\nboth unit: True True",
    "explanation": "Both endpoints are valid. Exp follows the geodesic for the supplied tangent length; normalization follows angle atan(0.8). They agree to first order near a zero step, not at arbitrary finite steps.",
    "language": "python"
  },
  "connection": {
    "title": "A straight line with nonzero coordinate acceleration",
    "question": "For the Cartesian path (t,1) at t=1, how can r and θ accelerate while the physical path stays straight?",
    "code": "import math\n\ndef polar_terms(time, height):\n    if height <= 0:\n        raise ValueError(\"This chart fixture uses height > 0.\")\n    r = math.hypot(time, height)\n    dr, dtheta = time/r, -height/r**2\n    ddr, ddtheta = height**2/r**3, 2*height*time/r**4\n    return (ddr, -r*dtheta*dtheta), (ddtheta, 2*dr*dtheta/r)\n\nradial, angular = polar_terms(1., 1.)\nprint(\"radial derivative, correction:\", *(round(x, 6) for x in radial))\nprint(\"angular derivative, correction:\", *(round(x, 6) for x in angular))\nprint(\"covariant acceleration:\", round(sum(radial), 12), round(sum(angular), 12))",
    "expected": "radial derivative, correction: 0.353553 -0.353553\nangular derivative, correction: 0.5 -0.5\ncovariant acceleration: 0.0 0.0",
    "explanation": "The moving basis contributes exactly the opposite terms. Nonzero Christoffel symbols describe this coordinate system even though the plane has zero Riemann curvature.",
    "language": "python"
  },
  "transport": {
    "title": "Carry an arrow around a spherical triangle",
    "question": "Does an arrow transported without turning locally return to its starting direction after N→A→B→N?",
    "code": "import math\nimport numpy as np\n\ndef transport(x, y, v):\n    x, y, v = map(lambda a: np.asarray(a, float), (x, y, v))\n    if x.shape != (3,) or y.shape != (3,) or v.shape != (3,) or not np.isfinite([x,y,v]).all():\n        raise ValueError(\"Use finite three-component inputs.\")\n    if max(abs(np.linalg.norm(x)-1), abs(np.linalg.norm(y)-1)) > 1e-10:\n        raise ValueError(\"Require unit endpoints.\")\n    if abs(x @ v) > 1e-10 or 1+x @ y < 1e-6:\n        raise ValueError(\"Require tangency and a resolved shorter route.\")\n    return v-(v @ y)/(1+x @ y)*(x+y)\n\nN, A, B = np.eye(3)[[2, 0, 1]]\narrow = np.array([1., 0, 0])\nfor x, y in [(N, A), (A, B), (B, N)]:\n    arrow = transport(x, y, arrow)\n    print(np.round(arrow, 6).tolist())\nturn = math.atan2(arrow[1], arrow[0])\nprint(\"turn in degrees:\", round(math.degrees(turn), 6))\nprint(\"norm:\", round(float(np.linalg.norm(arrow)), 6))",
    "expected": "[0.0, 0.0, -1.0]\n[0.0, 0.0, -1.0]\n[0.0, 1.0, 0.0]\nturn in degrees: 90.0\nnorm: 1.0",
    "explanation": "The arrow ends at (0,1,0), rotated by 90 degrees. Length was preserved on every leg. The result concerns this specified loop and its orientation, not merely its endpoint.",
    "language": "python"
  },
  "curvature": {
    "title": "Compute connection and curvature from a metric",
    "question": "Can nonzero connection coefficients coexist with zero curvature?",
    "code": "# Install once: python -m pip install sympy\nimport sympy as s\n\nu, v, R = s.symbols(\"u v R\", positive=True)\ndef curvature_from_metric(a):\n    coordinates = [u, v]\n    G, inverse = s.diag(1, a*a), s.diag(1, 1/(a*a))\n    gamma = [[[s.simplify(sum(inverse[k,l]*(s.diff(G[l,j], coordinates[i])\n               + s.diff(G[l,i], coordinates[j])-s.diff(G[i,j], coordinates[l]))\n               for l in range(2))/2) for j in range(2)] for i in range(2)] for k in range(2)]\n    numerator = s.diff(gamma[0][1][1], u)-s.diff(gamma[0][0][1], v)\n    numerator += sum(gamma[l][1][1]*gamma[0][0][l]\n                     - gamma[l][0][1]*gamma[0][1][l] for l in range(2))\n    return gamma[0][1][1], s.simplify(numerator/(a*a))\n\nfor name, a in [(\"plane\", u), (\"cylinder\", R),\n                (\"sphere\", R*s.sin(u/R)), (\"hyperbolic\", R*s.exp(u/R))]:\n    connection, K = curvature_from_metric(a)\n    print(name, \"Gamma_u_vv =\", connection, \"K =\", K)",
    "expected": "plane Gamma_u_vv = -u K = 0\ncylinder Gamma_u_vv = 0 K = 0\nsphere Gamma_u_vv = -R*sin(2*u/R)/2 K = R**(-2)\nhyperbolic Gamma_u_vv = -R*exp(2*u/R) K = -1/R**2",
    "explanation": "This program constructs the coefficients from the metric before forming curvature. The polar plane has Γᵘᵥᵥ=−u yet K=0; the declared sphere convention gives positive curvature.",
    "language": "python"
  },
  "means": {
    "title": "A mean depends on the loss and the geometry",
    "question": "For directions 0°, 0°, 90°, does averaging coordinates and renormalizing equal minimizing squared angular distances?",
    "code": "import math\n\nangles = [0., 0., math.pi/2]\nextrinsic = math.atan2(sum(math.sin(a) for a in angles),\n                      sum(math.cos(a) for a in angles))\nintrinsic = sum(angles)/len(angles)  # This data lies in one short arc.\ndef angular_loss(mean, data):\n    return sum(math.atan2(math.sin(a-mean), math.cos(a-mean))**2 for a in data)\n\nprint(\"extrinsic / intrinsic degrees:\", round(math.degrees(extrinsic), 6),\n      round(math.degrees(intrinsic), 6))\nprint(\"angular losses:\", round(angular_loss(extrinsic, angles), 6),\n      round(angular_loss(intrinsic, angles), 6))\nantipodal = [0., math.pi]\nprint(\"antipodal midpoint losses:\", round(angular_loss(math.pi/2, antipodal), 6),\n      round(angular_loss(-math.pi/2, antipodal), 6))",
    "expected": "extrinsic / intrinsic degrees: 26.565051 30.0\nangular losses: 1.655716 1.644934\nantipodal midpoint losses: 4.934802 4.934802",
    "explanation": "The extrinsic mean minimizes summed squared chord distances when the resultant is nonzero; the intrinsic mean minimizes squared angular distances. Antipodal data here has two intrinsic midpoint minimizers, while its extrinsic resultant is zero.",
    "language": "python"
  },
  "spd": {
    "title": "Use matrix functions for a covariance geodesic",
    "question": "What midpoint does the affine-invariant SPD metric choose between diag(1,4) and diag(4,1)?",
    "code": "import numpy as np\n\ndef matrix_power_spd(A, power):\n    A = np.asarray(A, float)\n    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.isfinite(A).all():\n        raise ValueError(\"Require a finite square matrix.\")\n    if not np.allclose(A, A.T, rtol=0, atol=1e-12):\n        raise ValueError(\"Require a symmetric matrix.\")\n    values, Q = np.linalg.eigh(A)\n    if np.min(values) <= 0:\n        raise ValueError(\"Require positive eigenvalues.\")\n    return (Q * values**power) @ Q.T\n\ndef affine_path(A, B, fraction):\n    if not 0 <= fraction <= 1:\n        raise ValueError(\"Use a path fraction in [0,1].\")\n    root, inverse = matrix_power_spd(A, .5), matrix_power_spd(A, -.5)\n    relative = inverse @ B @ inverse\n    return root @ matrix_power_spd(relative, fraction) @ root\n\ndef affine_distance(A, B):\n    inverse = matrix_power_spd(A, -.5)\n    relative = inverse @ B @ inverse\n    matrix_power_spd(relative, 1)  # Check its positive-definite domain.\n    return float(np.linalg.norm(np.log(np.linalg.eigvalsh(relative))))\n\nA, B = np.diag([1., 4.]), np.diag([4., 1.])\nprint(\"arithmetic midpoint:\", ((A+B)/2).tolist())\nprint(\"affine midpoint:\", np.round(affine_path(A, B, .5), 6).tolist())\nprint(\"affine distance:\", round(affine_distance(A, B), 6))",
    "expected": "arithmetic midpoint: [[2.5, 0.0], [0.0, 2.5]]\naffine midpoint: [[2.0, 0.0], [0.0, 2.0]]\naffine distance: 1.960516",
    "explanation": "The affine midpoint is 2I rather than 2.5I. Eigensystem powers are matrix functions, not entrywise powers. These small, well-conditioned float64 routines demonstrate the geometry; production covariance processing also needs a justified numerical conditioning policy.",
    "language": "python"
  },
  "fisher": {
    "title": "Flatten the Bernoulli Fisher metric locally",
    "question": "If the Fisher coefficient grows without bound near p=0, must that boundary be infinitely far away?",
    "code": "import math\n\ndef fisher_coordinate(p):\n    if not 0 < p < 1:\n        raise ValueError(\"The regular Bernoulli manifold has 0<p<1.\")\n    return 2*math.asin(math.sqrt(p))\n\nfor p in [.01, .25, .5]:\n    u = fisher_coordinate(p)\n    print(p, \"u =\", round(u, 6), \"distance to boundary limit =\", round(u, 6))\ndistance = abs(fisher_coordinate(.75)-fisher_coordinate(.25))\nprint(\"distance .25 to .75:\", round(distance, 6))\nprint(\"hyperbolic vertical distance 1 to 4:\", round(math.log(4), 6))",
    "expected": "0.01 u = 0.200335 distance to boundary limit = 0.200335\n0.25 u = 1.047198 distance to boundary limit = 1.047198\n0.5 u = 1.570796 distance to boundary limit = 1.570796\ndistance .25 to .75: 1.047198\nhyperbolic vertical distance 1 to 4: 1.386294",
    "explanation": "The coordinate u=2 asin√p makes the line element du². Its interval is (0,π), so the omitted endpoints are a finite distance away. The hyperbolic value concerns a different metric, (dx²+dy²)/y², along a vertical line.",
    "language": "python"
  },
  "laplacian": {
    "title": "Apply gradient and metric volume together",
    "question": "What does the sphere’s Laplace–Beltrami operator do to the height function cos θ?",
    "code": "import sympy as s\n\ntheta, phi, R = s.symbols(\"theta phi R\", positive=True)\nGinv = s.diag(1/R**2, 1/(R**2*s.sin(theta)**2))\nvolume = R**2*s.sin(theta)  # Chart 0 < theta < pi.\ncoordinates = [theta, phi]\ndef laplacian(f):\n    gradient = Ginv * s.Matrix([s.diff(f, c) for c in coordinates])\n    return s.simplify(sum(s.diff(volume*gradient[i], coordinates[i])\n                          for i in range(2))/volume)\n\nprint(\"Delta(cos theta):\", laplacian(s.cos(theta)))\nprint(\"Delta(constant):\", laplacian(s.Integer(1)))",
    "expected": "Delta(cos theta): -2*cos(theta)/R**2\nDelta(constant): 0",
    "explanation": "With Δ=div grad, the height function has eigenvalue −2/R²; the positive-semidefinite convention instead uses −Δ. The chart expression has coordinate singularities at poles, but this smooth function and its final result extend across them.",
    "language": "python"
  },
  "optimize": {
    "title": "A complete constrained optimization with an independent reference",
    "question": "Can a decreasing, valid sphere iteration locate the smallest Rayleigh value, and can stationarity alone certify it?",
    "code": "import numpy as np\n\ndef minimize_rayleigh(A, initial, tolerance=1e-8, limit=500):\n    A, x = np.asarray(A, float), np.asarray(initial, float).copy()\n    if A.ndim != 2 or A.shape != (x.size, x.size) or x.ndim != 1:\n        raise ValueError(\"Check the matrix/vector shapes.\")\n    if not np.isfinite(A).all() or not np.isfinite(x).all() or not np.allclose(A, A.T):\n        raise ValueError(\"Use finite data and a symmetric matrix.\")\n    if not np.isfinite(tolerance) or not isinstance(limit, int):\n        raise ValueError(\"Use a finite tolerance and integer work limit.\")\n    if np.linalg.norm(x) == 0 or tolerance <= 0 or not 1 <= limit <= 10000:\n        raise ValueError(\"Use a nonzero initial vector and positive work/tolerance.\")\n    x /= np.linalg.norm(x)\n    history = []\n    for iteration in range(limit):\n        value = float(x @ A @ x)\n        gradient = 2*(A @ x-value*x)\n        size = float(np.linalg.norm(gradient))\n        history.append((value, size, float(np.linalg.norm(x))))\n        if size <= tolerance:\n            return x, \"stationary\", history\n        step = 1.\n        for trial in range(50):\n            candidate = x-step*gradient\n            candidate /= np.linalg.norm(candidate)\n            candidate_value = float(candidate @ A @ candidate)\n            if candidate_value <= value-1e-4*step*size**2:\n                x = candidate\n                break\n            step *= .5\n        else:\n            return x, \"line-search limit\", history\n    return x, \"iteration limit\", history\n\nA = np.diag([1., 3., 6.])\nx, status, history = minimize_rayleigh(A, [1., 1., 1.])\nprint(\"status:\", status)\nprint(\"value / eigenvalue reference:\", round(float(x @ A @ x), 8),\n      round(float(np.linalg.eigvalsh(A)[0]), 8))\nprint(\"unit / monotone:\", abs(np.linalg.norm(x)-1) < 1e-12,\n      all(b[0] <= a[0]+1e-12 for a,b in zip(history, history[1:])))\nresidual = np.linalg.norm(2*(A @ x-float(x @ A @ x)*x))\nprint(\"gradient norm:\", format(float(residual), \".3e\"))\nprint(\"feasibility residual:\", format(abs(float(np.linalg.norm(x))-1), \".3e\"))\nmaximum, maximum_status, _ = minimize_rayleigh(A, [0., 0., 1.])\nprint(\"maximum start:\", maximum_status, float(maximum @ A @ maximum))\n# Independent changed task: rotate the eigenspaces and change the spectrum.\nQ = np.array([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]])\nchanged = Q @ np.diag([2., 5., 9.]) @ Q.T\nanswer, changed_status, _ = minimize_rayleigh(changed, [1., 2., 3.])\nprint(\"changed task:\", changed_status, round(float(answer @ changed @ answer), 8))\nchanged_residual = np.linalg.norm(2*(changed @ answer-float(answer @ changed @ answer)*answer))\nprint(\"changed gradient norm:\", format(float(changed_residual), \".3e\"))",
    "expected": "status: stationary\nvalue / eigenvalue reference: 1.0 1.0\nunit / monotone: True True\ngradient norm: 8.641e-09\nfeasibility residual: 0.000e+00\nmaximum start: stationary 6.0\nchanged task: line-search limit 2.0\nchanged gradient norm: 3.050e-08",
    "explanation": "The first and changed fixtures reach values 1 and 2, checked against independent eigenvalues. A maximum eigenvector also has zero gradient and immediately reports stationary at value 6. The status deliberately does not promise a global minimum; the Hessian and reference diagnose the difference.",
    "language": "python"
  }
};
