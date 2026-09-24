"""Write complete, actually executed topic-owned Python examples."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = {}


def add(key, title, question, code):
    PROGRAMS[key] = {"title": title, "question": question, "code": textwrap.dedent(code).strip(), "language": "python"}


TRIDIAGONAL = '''
def factor_spd(diagonal, off_diagonal):
    """Symmetric positive definite tridiagonal matrix, no pivoting."""
    if not diagonal or len(off_diagonal) != len(diagonal) - 1:
        raise ValueError("Invalid diagonal lengths")
    pivots = list(diagonal)
    multipliers = []
    for row in range(len(pivots)):
        if pivots[row] <= 0:
            raise ValueError("A positive LDL pivot is required")
        if row + 1 < len(pivots):
            factor = off_diagonal[row] / pivots[row]
            multipliers.append(factor)
            pivots[row + 1] -= factor * off_diagonal[row]
    return pivots, multipliers


def solve_factored(factor, rhs):
    pivots, multipliers = factor
    if len(rhs) != len(pivots):
        raise ValueError("Wrong right-hand-side length")
    values = list(rhs)
    for row in range(1, len(values)):
        values[row] -= multipliers[row - 1] * values[row - 1]
    values = [value / pivot for value, pivot in zip(values, pivots)]
    for row in range(len(values) - 2, -1, -1):
        values[row] -= multipliers[row] * values[row + 1]
    return values
'''

POISSON = '''
def poisson(intervals, source, left=0.0, right=0.0, length=1.0):
    if not isinstance(intervals, int) or intervals < 2 or length <= 0:
        raise ValueError("At least two intervals and a positive length are needed")
    h = length / intervals
    nodes = [j * h for j in range(intervals + 1)]
    # The prescribed domain endpoints are exact inputs, not accumulated products.
    nodes[0], nodes[-1] = 0.0, length
    loads = [h*h*source(x) for x in nodes[1:-1]]
    loads[0] += left
    loads[-1] += right
    factor = factor_spd([2.0]*(intervals-1), [-1.0]*(intervals-2))
    values = [left, *solve_factored(factor, loads), right]
    residual = [source(nodes[j]) - (2*values[j]-values[j-1]-values[j+1])/h**2
                for j in range(1, intervals)]
    return nodes, values, residual
'''

add("stencil", "Derive a row with exact fractions", "Which sign and boundary contribution make the quartic's defect positive?", r'''
from fractions import Fraction as F


def target(x):
    return x - 2*x**3 + x**4


for intervals in (4, 8):
    h = F(1, intervals)
    x = F(1, 2)
    approximation = (2*target(x)-target(x-h)-target(x+h))/h**2
    force = 12*x*(1-x)
    print(f"N={intervals}: -Dxx={approximation}, f={force}, defect={force-approximation}")

h = F(1, 4)
left = F(2)
force = F(2)
print("first scaled RHS with left=2:", h*h*force+left)
print("first scaled row: 2*U1 - U2 = 17/8")
''')

add("poisson", "Solve the tridiagonal problem", "Does changing an endpoint modify the operator, the load, or both in the reduced Dirichlet system?", TRIDIAGONAL + POISSON + '''
for left, right in [(0.0, 0.0), (2.0, -1.0)]:
    nodes, values, residual = poisson(4, lambda x: 2.0, left, right)
    print("endpoints:", left, right)
    print("nodes:", nodes)
    print("solution:", [round(value, 8) for value in values])
    print("physical residual:", f"{max(map(abs, residual)):.2e}")
    print("scaled residual:", f"{max(map(abs, residual))/16:.2e}")
''')

CERTIFICATE = '''
from fractions import Fraction as F


def exact_field_budget(values, length=1, scale=1):
    """Quartic family, exact rational residual of represented float values."""
    n = len(values) - 1
    domain, amplitude = F(length), F(scale)
    h = domain / n
    stored = list(map(F, values))
    residual = F(0)
    for j in range(1, n):
        t = F(j, n)
        force = 12*amplitude*t*(1-t)
        row = (2*stored[j]-stored[j-1]-stored[j+1])/h**2
        residual = max(residual, abs(force-row))
    discretization = amplitude*h*h/4
    interpolation = 3*amplitude*h*h/8
    algebraic = domain*domain*residual/8
    return residual, discretization, interpolation, algebraic


def jacobi_report(n, tolerance, budget, length=1.0, scale=1.0, left=0.0, right=0.0):
    if not isinstance(n, int) or n < 2 or budget < 0 or length <= 0 or scale < 0:
        raise ValueError("Invalid mesh, iteration budget, length or scale")
    tolerance = F(tolerance)
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")
    h = length / n
    loads = [h*h*12*scale*(j/n)*(1-j/n) for j in range(1, n)]
    loads[0] += left
    loads[-1] += right
    interior = [0.0]*(n-1)
    for iteration in range(budget + 1):
        # Exact checking every 25 updates avoids rational work at every step.
        if iteration % 25 == 0 or iteration == budget:
            values = [left, *interior, right]
            residual, mesh_error, interpolation, algebraic = exact_field_budget(values, length, scale)
            total = mesh_error + interpolation + algebraic
            if total <= tolerance or iteration == budget:
                return values, iteration, (residual, mesh_error, interpolation, algebraic), total <= tolerance
        interior = [(loads[j] + (interior[j-1] if j else 0)
                     + (interior[j+1] if j+1 < n-1 else 0))/2 for j in range(n-1)]
'''

add("certificate", "Stop only when the requested field is certified", "How much of a 1/1000 tolerance remains for the solver at N=32?", CERTIFICATE + '''
for budget in (25, 4000):
    values, steps, parts, certified = jacobi_report(32, F(1, 1000), budget)
    residual, nodal, interpolation, algebraic = parts
    print("budget/used/status:", budget, steps, "certified" if certified else "not certified")
    print("mesh plus interpolation:", nodal+interpolation)
    print("exact residual <= allowed:", residual <= F("0.0031171875"))
    print("total field bound:", f"{float(nodal+interpolation+algebraic):.9f}")
print("A residual allowance:", F(8)*(F(1, 1000)-F(5, 8*32**2)))
print("T residual allowance:", F(8)*(F(1, 1000)-F(5, 8*32**2))/32**2)
''')

add("refinement", "Separate a useful refinement test from an exactness trap", "Why does the quadratic fail to reveal this stencil's order, even though its reconstructed field is not exact?", '''
from fractions import Fraction as F
from math import log2
''' + TRIDIAGONAL + '''
for family in ("quartic", "quadratic"):
    print(family)
    previous = None
    for n in (8, 16, 32, 64):
        h = F(1, n)
        points = [j*h for j in range(1, n)]
        force = [12*x*(1-x) if family == "quartic" else F(2) for x in points]
        target = [x-2*x**3+x**4 if family == "quartic" else x*(1-x) for x in points]
        factor = factor_spd([F(2)]*(n-1), [F(-1)]*(n-2))
        values = solve_factored(factor, [h*h*f for f in force])
        error = max(abs(a-b) for a, b in zip(values, target))
        order = f"{log2(float(previous/error)):.6f}" if previous and error else "undefined"
        print(f"N={n}: exact nodal error={error}, order={order}")
        previous = error
''')

DIFFUSION = '''
from math import sin, pi, exp
''' + TRIDIAGONAL + '''

def diffuse(n, time_steps, final_time, mode, method):
    if n < 3 or time_steps < 1 or not 1 <= mode < n or final_time <= 0:
        raise ValueError("Invalid finite diffusion problem")
    ratio = final_time*n*n/time_steps
    initial = [sin(mode*pi*j/n) for j in range(1, n)]
    values = initial[:]
    theta = {"explicit": 0.0, "backward": 1.0, "crank": 0.5}[method]
    factor = factor_spd([1+2*theta*ratio]*(n-1), [-theta*ratio]*(n-2))
    for _ in range(time_steps):
        rhs = [(1-2*(1-theta)*ratio)*value
               + (1-theta)*ratio*((values[j-1] if j else 0)
               + (values[j+1] if j+1 < n-1 else 0))
               for j, value in enumerate(values)]
        values = solve_factored(factor, rhs)
    continuous = [value*exp(-(mode*pi)**2*final_time) for value in initial]
    eigenvalue = 4*n*n*sin(mode*pi/(2*n))**2
    spatial = [value*exp(-eigenvalue*final_time) for value in initial]
    return values, continuous, spatial, ratio
'''

add("diffusion", "Execute three different time updates", "Can an energy-stable method retain sign-alternating grid-scale noise?", DIFFUSION + '''
for method in ("explicit", "backward", "crank"):
    for steps, time in [(1, 0.05), (2, 0.1)]:
        values, exact, spatial, ratio = diffuse(8, steps, time, 7, method)
        error = max(abs(a-b) for a, b in zip(values, exact))
        print(method, "t=", time, "r=", round(ratio, 6), "first value=", f"{values[0]:.8f}", "error=", f"{error:.8f}")
print("Each pair uses the same dt; compare methods at the same physical time.")
''')

add("timeRefinement", "Refine time while keeping space fixed", "Which comparison isolates the time integrator rather than its unchanged spatial error?", DIFFUSION + '''
from math import log2
for method in ("explicit", "backward", "crank"):
    print(method)
    previous = None
    for steps in (16, 32, 64, 128):
        values, continuous, spatial, ratio = diffuse(16, steps, 0.02, 1, method)
        temporal = max(abs(a-b) for a, b in zip(values, spatial))
        order = "-" if previous is None else f"{log2(previous/temporal):.3f}"
        print(steps, f"time error={temporal:.8e}", f"order={order}")
        previous = temporal
''')

add("interface", "Combine resistances, not conductivities", "What changes if the material interface moves to one third of the rod?", '''
from fractions import Fraction as F


def interface_flux(position, k_left, k_right, left=F(1), right=F(0)):
    position, k_left, k_right = map(F, (position, k_left, k_right))
    if not 0 < position < 1 or min(k_left, k_right) <= 0:
        raise ValueError("Interior interface and positive conductivities required")
    resistance = position/k_left + (1-position)/k_right
    flux = (left-right)/resistance
    temperature = left-flux*position/k_left
    arithmetic_flux = (position*k_left+(1-position)*k_right)*(left-right)
    return flux, temperature, arithmetic_flux


for position in (F(1, 2), F(1, 3)):
    flux, temperature, arithmetic = interface_flux(position, 1, 10)
    print("interface:", position, "flux:", flux, "temperature:", temperature)
    print("arithmetic replacement flux:", arithmetic)
''')

add("neumann", "Reject incompatible fluxes before choosing a mean", "Can pinning one temperature hide a failed conservation equation?", '''
from fractions import Fraction as F
''' + TRIDIAGONAL + '''

def conservative_neumann(cells, source, outward_left, outward_right):
    source, outward_left, outward_right = map(F, (source, outward_left, outward_right))
    if cells < 2:
        raise ValueError("At least two cells required")
    if source != outward_left+outward_right:
        return None, source-outward_left-outward_right
    h = F(1, cells)
    loads = [source*h]*cells
    loads[0] -= outward_left
    loads[-1] -= outward_right
    diagonal = [2/h]*(cells-1)
    diagonal[-1] = 1/h
    factor = factor_spd(diagonal, [-1/h]*(cells-2))
    pinned = [F(0), *solve_factored(factor, loads[1:])]
    mean = sum(pinned)/cells
    values = [value-mean for value in pinned]
    faces = [-outward_left, *[(a-b)/h for a, b in zip(values, values[1:])], outward_right]
    assert all(faces[j+1]-faces[j] == source*h for j in range(cells))
    return values, F(0)


for outward_right in (1, 0):
    values, mismatch = conservative_neumann(4, 2, 1, outward_right)
    print("right outward:", outward_right, "mismatch:", mismatch)
    print("mean-zero values:", None if values is None else list(map(str, values)))
''')

ADVECTION = '''
from math import sin, pi


def pulse_average(left, right, shift):
    offset = shift % 1.0
    overlap = sum(max(0.0, min(right, 0.5+offset+copy)-max(left, 0.25+offset+copy))
                  for copy in (-1, 0, 1))
    return overlap/(right-left)


def transport(cells, courant, steps, velocity=1, scheme="upwind"):
    if cells < 4 or courant < 0 or steps < 0 or velocity not in (-1, 1):
        raise ValueError("Invalid bounded transport input")
    h = 1/cells
    values = [pulse_average(j*h, (j+1)*h, 0) for j in range(cells)]
    for _ in range(steps):
        if scheme == "upwind":
            values = [(1-courant)*value+courant*values[(j-velocity)%cells]
                      for j, value in enumerate(values)]
        elif scheme == "centered":
            values = [value-velocity*courant/2*(values[(j+1)%cells]-values[(j-1)%cells])
                      for j, value in enumerate(values)]
        else:
            raise ValueError("Unknown scheme")
    exact = [pulse_average(j*h, (j+1)*h, velocity*steps*courant*h) for j in range(cells)]
    return values, exact
'''

add("advection", "Check mass and shape separately", "Does unchanged total mass imply an accurate or nonnegative transported pulse?", ADVECTION + '''
for scheme, courant, velocity in [("upwind", 0.75, 1), ("centered", 0.75, 1), ("upwind", 1.0, -1)]:
    values, exact = transport(16, courant, 8, velocity, scheme)
    print(scheme, "c=", courant, "v=", velocity)
    print("mass/min/max:", *(f"{value:.6f}" for value in (sum(values)/16, min(values), max(values))))
    print("cell-average L1 error:", f"{sum(abs(a-b) for a, b in zip(values, exact))/16:.6f}")
''')

add("godunov", "Read the entropy solution at one cell face", "Why does a sign-changing rarefaction have zero face flux while the reversed states do not?", '''
from fractions import Fraction as F


def godunov(left, right):
    left, right = F(left), F(right)
    if left <= right:
        state = max(left, min(F(0), right))
        return state*state/2, "rarefaction" if left < right else "constant"
    speed = (left+right)/2
    state = left if speed >= 0 else right
    return state*state/2, "shock"


for left, right in [(-1, 2), (2, -1), (-3, -1), (1, 3), (2, -2)]:
    flux, kind = godunov(left, right)
    print(left, right, kind, "flux=", flux)
''')

FEM = '''
import numpy as np


def assemble_p1(nodes, source=2.0, point=None, strength=1.0, left=0.0, right=0.0, conductivity=1.0):
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or len(nodes) < 3 or not np.all(np.isfinite(nodes)):
        raise ValueError("Need finite one-dimensional mesh nodes")
    if nodes[0] != 0 or nodes[-1] != 1 or np.any(np.diff(nodes) <= 0) or conductivity <= 0:
        raise ValueError("Increasing [0,1] mesh and positive conductivity required")
    size = len(nodes)
    matrix = np.zeros((size, size))
    load = np.zeros(size)
    for j, width in enumerate(np.diff(nodes)):
        matrix[j:j+2, j:j+2] += conductivity/width*np.array([[1., -1.], [-1., 1.]])
        if point is None:
            load[j:j+2] += source*width/2
    if point is not None:
        if not 0 < point < 1:
            raise ValueError("Point load must lie inside the domain")
        j = np.searchsorted(nodes, point, side="left")
        weight = (point-nodes[j-1])/(nodes[j]-nodes[j-1])
        load[j-1:j+1] += strength*np.array([1-weight, weight])
    rhs = load[1:-1]-matrix[1:-1, 0]*left-matrix[1:-1, -1]*right
    values = np.r_[left, np.linalg.solve(matrix[1:-1, 1:-1], rhs), right]
    return matrix, load, values
'''

add("finiteElement", "Assemble overlapping hats on an actual mesh", "Which load entries change when a point source crosses an element, and why is its total strength preserved?", FEM + '''
np.set_printoptions(precision=6, suppress=True)
nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
matrix, load, values = assemble_p1(nodes, point=1/3)
print("stiffness:\\n", matrix, sep="")
print("point load:", load)
print("nodal values:", values)
at_source = np.interp(1/3, nodes, values)
print("peak error:", f"{2/9-at_source:.8f}")
aligned = sorted(set(nodes+[1/3]))
_, _, aligned_values = assemble_p1(aligned, point=1/3)
print("aligned peak error:", f"{abs(2/9-aligned_values[aligned.index(1/3)]):.8f}")
matrix, load, values = assemble_p1([0, 0.2, 0.6, 1], source=2, left=2, right=-1)
print("nonuniform changed boundary values:", values)
''')

add("norms", "Integrate field and slope errors", "How can the nodal error be zero while the field and gradient have different convergence orders?", '''
from fractions import Fraction as F
from math import sqrt


def multiply(first, second):
    result = [F(0)]*(len(first)+len(second)-1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            result[i+j] += a*b
    return result


def integral(coefficients, right):
    return sum(value*right**(power+1)/(power+1) for power, value in enumerate(coefficients))


for n in (4, 8, 16):
    h = F(1, n)
    # On each cell, quadratic minus its chord is t*(h-t).
    field = [F(0), h, F(-1)]
    derivative = [h, F(-2)]
    l2_squared = n*integral(multiply(field, field), h)
    energy_squared = n*integral(multiply(derivative, derivative), h)
    print(n, "nodal=0", "sup=", h*h/4,
          "L2=", f"{sqrt(l2_squared):.8f}", "energy=", f"{sqrt(energy_squared):.8f}")
''')

add("poisson2D", "Solve a sparse rectangular problem", "Do unequal spacings, nonzero boundary values and row-major indexing survive together?", '''
import numpy as np
import scipy
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve


def solve_rectangle(nx, ny, length_x=2.0, length_y=1.0, mode_x=1, mode_y=2):
    if nx < 2 or ny < 2 or not 1 <= mode_x < nx or not 1 <= mode_y < ny:
        raise ValueError("Interior nodes and representable modes required")
    hx, hy = length_x/nx, length_y/ny
    index = lambda i, j: (j-1)*(nx-1)+i-1
    wave = lambda x, y: np.sin(mode_x*np.pi*x/length_x)*np.sin(mode_y*np.pi*y/length_y)
    boundary = lambda x, y: 0.4*x-0.2*y
    eigenvalue = (mode_x*np.pi/length_x)**2+(mode_y*np.pi/length_y)**2
    size = (nx-1)*(ny-1)
    rows, columns, entries = [], [], []
    rhs = np.zeros(size)
    target = np.zeros(size)
    for j in range(1, ny):
        for i in range(1, nx):
            row = index(i, j)
            x, y = i*hx, j*hy
            rhs[row] = eigenvalue*wave(x, y)
            target[row] = wave(x, y)+boundary(x, y)
            for di, dj, coefficient in [(0, 0, 2/hx**2+2/hy**2), (-1, 0, -1/hx**2),
                                         (1, 0, -1/hx**2), (0, -1, -1/hy**2), (0, 1, -1/hy**2)]:
                ni, nj = i+di, j+dj
                if ni in (0, nx) or nj in (0, ny):
                    rhs[row] -= coefficient*boundary(ni*hx, nj*hy)
                else:
                    rows.append(row); columns.append(index(ni, nj)); entries.append(coefficient)
    matrix = coo_array((entries, (rows, columns)), shape=(size, size)).tocsr()
    values = spsolve(matrix, rhs)
    return matrix, rhs, values, target


for nx, ny in [(16, 10), (32, 20)]:
    matrix, rhs, values, target = solve_rectangle(nx, ny)
    print("shape/nonzeros:", matrix.shape, matrix.nnz)
    print("nodal error:", f"{np.max(np.abs(values-target)):.8f}")
    print("residual below 1e-10:", np.max(np.abs(rhs-matrix@values)) < 1e-10)
print("Runtime packages: NumPy and SciPy; version details belong in your run report.")
''')

add("triangle", "Calculate a triangle's basis gradients", "Why does stretching the triangle change stiffness even when its three node values stay the same?", '''
import numpy as np


def triangle_stiffness(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2) or not np.all(np.isfinite(vertices)):
        raise ValueError("Three finite plane points required")
    jacobian = np.column_stack((vertices[1]-vertices[0], vertices[2]-vertices[0]))
    determinant = np.linalg.det(jacobian)
    if determinant == 0:
        raise ValueError("Degenerate triangle")
    gradients = np.linalg.solve(jacobian.T, np.array([[-1., 1., 0.], [-1., 0., 1.]]))
    area = abs(determinant)/2
    return area, gradients, area*gradients.T@gradients


np.set_printoptions(precision=6, suppress=True)
for vertices in [[[0, 0], [1, 0], [0, 1]], [[1, 2], [3, 2], [1, 5]]]:
    area, gradients, matrix = triangle_stiffness(vertices)
    print("area:", round(area, 6))
    print("stiffness:\\n", matrix, sep="")
    print("absolute constant-field energy residual:", f"{abs(float(np.ones(3)@matrix@np.ones(3))):.2e}")
''')

add("twoGrid", "Compute one explicitly defined coarse correction", "Which error remains after smoothing, and what can the coarse piecewise-linear space remove?", '''
import numpy as np

n = 8
h = 1/n
matrix = (2*np.eye(n-1)-np.eye(n-1, k=1)-np.eye(n-1, k=-1))/h**2
prolongation = np.array([[max(0, 1-abs((i+1)/2-(j+1))) for j in range(3)] for i in range(7)])
restriction = 0.5*prolongation.T
coarse_matrix = restriction@matrix@prolongation
norm = lambda values: np.sqrt(h*np.dot(values, values))
energy = lambda values: np.sqrt(h*np.dot(values, matrix@values))
for mode in (1, 7):
    error = np.sin(mode*np.pi*np.arange(1, n)/n)
    smooth = error-(2/3)*(matrix@error)/np.diag(matrix)
    coarse_error = np.linalg.solve(coarse_matrix, restriction@matrix@smooth)
    corrected = smooth-prolongation@coarse_error
    print("mode", mode, "norms:", *(f"{norm(v):.8f}" for v in (error, smooth, corrected)))
    print("energy norms:", *(f"{energy(v):.8f}" for v in (error, smooth, corrected)))
    print("coarse residual removed:", np.max(np.abs(restriction@matrix@corrected)) < 1e-12)
print("This is one fixed two-grid operation, not a general multigrid complexity claim.")
''')

add("changedReport", "Deliver a changed field-tolerance report", "With a longer rod, stronger source and nonzero endpoints, does the old residual allowance still certify the requested field?", CERTIFICATE + '''
for budget in (200, 12000):
    values, used, parts, certified = jacobi_report(64, F(1, 1000), budget,
                                                  length=1.5, scale=2.0, left=2.0, right=-1.0)
    residual, nodal, interpolation, algebraic = parts
    print("budget/used/certified:", budget, used, certified)
    print("mesh+interpolation:", nodal+interpolation)
    print("residual:", f"{float(residual):.8f}")
    print("whole-field bound:", f"{float(nodal+interpolation+algebraic):.9f}")
print("A-equation allowance:", F(8, 1)/F(3, 2)**2*(F(1, 1000)-F(5, 8)*2*(F(3, 2)/64)**2))
''')

folder = ROOT / "scratch/numerical-pde-native/programs"
folder.mkdir(parents=True, exist_ok=True)
runs = []
for key, example in PROGRAMS.items():
    path = folder / f"{key}.py"
    path.write_text(example["code"] + "\n", encoding="utf-8")
    run = subprocess.run([sys.executable, "-X", "utf8", "-I", str(path)], capture_output=True, text=True, encoding="utf-8", timeout=90, check=True)
    if run.stderr:
        raise AssertionError((key, run.stderr))
    example["expected"] = run.stdout.rstrip()
    runs.append({"key": key, "codeSha256": hashlib.sha256(example["code"].encode()).hexdigest(), "stdout": example["expected"]})
target = ROOT / "src/learn/data/numerical-pde-examples.js"
target.write_text("export const numericalPdeExamples = " + json.dumps(PROGRAMS, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
record = {"checkedAt": datetime.now(timezone.utc).isoformat(), "passed": True, "programs": runs, "python": sys.version, "exampleFileSha256": hashlib.sha256(target.read_bytes()).hexdigest()}
(folder.parent / "example-runs.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
print(f"Executed and saved {len(PROGRAMS)} complete Python programs.")
