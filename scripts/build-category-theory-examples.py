"""Build complete lesson-owned programs and capture their real Python output."""
import contextlib
import io
import json
from pathlib import Path
import textwrap

examples = {}


def add(key, title, question, code, interpretation):
    program = textwrap.dedent(code).strip() + "\n"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(program, f"<category-example-{key}>", "exec"), {})
    examples[key] = {
        "title": title, "question": question, "language": "python",
        "code": program, "expected": output.getvalue().rstrip(),
        "interpretation": interpretation,
    }


original = json.loads(Path("docs/teaching/evidence/category-theory-original-content.json").read_text(encoding="utf-8"))
add("original", "Regroup the same three functions",
    "Predict both outputs before running. Would exchanging f and g preserve the first result?",
    original["blocks"][0]["code"],
    "Both parenthesizations produce 7; either identity composite produces 5. Function equality here means equal outputs for every input, not equality of Python function objects. The displayed program is preserved from the original lesson.")

add("schema", "Check every path in a finite schema",
    "Which sensor violates the requirement that its direct site equal the site of its device?",
    '''
    def compose(after, before):
        return tuple(after[index] for index in before)

    def identity(size):
        return tuple(range(size))

    def audit(device_of, site_of, direct_site):
        if any(not 0 <= device < len(site_of) for device in device_of):
            raise ValueError("Every sensor must reference an existing device.")
        if len(direct_site) != len(device_of):
            raise ValueError("A direct site is required for every sensor.")
        expected = compose(site_of, device_of)
        return [sensor for sensor in range(len(expected))
                if direct_site[sensor] != expected[sensor]]

    # Objects are labelled sets: Sensor={s0,s1,s2}, Device={d0,d1},
    # Site={North,South}. The integers below are indices within those sets.
    device_of = (0, 0, 1)
    site_of = (0, 1)
    direct_site = (0, 1, 1)
    print("violating sensors:", audit(device_of, site_of, direct_site))
    repaired = compose(site_of, device_of)
    print("repaired direct map:", repaired)
    print("violations after repair:", audit(device_of, site_of, repaired))
    print("left identity:", compose(identity(2), device_of) == device_of)
    print("right identity:", compose(device_of, identity(3)) == device_of)
    ''',
    "Sensor s1 claims South while d0 belongs to North. Repairing its direct site enforces the source schema's equation. Merely drawing a triangle would not enforce it; the finite instance must interpret both paths equally.")

add("naturality", "Try reversal and sorting against every tiny map",
    "For the list [2,0,1], which operation survives mapping x to 2-x? Does an empty list refute either one?",
    '''
    from itertools import product

    def square(values, mapping, operation):
        top_right = [mapping[x] for x in operation(values)]
        bottom_left = operation([mapping[x] for x in values])
        return top_right, bottom_left

    values = [2, 0, 1]
    flip = (2, 1, 0)
    for name, operation in (("reverse", lambda xs: list(reversed(xs))),
                            ("sort", sorted)):
        left, right = square(values, flip, operation)
        print(name, left, right, left == right)

    # All functions on {0,1,2}, and all lists of length at most 3.
    tested = 0
    for mapping in product(range(3), repeat=3):
        for length in range(4):
            for values in product(range(3), repeat=length):
                left, right = square(values, mapping, lambda xs: list(reversed(xs)))
                assert left == right
                tested += 1
    print("finite reversal checks:", tested)
    print("empty sort:", square([], flip, sorted))
    ''',
    "Sorting changes which positions are occupied by mapped elements, so its two routes disagree under a decreasing map. Reversal only permutes positions and commutes with every element map; the proof in the text establishes that general fact. The 1,080 checks are finite evidence, not its proof.")

add("universal", "Count mediators and build a compatible-pair join",
    "Which pair has no mediator after p00 is removed? What changes if a second distinct point has its same two projections?",
    '''
    from itertools import product

    def witnesses(points, row, column):
        return [name for name, a, b in points if (a, b) == (row, column)]

    complete = [(f"p{a}{b}", a, b) for a, b in product(range(2), repeat=2)]
    candidates = {
        "complete": complete,
        "missing": complete[1:],
        "duplicate": complete + [("q00", 0, 0)],
    }
    for name, points in candidates.items():
        counts = [len(witnesses(points, a, b)) for a, b in product(range(2), repeat=2)]
        print(name, "pair counts", counts, "is product", all(n == 1 for n in counts))

    # A sensor and a technician can be paired only at the same site.
    sensor_site = {"s0": "North", "s1": "North", "s2": "South"}
    technician_site = {"t0": "North", "t1": "South"}
    pairs = [(s, t) for s in sensor_site for t in technician_site
             if sensor_site[s] == technician_site[t]]
    print("compatible pairs:", pairs)
    # A compatible cone from jobs has a forced factorization.
    job_sensor = {"j0": "s1", "j1": "s2"}
    job_technician = {"j0": "t0", "j1": "t1"}
    mediator = {j: (job_sensor[j], job_technician[j]) for j in job_sensor}
    assert all(pair in pairs for pair in mediator.values())
    print("unique job map:", mediator)
    ''',
    "The selected point in a product is determined by its two projections. The pullback keeps only compatible pairs, and the job map is forced point by point. This model uses total maps and ordinary finite sets; it is not a claim about SQL bag or NULL semantics.")

add("homology", "Follow a cycle through an inclusion",
    "If an edge cycle is unchanged as a chain, how can its homology class become zero after a face is filled?",
    '''
    from itertools import combinations

    def boundary(chain):
        result = set()
        for simplex in chain:
            if len(simplex) > 1:
                result.symmetric_difference_update(combinations(simplex, len(simplex) - 1))
        return frozenset(result)

    def span(chains):
        values = {frozenset()}
        for chain in chains:
            values |= {value.symmetric_difference(chain) for value in tuple(values)}
        return values

    def homology_classes(edges, faces):
        cycles = {chain for chain in span([frozenset([edge]) for edge in edges])
                  if not boundary(chain)}
        boundaries = span([boundary([face]) for face in faces])
        assert boundaries <= cycles
        classes = {frozenset(cycle.symmetric_difference(b) for b in boundaries)
                   for cycle in cycles}
        return cycles, boundaries, classes

    edges = [(0, 1), (0, 2), (1, 2)]
    cycle = frozenset(edges)
    source = homology_classes(edges, [])
    target = homology_classes(edges, [(0, 1, 2)])
    print("edge-space H1 class count:", len(source[2]))
    print("filled-space H1 class count:", len(target[2]))
    print("same cycle becomes a boundary:", cycle in target[1])
    # Check the chain-map square under a distinct injective relabelling.
    # The image simplex remains present in the larger target complex.
    vertices = {0: 3, 1: 4, 2: 5}
    def induced_chain(chain):
        return frozenset(tuple(sorted(vertices[v] for v in cell)) for cell in chain)
    for chain in span([frozenset([cell]) for cell in edges + [(0, 1, 2)]]):
        assert boundary(induced_chain(chain)) == induced_chain(boundary(chain))
    print("boundary of filled face:", sorted(boundary([(0, 1, 2)])))
    print("boundary twice:", sorted(boundary(boundary([(0, 1, 2)]))))
    ''',
    "Over F2, a one-dimensional homology vector space has two elements; the zero-dimensional one has one. The inclusion preserves the actual cycle but its target class contains a new boundary, so the induced linear map sends the nonzero class to zero. The code illustrates this quotient; the chain-map proof in the lesson establishes functoriality.")

add("option", "Compose total functions whose results may be missing",
    "Where does failure occur for x=-1, x=0 and x=3? Does regrouping change that result?",
    '''
    from dataclasses import dataclass
    from fractions import Fraction

    @dataclass(frozen=True)
    class Some:
        value: object

    # None means failure; Some(None) would still be a successful result.
    def pure(value):
        return Some(value)

    def bind(result, function):
        return None if result is None else function(result.value)

    def kleisli(after, before):
        return lambda x: bind(before(x), after)

    def nonnegative(x):
        return Some(x) if x >= 0 else None

    def reciprocal(x):
        return None if x == 0 else Some(Fraction(1, x))

    def add_one(x):
        return Some(x + 1)

    left = kleisli(add_one, kleisli(reciprocal, nonnegative))
    right = kleisli(kleisli(add_one, reciprocal), nonnegative)
    for x in (-1, 0, 3):
        assert left(x) == right(x)
        assert kleisli(pure, nonnegative)(x) == nonnegative(x)
        assert kleisli(nonnegative, pure)(x) == nonnegative(x)
        print(x, left(x))
    print("Some(None) stays successful:", bind(Some(None), pure))
    ''',
    "Ordinary arrows here are total functions into an explicit Option set. The special composition unwraps success and propagates failure. No division is attempted after failure. None and Some(None) are different states; this prevents a hidden ambiguity in the interface.")

add("probability", "Compose channels, then compare two joint distributions",
    "Can the same two marginals determine whether two bits always agree? Does conditioning undo a noisy channel?",
    '''
    from fractions import Fraction as Q

    def compose(first, second):
        assert len(first[0]) == len(second)
        return [[sum(first[x][y] * second[y][z] for y in range(len(second)))
                 for z in range(len(second[0]))] for x in range(len(first))]

    def display(matrix):
        return [[str(value) for value in row] for row in matrix]

    p = Q(1, 3)
    marginal = [1-p, p]
    copied = [[1-p, Q(0)], [Q(0), p]]
    independent = [[a*b for b in marginal] for a in marginal]
    print("copied joint:", display(copied))
    print("independent joint:", display(independent))
    for joint in (copied, independent):
        assert [sum(row) for row in joint] == marginal
        assert [sum(row[j] for row in joint) for j in range(2)] == marginal
    channel = [[Q(3,4), Q(1,4)], [Q(1,4), Q(3,4)]]
    evidence = compose([marginal], channel)[0]
    # Reversed row y represents P(X=x | Y=y), relative to this prior.
    reverse = [[marginal[x] * channel[x][y] / evidence[y] for x in range(2)]
               if evidence[y] else None for y in range(2)]
    print("output distribution:", [str(value) for value in evidence])
    print("Bayes reverse:", display(reverse))
    print("forward then reverse:", display(compose(channel, reverse)))
    print("two-channel composition:", display(compose(channel, channel)))
    ''',
    "Both joints have marginal [2/3,1/3], but only copying puts zero mass on disagreement. The reversed channel depends on the stated prior. Forward then reversed is not identity. If an output has zero evidence, its conditional row is undefined; the program's explicit None branch must be handled before any further composition.")

add("tangent", "Carry primal values through forward and reverse differentiation",
    "At x=0, why is the derivative of (x+1)^2 equal to 2 even though the derivative of y^2 at y=0 is zero?",
    '''
    from fractions import Fraction as Q

    def tangent(function, derivative, point, direction):
        return function(point), derivative(point) * direction

    f, df = lambda x: x+1, lambda x: Q(1)
    g, dg = lambda y: y*y, lambda y: 2*y
    y, dy = tangent(f, df, Q(0), Q(1))
    z, dz = tangent(g, dg, y, dy)
    print("intermediate (point,tangent):", y, dy)
    print("output (point,tangent):", z, dz)
    print("wrong derivative using dg(0):", df(Q(0))*dg(Q(0)))

    # Shared parameter: prediction = w*(w*x + b); loss = (prediction-target)^2/2.
    w, x, b, target = Q(2), Q(3), Q(1), Q(10)
    hidden = w*x + b
    prediction = w*hidden
    output_bar = prediction-target
    hidden_bar = w*output_bar
    w_gradient = hidden*output_bar + x*hidden_bar  # Both uses of w.
    b_gradient = hidden_bar
    x_gradient = w*hidden_bar
    direct_w = (prediction-target)*(2*w*x+b)
    assert w_gradient == direct_w
    step = Q(1,100)
    new_w, new_b = w-step*w_gradient, b-step*b_gradient
    new_prediction = new_w*(new_w*x+new_b)
    print("prediction and loss:", prediction, (prediction-target)**2/2)
    print("gradients w,b,x:", w_gradient, b_gradient, x_gradient)
    print("one simultaneous update:", new_w, new_b)
    print("updated loss:", (new_prediction-target)**2/2)
    ''',
    "The shared parameter has two incoming gradient contributions. The update computes all gradients at the old parameters, then changes them simultaneously. Exact fractions establish this particular result; a fixed positive step does not guarantee descent for every objective or state.")

add("adjunction", "Turn a forward coverage question into a backward filter",
    "Can f({0}) and f({1}) overlap when {0} and {1} do not? Which source points pass a selected target filter?",
    '''
    from itertools import combinations

    def subsets(values):
        return [set(group) for size in range(len(values)+1)
                for group in combinations(values, size)]

    mapping = [0, 0, 1, 2]
    def image(source):
        return {mapping[x] for x in source}

    def preimage(target):
        return {x for x, y in enumerate(mapping) if y in target}

    source, target = {0}, {0}
    print("direct image:", sorted(image(source)))
    print("preimage:", sorted(preimage(target)))
    print("round-trip saturation:", sorted(preimage(image(source))))
    print("image of intersection:", sorted(image({0} & {1})))
    print("intersection of images:", sorted(image({0}) & image({1})))
    count = 0
    for source in subsets(list(range(4))):
        for target in subsets(list(range(3))):
            assert (image(source) <= target) == (source <= preimage(target))
            assert source <= preimage(image(source))
            assert image(preimage(target)) <= target
            count += 1
    print("finite adjunction pairs:", count)
    ''',
    "The preimage of {0} contains both source points mapped there. A round trip can enlarge a source set, so these adjoints are not inverses. All 128 subset pairs satisfy the same containment equivalence; the arbitrary-element argument proves why.")

add("yoneda", "Reconstruct every compatible tiny probe family",
    "If a transformation changes every map into A in a compatible way, how much is left to choose after its effect on id_A is fixed?",
    '''
    from itertools import product

    # Full subcategory of FinSet on objects 0,1,2, including all functions.
    def maps(source, target):
        return tuple(product(range(target), repeat=source))

    def compose(after, before):
        return tuple(after[i] for i in before)

    A = B = 2
    probes = {n: maps(n, A) for n in range(3)}
    outputs = {n: maps(n, B) for n in range(3)}
    # A component maps each probe to an output probe of the same domain.
    components = {n: tuple(product(outputs[n], repeat=len(probes[n]))) for n in range(3)}
    indices = {n: {probe: i for i, probe in enumerate(probes[n])} for n in range(3)}

    def is_natural(family):
        for source in range(3):
            for target in range(3):
                for arrow in maps(source, target):
                    for probe in probes[target]:
                        left = family[source][indices[source][compose(probe, arrow)]]
                        right = compose(family[target][indices[target][probe]], arrow)
                        if left != right:
                            return False
        return True

    candidates = compatible = 0
    reconstructed = []
    for choice in product(*(components[n] for n in range(3))):
        candidates += 1
        if is_natural(choice):
            compatible += 1
            arrow = choice[A][indices[A][tuple(range(A))]]
            assert all(choice[n][i] == compose(arrow, probe)
                       for n in range(3) for i, probe in enumerate(probes[n]))
            reconstructed.append(arrow)
    print("candidate families:", candidates)
    print("compatible families:", compatible)
    print("recovered maps A to B:", sorted(reconstructed))
    ''',
    "There are 1,024 candidate families but only four natural ones, one per function from the two-element A to B. Every component is forced by postcomposing with the recovered map. This verifies one finite representable-to-representable example; the general Yoneda statement is proved separately.")

add("capstone", "Audit a migration and the join it induces",
    "If two sites are merged, must every pair allowed afterward come from a pair allowed before? First check whether the migration respects the schema.",
    '''
    def commute(arrow_before, arrow_after, source_change, target_change):
        return {x: (target_change[arrow_before[x]],
                    arrow_after[source_change[x]])
                for x in arrow_before
                if target_change[arrow_before[x]] != arrow_after[source_change[x]]}

    old_sensor = {"s0": "North", "s1": "South"}
    old_tech = {"t0": "North", "t1": "South"}
    sensor_change = {"s0": "S0", "s1": "S1"}
    tech_change = {"t0": "T0", "t1": "T1"}
    site_change = {"North": "Region", "South": "Region"}
    new_sensor = {"S0": "Region", "S1": "Wrong"}
    new_tech = {"T0": "Region", "T1": "Region"}
    print("bad sensor square:", commute(old_sensor, new_sensor, sensor_change, site_change))
    new_sensor["S1"] = "Region"
    assert not commute(old_sensor, new_sensor, sensor_change, site_change)
    assert not commute(old_tech, new_tech, tech_change, site_change)
    print("repaired squares commute")
    old_pairs = {(s, t) for s in old_sensor for t in old_tech
                 if old_sensor[s] == old_tech[t]}
    new_pairs = {(s, t) for s in new_sensor for t in new_tech
                 if new_sensor[s] == new_tech[t]}
    induced = {(sensor_change[s], tech_change[t]) for s, t in old_pairs}
    assert induced <= new_pairs
    print("old compatible pairs:", sorted(old_pairs))
    print("induced pair images:", sorted(induced))
    print("new extra pairs:", sorted(new_pairs-induced))
    ''',
    "Commuting component squares induce a well-defined map on compatible pairs. Merging sites allows new pairs that had no old counterpart, so the induced map need not be surjective. This is a concrete information-loss consequence, not a promise that a natural transformation preserves every database query result exactly.")

destination = Path("src/learn/data/category-theory-examples.js")
destination.write_text("// Complete standalone programs with actual captured Python output.\n"
                       "export const categoryTheoryExamples = "
                       + json.dumps(examples, indent=2, ensure_ascii=False) + ";\n", encoding="utf-8")
print(json.dumps({key: value["expected"] for key, value in examples.items()}, indent=2))
