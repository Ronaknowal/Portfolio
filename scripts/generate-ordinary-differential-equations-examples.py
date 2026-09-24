"""Format and execute the complete ODE lesson's actual displayed programs."""
import contextlib
import io
import json
import textwrap
from pathlib import Path

import black

examples = {}


def add(key, title, question, source):
    code = black.format_str(textwrap.dedent(source).strip() + "\n", mode=black.Mode(line_length=80))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, key, "exec"), {})
    examples[key] = dict(title=title, question=question, code=code.rstrip(), expected=output.getvalue().rstrip(), language="python")


add("thermal", "Convert the energy balance before solving", "Will the first minute remove exactly 8 K? Compare the instantaneous prediction with the solution.", '''
    from math import exp, log

    capacity = 600.0  # J/K
    conductance = 2.0  # W/K = J/(s K)
    initial = 40.0  # K above a constant room temperature
    rate_per_minute = 60 * conductance / capacity
    first_rate = -rate_per_minute * initial
    after_one_minute = initial * exp(-rate_per_minute)
    print(f"rate constant: {rate_per_minute:.3f} /min")
    print(f"initial derivative: {first_rate:.3f} K/min")
    print(f"one-minute tangent estimate: {initial + first_rate:.6f} K")
    print(f"one-minute exact state: {after_one_minute:.6f} K")
    print(f"half-time: {log(2) / rate_per_minute:.6f} min")
    power = 20.0  # W
    equilibrium = power / conductance
    state = equilibrium + (initial - equilibrium) * exp(-rate_per_minute * 5)
    print(f"with 20 W, state after 5 min: {state:.6f} K")
''')

add("domains", "One equation can have several solutions—or a finite lifetime", "Which candidates share the same initial value, and where does the blow-up formula stop belonging to that initial-value problem?", '''
    from math import sqrt

    def waiting_solution(time, departure):
        elapsed = max(0.0, time - departure)
        state = elapsed * elapsed
        derivative = 2 * elapsed
        assert abs(derivative - 2 * sqrt(abs(state))) < 1e-12
        return state

    for departure in (0.0, 1.0, 2.0):
        assert waiting_solution(0.0, departure) == 0.0
        values = [waiting_solution(time, departure) for time in (0.0, 1.0, 2.0, 3.0)]
        print(f"departure {departure:g}: {values}")

    def growing_solution(time, initial=1.0):
        if initial <= 0:
            raise ValueError("This helper is for a positive initial value.")
        denominator = 1 - initial * time
        if denominator <= 0:
            raise ValueError("Outside the maximal interval through time zero.")
        return initial / denominator

    print("blow-up samples:", [round(growing_solution(t), 6) for t in (0.0, 0.5, 0.9)])
    try:
        growing_solution(1.1)
    except ValueError as error:
        print("rejected:", error)
''')

add("scalar", "Verify a variable-coefficient integrating factor", "At t = 1, do both the equation and the initial condition hold? Why is t = −1 excluded?", '''
    def solution(time):
        if time <= -1:
            raise ValueError("Use the interval (-1, infinity) through the initial time.")
        shifted = 1 + time
        return shifted**2 / 4 + 3 / (4 * shifted**2)

    def derivative(time):
        shifted = 1 + time
        return shifted / 2 - 3 / (2 * shifted**3)

    assert solution(0) == 1
    for time in (0.0, 0.5, 1.0, 2.0):
        value = solution(time)
        residual = derivative(time) + 2 * value / (1 + time) - (1 + time)
        assert abs(residual) < 1e-12
        print(f"t={time:.1f}: y={value:.6f}, equation residual={residual:.2e}")
''')

add("oscillator", "Recover motion from position and velocity", "How do the damping choices change motion and energy, while leaving the initial position fixed?", '''
    import numpy as np
    from scipy.linalg import expm

    def motion(time, mass, damping, stiffness, initial):
        if mass <= 0 or damping < 0 or stiffness <= 0:
            raise ValueError("Use mass > 0, damping >= 0 and stiffness > 0.")
        matrix = np.array([[0.0, 1.0], [-stiffness / mass, -damping / mass]])
        position, velocity = expm(time * matrix) @ np.asarray(initial, dtype=float)
        energy = (mass * velocity**2 + stiffness * position**2) / 2
        return position, velocity, energy, -damping * velocity**2

    for damping in (0.0, 2.0, 4.0, 6.0):
        position, velocity, energy, energy_rate = motion(1.0, 1.0, damping, 4.0, [1.0, 0.0])
        print(f"c={damping:g}: q={position:.6f}, v={velocity:.6f}, E={energy:.6f}, E'={energy_rate:.6f}")
    critical = motion(1.0, 1.0, 4.0, 4.0, [1.0, 0.0])[0]
    assert np.isclose(critical, 3 * np.exp(-2))
''')

add("matrices", "The exponential carries every initial column", "Can a repeated eigenvalue still produce a time factor? What quantity stays fixed in the two-compartment system?", '''
    import numpy as np
    from scipy.linalg import expm

    shear = np.array([[-1.0, 3.0], [0.0, -1.0]])
    transition = expm(shear)
    initial = np.array([0.0, 1.0])
    state = transition @ initial
    assert np.allclose(state, [3 / np.e, 1 / np.e])
    assert np.allclose(expm(0.4 * shear) @ expm(0.6 * shear), transition)
    print("shear state at t=1:", np.round(state, 6).tolist())
    print("first initial column:", np.round(transition[:, 0], 6).tolist())
    print("second initial column:", np.round(transition[:, 1], 6).tolist())
    transfer = np.array([[-1.0, 2.0], [1.0, -2.0]])
    for time in (0.0, 0.5, 1.0):
        amounts = expm(time * transfer) @ [9.0, 0.0]
        assert np.isclose(sum(amounts), 9.0)
        print(f"t={time:g}: amounts={np.round(amounts, 6).tolist()}, total={sum(amounts):.6f}")
''')

add("forcing", "Add initial response and input response once", "Does the same total heater energy imply the same final temperature when its timing changes?", '''
    from math import exp, expm1

    def heated_interval(initial, power, duration, rate=0.2, conductance=2.0):
        equilibrium = power / conductance
        return initial * exp(-rate * duration) - equilibrium * expm1(-rate * duration)

    def schedule(initial, intervals):
        state = initial
        for duration, power in intervals:
            state = heated_interval(state, power, duration)
        return state

    early = schedule(40.0, [(3.0, 20.0), (3.0, 0.0)])
    late = schedule(40.0, [(3.0, 0.0), (3.0, 20.0)])
    initial_response = 40 * exp(-0.2 * 6)
    early_input_response = 10 * (-expm1(-0.2 * 3)) * exp(-0.2 * 3)
    assert abs(early - initial_response - early_input_response) < 1e-12
    print(f"same input energy: {20 * 3 * 60:.0f} J")
    print(f"early heating, final state: {early:.6f} K")
    print(f"late heating, final state: {late:.6f} K")
    print(f"initial contribution: {initial_response:.6f} K")
    print(f"early input contribution: {early_input_response:.6f} K")
''')

add("heldInput", "Use a held input even when A has no inverse", "What final position and velocity result from constant acceleration 2 for three seconds?", '''
    import numpy as np
    from scipy.linalg import expm

    def held_input_matrices(matrix, input_matrix, duration):
        matrix = np.asarray(matrix, dtype=float)
        input_matrix = np.asarray(input_matrix, dtype=float)
        size, inputs = input_matrix.shape
        if matrix.shape != (size, size) or duration < 0:
            raise ValueError("Check the shapes and nonnegative duration.")
        augmented = np.zeros((size + inputs, size + inputs))
        augmented[:size, :size] = matrix
        augmented[:size, size:] = input_matrix
        transition = expm(duration * augmented)
        return transition[:size, :size], transition[:size, size:]

    matrix = [[0.0, 1.0], [0.0, 0.0]]
    input_matrix = [[0.0], [1.0]]
    transition, response = held_input_matrices(matrix, input_matrix, 3.0)
    final = transition @ [1.0, -1.0] + response @ [2.0]
    print("state transition:", transition.tolist())
    print("held-input column:", response.ravel().tolist())
    print("final [position, velocity]:", final.tolist())
    assert np.allclose(final, [7, 5])
    # In y = Cx + Du, a direct term is added once, separately from the state.
    print("output for C=[1,0], D=0.5:", float(final[0] + 0.5 * 2))
''')

add("timeOrder", "Later evolution multiplies on the left", "Swap two one-unit shears. Does the final state depend only on the sum of their matrices?", '''
    import numpy as np
    from scipy.linalg import expm

    upper = np.array([[0.0, 1.0], [0.0, 0.0]])
    lower = np.array([[0.0, 0.0], [1.0, 0.0]])
    initial = np.array([1.0, 0.0])
    upper_then_lower = expm(lower) @ expm(upper) @ initial
    lower_then_upper = expm(upper) @ expm(lower) @ initial
    tempting_shortcut = expm(upper + lower) @ initial
    print("upper then lower:", upper_then_lower.tolist())
    print("lower then upper:", lower_then_upper.tolist())
    print("exp(sum) shortcut:", np.round(tempting_shortcut, 6).tolist())
    assert not np.allclose(upper_then_lower, lower_then_upper)
    assert not np.allclose(upper_then_lower, tempting_shortcut)
''')

add("fixedSteps", "Inspect actual stages and the same final time", "When the requested step does not divide the horizon, what should the final step be? Compare errors at a shared time.", '''
    from math import exp, isfinite

    def step(method, function, time, state, width):
        first = function(time, state)
        if method == "euler":
            average = first
        elif method == "midpoint":
            average = function(time + width / 2, state + width * first / 2)
        elif method == "rk4":
            second = function(time + width / 2, state + width * first / 2)
            third = function(time + width / 2, state + width * second / 2)
            fourth = function(time + width, state + width * third)
            average = (first + 2 * second + 2 * third + fourth) / 6
        else:
            raise ValueError("Unknown method.")
        return state + width * average

    def integrate(method, function, initial, end, width, budget=10000):
        if not all(isfinite(value) for value in (initial, end, width)) or end < 0 or width <= 0:
            raise ValueError("Require finite data, end >= 0 and width > 0.")
        time, state = 0.0, initial
        history = [(time, state)]
        while time < end:
            if len(history) - 1 >= budget:
                return history, "step budget exhausted"
            actual_width = min(width, end - time)
            if time + actual_width == time:
                return history, "time cannot advance in this arithmetic"
            state = step(method, function, time, state, actual_width)
            if not isfinite(state):
                return history, "nonfinite state"
            time += actual_width
            history.append((time, state))
        return history, "reached horizon"

    for method in ("euler", "midpoint", "rk4"):
        for width in (0.7, 0.35):
            history, status = integrate(method, lambda t, y: -0.2 * y, 40.0, 5.0, width)
            time, state = history[-1]
            error = abs(state - 40 * exp(-0.2 * time))
            print(f"{method}, h={width:g}: steps={len(history)-1}, t={time:.1f}, error={error:.8f}, {status}")
    history, status = integrate("euler", lambda t, y: -y, 1.0, 2.0, 0.1, budget=3)
    print(f"limited run: t={history[-1][0]:.1f}, {status}")
''')

add("adaptive", "Read an adaptive solver's contract and diagnostics", "A slowly moving exact solution can still punish explicit steps. Which output measures error, and which only counts work?", '''
    import numpy as np
    from scipy.integrate import solve_ivp

    def run(method, stiffness=80.0, initial=1.0):
        def function(time, state):
            return -stiffness * (state - np.cos(time)) - np.sin(time)

        times = np.linspace(0.0, 3.0, 301)
        result = solve_ivp(function, (0.0, 3.0), [initial], method=method,
                           rtol=1e-7, atol=1e-9, t_eval=times)
        exact = np.cos(result.t) + (initial - 1) * np.exp(-stiffness * result.t)
        if not result.success:
            raise RuntimeError(result.message)
        error = np.max(np.abs(result.y[0] - exact))
        return result, error

    for initial in (1.0, 2.0):
        for method in ("RK45", "Radau"):
            result, error = run(method, initial=initial)
            print(f"initial={initial:g}, {method}: status={result.status}, last={result.t[-1]:.1f}, nfev={result.nfev}, max sampled error={error:.3e}")
    print("t_eval requests output locations; it is not a list of the solver's accepted internal steps.")
''')

add("events", "A successful terminal event is different from reaching the horizon", "Will an event finder see every root between accepted steps? Compare an actual threshold with deliberately hidden crossings.", '''
    from math import log
    import numpy as np
    from scipy.integrate import solve_ivp

    def threshold(time, state):
        return state[0] - 10.0

    threshold.terminal = True
    threshold.direction = -1
    result = solve_ivp(lambda t, y: -0.2 * y, (0, 20), [40.0], events=threshold,
                       rtol=1e-9, atol=1e-11, dense_output=True)
    event_time = result.t_events[0][0]
    print(f"threshold: status={result.status}, success={result.success}, time={event_time:.8f}")
    print(f"analytic time={log(4)/0.2:.8f}; absolute error={abs(event_time-log(4)/0.2):.3e}")
    print("message:", result.message)

    # The state is constant. Only the artificial event function oscillates.
    def hidden_crossings(time, state):
        return (time - 0.25) * (time - 0.75)

    missed = solve_ivp(lambda t, y: [0.0], (0, 1), [1.0], first_step=1.0,
                       max_step=1.0, events=hidden_crossings)
    exposed = solve_ivp(lambda t, y: [0.0], (0, 1), [1.0], max_step=0.1,
                        events=hidden_crossings)
    print("one accepted step, detected roots:", missed.t_events[0].tolist())
    print("max_step=0.1, detected roots:", np.round(exposed.t_events[0], 6).tolist())
    assert len(missed.t_events[0]) == 0 and len(exposed.t_events[0]) == 2
    print("Reducing max_step exposes these roots; it is not a universal event-detection proof.")
''')

add("series", "Generate two independent ordinary-point series", "Why do two freely chosen initial coefficients determine all the rest for y'' = ty?", '''
    from fractions import Fraction
    from math import factorial
    import numpy as np
    from scipy.integrate import solve_ivp

    def coefficients(initial_value, initial_slope, degree):
        values = [Fraction(0) for _ in range(degree + 1)]
        values[0], values[1] = Fraction(initial_value), Fraction(initial_slope)
        for index in range(degree - 1):
            previous = values[index - 1] if index >= 1 else Fraction(0)
            values[index + 2] = previous / ((index + 2) * (index + 1))
        return values

    def evaluate(values, time):
        return sum(float(value) * time**power for power, value in enumerate(values))

    for initial in ((1, 0), (0, 1)):
        values = coefficients(*initial, 15)
        nonzero = [(index, str(value)) for index, value in enumerate(values) if value]
        reference = solve_ivp(lambda t, state: [state[1], t * state[0]],
                              (0, 1), initial, method="DOP853", rtol=1e-12, atol=1e-14)
        error = abs(evaluate(values, 1.0) - reference.y[0, -1])
        print(f"initial={initial}, coefficients={nonzero}")
        print(f"degree-15 value at 1: {evaluate(values, 1.0):.10f}; reference difference={error:.3e}")
    print("A small differential residual alone is not a general error bound.")
''')

add("changedPractice", "Check changed parameters independently", "Derive these answers first: changed damping, transfer rates, Jordan coupling, resonance and heating time.", '''
    from math import cos, exp, log, sin, sqrt
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    print(f"changed cooling: initial rate={-0.3*24:.6f}, time to 6={log(4)/0.3:.6f}")
    for time in (0.5, 1.0):
        state = expm(time * np.array([[0.0, 1.0], [-4.0, -2.0]])) @ [1.0, -1.0]
        assert np.isclose(state[0], exp(-time) * cos(sqrt(3) * time))
        transfer = expm(time * np.array([[-2.0, 1.0], [2.0, -1.0]])) @ [0.0, 12.0]
        assert np.isclose(transfer[0], 4 * (1 - exp(-3 * time)))
        print(f"t={time:g}: changed spring q={state[0]:.6f}, transfer first={transfer[0]:.6f}")
    for coupling in (4.0, 8.0):
        state = expm(0.5 * np.array([[-2.0, coupling], [0.0, -2.0]])) @ [0.0, 1.0]
        print(f"coupling={coupling:g}: norm at 0.5={np.linalg.norm(state):.6f}")
    resonant = solve_ivp(lambda t, state: [state[1], 3 * cos(2 * t) - 4 * state[0]],
                         (0, 2), [0, 0], method="DOP853", rtol=1e-12, atol=1e-14)
    print(f"resonance q(2)={resonant.y[0,-1]:.6f}; analytic={0.75*2*sin(4):.6f}")
    # Changed capstone: initial excess 24 K, 30 W for two minutes, then no power for four.
    at_switch = 15 + (24 - 15) * exp(-0.2 * 2)
    final = at_switch * exp(-0.2 * 4)
    delay_same_energy = 24 * exp(-0.2 * 6) + 15 * (1 - exp(-0.2 * 2))
    print(f"changed heating: switch={at_switch:.6f}, final={final:.6f}, delayed={delay_same_energy:.6f}")
''')

root = Path(__file__).resolve().parents[1]
target = root / 'src/learn/data/ordinary-differential-equations-examples.js'
target.write_text('export const ordinaryDifferentialEquationsExamples = ' + json.dumps(examples, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
directory = root / 'scratch/ordinary-differential-equations-verification'
directory.mkdir(parents=True, exist_ok=True)
(directory / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Executed and captured {len(examples)} complete programs.')
