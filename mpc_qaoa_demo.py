"""Discretized MPC for a 1D point robot solved classically and with QAOA.

This demo shows the full modeling chain:

1. Finite-horizon MPC with a discretized action set a_k in {-1, 0, +1}.
2. One-hot binary encoding of each action choice.
3. Quadratic cost assembly into a QUBO over binary variables.
4. Conversion from QuadraticProgram -> QUBO -> Ising Hamiltonian in Qiskit.
5. Exact classical and QAOA-based solution of the QUBO.
6. Receding-horizon closed-loop control with exact and QAOA-backed solvers.

The goal is educational rather than performance-oriented. The problem sizes are
kept intentionally small so the complete pipeline can be inspected and verified.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


ACTIONS: tuple[int, int, int] = (-1, 0, 1)
ACTION_LABELS: tuple[str, str, str] = ("-1", "0", "+1")


@dataclass(frozen=True)
class MPCParams:
    """Configuration for the 1D point-robot MPC problem."""

    dt: float = 1.0
    horizon: int = 4
    p0: float = -2.0
    v0: float = 0.0
    p_ref: float = 5.0
    q_p: float = 1.0
    q_v: float = 0.2
    r: float = 0.1
    q_pf: float = 4.0
    q_vf: float = 1.0
    lambda_onehot: float = 25.0
    v_max: float | None = None


@dataclass
class RolloutResult:
    """State and cost data for a single control sequence rollout."""

    controls: list[int]
    positions: list[float]
    velocities: list[float]
    stage_costs: list[float]
    total_cost: float


@dataclass
class BaselineResult:
    """Result of the brute-force search over all control sequences."""

    rollout: RolloutResult
    runtime_s: float
    evaluated_sequences: int


@dataclass
class QuboModel:
    """Structured QUBO data with metadata needed for decoding and inspection."""

    params: MPCParams
    var_names: list[str]
    linear: dict[str, float]
    quadratic: dict[tuple[str, str], float]
    constant: float
    action_var_map: dict[tuple[int, int], str]
    qubo_matrix: np.ndarray


@dataclass
class SolverOutcome:
    """Normalized solver output used for reporting and closed-loop control."""

    name: str
    bit_values: list[int]
    controls: list[int]
    feasible: bool
    qubo_objective: float
    mpc_cost: float
    rollout: RolloutResult
    runtime_s: float
    extra: dict[str, float | str | int]


@dataclass
class ClosedLoopResult:
    """Closed-loop trajectory produced by a receding-horizon controller."""

    name: str
    positions: list[float]
    velocities: list[float]
    applied_controls: list[int]
    stage_costs: list[float]
    solve_times_s: list[float]
    feasibility_flags: list[bool]


@dataclass
class AffineExpr:
    """Affine expression c + sum_i a_i z_i over binary variables."""

    constant: float
    coeffs: dict[str, float]


def check_qiskit_available() -> None:
    """Raise a friendly error if Qiskit packages are missing."""

    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Qiskit dependencies are not installed. Install the packages from "
            "`requirements.txt`, then rerun this script."
        )


def rollout_dynamics(params: MPCParams, controls: Sequence[int]) -> RolloutResult:
    """Simulate the dynamics and evaluate the MPC cost for a control sequence."""

    p = params.p0
    v = params.v0
    positions = [p]
    velocities = [v]
    stage_costs: list[float] = []

    for a in controls:
        stage_cost = (
            params.q_p * (p - params.p_ref) ** 2
            + params.q_v * v**2
            + params.r * a**2
        )
        stage_costs.append(stage_cost)
        p_next = p + params.dt * v
        v_next = v + params.dt * a
        p, v = p_next, v_next
        positions.append(p)
        velocities.append(v)

    terminal_cost = params.q_pf * (positions[-1] - params.p_ref) ** 2 + params.q_vf * velocities[-1] ** 2
    total_cost = float(sum(stage_costs) + terminal_cost)
    return RolloutResult(
        controls=list(controls),
        positions=positions,
        velocities=velocities,
        stage_costs=stage_costs,
        total_cost=total_cost,
    )


def brute_force_baseline(params: MPCParams) -> BaselineResult:
    """Enumerate all action sequences and return the exact optimum."""

    start = time.perf_counter()
    best_rollout: RolloutResult | None = None
    count = 0
    for controls in itertools.product(ACTIONS, repeat=params.horizon):
        rollout = rollout_dynamics(params, controls)
        count += 1
        if best_rollout is None or rollout.total_cost < best_rollout.total_cost:
            best_rollout = rollout
    runtime_s = time.perf_counter() - start
    assert best_rollout is not None
    return BaselineResult(best_rollout, runtime_s, count)


def variable_name(k: int, action: int) -> str:
    """Return the binary-variable name for timestep k and action value."""

    label = { -1: "m1", 0: "0", 1: "p1" }[action]
    return f"z_{k}_{label}"


def build_variable_order(horizon: int) -> tuple[list[str], dict[tuple[int, int], str]]:
    """Create a consistent variable ordering for all one-hot bits."""

    names: list[str] = []
    mapping: dict[tuple[int, int], str] = {}
    for k in range(horizon):
        for action in ACTIONS:
            name = variable_name(k, action)
            names.append(name)
            mapping[(k, action)] = name
    return names, mapping


def encode_control_sequence(controls: Sequence[int], horizon: int) -> list[int]:
    """Encode a control sequence into the one-hot binary vector."""

    if len(controls) != horizon:
        raise ValueError(f"Expected {horizon} controls, got {len(controls)}.")
    bits: list[int] = []
    for action in controls:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}. Allowed actions are {ACTIONS}.")
        bits.extend([1 if action == candidate else 0 for candidate in ACTIONS])
    return bits


def validate_one_hot(bit_values: Sequence[int], horizon: int) -> bool:
    """Return True iff every timestep has exactly one active action bit."""

    if len(bit_values) != 3 * horizon:
        return False
    for k in range(horizon):
        block = bit_values[3 * k : 3 * (k + 1)]
        if sum(block) != 1 or any(bit not in (0, 1) for bit in block):
            return False
    return True


def decode_binary_vector(bit_values: Sequence[int], horizon: int) -> list[int]:
    """Decode a one-hot binary vector into a control sequence."""

    if not validate_one_hot(bit_values, horizon):
        raise ValueError("Bit vector is not one-hot feasible.")
    controls: list[int] = []
    for k in range(horizon):
        block = bit_values[3 * k : 3 * (k + 1)]
        active_idx = int(np.argmax(block))
        controls.append(ACTIONS[active_idx])
    return controls


def format_bitstring(bit_values: Sequence[int]) -> str:
    """Format a binary vector as a compact bitstring."""

    return "".join(str(int(bit)) for bit in bit_values)


def affine_constant(value: float) -> AffineExpr:
    """Construct a constant affine expression."""

    return AffineExpr(float(value), {})


def affine_add(lhs: AffineExpr, rhs: AffineExpr) -> AffineExpr:
    """Add two affine expressions."""

    coeffs = dict(lhs.coeffs)
    for name, value in rhs.coeffs.items():
        coeffs[name] = coeffs.get(name, 0.0) + value
    return AffineExpr(lhs.constant + rhs.constant, coeffs)


def affine_scale(expr: AffineExpr, scalar: float) -> AffineExpr:
    """Scale an affine expression by a constant."""

    return AffineExpr(
        expr.constant * scalar,
        {name: coeff * scalar for name, coeff in expr.coeffs.items()},
    )


def affine_square(expr: AffineExpr) -> tuple[float, dict[str, float], dict[tuple[str, str], float]]:
    """Expand (c + a^T z)^2 using z_i^2 = z_i for binary variables."""

    constant = expr.constant**2
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    items = list(expr.coeffs.items())

    for name, coeff in items:
        linear[name] = linear.get(name, 0.0) + 2.0 * expr.constant * coeff + coeff**2

    for idx, (name_i, coeff_i) in enumerate(items):
        for name_j, coeff_j in items[idx + 1 :]:
            key = tuple(sorted((name_i, name_j)))
            quadratic[key] = quadratic.get(key, 0.0) + 2.0 * coeff_i * coeff_j

    return constant, linear, quadratic


def add_quadratic_term(
    constant: float,
    linear: dict[str, float],
    quadratic: dict[tuple[str, str], float],
    scale: float,
    square_term: tuple[float, dict[str, float], dict[tuple[str, str], float]],
) -> tuple[float, dict[str, float], dict[tuple[str, str], float]]:
    """Accumulate a scaled squared-affine term into QUBO coefficients."""

    term_const, term_linear, term_quadratic = square_term
    constant += scale * term_const
    for name, value in term_linear.items():
        linear[name] = linear.get(name, 0.0) + scale * value
    for key, value in term_quadratic.items():
        quadratic[key] = quadratic.get(key, 0.0) + scale * value
    return constant, linear, quadratic


def build_affine_dynamics(params: MPCParams, action_var_map: dict[tuple[int, int], str]) -> tuple[list[AffineExpr], list[AffineExpr], list[AffineExpr]]:
    """Build affine expressions for p_k, v_k, and a_k in terms of one-hot bits."""

    p_exprs = [affine_constant(params.p0)]
    v_exprs = [affine_constant(params.v0)]
    a_exprs: list[AffineExpr] = []

    for k in range(params.horizon):
        coeffs = {
            action_var_map[(k, -1)]: -1.0,
            action_var_map[(k, 0)]: 0.0,
            action_var_map[(k, 1)]: 1.0,
        }
        coeffs = {name: coeff for name, coeff in coeffs.items() if abs(coeff) > 0.0}
        a_expr = AffineExpr(0.0, coeffs)
        a_exprs.append(a_expr)

        p_next = affine_add(p_exprs[-1], affine_scale(v_exprs[-1], params.dt))
        v_next = affine_add(v_exprs[-1], affine_scale(a_expr, params.dt))
        p_exprs.append(p_next)
        v_exprs.append(v_next)

    return p_exprs, v_exprs, a_exprs


def build_manual_qubo(params: MPCParams) -> QuboModel:
    """Assemble the penalized QUBO directly from the robot MPC equations."""

    var_names, action_var_map = build_variable_order(params.horizon)
    p_exprs, v_exprs, a_exprs = build_affine_dynamics(params, action_var_map)

    constant = 0.0
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}

    for k in range(params.horizon):
        p_tracking = affine_add(p_exprs[k], affine_constant(-params.p_ref))
        constant, linear, quadratic = add_quadratic_term(
            constant,
            linear,
            quadratic,
            params.q_p,
            affine_square(p_tracking),
        )
        constant, linear, quadratic = add_quadratic_term(
            constant,
            linear,
            quadratic,
            params.q_v,
            affine_square(v_exprs[k]),
        )
        constant, linear, quadratic = add_quadratic_term(
            constant,
            linear,
            quadratic,
            params.r,
            affine_square(a_exprs[k]),
        )

        onehot_expr = AffineExpr(
            -1.0,
            {
                action_var_map[(k, -1)]: 1.0,
                action_var_map[(k, 0)]: 1.0,
                action_var_map[(k, 1)]: 1.0,
            },
        )
        constant, linear, quadratic = add_quadratic_term(
            constant,
            linear,
            quadratic,
            params.lambda_onehot,
            affine_square(onehot_expr),
        )

    terminal_tracking = affine_add(p_exprs[-1], affine_constant(-params.p_ref))
    constant, linear, quadratic = add_quadratic_term(
        constant,
        linear,
        quadratic,
        params.q_pf,
        affine_square(terminal_tracking),
    )
    constant, linear, quadratic = add_quadratic_term(
        constant,
        linear,
        quadratic,
        params.q_vf,
        affine_square(v_exprs[-1]),
    )

    qubo_matrix = qubo_terms_to_matrix(var_names, linear, quadratic)
    return QuboModel(
        params=params,
        var_names=var_names,
        linear=linear,
        quadratic=quadratic,
        constant=constant,
        action_var_map=action_var_map,
        qubo_matrix=qubo_matrix,
    )


def qubo_terms_to_matrix(
    var_names: Sequence[str],
    linear: dict[str, float],
    quadratic: dict[tuple[str, str], float],
) -> np.ndarray:
    """Create a readable upper-triangular QUBO matrix."""

    index = {name: idx for idx, name in enumerate(var_names)}
    matrix = np.zeros((len(var_names), len(var_names)))
    for name, coeff in linear.items():
        matrix[index[name], index[name]] += coeff
    for (name_i, name_j), coeff in quadratic.items():
        i = index[name_i]
        j = index[name_j]
        if i <= j:
            matrix[i, j] += coeff
        else:
            matrix[j, i] += coeff
    return matrix


def evaluate_qubo(model: QuboModel, bit_values: Sequence[int]) -> float:
    """Evaluate the penalized QUBO objective for a binary vector."""

    if len(bit_values) != len(model.var_names):
        raise ValueError("Bit vector length does not match the QUBO variable count.")

    value = model.constant
    assignment = {name: float(bit) for name, bit in zip(model.var_names, bit_values)}
    for name, coeff in model.linear.items():
        value += coeff * assignment[name]
    for (name_i, name_j), coeff in model.quadratic.items():
        value += coeff * assignment[name_i] * assignment[name_j]
    return float(value)


def build_quadratic_program_constrained(params: MPCParams) -> QuadraticProgram:
    """Build the constrained binary program before penalties are added."""

    check_qiskit_available()
    model = build_manual_qubo(MPCParams(**{**params.__dict__, "lambda_onehot": 0.0}))
    qp = QuadraticProgram("discretized_point_robot_mpc")

    for name in model.var_names:
        qp.binary_var(name=name)

    qp.minimize(constant=model.constant, linear=model.linear, quadratic=model.quadratic)

    for k in range(params.horizon):
        coeffs = {
            variable_name(k, -1): 1.0,
            variable_name(k, 0): 1.0,
            variable_name(k, 1): 1.0,
        }
        qp.linear_constraint(linear=coeffs, sense="==", rhs=1.0, name=f"onehot_{k}")

    return qp


def build_qiskit_qubo(params: MPCParams) -> tuple[QuadraticProgram, QuadraticProgram]:
    """Build the constrained model and convert it to a QUBO in Qiskit."""

    check_qiskit_available()
    qp = build_quadratic_program_constrained(params)
    converter = QuadraticProgramToQubo(penalty=params.lambda_onehot)
    qubo = converter.convert(qp)
    return qp, qubo


def decode_assignment_from_values(values: Sequence[float], horizon: int) -> list[int]:
    """Convert solver values into integer bits."""

    bits = [int(round(v)) for v in values]
    if len(bits) != 3 * horizon:
        raise ValueError("Unexpected assignment length returned by solver.")
    return bits


def best_feasible_sample(
    samples: Iterable,
    horizon: int,
    model: QuboModel,
) -> tuple[list[int], float] | None:
    """Pick the lowest-QUBO feasible sample from Qiskit sample data."""

    best: tuple[list[int], float] | None = None
    for sample in samples:
        bits = decode_assignment_from_values(sample.x, horizon)
        if not validate_one_hot(bits, horizon):
            continue
        fval = evaluate_qubo(model, bits)
        if best is None or fval < best[1]:
            best = (bits, fval)
    return best


def outcome_from_bits(name: str, bit_values: Sequence[int], model: QuboModel, runtime_s: float, extra: dict[str, float | str | int] | None = None) -> SolverOutcome:
    """Build a normalized outcome object from a chosen bit vector."""

    feasible = validate_one_hot(bit_values, model.params.horizon)
    controls = decode_binary_vector(bit_values, model.params.horizon) if feasible else []
    rollout = rollout_dynamics(model.params, controls) if feasible else rollout_dynamics(model.params, [0] * model.params.horizon)
    return SolverOutcome(
        name=name,
        bit_values=list(bit_values),
        controls=controls,
        feasible=feasible,
        qubo_objective=evaluate_qubo(model, bit_values),
        mpc_cost=rollout.total_cost if feasible else math.inf,
        rollout=rollout,
        runtime_s=runtime_s,
        extra=extra or {},
    )


def solve_qubo_exact_qiskit(params: MPCParams, model: QuboModel) -> SolverOutcome:
    """Solve the QUBO exactly using NumPyMinimumEigensolver via Qiskit."""

    check_qiskit_available()
    _, qubo = build_qiskit_qubo(params)
    start = time.perf_counter()
    exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = exact_solver.solve(qubo)
    runtime_s = time.perf_counter() - start

    chosen = best_feasible_sample(result.samples, params.horizon, model)
    if chosen is None:
        bits = decode_assignment_from_values(result.x, params.horizon)
    else:
        bits = chosen[0]

    return outcome_from_bits(
        name="Exact QUBO (Qiskit)",
        bit_values=bits,
        model=model,
        runtime_s=runtime_s,
        extra={"solver_status": str(result.status)},
    )


def solve_qubo_qaoa(
    params: MPCParams,
    model: QuboModel,
    reps: int,
    maxiter: int,
    seed: int,
) -> SolverOutcome:
    """Solve the QUBO approximately with QAOA on a simulator-backed sampler."""

    check_qiskit_available()
    _, qubo = build_qiskit_qubo(params)
    start = time.perf_counter()
    initial_point = np.concatenate([0.2 * np.ones(reps), 0.3 * np.ones(reps)])
    qaoa_mes = QAOA(
        sampler=StatevectorSampler(seed=seed),
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
        initial_point=initial_point,
    )
    optimizer = MinimumEigenOptimizer(qaoa_mes)
    result = optimizer.solve(qubo)
    runtime_s = time.perf_counter() - start

    chosen = best_feasible_sample(result.samples, params.horizon, model)
    if chosen is None:
        bits = decode_assignment_from_values(result.x, params.horizon)
    else:
        bits = chosen[0]

    outcome = outcome_from_bits(
        name=f"QAOA reps={reps}",
        bit_values=bits,
        model=model,
        runtime_s=runtime_s,
        extra={
            "reps": reps,
            "solver_status": str(result.status),
            "num_samples": len(result.samples),
        },
    )
    return outcome


def run_closed_loop(
    name: str,
    params: MPCParams,
    steps: int,
    solver_factory: Callable[[MPCParams], SolverOutcome],
) -> ClosedLoopResult:
    """Run a receding-horizon controller that re-solves MPC at each step."""

    current_p = params.p0
    current_v = params.v0
    positions = [current_p]
    velocities = [current_v]
    applied_controls: list[int] = []
    stage_costs: list[float] = []
    solve_times_s: list[float] = []
    feasibility_flags: list[bool] = []

    for _ in range(steps):
        step_params = MPCParams(**{**params.__dict__, "p0": current_p, "v0": current_v})
        outcome = solver_factory(step_params)
        chosen_control = outcome.controls[0] if outcome.feasible and outcome.controls else 0

        stage_cost = (
            params.q_p * (current_p - params.p_ref) ** 2
            + params.q_v * current_v**2
            + params.r * chosen_control**2
        )
        stage_costs.append(stage_cost)
        solve_times_s.append(outcome.runtime_s)
        feasibility_flags.append(outcome.feasible)
        applied_controls.append(chosen_control)

        next_p = current_p + params.dt * current_v
        next_v = current_v + params.dt * chosen_control
        current_p, current_v = next_p, next_v
        positions.append(current_p)
        velocities.append(current_v)

    return ClosedLoopResult(
        name=name,
        positions=positions,
        velocities=velocities,
        applied_controls=applied_controls,
        stage_costs=stage_costs,
        solve_times_s=solve_times_s,
        feasibility_flags=feasibility_flags,
    )


def print_section(title: str) -> None:
    """Pretty section divider for console output."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_problem_story(params: MPCParams) -> None:
    """Explain the modeling intent in presentation-friendly language."""

    print_section("1. Robotics Problem Definition")
    print("Continuous MPC is typically posed over continuous control variables and solved")
    print("with continuous optimization routines. That is not naturally a QAOA problem.")
    print("Here we discretize the control set to {-1, 0, +1}, which turns the MPC action")
    print("selection into a finite combinatorial search over control sequences.")
    print()
    print(f"Horizon N = {params.horizon}, dt = {params.dt}")
    print(f"Initial state: p0 = {params.p0}, v0 = {params.v0}")
    print(f"Reference: p_ref = {params.p_ref}, terminal velocity target = 0")
    print(
        "Weights: "
        f"q_p={params.q_p}, q_v={params.q_v}, r={params.r}, "
        f"q_pf={params.q_pf}, q_vf={params.q_vf}"
    )
    print(f"One-hot penalty lambda = {params.lambda_onehot}")


def print_baseline(baseline: BaselineResult) -> None:
    """Print the brute-force optimum and its trajectory."""

    print_section("2. Exact Brute-Force Baseline")
    print(f"Enumerated {baseline.evaluated_sequences} control sequences.")
    print(f"Best exact sequence: {baseline.rollout.controls}")
    print(f"Best exact cost: {baseline.rollout.total_cost:.6f}")
    print(f"Runtime: {baseline.runtime_s:.6f} s")
    print(f"Positions: {np.round(baseline.rollout.positions, 3).tolist()}")
    print(f"Velocities: {np.round(baseline.rollout.velocities, 3).tolist()}")


def print_binary_encoding_demo(params: MPCParams, controls: Sequence[int]) -> None:
    """Show how the one-hot encoding maps between controls and bits."""

    print_section("3. Binary Encoding")
    encoded = encode_control_sequence(controls, params.horizon)
    print("Each timestep has three binary variables:")
    for k in range(params.horizon):
        names = [variable_name(k, action) for action in ACTIONS]
        print(f"  k={k}: {names}, one-hot sum must equal 1")
    print()
    print(f"Example exact-optimal control sequence: {list(controls)}")
    print(f"Encoded one-hot bits: {encoded}")
    print(f"Encoded bitstring: {format_bitstring(encoded)}")
    print(f"Decoding gives: {decode_binary_vector(encoded, params.horizon)}")
    print(f"One-hot feasible: {validate_one_hot(encoded, params.horizon)}")


def print_qubo_summary(model: QuboModel) -> None:
    """Print human-readable QUBO details."""

    print_section("4. Manual QUBO Construction")
    print("The QUBO objective is a quadratic polynomial over binary variables only.")
    print("One-hot constraints are enforced as quadratic penalties of the form")
    print("lambda * (z_{k,-1} + z_{k,0} + z_{k,+1} - 1)^2.")
    print()
    print(f"Number of binary variables: {len(model.var_names)}")
    print(f"Variable names: {model.var_names}")
    print(f"Penalty weight: {model.params.lambda_onehot}")
    print(f"QUBO constant term: {model.constant:.6f}")
    print()
    if len(model.var_names) <= 12:
        matrix_str = np.array2string(np.round(model.qubo_matrix, 3), precision=3, suppress_small=True)
        print("Upper-triangular readable QUBO matrix Q where")
        print("J(z) = constant + sum_i Q_ii z_i + sum_{i<j} Q_ij z_i z_j")
        print(matrix_str)
    else:
        print("QUBO matrix omitted because the problem is larger than the demo print threshold.")


def print_qiskit_model_summary(params: MPCParams) -> tuple[QuadraticProgram, QuadraticProgram, object, float]:
    """Build and print the constrained model, QUBO model, and Ising operator."""

    check_qiskit_available()
    print_section("5. Qiskit Model -> QUBO -> Ising")
    qp, qubo = build_qiskit_qubo(params)
    print("QuadraticProgram with explicit one-hot equality constraints:")
    print(qp.prettyprint())
    print()
    print("Converted QUBO:")
    print(qubo.prettyprint())
    print()
    qubit_op, offset = to_ising(qubo)
    print("Ising operator summary:")
    print(f"Qubits: {qubit_op.num_qubits}")
    print(f"Pauli terms: {len(qubit_op)}")
    print(f"Offset: {offset:.6f}")
    print(qubit_op)
    print()
    print("QAOA operates on the Ising Hamiltonian because binary optimization is mapped")
    print("into finding the low-energy state of a qubit Hamiltonian.")
    return qp, qubo, qubit_op, offset


def print_solver_outcome(outcome: SolverOutcome, exact_cost: float | None = None) -> None:
    """Print a normalized solver result."""

    print_section(f"Solver Result: {outcome.name}")
    print(f"Bitstring: {format_bitstring(outcome.bit_values)}")
    print(f"One-hot feasible: {outcome.feasible}")
    print(f"Decoded control sequence: {outcome.controls if outcome.feasible else 'N/A'}")
    print(f"QUBO objective: {outcome.qubo_objective:.6f}")
    print(f"MPC cost: {outcome.mpc_cost:.6f}" if math.isfinite(outcome.mpc_cost) else "MPC cost: infeasible")
    if exact_cost is not None and math.isfinite(outcome.mpc_cost):
        gap = outcome.mpc_cost - exact_cost
        rel_gap = gap / max(abs(exact_cost), 1e-9)
        print(f"Approximation gap vs exact: {gap:.6f}")
        print(f"Relative gap vs exact: {100.0 * rel_gap:.3f}%")
    print(f"Runtime: {outcome.runtime_s:.6f} s")
    if outcome.extra:
        print(f"Extra info: {outcome.extra}")
    if outcome.feasible:
        print(f"Positions: {np.round(outcome.rollout.positions, 3).tolist()}")
        print(f"Velocities: {np.round(outcome.rollout.velocities, 3).tolist()}")


def penalty_feasibility_experiment(
    base_params: MPCParams,
    penalties: Sequence[float],
) -> list[tuple[float, bool, list[int], float]]:
    """Run a short experiment showing how penalty size affects feasibility."""

    results: list[tuple[float, bool, list[int], float]] = []
    for penalty in penalties:
        params = MPCParams(**{**base_params.__dict__, "lambda_onehot": penalty})
        model = build_manual_qubo(params)
        best_value = math.inf
        best_bits: list[int] | None = None
        for bits in itertools.product((0, 1), repeat=3 * params.horizon):
            bits_list = list(bits)
            value = evaluate_qubo(model, bits_list)
            if value < best_value:
                best_value = value
                best_bits = bits_list
        assert best_bits is not None
        results.append((penalty, validate_one_hot(best_bits, params.horizon), best_bits, best_value))
    return results


def save_one_shot_plots(
    output_dir: Path,
    exact_rollout: RolloutResult,
    qaoa_rollout: RolloutResult | None,
    costs: dict[str, float],
    runtimes: dict[str, float],
) -> None:
    """Save one-shot comparison plots for trajectory, cost, and runtime."""

    output_dir.mkdir(parents=True, exist_ok=True)
    horizon_axis = np.arange(len(exact_rollout.positions))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(horizon_axis, exact_rollout.positions, marker="o", label="Exact")
    if qaoa_rollout is not None:
        axes[0].plot(horizon_axis, qaoa_rollout.positions, marker="s", label="QAOA")
    axes[0].set_title("One-Shot Position Trajectory")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Position")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    labels = list(costs.keys())
    axes[1].bar(labels, [costs[label] for label in labels], color=["tab:blue", "tab:green", "tab:orange"][: len(labels)])
    axes[1].set_title("Cost Comparison")
    axes[1].set_ylabel("Objective value")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "one_shot_summary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    runtime_labels = list(runtimes.keys())
    ax.bar(runtime_labels, [runtimes[label] for label in runtime_labels], color=["tab:blue", "tab:green", "tab:orange"][: len(runtime_labels)])
    ax.set_title("Runtime Comparison")
    ax.set_ylabel("Seconds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_comparison.png", dpi=160)
    plt.close(fig)


def save_closed_loop_plots(
    output_dir: Path,
    exact_loop: ClosedLoopResult,
    qaoa_loop: ClosedLoopResult | None,
    p_ref: float,
) -> None:
    """Save closed-loop trajectory plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    t_state = np.arange(len(exact_loop.positions))
    t_control = np.arange(len(exact_loop.applied_controls))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t_state, exact_loop.positions, marker="o", label="Exact")
    if qaoa_loop is not None:
        axes[0, 0].plot(t_state, qaoa_loop.positions, marker="s", label="QAOA")
    axes[0, 0].axhline(p_ref, color="black", linestyle="--", alpha=0.6, label="Target")
    axes[0, 0].set_title("Closed-Loop Position")
    axes[0, 0].set_xlabel("Time step")
    axes[0, 0].set_ylabel("Position")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(t_state, exact_loop.velocities, marker="o", label="Exact")
    if qaoa_loop is not None:
        axes[0, 1].plot(t_state, qaoa_loop.velocities, marker="s", label="QAOA")
    axes[0, 1].set_title("Closed-Loop Velocity")
    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].set_ylabel("Velocity")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].step(t_control, exact_loop.applied_controls, where="post", label="Exact")
    if qaoa_loop is not None:
        axes[1, 0].step(t_control, qaoa_loop.applied_controls, where="post", label="QAOA")
    axes[1, 0].set_title("Applied Control")
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("Acceleration")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    exact_distance = [abs(p - p_ref) for p in exact_loop.positions]
    axes[1, 1].plot(t_state, exact_distance, marker="o", label="Exact")
    if qaoa_loop is not None:
        qaoa_distance = [abs(p - p_ref) for p in qaoa_loop.positions]
        axes[1, 1].plot(t_state, qaoa_distance, marker="s", label="QAOA")
    axes[1, 1].set_title("Distance to Target")
    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].set_ylabel("|p - p_ref|")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "closed_loop_comparison.png", dpi=160)
    plt.close(fig)


def save_closed_loop_animation(
    output_dir: Path,
    exact_loop: ClosedLoopResult,
    qaoa_loop: ClosedLoopResult | None,
    p_ref: float,
    fps: int = 2,
) -> Path:
    """Save a presentation-friendly GIF of the 1D closed-loop trajectories."""

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "closed_loop_animation.gif"

    all_positions = list(exact_loop.positions)
    if qaoa_loop is not None:
        all_positions.extend(qaoa_loop.positions)
    x_min = min(min(all_positions), p_ref) - 1.5
    x_max = max(max(all_positions), p_ref) + 1.5

    num_frames = len(exact_loop.positions)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.8, 1.8 if qaoa_loop is not None else 0.8)
    ax.set_xlabel("Position")
    ax.set_title("Closed-Loop 1D Point Robot Animation")
    ax.grid(True, axis="x", alpha=0.25)
    ax.axvline(p_ref, color="black", linestyle="--", linewidth=1.5, label="Target")

    if qaoa_loop is not None:
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Exact MPC", "QAOA MPC"])
    else:
        ax.set_yticks([0])
        ax.set_yticklabels(["Exact MPC"])

    exact_lane = 0.0
    qaoa_lane = 1.0
    ax.hlines(exact_lane, x_min, x_max, color="tab:blue", alpha=0.2, linewidth=3)
    if qaoa_loop is not None:
        ax.hlines(qaoa_lane, x_min, x_max, color="tab:orange", alpha=0.2, linewidth=3)

    exact_marker, = ax.plot([], [], marker="o", markersize=12, color="tab:blue", linestyle="None", label="Exact")
    exact_trail, = ax.plot([], [], color="tab:blue", linewidth=2, alpha=0.5)

    if qaoa_loop is not None:
        qaoa_marker, = ax.plot([], [], marker="s", markersize=12, color="tab:orange", linestyle="None", label="QAOA")
        qaoa_trail, = ax.plot([], [], color="tab:orange", linewidth=2, alpha=0.5)
    else:
        qaoa_marker = None
        qaoa_trail = None

    info_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    ax.legend(loc="lower right")

    def build_text(frame_idx: int) -> str:
        exact_u = exact_loop.applied_controls[frame_idx] if frame_idx < len(exact_loop.applied_controls) else "-"
        text = (
            f"Step {frame_idx}\n"
            f"Exact: p={exact_loop.positions[frame_idx]:.2f}, "
            f"v={exact_loop.velocities[frame_idx]:.2f}, u={exact_u}"
        )
        if qaoa_loop is not None:
            qaoa_u = qaoa_loop.applied_controls[frame_idx] if frame_idx < len(qaoa_loop.applied_controls) else "-"
            text += (
                f"\nQAOA: p={qaoa_loop.positions[frame_idx]:.2f}, "
                f"v={qaoa_loop.velocities[frame_idx]:.2f}, u={qaoa_u}"
            )
        return text

    def update(frame_idx: int):
        exact_positions = exact_loop.positions[: frame_idx + 1]
        exact_marker.set_data([exact_loop.positions[frame_idx]], [exact_lane])
        exact_trail.set_data(exact_positions, [exact_lane] * len(exact_positions))
        artists = [exact_marker, exact_trail]

        if qaoa_loop is not None and qaoa_marker is not None and qaoa_trail is not None:
            qaoa_positions = qaoa_loop.positions[: frame_idx + 1]
            qaoa_marker.set_data([qaoa_loop.positions[frame_idx]], [qaoa_lane])
            qaoa_trail.set_data(qaoa_positions, [qaoa_lane] * len(qaoa_positions))
            artists.extend([qaoa_marker, qaoa_trail])

        info_text.set_text(build_text(frame_idx))
        artists.append(info_text)
        return artists

    animation = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=False, repeat=True)
    animation.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return gif_path


def print_closed_loop_summary(result: ClosedLoopResult, p_ref: float) -> None:
    """Print the closed-loop outcome in a compact way."""

    terminal_distance = abs(result.positions[-1] - p_ref)
    mean_solve_time = float(np.mean(result.solve_times_s)) if result.solve_times_s else 0.0
    feasibility_rate = float(np.mean(result.feasibility_flags)) if result.feasibility_flags else 0.0
    print_section(f"Closed-Loop Summary: {result.name}")
    print(f"Positions: {np.round(result.positions, 3).tolist()}")
    print(f"Velocities: {np.round(result.velocities, 3).tolist()}")
    print(f"Applied controls: {result.applied_controls}")
    print(f"Terminal distance to target: {terminal_distance:.6f}")
    print(f"Mean MPC solve time: {mean_solve_time:.6f} s")
    print(f"Per-step one-hot feasibility rate: {100.0 * feasibility_rate:.1f}%")


def print_final_summary(
    params: MPCParams,
    baseline: BaselineResult,
    exact_outcome: SolverOutcome | None,
    qaoa_outcome: SolverOutcome | None,
) -> None:
    """Print the concise presentation-ready final summary."""

    print_section("Final Summary")
    print(f"Horizon N: {params.horizon}")
    print(f"Number of binary variables: {3 * params.horizon}")
    print(f"Best exact brute-force sequence: {baseline.rollout.controls}")
    print(f"Exact brute-force cost: {baseline.rollout.total_cost:.6f}")
    if exact_outcome is not None:
        print(f"Best exact QUBO sequence: {exact_outcome.controls}")
        print(f"Exact QUBO cost: {exact_outcome.mpc_cost:.6f}")
        print(f"Exact QUBO feasible: {exact_outcome.feasible}")
    if qaoa_outcome is not None:
        gap = qaoa_outcome.mpc_cost - baseline.rollout.total_cost if math.isfinite(qaoa_outcome.mpc_cost) else math.inf
        rel_gap = gap / max(abs(baseline.rollout.total_cost), 1e-9) if math.isfinite(gap) else math.inf
        print(f"Best QAOA sequence: {qaoa_outcome.controls if qaoa_outcome.feasible else 'N/A'}")
        print(f"QAOA cost: {qaoa_outcome.mpc_cost:.6f}" if math.isfinite(qaoa_outcome.mpc_cost) else "QAOA cost: infeasible")
        print(f"Relative gap: {100.0 * rel_gap:.3f}%" if math.isfinite(rel_gap) else "Relative gap: infeasible")
        print(f"QAOA feasible: {qaoa_outcome.feasible}")
    print()
    print("This is a proof of concept rather than a practical real-time controller.")
    print("The action set is discretized, the horizon is tiny, and the QAOA loop is run")
    print("on a simulator. The value here is in making the MPC -> binary -> QUBO -> Ising")
    print("pipeline explicit and inspectable.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=4, help="Finite-horizon length N.")
    parser.add_argument("--dt", type=float, default=1.0, help="Discrete-time step.")
    parser.add_argument("--p0", type=float, default=-2.0, help="Initial position.")
    parser.add_argument("--v0", type=float, default=0.0, help="Initial velocity.")
    parser.add_argument("--p-ref", dest="p_ref", type=float, default=5.0, help="Target position.")
    parser.add_argument("--q-p", dest="q_p", type=float, default=1.0, help="Stage position weight.")
    parser.add_argument("--q-v", dest="q_v", type=float, default=0.2, help="Stage velocity weight.")
    parser.add_argument("--r", type=float, default=0.1, help="Control effort weight.")
    parser.add_argument("--q-pf", dest="q_pf", type=float, default=4.0, help="Terminal position weight.")
    parser.add_argument("--q-vf", dest="q_vf", type=float, default=1.0, help="Terminal velocity weight.")
    parser.add_argument("--lambda-onehot", dest="lambda_onehot", type=float, default=25.0, help="One-hot penalty.")
    parser.add_argument("--qaoa-reps", type=int, default=1, help="QAOA depth parameter.")
    parser.add_argument("--qaoa-maxiter", type=int, default=100, help="Maximum COBYLA iterations.")
    parser.add_argument("--closed-loop-steps", type=int, default=6, help="Receding-horizon simulation length.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for QAOA sampler.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_mpc_qaoa"),
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--skip-qiskit",
        action="store_true",
        help="Run only the brute-force/binary/QUBO assembly stages without Qiskit solvers.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full demo."""

    args = parse_args()
    params = MPCParams(
        dt=args.dt,
        horizon=args.horizon,
        p0=args.p0,
        v0=args.v0,
        p_ref=args.p_ref,
        q_p=args.q_p,
        q_v=args.q_v,
        r=args.r,
        q_pf=args.q_pf,
        q_vf=args.q_vf,
        lambda_onehot=args.lambda_onehot,
    )

    print_problem_story(params)
    baseline = brute_force_baseline(params)
    print_baseline(baseline)
    print_binary_encoding_demo(params, baseline.rollout.controls)

    manual_qubo = build_manual_qubo(params)
    print_qubo_summary(manual_qubo)

    print_section("Penalty Sensitivity Check")
    for penalty, feasible, bits, value in penalty_feasibility_experiment(params, [0.5, 2.0, params.lambda_onehot]):
        print(
            f"lambda={penalty:>6.2f} -> feasible optimum={feasible}, "
            f"best bitstring={format_bitstring(bits)}, qubo_value={value:.6f}"
        )

    exact_outcome: SolverOutcome | None = None
    qaoa_outcome: SolverOutcome | None = None
    exact_loop: ClosedLoopResult | None = None
    qaoa_loop: ClosedLoopResult | None = None

    if args.skip_qiskit:
        print_section("Qiskit Stages Skipped")
        print("Only the exact brute-force and manual QUBO construction stages were run.")
    else:
        check_qiskit_available()
        print_qiskit_model_summary(params)
        exact_outcome = solve_qubo_exact_qiskit(params, manual_qubo)
        print_solver_outcome(exact_outcome, exact_cost=baseline.rollout.total_cost)
        qaoa_outcome = solve_qubo_qaoa(params, manual_qubo, reps=args.qaoa_reps, maxiter=args.qaoa_maxiter, seed=args.seed)
        print_solver_outcome(qaoa_outcome, exact_cost=baseline.rollout.total_cost)

        exact_loop = run_closed_loop(
            name="Exact Receding-Horizon MPC",
            params=params,
            steps=args.closed_loop_steps,
            solver_factory=lambda step_params: solve_qubo_exact_qiskit(step_params, build_manual_qubo(step_params)),
        )
        qaoa_loop = run_closed_loop(
            name="QAOA Receding-Horizon MPC",
            params=params,
            steps=args.closed_loop_steps,
            solver_factory=lambda step_params: solve_qubo_qaoa(
                step_params,
                build_manual_qubo(step_params),
                reps=args.qaoa_reps,
                maxiter=args.qaoa_maxiter,
                seed=args.seed,
            ),
        )
        print_closed_loop_summary(exact_loop, params.p_ref)
        print_closed_loop_summary(qaoa_loop, params.p_ref)

        costs = {
            "Brute force": baseline.rollout.total_cost,
            "Exact QUBO": exact_outcome.mpc_cost,
            "QAOA": qaoa_outcome.mpc_cost,
        }
        runtimes = {
            "Brute force": baseline.runtime_s,
            "Exact QUBO": exact_outcome.runtime_s,
            "QAOA": qaoa_outcome.runtime_s,
        }
        save_one_shot_plots(
            args.output_dir,
            exact_rollout=baseline.rollout,
            qaoa_rollout=qaoa_outcome.rollout if qaoa_outcome.feasible else None,
            costs=costs,
            runtimes=runtimes,
        )
        save_closed_loop_plots(args.output_dir, exact_loop, qaoa_loop, params.p_ref)
        print(f"\nSaved plots to: {args.output_dir.resolve()}")

    print_final_summary(params, baseline, exact_outcome, qaoa_outcome)


if __name__ == "__main__":
    main()
