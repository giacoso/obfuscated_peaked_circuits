"""Die zwei verschiedenen Optimierungsprobleme aus Kapitel 3.

Peaking: -|<s|P(theta) R|0>|² (zustandsabhängig; gleiche Minima wie 1-p).
Patch:   1 - |Tr(T† T_tilde(theta))|/d (volle Unitäre, Gleichung 1).
Diese Ziele dürfen nicht vertauscht werden: Ein Patch, der nur auf |0>
funktioniert, kann innerhalb eines verschränkten Schaltkreises falsch sein.
"""
from dataclasses import dataclass
from time import monotonic
import numpy as np
from scipy.optimize import minimize
from .circuit import Circuit, apply_matrix, statevector, unitary
from .simulation import AmplitudePlan, Budget


@dataclass
class Fit:
    circuit: Circuit
    initial_loss: float
    final_loss: float
    history: list[float]
    iterations: int
    success: bool
    message: str
    seconds: float
    initial_probability: float | None = None
    final_probability: float | None = None
    probability_history: list[float] | None = None
    rescalings: int = 0


def _minimize(circuit, objective, maxiter, seconds=120, restart_gain=None):
    """Beste gefundene Parameter aufbewahren, auch bei Iterations-/Zeitlimit.

    Ein Limit ist kein Konvergenznachweis. success und message werden separat
    protokolliert. Das Zeitlimit ist kooperativ zwischen Zielfunktionsaufrufen;
    eine einzelne BLAS-/Kontraktionsoperation kann es überschreiten.
    """
    if maxiter < 0 or seconds <= 0:
        raise ValueError("Ungültiges Optimierungsbudget")
    start = monotonic()
    x0 = circuit.parameters()
    value0, _ = objective(x0)
    history = [float(value0)]
    best = [float(value0), x0.copy()]
    nit = [0]

    class TimeLimit(Exception):
        pass

    class Rescale(Exception):
        pass

    def fun(x):
        if monotonic() - start > seconds:
            raise TimeLimit()
        value, gradient = objective(x)
        value = float(value)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Nichtendlicher Optimierungswert oder Gradient")
        if value < best[0]:
            best[:] = [value, x.copy()]
        if restart_gain is not None and value < -restart_gain:
            # Eine Verbesserung um viele Größenordnungen macht die alte
            # feste Skalierung unpraktisch. Beste Parameter bleiben erhalten.
            raise Rescale()
        return value, np.asarray(gradient, dtype=float)

    def callback(x):
        nit[0] += 1
        history.append(best[0])

    if maxiter == 0 or len(x0) == 0:
        return Fit(circuit, float(value0), float(value0), history, 0, False, "Kein Optimierungsschritt angefordert", monotonic() - start)
    try:
        result = minimize(fun, x0, jac=True, method="L-BFGS-B", callback=callback,
                          options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-9, "maxls": 30})
        success, message = bool(result.success), str(result.message)
    except Rescale:
        success, message = False, "RESCALE"
    except TimeLimit:
        success, message = False, "Zeitbudget erreicht; beste Parameter zurückgegeben"
    history.append(best[0])
    return Fit(circuit.bind(best[1]), float(value0), best[0], history, nit[0], success, message, monotonic() - start)


def dense_peaking_objective(random, peaking, target):
    """Adjungiertenverfahren, O(G * 2^n), nur für kleine Kontrollsysteme.

    Speichert Vorwärtszustände pro P-Gate. Rückwärts propagiert eine Bra vom
    Zielzustand. d|a|²/dθ = 2 Re(a* da/dθ); das Minus kommt von der Loss.
    """
    if random.n > 14:
        raise MemoryError("Dichtes Training wegen gespeicherter Zwischenzustände auf 14 Qubits begrenzt")
    initial = statevector(random)
    target_index = int(target, 2)

    def objective(parameters):
        bound = peaking.bind(parameters)
        states = [initial]
        for gate in bound.gates:
            states.append(apply_matrix(states[-1], gate.matrix(), gate.qubits, bound.n))
        amp = states[-1][target_index]
        back = np.zeros_like(initial)
        back[target_index] = 1
        derivatives = []
        for k in reversed(range(len(bound.gates))):
            gate = bound.gates[k]
            if gate.trainable:
                dstate = apply_matrix(states[k], gate.derivative(), gate.qubits, bound.n)
                derivatives.append(-2 * np.real(np.conj(amp) * np.vdot(back, dstate)))
            back = apply_matrix(back, gate.matrix().conj().T, gate.qubits, bound.n)
        return -float(abs(amp) ** 2), np.array(derivatives[::-1])
    return objective


def train_peaking(random, peaking, target, backend="tensor", maxiter=100, seconds=120, budget=Budget(), rescale=True):
    if random.n != peaking.n or len(target) != random.n or set(target) - {"0", "1"}:
        raise ValueError("Unpassende Register oder Ziel-Bitstring")
    if backend == "dense":
        objective = dense_peaking_objective(random, peaking, target)
    elif backend == "tensor":
        from autograd import value_and_grad
        import autograd.numpy as anp
        combined = random.then(peaking)
        plan = AmplitudePlan(combined, target, budget)
        constant = random.parameters()

        def loss(params):
            amplitude = plan.amplitude(anp.concatenate([constant, params]))
            return -anp.real(amplitude * anp.conj(amplitude))
        objective = value_and_grad(loss)
    else:
        raise ValueError("backend muss dense oder tensor sein")
    # -p vermeidet Auslöschung. Bei enormer Verbesserung starten wir L-BFGS
    # mit einer neuen POSITIVEN KONSTANTEN Skalierung und leerer Hessian-Historie.
    # Die Zielfunktion bleibt in jeder Phase proportional zum Paper-Ziel.
    # Dies ist keine Ersetzung durch log(p) oder ein lokales Ersatztraining.
    if maxiter < 0 or seconds <= 0:
        raise ValueError("Ungültiges Optimierungsbudget")
    start = monotonic()
    initial_value, _ = objective(peaking.parameters())
    initial_probability = -float(initial_value)
    current, history, used, rescalings = peaking, [], 0, 0
    while True:
        value, _ = objective(current.parameters())
        scale = max(abs(float(value)), 1e-300)
        def normalized(params):
            loss, gradient = objective(params)
            return loss / scale, gradient / scale
        fit = _minimize(current, normalized, max(0,maxiter-used),
                        max(0.001,seconds-(monotonic()-start)), restart_gain=1000 if rescale else None)
        history.extend([-loss*scale for loss in fit.history])
        current = fit.circuit
        used += fit.iterations
        if fit.message != "RESCALE":
            break
        rescalings += 1
        # Ein Rescaling zählt separat; die zusätzliche Einheit begrenzt auch
        # sehr viele erfolgreiche Neustarts innerhalb des Iterationsbudgets.
        used += 1
        if used >= maxiter or monotonic()-start >= seconds:
            fit.message = "Budget nach Neuskalierung erreicht"
            break
    fit.initial_probability = initial_probability
    fit.final_probability = history[-1]
    fit.probability_history = history
    fit.initial_loss = 1-initial_probability
    fit.final_loss = 1-fit.final_probability
    fit.history = [1-p for p in history]
    fit.iterations = used
    fit.rescalings = rescalings
    fit.seconds = monotonic()-start
    return fit


def trace_loss(target, candidate):
    """Gleichung (1), OHNE Quadrat. Invariant unter globaler Phase."""
    if target.shape != candidate.shape or target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("Patch-Matrizen müssen quadratisch und gleich groß sein")
    return float(1 - abs(np.vdot(target, candidate)) / target.shape[0])


def patch_objective(target, ansatz):
    if ansatz.n > 8:
        raise MemoryError("Patch-Optimierung auf höchstens 8 Qubits begrenzt")
    d = 2 ** ansatz.n
    if target.shape != (d, d) or not np.allclose(target.conj().T @ target, np.eye(d), atol=1e-10):
        raise ValueError("Zielpatch muss eine passende Unitäre sein")

    def objective(parameters):
        bound = ansatz.bind(parameters)
        forward = [np.eye(d, dtype=complex)]
        for gate in bound.gates:
            forward.append(apply_matrix(forward[-1], gate.matrix(), gate.qubits, bound.n))
        overlap = np.vdot(target, forward[-1])
        # Hilbert-Schmidt-Bra rückwärts durch die Gate-Folge propagieren.
        back = target.copy()
        gradient = []
        for k in reversed(range(len(bound.gates))):
            gate = bound.gates[k]
            if gate.trainable:
                derivative = np.vdot(back, apply_matrix(forward[k], gate.derivative(), gate.qubits, bound.n))
                # |z| ist bei z=0 nicht differenzierbar. Null ist ein zulässiger
                # Subgradient; zufällige Starts vermeiden diesen Sonderfall meist.
                grad = -np.real(np.conj(overlap) * derivative) / (d * abs(overlap)) if abs(overlap) > 1e-15 else 0.0
                gradient.append(grad)
            back = apply_matrix(back, gate.matrix().conj().T, gate.qubits, bound.n)
        return 1 - float(abs(overlap)) / d, np.array(gradient[::-1])
    return objective


def fit_patch(original, candidate, maxiter=100, seconds=30):
    if original.n != candidate.n:
        raise ValueError("Patch-Register stimmen nicht überein")
    return _minimize(candidate, patch_objective(unitary(original), candidate), maxiter, seconds)
