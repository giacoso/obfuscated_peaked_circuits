"""Kleine, simulatorunabhängige Schaltkreisdarstellung.

Konvention: |q0 q1 ... q(n-1)>; q0 ist das höchstwertige Bit (big endian).
Eine Zwei-Qubit-Matrix benutzt die Reihenfolge ihres qubits-Tupels. Qiskit
verwendet eine andere Konvention; die Umrechnung steht in references.py.
Unveränderliche Dataclasses verhindern versehentliche Änderungen an U, wenn
später T[U] optimiert wird. Damit bleibt U† wirklich die ursprüngliche Inverse.
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
SWAP = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
GENERATORS = {"RX": X, "RY": Y, "RZ": Z, "RZZ": np.kron(Z, Z)}


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    theta: float = 0.0
    block: str = ""
    layer: int = 0

    def __post_init__(self):
        arity = 2 if self.name in ("RZZ", "SWAP") else 1
        if self.name not in {*GENERATORS, "X", "SWAP"}:
            raise ValueError(f"Unbekanntes Gate: {self.name}")
        if len(self.qubits) != arity or len(set(self.qubits)) != arity:
            raise ValueError("Falsche Anzahl oder doppelte Qubit-Indizes")
        if any(q < 0 for q in self.qubits) or not np.isfinite(self.theta):
            raise ValueError("Qubit-Indizes und Winkel müssen gültig sein")

    @property
    def trainable(self):
        return self.name in GENERATORS

    def matrix(self):
        if self.name == "X":
            return X.copy()
        if self.name == "SWAP":
            return SWAP.copy()
        # exp(-i theta A/2), A²=I. Diese Form ermöglicht exakte Ableitungen.
        a = GENERATORS[self.name]
        return np.cos(self.theta / 2) * np.eye(len(a)) - 1j * np.sin(self.theta / 2) * a

    def derivative(self):
        return (-0.5j * GENERATORS[self.name]) @ self.matrix()

    def inverse(self):
        return replace(self, theta=-self.theta) if self.trainable else self


@dataclass(frozen=True)
class Circuit:
    n: int
    gates: tuple[Gate, ...] = ()

    def __post_init__(self):
        if self.n < 1 or any(max(g.qubits) >= self.n for g in self.gates):
            raise ValueError("Ungültige Registergröße oder Qubit außerhalb des Registers")

    def then(self, other):
        if self.n != other.n:
            raise ValueError("Registergrößen stimmen nicht überein")
        return Circuit(self.n, self.gates + other.gates)

    def inverse(self, block=None):
        return Circuit(self.n, tuple(replace(g.inverse(), block=block or g.block)
                                     for g in reversed(self.gates)))

    def parameters(self):
        return np.array([g.theta for g in self.gates if g.trainable])

    def bind(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != self.parameters().shape:
            raise ValueError("Falsche Anzahl Parameter")
        it = iter(values)
        return Circuit(self.n, tuple(replace(g, theta=float(next(it))) if g.trainable
                                     else g for g in self.gates))

    def relabel(self, permutation):
        if sorted(permutation) != list(range(self.n)):
            raise ValueError("Keine Permutation")
        return Circuit(self.n, tuple(replace(g, qubits=tuple(permutation[q] for q in g.qubits))
                                     for g in self.gates))

    def save(self, path):
        """Lesbare Checkpoints: Winkel, Wire-Indizes und Herkunft bleiben erhalten."""
        from dataclasses import asdict
        Path(path).write_text(json.dumps({"n": self.n, "gates": [asdict(g) for g in self.gates]},
                                        indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["n"], tuple(Gate(**{**g, "qubits": tuple(g["qubits"])}) for g in data["gates"]))


def apply_matrix(state, matrix, qubits, n):
    """Wendet eine lokale Matrix an; keine globale 2^n x 2^n-Matrix.

    Zusätzliche trailing-Achsen sind erlaubt, z.B. die Spalten einer lokalen
    Patch-Unitären. Erst aktive Achsen nach vorn, dann multiplizieren, dann
    die ursprüngliche Achsenreihenfolge wiederherstellen.
    """
    shape = state.shape
    trailing = shape[1:]
    tensor = state.reshape((2,) * n + trailing)
    axes = list(qubits) + [q for q in range(n) if q not in qubits] + list(range(n, tensor.ndim))
    moved = tensor.transpose(axes)
    out = matrix @ moved.reshape(2 ** len(qubits), -1)
    return out.reshape(moved.shape).transpose(np.argsort(axes)).reshape(shape)


def statevector(circuit, initial=None, max_qubits=20):
    """Exakte Kleinsystem-Referenz. Große Register werden vor Allokation abgelehnt."""
    if circuit.n > max_qubits:
        raise MemoryError(f"Statevector auf {max_qubits} Qubits begrenzt; Tensor-Backend verwenden")
    state = np.zeros(2 ** circuit.n, dtype=complex) if initial is None else np.array(initial, dtype=complex)
    if initial is None:
        state[0] = 1
    if state.shape != (2 ** circuit.n,):
        raise ValueError("Initialzustand hat falsche Dimension")
    for gate in circuit.gates:
        state = apply_matrix(state, gate.matrix(), gate.qubits, circuit.n)
    return state


def unitary(circuit, max_qubits=8):
    """Nur für kleine Patches und Tests, niemals für das gesamte große Register."""
    if circuit.n > max_qubits:
        raise MemoryError("Patch zu breit: vollständige Unitäre benötigt 16 * 4^n Bytes")
    result = np.eye(2 ** circuit.n, dtype=complex)
    for gate in circuit.gates:
        result = apply_matrix(result, gate.matrix(), gate.qubits, circuit.n)
    return result
