"""Exakte Tensor-Kontraktion mit expliziter Ressourcenprüfung.

Das Backend berechnet eine EINZELNE Amplitude, keinen Vektor mit 2^n Einträgen.
'Exakt' bedeutet ohne Bond-Trunkierung, bis auf Gleitkommafehler. Auch dieser
Algorithmus kann exponentiell teuer werden. Vor Ausführung prüfen wir den
Kontraktionsbaum auf größte Zwischentensoren und geschätzte Rechenoperationen.
"""
from dataclasses import dataclass
import numpy as np
import quimb.tensor as qtn
from .circuit import GENERATORS, statevector


@dataclass(frozen=True)
class Budget:
    max_tensor_bytes: int = 256 * 1024**2
    max_flops: float = 2e9


class ContractionBudgetExceeded(RuntimeError):
    pass


def tensor_network(circuit, output=None):
    """Explizites Netzwerk für Visualisierung UND Kontraktion.

    Jeder Gate-Tensor hat erst Ausgangs-, dann Eingangsindizes. Ein Index
    verbindet immer aufeinanderfolgende Operationen desselben Qubits. Mit
    output=None bleiben n Ausgänge offen; das allein kontrahiert noch nichts.
    """
    if output is not None and (len(output) != circuit.n or set(output) - {"0", "1"}):
        raise ValueError("Ausgabe-Bitstring muss n binäre Zeichen enthalten")
    wires = [f"q{q}:0" for q in range(circuit.n)]
    tensors = [qtn.Tensor(np.array([1., 0.], dtype=complex), inds=(w,), tags={"INPUT", f"Q{q}"})
               for q, w in enumerate(wires)]
    for k, gate in enumerate(circuit.gates):
        incoming = tuple(wires[q] for q in gate.qubits)
        outgoing = tuple(f"q{q}:g{k}" for q in gate.qubits)
        tensors.append(qtn.Tensor(gate.matrix().reshape((2,) * (2 * len(gate.qubits))),
                                  inds=outgoing + incoming, tags={f"G{k}", gate.block or "GATE"}))
        for q, w in zip(gate.qubits, outgoing):
            wires[q] = w
    if output is not None:
        for q, bit in enumerate(output):
            tensors.append(qtn.Tensor(np.eye(2, dtype=complex)[int(bit)], inds=(wires[q],), tags={"OUTPUT"}))
    return qtn.TensorNetwork(tensors)


class AmplitudePlan:
    """Einmal Kontraktionspfad suchen; während Training nur Zahlen austauschen.

    Autograd differenziert die Kontraktion rückwärts. So braucht ein kompletter
    Gradient keinen separaten Parameter-Shift-Lauf für jeden Winkel. Der Test
    gegen Parameter Shift bleibt eine unabhängige mathematische Kontrolle.
    """
    def __init__(self, circuit, target, budget=Budget()):
        self.circuit = circuit
        self.network = tensor_network(circuit, target)
        self.tree = self.network.contraction_tree(optimize="greedy", output_inds=())
        self.stats = {"largest_tensor_bytes": int(self.tree.max_size()) * 16,
                      "estimated_flops": float(self.tree.contraction_cost()),
                      "num_tensors": self.network.num_tensors}
        # Der erste Wert ist KEIN Prozess-RAM-Limit: Rückwärtsableitungen und
        # mehrere gleichzeitig lebende Tensoren können zusätzlichen RAM kosten.
        if self.stats["largest_tensor_bytes"] > budget.max_tensor_bytes or self.stats["estimated_flops"] > budget.max_flops:
            raise ContractionBudgetExceeded(f"Kontraktion überschreitet Budget: {self.stats}")
        self.arrays = list(self.network.arrays)

    def amplitude(self, parameters=None):
        import autograd.numpy as anp
        arrays = list(self.arrays)
        if parameters is not None:
            j = 0
            for k, gate in enumerate(self.circuit.gates):
                if gate.trainable:
                    a = GENERATORS[gate.name]
                    t = parameters[j]
                    matrix = anp.cos(t / 2) * np.eye(len(a)) - 1j * anp.sin(t / 2) * a
                    arrays[self.circuit.n + k] = anp.reshape(matrix, (2,) * (2 * len(gate.qubits)))
                    j += 1
        return self.tree.contract(arrays, backend="autograd" if parameters is not None else "numpy")


def peak_probability(circuit, target, backend="tensor", budget=Budget()):
    if len(target) != circuit.n or set(target) - {"0", "1"}:
        raise ValueError("Ungültiger Ziel-Bitstring")
    if backend == "dense":
        return float(abs(statevector(circuit)[int(target, 2)]) ** 2)
    if backend != "tensor":
        raise ValueError("backend muss dense oder tensor sein")
    return float(abs(AmplitudePlan(circuit, target, budget).amplitude()) ** 2)
