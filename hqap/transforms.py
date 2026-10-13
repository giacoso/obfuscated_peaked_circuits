"""Tensor-Patches (§3.1) und physisch konsistente SWAPs (§3.2).

Patches enden an jedem Gate, das ihren Support mit der Außenwelt verbindet.
Disjunkte Gates dazwischen kommutieren und dürfen beim Ersetzen stehen bleiben.
"""
from dataclasses import replace
import networkx as nx
import numpy as np
from .circuit import Circuit, Gate, unitary
from .optimization import fit_patch
from .topology import ansatz, validate_connectivity


def patch_candidates(circuit, graph, width=2):
    """Wähle kausal abgeschlossene Teilfolgen auf kleinen verbundenen Supports."""
    if not 2 <= width <= 5:
        raise ValueError("Patchbreite muss zwischen 2 und 5 liegen")
    supports = set()
    for edge in sorted(graph.edges()):
        support = set(edge)
        while len(support) < min(width, len(graph)):
            neighbors = sorted({v for u in support for v in graph[u]} - support)
            if not neighbors:
                break
            support.add(neighbors[0])
        supports.add(tuple(sorted(support)))
    candidates = []
    for support in sorted(supports):
        selected = []
        def finish():
            if len(selected) >= 3 and any(len(circuit.gates[i].qubits) == 2 for i in selected):
                candidates.append((tuple(selected), support))
        for i, gate in enumerate(circuit.gates):
            if not (set(gate.qubits) & set(support)):
                continue
            if set(gate.qubits) <= set(support):
                selected.append(i)
            else:
                finish()
                selected = []
        finish()
    return candidates


def transform_patches(circuit, graph, kind, count, rng, width=2, kick=0.35,
                      tolerance=1e-5, maxiter=150, seconds=20):
    """Sweeping: Winkel kicken. Masking: andere lokale Gate-Folge trainieren.

    Ungenaue Ersetzungen werden abgelehnt und trotzdem im Bericht erfasst.
    Der phasenbereinigte Operatornormfehler ermöglicht eine globale Schranke.
    """
    if kind not in ("sweep", "mask") or count < 0 or tolerance < 0 or kick < 0:
        raise ValueError("Ungültige Patch-Einstellungen")
    current, reports = circuit, []
    for attempt in range(count):
        candidates = patch_candidates(current, graph, width)
        if not candidates:
            reports.append({"kind": kind, "accepted": False, "reason": "Kein gültiger Patch"})
            break
        indices, support = candidates[int(rng.integers(len(candidates)))]
        local_index = {q: j for j, q in enumerate(support)}
        original = Circuit(len(support), tuple(replace(current.gates[i], qubits=tuple(local_index[q] for q in current.gates[i].qubits)) for i in indices))
        if kind == "sweep":
            proposal = original.bind(original.parameters() + rng.normal(0, kick, len(original.parameters())))
        else:
            local_graph = nx.relabel_nodes(graph.subgraph(support), local_index)
            # Eigene Architekturwahl: zusätzliche Euler/RZZ-Layer statt der
            # ursprünglichen Gate-Folge. Keine Kopie mit versteckten Nullgates.
            depth = max(6, 2 * sum(len(g.qubits) == 2 for g in original.gates))
            proposal = ansatz(local_graph, depth, rng, "MASK", angle_scale=0.5, start_layer=1)
        fit = fit_patch(original, proposal, maxiter=maxiter, seconds=seconds)
        accepted = fit.final_loss <= tolerance
        old, new = unitary(original), unitary(fit.circuit)
        phase = np.exp(-1j * np.angle(np.vdot(old, new)))
        operator_error = float(np.linalg.norm(phase * new - old, ord=2))
        reports.append({"kind": kind, "attempt": attempt, "support": list(support),
                        "gate_indices": list(indices), "initial_loss": fit.initial_loss,
                        "final_loss": fit.final_loss, "accepted": accepted,
                        "operator_error": operator_error if accepted else 0.0,
                        "candidate_operator_error": operator_error,
                        "iterations": fit.iterations, "optimizer_success": fit.success,
                        "message": fit.message, "seconds": fit.seconds, "history": fit.history,
                        "original_gates": len(original.gates), "candidate_gates": len(proposal.gates)})
        if accepted:
            replacement = tuple(replace(g, qubits=tuple(support[q] for q in g.qubits), block="SWEEP" if kind == "sweep" else "MASK") for g in fit.circuit.gates)
            removed, gates = set(indices), []
            for i, gate in enumerate(current.gates):
                if i == indices[0]:
                    gates.extend(replacement)
                if i not in removed:
                    gates.append(gate)
            current = Circuit(current.n, tuple(gates))
    validate_connectivity(current, graph)
    return current, reports


def route_to_graph(circuit, graph):
    """Nichtlokales Gate durch lokale SWAPs erreichbar machen, danach rückführen.

    Das erhält an jeder Gate-Grenze die logische Zuordnung. Dieses transparente
    Referenzrouting kostet mehr Gates als ein optimierender Hardware-Compiler.
    """
    gates = []
    for gate in circuit.gates:
        if len(gate.qubits) == 1 or graph.has_edge(*gate.qubits):
            gates.append(gate)
            continue
        a, b = gate.qubits
        path = nx.shortest_path(graph, a, b)
        swaps = [Gate("SWAP", (path[j], path[j + 1]), block="ROUTE") for j in range(len(path) - 2)]
        gates.extend(swaps)
        gates.append(replace(gate, qubits=(path[-2], b)))
        gates.extend(reversed(swaps))
    result = Circuit(circuit.n, tuple(gates))
    validate_connectivity(result, graph)
    return result


def swap_transform(circuit, graph, swaps):
    """Gleichung (2): X_sigma = S X S, folglich S X_sigma S = X.

    BEIDE Rand-SWAPs bleiben enthalten: Gleichheit gilt für jeden Eingang,
    auch mitten im Circuit. Nur ganz am Anfang bei |0...0> wäre einer entbehrlich.
    Relabelte Gates werden lokal geroutet, sonst wäre Heavy Hex verletzt.
    """
    transformed = circuit
    for a, b in swaps:
        if not graph.has_edge(a, b):
            raise ValueError("Rand-SWAP muss auf einer Heavy-Hex-Kante liegen")
        permutation = list(range(circuit.n))
        permutation[a], permutation[b] = b, a
        boundary = Circuit(circuit.n, (Gate("SWAP", (a, b), block="SWAP"),))
        transformed = boundary.then(route_to_graph(transformed.relabel(permutation), graph)).then(boundary)
    return transformed
