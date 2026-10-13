"""Vergleiche mit unabhängigen Simulatoren und bereitgestelltem Altcode.

Referenzschaltungen haben nicht dieselben Zufallszahlen/Architekturen wie
unsere Konstruktion. Aussagekräftig ist der Vergleich IDENTISCHER Gate-Folgen,
nicht die Gleichheit zweier unabhängig trainierter Peakgewichte.
"""
import importlib.util
from pathlib import Path
import numpy as np
import quimb.tensor as qtn
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector
from .circuit import apply_matrix, statevector


def to_qiskit(circuit):
    """Erhalte physische Qubit-IDs. Qiskit-Bitstrings sind q(n-1)...q0."""
    qc = QuantumCircuit(circuit.n)
    for gate in circuit.gates:
        method = getattr(qc, gate.name.lower())
        if gate.trainable:
            method(gate.theta, *gate.qubits)
        else:
            method(*gate.qubits)
    return qc


def qiskit_state(circuit):
    # Bit-Reversal nur am API-Rand, keine versteckte Änderung der Topologie.
    state = np.asarray(Statevector.from_instruction(to_qiskit(circuit)).data)
    return state.reshape((2,) * circuit.n).transpose(tuple(reversed(range(circuit.n)))).reshape(-1)


def compare_simulators(circuit):
    """NumPy, Quimb und Qiskit für denselben kleinen Circuit vergleichen."""
    if circuit.n > 14:
        raise MemoryError("Vollständiger Referenzvergleich ist auf 14 Qubits begrenzt")
    own = statevector(circuit)
    qc_state = qiskit_state(circuit)
    tn = qtn.Circuit(circuit.n)
    for gate in circuit.gates:
        tn.apply_gate_raw(gate.matrix(), gate.qubits)
    tensor = np.asarray(tn.to_dense(optimize="greedy")).reshape(-1)
    return {"qiskit_max_abs_error": float(np.max(abs(own - qc_state))),
            "quimb_max_abs_error": float(np.max(abs(own - tensor))),
            "norm": float(np.vdot(own, own).real)}


def compare_legacy(directory, n=6, depth=4, seed=13):
    """Führe functions.py UND circuits.py mit identischen Haar-U(4)-Gates aus.

    Der RNG des alten Codes wird bewusst umgangen: seed_init wird dort zwar
    weitergereicht, setzt aber keinen Zufallsgenerator. Wir übergeben uni_list.
    Damit prüft der Vergleich die tatsächliche Gate-/Tensor-Reihenfolge.
    Die vorhandenen Dateien werden weder geändert noch als Skript ausgeführt.
    """
    rng = np.random.default_rng(seed)
    pairs = [(q, q + 1) for layer in range(depth) for q in range(layer % 2, n - 1, 2)]
    matrices = []
    for _ in pairs:
        z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        q, r = np.linalg.qr(z)
        matrices.append(q @ np.diag(np.diag(r) / abs(np.diag(r))))
    own = np.zeros(2**n, dtype=complex)
    own[0] = 1
    for pair, matrix in zip(pairs, matrices):
        own = apply_matrix(own, matrix, pair, n)
    reports = {}
    for name in ("circuits", "functions"):
        path = Path(directory) / (name + ".py")
        spec = importlib.util.spec_from_file_location("legacy_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tn = module.qmps_f(L=n, in_depth=depth, n_Qbit=n-1, uni_list=matrices, rand=False)
        # Explizite Indexreihenfolge statt blindem reshape einer kontrahierten TN.
        tensor = tn.contract(all)
        result = np.asarray(tensor.transpose(*[f"k{q}" for q in range(n)]).data).reshape(-1)
        reports[name] = {"max_abs_error": float(np.max(abs(own-result))), "n": n,
                         "gates": len(pairs), "seed": seed}
    return reports


def export_qasm(circuit, path):
    """Portables OpenQASM 2: RZZ und SWAP explizit in CX/Rotationen zerlegen.

    Qiskits Exporter nimmt sonst erweiterte qelib1-Definitionen an, die ein
    strikter QASM-2-Importer nicht automatisch kennt. Die Zerlegungen ändern
    die Unitäre nicht. Der originale Gate-Satz bleibt im JSON-Checkpoint.
    """
    qc = QuantumCircuit(circuit.n)
    for gate in circuit.gates:
        if gate.name == "SWAP":
            a,b = gate.qubits
            qc.cx(a,b); qc.cx(b,a); qc.cx(a,b)
        elif gate.name == "RZZ":
            a,b = gate.qubits
            qc.cx(a,b); qc.rz(gate.theta,b); qc.cx(a,b)
        elif gate.trainable:
            getattr(qc,gate.name.lower())(gate.theta,*gate.qubits)
        else:
            qc.x(*gate.qubits)
    qasm2.dump(qc, str(path))


def inspect_external_qasm(path, max_window_qubits=8, window_gates=60):
    """Prüfe veröffentlichte QASM-Dateien, ohne 49/56 Qubits dicht zu simulieren.

    Der komplette Circuit wird geparst und strukturell vermessen. Zusätzlich
    wird ab dem ersten 2-Qubit-Gate ein zusammenhängendes begrenztes Zeitfenster
    mit höchstens max_window_qubits als eigenständiger Circuit gegengeprüft.
    Dieses Fenster ist KEIN Nachweis für den Peak des vollständigen Benchmarks.
    """
    import networkx as nx
    import hashlib
    source = Path(path).read_text(encoding="utf-8-sig")
    if source.lstrip().startswith("OPENQASM 3"):
        from qiskit import qasm3
        # Diese öffentliche Datei benutzt den alten Alias u3 mit QASM-3-Header.
        # U hat exakt dieselbe Matrix. Reparatur nur im Speicher, Original bleibt.
        import re
        source = re.sub(r"\bu3\s*\(", "U(", source)
        qc = qasm3.loads(source)
    else:
        qc = qasm2.loads(source)
    graph = nx.Graph()
    graph.add_nodes_from(range(qc.num_qubits))
    for inst in qc.data:
        ids = [qc.find_bit(q).index for q in inst.qubits]
        if len(ids) == 2:
            graph.add_edge(*ids)
    start = next((i for i, inst in enumerate(qc.data) if len(inst.qubits) == 2), 0)
    selected, active = [], set()
    for inst in qc.data[start:]:
        if inst.operation.name in ("measure", "reset", "barrier"):
            break
        ids = [qc.find_bit(q).index for q in inst.qubits]
        if len(active | set(ids)) > max_window_qubits:
            break
        active.update(ids)
        selected.append(inst)
        if len(selected) >= window_gates:
            break
    active = sorted(active)
    remap = {q: i for i, q in enumerate(active)}
    small = QuantumCircuit(len(active))
    own = np.zeros(2**len(active), dtype=complex)
    own[0] = 1
    tn = qtn.Circuit(len(active))
    for inst in selected:
        ids = [remap[qc.find_bit(q).index] for q in inst.qubits]
        small.append(inst.operation, ids)
        matrix = np.asarray(inst.operation.to_matrix())
        # Qiskit-Matrizen ordnen lokale Basisbits entgegengesetzt zu NumPy.
        own = apply_matrix(own, matrix, tuple(reversed(ids)), len(active))
        tn.apply_gate_raw(matrix, tuple(reversed(ids)))
    expected = np.asarray(Statevector.from_instruction(small).data).reshape((2,) * len(active)).transpose(tuple(reversed(range(len(active))))).reshape(-1)
    tensor = np.asarray(tn.to_dense(optimize="greedy")).reshape(-1)
    return {"file": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "n_qubits": qc.num_qubits, "gates": len(qc.data), "gate_counts": dict(qc.count_ops()),
            "interaction_edges": graph.number_of_edges(), "max_degree": max(dict(graph.degree()).values()),
            "window_start": start, "window_gates": len(selected), "window_qubits": len(active),
            "window_qiskit_error": float(np.max(abs(own-expected))),
            "window_quimb_error": float(np.max(abs(own-tensor))),
            "full_peak_verified": False}

