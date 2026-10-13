"""
Peaked-Circuit-Demo für ein überschaubares System.

WICHTIG:
- Die Funktionen aus circuits.py werden NICHT verändert.
- Wir benutzen qmps_f(), norm_f() und negative_overlap() genau so,
  wie sie in der bereitgestellten Datei definiert sind.
- "10 random" und "5 peaking" werden hier als 10 bzw. 5 BRICKWALL-LAYER
  interpretiert, weil qmps_f(..., in_depth=...) in deinem Code die Tiefe
  in Layern parametrisiert.

Für L = 20 bedeutet das:
- 10 Random-Layer  -> 95 Zwei-Qubit-Gates           besser wären min 20 random layer, damit die Verschränkiung überall ist
- 5 Peaking-Layer  -> 48 Zwei-Qubit-Gates
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import re
import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer

# Die folgenden Funktionen kommen unverändert aus deiner Datei:
from circuits import qmps_f, norm_f, negative_overlap


# ============================================================
# 2. EINSTELLUNGEN
# ============================================================

L = 20

RANDOM_DEPTH = 10
PEAKING_DEPTH = 5

N_OPT_STEPS = 200

# Reproduzierbarkeit für alles, was über NumPy läuft.
np.random.seed(1)


# ============================================================
# 3. HILFSFUNKTIONEN
#    Diese Funktionen sind NEU und ändern deine Originalfunktionen nicht.
# ============================================================

def brickwall_schedule(L, depth):
    """
    Gibt nur die Reihenfolge der Qubit-Paare zurück,
    auf die qmps_f(..., n_Qbit=L-1, qmps_structure='brickwall')
    seine Zwei-Qubit-Gates anwendet.

    Beispiel:
    Layer 0: (0,1), (2,3), ...
    Layer 1: (1,2), (3,4), ...
    """
    schedule = []

    n_Qbit = L - 1
    i_start = 0

    for r in range(depth):
        if r % 2 == 0:
            for i in range(i_start, i_start + n_Qbit, 2):
                schedule.append((i, i + 1))
        else:
            for i in range(i_start, i_start + n_Qbit - 1, 2):
                schedule.append((i + 1, i + 2))

    return schedule


def extract_gate_matrices_in_G_order(tn):
    """
    Liest die Gate-Tensoren G0, G1, G2, ... aus einem Tensor Network aus.

    Das ist absichtlich ähnlich zur Logik von save_para(), nur dass wir
    hier die Gates explizit nach ihrer G-Nummer sortieren.
    """
    gate_tags = [
        tag for tag in tn.tags
        if re.fullmatch(r"G\d+", tag)
    ]

    gate_tags = sorted(
        gate_tags,
        key=lambda tag: int(tag[1:])
    )

    gate_matrices = []

    for tag in gate_tags:
        tensor = tn[tag]

        # Kompatibel mit dem Stil aus save_para():
        if isinstance(tensor, tuple):
            tensor = tensor[0]

        U = np.asarray(tensor.data).reshape(4, 4)
        gate_matrices.append(U)

    return gate_matrices


def apply_inverse_pqc_to_state(state, optimized_pqc, L, depth):
    """
    Aus dem trainierten PQC V bauen wir den eigentlichen Peaking-Teil V†.

    Training:
        V |0...0>  ≈  R |0...0>

    Danach:
        V† R |0...0>  ≈  |0...0>

    Genau deshalb entsteht am Ende ein Peak auf 0^L.
    """
    result = state.copy()

    schedule = brickwall_schedule(L, depth)
    gates = extract_gate_matrices_in_G_order(optimized_pqc)

    if len(schedule) != len(gates):
        raise RuntimeError(
            f"Gate-Anzahl passt nicht: Schedule={len(schedule)}, "
            f"extrahierte Gates={len(gates)}"
        )

    # Für V† müssen Reihenfolge UND Gates invertiert werden.
    for j, (pair, U) in enumerate(
        zip(reversed(schedule), reversed(gates))
    ):
        U_dagger = U.conj().T

        result.gate_(
            U_dagger,
            pair,
            tags={"PEAKING", f"PG{j}"}
        )

    return result


def peak_weight_zero(state, L):
    """
    Wahrscheinlichkeit für den Bitstring 00...0:

        delta_0 = |<0^L | psi>|^2
    """
    zero_state = qtn.MPS_computational_state("0" * L).astype_("complex128")

    # negative_overlap gibt per Definition das Negative zurück.
    return float(-negative_overlap(state, zero_state))


def draw_network(tn, title):
    fig, ax = plt.subplots(figsize=(14, 7))

    tn.draw(
        color=["MPS", "U"],
        ax=ax,
        show_tags=False,
        show_inds=False,
        legend=True,
    )

    ax.set_title(title)

    plt.tight_layout()
    plt.show()


# ============================================================
# 4. RANDOM CIRCUIT R ERZEUGEN
# ============================================================

print("============================================================")
print("1) RANDOM CIRCUIT R")
print("============================================================")

random_schedule = brickwall_schedule(L, RANDOM_DEPTH)

print(f"L = {L}")
print(f"Random-Tiefe = {RANDOM_DEPTH} Layer")
print(f"Anzahl Zwei-Qubit-Gates in R = {len(random_schedule)}")

rqc = qmps_f(
    L,
    in_depth=RANDOM_DEPTH,
    n_Qbit=L - 1,
    qmps_structure="brickwall",
)

draw_network(
    rqc,
    "Tensor Network nach dem Random Circuit R"
)

delta_before = peak_weight_zero(rqc, L)
print(f"Gewicht von |0...0> direkt nach R: {delta_before:.8e}")


# ============================================================
# 5. TRAINIERBARES PQC V AUFBAUEN
# ============================================================

print("\n============================================================")
print("2) TRAINIERBARES PQC V")
print("============================================================")

peaking_schedule = brickwall_schedule(L, PEAKING_DEPTH)

print(f"Peaking-Tiefe = {PEAKING_DEPTH} Layer")
print(f"Anzahl trainierbarer Zwei-Qubit-Gates = {len(peaking_schedule)}")

pqc = qmps_f(
    L,
    in_depth=PEAKING_DEPTH,
    n_Qbit=L - 1,
    qmps_structure="brickwall",
)

draw_network(
    pqc,
    "Trainierbares PQC V vor der Optimierung"
)


# ============================================================
# 6. PQC TRAINIEREN
# ============================================================

print("\n============================================================")
print("3) OPTIMIERUNG")
print("============================================================")

print(
    "Wir trainieren V so, dass V|0...0> möglichst gut "
    "mit R|0...0> überlappt."
)

optmzr = TNOptimizer(
    pqc,
    loss_fn=negative_overlap,
    norm_fn=norm_f,
    loss_target=-1 + 1e-2,
    constant_tags=["MPS"],
    loss_constants={"target": rqc},
    autodiff_backend="tensorflow",
    optimizer="L-BFGS-B",
)

pqc_optimized = optmzr.optimize(
    N_OPT_STEPS,
    tol=1e-10,
)

training_fidelity = float(
    -negative_overlap(pqc_optimized, rqc)
)

print(
    "Erreichte Trainings-Fidelity "
    "|<V0|R0>|^2 = "
    f"{training_fidelity:.8f}"
)

draw_network(
    pqc_optimized,
    "PQC V nach der Optimierung"
)


# ============================================================
# 7. OPTIONAL: OPTIMIERUNGSVERLAUF
# ============================================================

if hasattr(optmzr, "losses") and len(optmzr.losses) > 0:
    losses = np.asarray(optmzr.losses, dtype=float)

    plt.figure()
    plt.plot(-losses)
    plt.xlabel("Optimierungsschritt")
    plt.ylabel(r"$|\langle V0|R0\rangle|^2$")
    plt.title("Training des Peaking-PQC")
    plt.grid(True)
    plt.show()


# ============================================================
# 8. AUS V DEN EIGENTLICHEN PEAKING CIRCUIT V† BAUEN
# ============================================================

print("\n============================================================")
print("4) PEAKED CIRCUIT C = V† R")
print("============================================================")

peaked_state = apply_inverse_pqc_to_state(
    state=rqc,
    optimized_pqc=pqc_optimized,
    L=L,
    depth=PEAKING_DEPTH,
)

draw_network(
    peaked_state,
    "Gesamtes Tensor Network des peaked circuits: V† R |0...0>"
)


# ============================================================
# 9. PEAKGEWICHT AUSWERTEN
# ============================================================

delta_after = peak_weight_zero(peaked_state, L)

print("\n============================================================")
print("5) ERGEBNIS")
print("============================================================")

print(f"Peakgewicht vorher : {delta_before:.8e}")
print(f"Peakgewicht nachher: {delta_after:.8f}")
print(f"Trainings-Fidelity : {training_fidelity:.8f}")

print(
    "\nTheoretisch sollten Trainings-Fidelity und finales "
    "Peakgewicht nahezu gleich sein, denn:"
)

print(
    "  |<0| V† R |0>|² = |<V0|R0>|²"
)

print(
    "\nDer konstruierte Circuit ist also auf den Bitstring "
    + "0" * L
    + " gepeakt."
)