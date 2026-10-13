"""Wiederverwendbare Grafiken für beliebige Zwischenstände.

Alle Funktionen nehmen den aktuellen Circuit explizit entgegen. Keine globale
Variable und keine automatische Neuberechnung des Experiments. PNG zur Ansicht,
SVG zur verlustfreien Verwendung in der Bachelorarbeit.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from .simulation import tensor_network

COLORS = {"R": "#3478c8", "U": "#a562d4", "U_DAGGER": "#2b9b83", "P": "#df665f",
          "SWEEP": "#dc9b30", "MASK": "#d56b9d", "SWAP": "#424b5a", "ROUTE": "#9199a7"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(path.with_suffix("." + extension), dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_lattice(graph, path, highlighted=()):
    """Zeige tatsächliche Kopplungskanten und optional einen Patch-Support."""
    # Kamada-Kawai nutzt Graphdistanzen und zeigt die Heavy-Hex-Zellen deutlich.
    pos = nx.get_node_attributes(graph, "pos") or nx.kamada_kawai_layout(graph)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#df665f" if q in highlighted else "#3478c8" for q in graph]
    nx.draw_networkx(graph, pos, ax=ax, node_color=colors, font_color="white",
                     node_size=430 if len(graph) < 80 else 130, font_size=8,
                     width=1.7, edge_color="#bcc7d5", with_labels=len(graph) <= 200)
    ax.set_title(f"Heavy-Hex-Kopplungsgraph | {len(graph)} Qubits, {graph.number_of_edges()} Kanten", loc="left", pad=20)
    ax.axis("off")
    if highlighted:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#df665f",label="Hervorgehobener Ausschnitt"),
                           Patch(color="#3478c8",label="Übriges Gitter")],
                  loc="lower right",frameon=False)
    fig.tight_layout()
    _save(fig, path)


def draw_circuit(circuit, path, title="Schaltkreis", start=0, stop=None, max_gates=100):
    """Ein lesbares Gate-Fenster; Kürzung und tatsächliche Indizes sind beschriftet.

    Gates auf disjunkten Qubits werden für die Darstellung in dieselbe Spalte
    gelegt. Die Reihenfolge auf jeder Wire bleibt unverändert.
    """
    stop = min(len(circuit.gates), start + max_gates) if stop is None else min(stop, len(circuit.gates))
    if not 0 <= start <= stop:
        raise ValueError("Ungültiges Gate-Fenster")
    selected = circuit.gates[start:stop]
    active = sorted({q for g in selected for q in g.qubits})
    if not active:
        return
    ypos = {q: len(active)-i-1 for i, q in enumerate(active)}
    clocks = {q: 0 for q in active}
    placed = []
    for i, gate in enumerate(selected, start):
        # Vertikale Gate-Verbindungen dürfen andere Symbole nicht kreuzen.
        lo, hi = sorted([ypos[q] for q in gate.qubits]) if len(gate.qubits) == 2 else (ypos[gate.qubits[0]],)*2
        span = [q for q in active if lo <= ypos[q] <= hi]
        x = max(clocks[q] for q in span) + 1
        for q in span:
            clocks[q] = x
        placed.append((x, gate, i))
    width = max(x for x, _, _ in placed) + 1
    fig, ax = plt.subplots(figsize=(min(26, max(10, width*.44)), max(3.4, len(active)*.45)))
    for q in active:
        ax.hlines(ypos[q], 0, width, color="#dce2ea", lw=1)
        ax.text(-.35, ypos[q], f"q{q}", ha="right", va="center")
    for x, gate, index in placed:
        color = COLORS.get(gate.block, "#63768b")
        ys = [ypos[q] for q in gate.qubits]
        ax.plot([x]*len(ys), ys, color=color, lw=1.8)
        for y in ys:
            ax.text(x, y, gate.name, ha="center", va="center", fontsize=6,
                    bbox={"boxstyle": "round,pad=.25", "fc": "white", "ec": color, "lw":1.5})
    ax.set_title(f"{title}\nGate-Fenster [{start}, {stop}) von {len(circuit.gates)} Gates; Zeit nach rechts", loc="left", pad=20)
    ax.set_xlim(-1, width+.5)
    ax.set_ylim(-.8, len(active)-.1)
    ax.axis("off")
    from matplotlib.patches import Patch
    blocks = list(dict.fromkeys(g.block for g in selected))
    ax.legend(handles=[Patch(color=COLORS.get(b, "#63768b"), label=b) for b in blocks],
              loc="upper center", bbox_to_anchor=(.5, -.015), ncol=min(8,len(blocks)), frameon=False)
    _save(fig, path)


def draw_tensor_network(circuit, path, max_gates=70, start=0):
    """Echte Tensor-Indizes als Kanten, Gate-Tensoren als Knoten.

    Gezeigt wird ein eigenständiges Gate-Fenster mit offenen Eingängen und
    Ausgängen. Die Eingangsknoten symbolisieren Schnittindizes, nicht die
    Behauptung, der Zwischenzustand sei ein Produktzustand.
    """
    from .circuit import Circuit
    cut = circuit.gates[start:start+max_gates]
    sub = Circuit(circuit.n, cut)
    network = tensor_network(sub)
    graph, pos, labels, colors = nx.MultiGraph(), {}, {}, []
    active = sorted({q for g in cut for q in g.qubits})
    if not active:
        return
    row = {q: i for i, q in enumerate(active)}
    previous = {}
    clocks = {q: 0 for q in active}
    for q in active:
        node = f"in{q}"
        graph.add_node(node, block="INPUT")
        pos[node], labels[node], previous[q] = (0, -row[q]), f"q{q}", node
    for i, gate in enumerate(cut):
        node = f"G{i}"
        x = i+1  # Eigene Zeitspalte je Gate: keine verdeckten Tensor-Knoten.
        graph.add_node(node, block=gate.block)
        pos[node] = (x, -np.mean([row[q] for q in gate.qubits]))
        labels[node] = gate.name
        for q in gate.qubits:
            graph.add_edge(previous[q], node)
            previous[q], clocks[q] = node, x
    end = max(clocks.values())+1
    for q in active:
        node = f"out{q}"
        graph.add_node(node, block="OUTPUT")
        graph.add_edge(previous[q], node)
        pos[node], labels[node] = (end, -row[q]), f"q{q}"
    fig, ax = plt.subplots(figsize=(max(12, min(32, len(cut)*.46)), max(4, len(active)*.55)))
    colors = [COLORS.get(graph.nodes[v]["block"], "#bdc8d4") for v in graph]
    nx.draw_networkx(graph, pos=pos, labels=labels, node_color=colors, node_size=260,
                     font_size=6, font_color="white", edge_color="#bbc5d1", ax=ax,
                     connectionstyle=["arc3,rad=0.06","arc3,rad=-0.06"])
    ax.set_title(f"Tensor-Netzwerk: Gate-Fenster [{start}, {start+len(cut)})\nKnoten = Tensoren, Kanten = Qubit-Indizes (Dimension 2); offene Schnittgrenzen", loc="left", pad=20)
    ax.axis("off")
    _save(fig, path)


def draw_summary(result, path):
    """Training, lokale Verluste, Protokoll und überprüftes Peakgewicht."""
    report = result.report
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    blocks = [("T[R]", "#3478c8"), ("T[U]", "#a562d4"), ("U†", "#2b9b83"), ("P", "#df665f")]
    for i, (label, color) in enumerate(blocks):
        ax.barh(0, .85, left=i, color=color, height=.55)
        ax.text(i+.425, 0, label, ha="center", va="center", color="white", fontsize=18)
    ax.text(1.9, -.52, "Zeit →   |   Sweeping / Masking / SWAPs", ha="center", fontsize=10)
    ax.set_ylim(-.8,.7); ax.axis("off")
    ax.set_title(f"HQAP-Protokoll | {report['n_qubits']} Qubits", loc="left")
    ax = axes[0, 1]
    history = np.array(report["training"].get("probability_history") or (1-np.array(report["training"]["history"])))
    ax.plot(history, color="#3478c8", lw=2)
    if report["n_qubits"] > 20 and np.all(history > 0):
        ax.set_yscale("log")
    ax.set(xlabel="Optimiererschritte / Neuskalierungen", ylabel="Zielwahrscheinlichkeit",
           title="Training des flachen R ▷ P")
    ax.grid(alpha=.2)
    ax = axes[1, 0]
    for i, patch in enumerate(report["patches"]):
        if "history" in patch:
            ax.semilogy(np.maximum(patch["history"], 1e-16),
                        label=f"{patch['kind']} {i+1} ({'akzeptiert' if patch['accepted'] else 'abgelehnt'})")
    ax.set(xlabel="Iteration", ylabel="1 - |Tr(T† T̃)| / d", title="Lokale Tensor-Patch-Optimierung")
    if report["patches"]:
        ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = axes[1, 1]
    lo, hi = report["probability_lower_bound"], report["probability_upper_bound"]
    labels = ["Flach trainiert", "Finale Schranke"]
    values = [report["shallow_probability"], (lo+hi)/2]
    ax.bar([0,1], values, color=["#3478c8","#2b9b83"], width=.5)
    ax.errorbar([1], [(lo+hi)/2], yerr=[(hi-lo)/2], color="black", capsize=8)
    if report["final_probability"] is not None:
        labels.append("Final exakt")
        ax.bar(2, report["final_probability"], color="#df665f", width=.5)
    ax.set_xticks(range(len(labels)), labels)
    ax.set(ylabel="Zielwahrscheinlichkeit", title="Peak-Erhaltung", ylim=(0, min(1.05,max(hi,values[0]) * 1.3 + 1e-12)))
    fig.suptitle("Forschungsdiagnostik · Kapitel 3 · arXiv:2510.25838v1", x=.06, ha="left", fontsize=16)
    fig.tight_layout(rect=(0,0,1,.95), h_pad=3)
    _save(fig, path)



def draw_full_circuit(circuit, path, title="Gesamter Schaltkreis"):
    """Alle Gates als zeitliches Netzwerk, ohne verdeckte Kürzung.

    X ist der serielle Gate-Index (keine Hardware-Laufzeit), Y die Qubit-ID.
    Einzelne Winkel lassen sich im JSON oder im beschrifteten Fensterplot lesen.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(16, max(4, min(11, circuit.n*.30))))
    for q in range(circuit.n):
        ax.axhline(q, color="#e2e7ed", linewidth=.4, zorder=0)
    points = {}
    segments, edgecolors = [], []
    for i, gate in enumerate(circuit.gates):
        color = COLORS.get(gate.block, "#63768b")
        points.setdefault(gate.block, ([], []))
        for q in gate.qubits:
            points[gate.block][0].append(i)
            points[gate.block][1].append(q)
        if len(gate.qubits) == 2:
            segments.append([(i, gate.qubits[0]), (i, gate.qubits[1])])
            edgecolors.append(color)
    ax.add_collection(LineCollection(segments, colors=edgecolors, linewidths=.6, alpha=.75))
    for block, (xs, ys) in points.items():
        ax.scatter(xs, ys, s=8 if len(circuit.gates)<1500 else 2,
                   color=COLORS.get(block,"#63768b"), marker="s", linewidths=0)
    ax.set(xlabel="Serieller Gate-Index (Zeit nach rechts)", ylabel="Qubit-ID",
           title=f"{title} | {circuit.n} Qubits, {len(circuit.gates)} Gates",
           xlim=(-2,max(1,len(circuit.gates))), ylim=(circuit.n-.4,-.6))
    ax.legend(handles=[Patch(color=COLORS.get(b,"#63768b"),label=b) for b in points],
              ncol=min(8,len(points)),frameon=False,loc="upper center",bbox_to_anchor=(.5,-.14))
    fig.tight_layout()
    _save(fig,path)
