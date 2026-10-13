"""Ideales IBM-Heavy-Hex-Gitter: jede Hexagonkante erhält ein weiteres Qubit.

Quelle: IBM-Blog vom 7. Juli 2021. cells bezeichnet Hexagon-Zellen pro Seite,
KEINE Code-Distanz und keinen IBM-Gerätenamen. Ganze Gitter haben
N=5*cells²+8*cells-1 Qubits. Ausschnitte bleiben verbundene induzierte Teilgraphen.
Qiskits from_heavy_hex erzeugt einen anderen, auf Fehlerkorrektur bezogenen
Codegraphen; dessen Randzyklen werden hier nicht mit Hardwarezellen verwechselt.
"""
import networkx as nx
from .circuit import Circuit, Gate


def heavy_hex(cells=2, max_qubits=None):
    if not isinstance(cells, int) or isinstance(cells, bool) or cells < 1:
        raise ValueError("cells muss eine positive ganze Zahl sein")
    honeycomb = nx.hexagonal_lattice_graph(cells, cells, with_positions=True)
    original = sorted(honeycomb.nodes())
    ids = {q: i for i, q in enumerate(original)}
    graph = nx.Graph()
    for q in original:
        graph.add_node(ids[q], pos=honeycomb.nodes[q]["pos"], kind="vertex")
    # Jede ursprüngliche Kante wird durch zwei Kanten ersetzt. Die Position
    # des zusätzlichen Qubits liegt genau zwischen den ursprünglichen Ecken.
    for k, (a, b) in enumerate(sorted(honeycomb.edges())):
        middle = len(original)+k
        pa, pb = honeycomb.nodes[a]["pos"], honeycomb.nodes[b]["pos"]
        graph.add_node(middle, pos=((pa[0]+pb[0])/2,(pa[1]+pb[1])/2), kind="edge")
        graph.add_edges_from([(ids[a],middle),(middle,ids[b])])
    nx.set_node_attributes(graph, {q:q for q in graph}, "physical_id")
    if max_qubits is not None:
        if not isinstance(max_qubits,int) or not 1 <= max_qubits <= len(graph):
            raise ValueError("max_qubits liegt außerhalb des Gitters")
        nodes = list(nx.bfs_tree(graph,0))[:max_qubits]
        graph = nx.convert_node_labels_to_integers(graph.subgraph(nodes).copy())
    graph.graph.update(cells=cells, kind="subdivided_hexagonal_lattice")
    return graph


def edge_layers(graph):
    """Zerlegt Kanten deterministisch in Matchings: kein Qubit doppelt im Layer.

    Ein bipartiter Graph hat eine Δ-Kantenfärbung. Wiederholte Matchings,
    die alle Knoten des aktuellen Maximalgrades abdecken, reduzieren den
    Maximalgrad in jedem Schritt. Maximum-weight matching priorisiert diese
    Knoten; ihre gleichzeitige Abdeckung existiert für bipartite Graphen.
    """
    work = graph.copy()
    layers = []
    while work.number_of_edges():
        degree = dict(work.degree())
        delta = max(degree.values())
        for a, b in work.edges():
            work[a][b]["weight"] = int(degree[a] == delta) + int(degree[b] == delta)
        matching = nx.max_weight_matching(work, maxcardinality=True)
        layer = sorted(tuple(sorted(e)) for e in matching)
        layers.append(layer)
        work.remove_edges_from(layer)
    return layers


def ansatz(graph, depth, rng, block, angle_scale=1.0, start_layer=0):
    """Eigene konkrete Ansatzwahl, da Kapitel 3 diese nicht vollständig festlegt.

    Pro Layer: RZ-RY-RZ auf jedem Qubit, danach parametrisierte RZZ-Gates auf
    einem Matching. Die Matchings rotieren über das gesamte Heavy-Hex-Gitter.
    Ein 'Layer' bezeichnet hier EIN Matching, nicht einen vollständigen Zyklus.
    Beliebige Einzelqubitrotationen + ZZ bilden einen universellen Gate-Satz.
    angle_scale skaliert uniforme Winkel aus [-pi, pi].
    """
    import numpy as np
    if depth < 0 or angle_scale < 0:
        raise ValueError("Tiefe und Winkelskala dürfen nicht negativ sein")
    matchings = edge_layers(graph)
    gates = []
    for layer in range(depth):
        for q in graph:
            for name in ("RZ", "RY", "RZ"):
                gates.append(Gate(name, (q,), float(rng.uniform(-np.pi, np.pi) * angle_scale), block, layer))
        if matchings:
            for pair in matchings[(layer + start_layer) % len(matchings)]:
                gates.append(Gate("RZZ", pair, float(rng.uniform(-np.pi, np.pi) * angle_scale), block, layer))
    return Circuit(len(graph), tuple(gates))


def validate_connectivity(circuit, graph):
    if circuit.n != len(graph):
        raise ValueError("Schaltkreis und Graph besitzen unterschiedliche Registergrößen")
    for gate in circuit.gates:
        if len(gate.qubits) == 2 and not graph.has_edge(*gate.qubits):
            raise ValueError(f"Nichtlokales Gate auf {gate.qubits}")
