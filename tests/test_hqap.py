"""Mathematische Regressionen, unabhängige Backends und Referenzcode."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import networkx as nx
import pytest
from scipy.optimize._numdiff import approx_derivative
from hqap import Circuit, Gate, Config, build, heavy_hex
from hqap.circuit import apply_matrix, statevector, unitary
from hqap.optimization import dense_peaking_objective, patch_objective, fit_patch, train_peaking, trace_loss
from hqap.simulation import AmplitudePlan, Budget, ContractionBudgetExceeded, peak_probability
from hqap.topology import ansatz, edge_layers, validate_connectivity
from hqap.transforms import swap_transform, route_to_graph, transform_patches, patch_candidates
from hqap.references import compare_simulators, compare_legacy, inspect_external_qasm, to_qiskit

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,qubits", [("RX",(0,)),("RY",(1,)),("RZ",(0,)),("RZZ",(1,0)),("X",(1,)),("SWAP",(0,1))])
def test_gates_and_inverse(name,qubits):
    gate = Gate(name,qubits,.713)
    assert np.allclose(gate.matrix().conj().T @ gate.matrix(), np.eye(len(gate.matrix())), atol=1e-13)
    assert np.allclose(gate.inverse().matrix(), gate.matrix().conj().T)


@pytest.mark.parametrize("distance,n", [(1,12),(2,35),(3,68),(4,111)])
def test_heavy_hex_geometry(distance,n):
    graph = heavy_hex(distance)
    assert len(graph)==n and nx.is_connected(graph) and nx.is_bipartite(graph)
    assert max(dict(graph.degree()).values()) <= 3
    layers = edge_layers(graph)
    assert len(layers) <= 3
    assert {tuple(sorted(e)) for layer in layers for e in layer} == set(tuple(sorted(e)) for e in graph.edges())
    for layer in layers:
        assert len(set(q for e in layer for q in e)) == 2*len(layer)
    if distance >= 1:
        # Eine Heavy-Hex-Zelle ist ein Hexagon mit zusätzlichen Kantenqubits.
        assert min(map(len,nx.minimum_cycle_basis(graph))) == 12


def example(n=5, seed=12):
    graph = heavy_hex(3,n)
    return graph, ansatz(graph,4,np.random.default_rng(seed),"R")


@pytest.mark.parametrize("n", [2,4,7])
def test_three_simulators_and_amplitudes(n):
    _, circuit = example(n)
    report = compare_simulators(circuit)
    assert report["qiskit_max_abs_error"] < 1e-11
    assert report["quimb_max_abs_error"] < 1e-11
    for target in ("0"*n, "1"*n, ("01"*n)[:n]):
        assert peak_probability(circuit,target) == pytest.approx(peak_probability(circuit,target,"dense"), abs=1e-11)


def test_bit_order_and_nonzero_target():
    c = Circuit(3,(Gate("X",(0,)),))
    assert np.argmax(abs(statevector(c))) == 4
    assert peak_probability(c,"100") == pytest.approx(1)
    assert peak_probability(c,"001") == pytest.approx(0)


def test_unitary_identity_and_composition():
    _, circuit = example(5)
    assert np.allclose(unitary(circuit.then(circuit.inverse())), np.eye(32), atol=1e-12)
    state = np.random.default_rng(3).normal(size=32)+1j*np.random.default_rng(5).normal(size=32)
    state /= np.linalg.norm(state)
    assert np.allclose(statevector(circuit, state), unitary(circuit)@state)


def test_adjoint_gradient_and_parameter_shift():
    graph, random = example(4)
    peaking = ansatz(graph,2,np.random.default_rng(9),"P")
    objective = dense_peaking_objective(random,peaking,"1010")
    x = peaking.parameters()
    loss, gradient = objective(x)
    numerical = approx_derivative(lambda p: objective(p)[0],x,method="3-point").ravel()
    assert np.allclose(gradient,numerical,atol=1e-8)
    for j in (0,4,10):
        step = np.zeros(len(x)); step[j] = np.pi/2
        expected = (objective(x+step)[0]-objective(x-step)[0])/2
        assert gradient[j] == pytest.approx(expected,abs=1e-10)


def test_tensor_autodiff_gradient():
    from autograd import value_and_grad
    import autograd.numpy as anp
    graph, random = example(4)
    peaking = ansatz(graph,2,np.random.default_rng(9),"P")
    plan = AmplitudePlan(random.then(peaking),"1010")
    def loss(x):
        a = plan.amplitude(anp.concatenate([random.parameters(), x]))
        return -anp.real(a*anp.conj(a))
    actual, gradient = value_and_grad(loss)(peaking.parameters())
    expected, refgrad = dense_peaking_objective(random,peaking,"1010")(peaking.parameters())
    assert actual == pytest.approx(expected,abs=1e-12)
    assert np.allclose(gradient,refgrad,atol=1e-10)


def test_patch_gradient_and_global_phase():
    graph, circuit = example(3)
    target = unitary(circuit)
    proposal = ansatz(graph,3,np.random.default_rng(91),"MASK")
    objective = patch_objective(target,proposal)
    x = proposal.parameters()
    _,gradient = objective(x)
    numerical = approx_derivative(lambda p: objective(p)[0],x,method="3-point").ravel()
    assert np.allclose(gradient,numerical,atol=1e-8)
    assert trace_loss(target,np.exp(.31j)*target) == pytest.approx(0,abs=1e-13)
    # Hilbert-Schmidt-Overlap ist unquadriert: abs(Tr(RX(theta)))/2 = cos(theta/2).
    assert trace_loss(np.eye(2),Gate("RX",(0,),.8).matrix()) == pytest.approx(1-np.cos(.4))


@pytest.mark.parametrize("backend",["dense","tensor"])
def test_training_improves_nonzero_peak(backend):
    graph = heavy_hex(3,3)
    random = ansatz(graph,2,np.random.default_rng(17),"R")
    peaking = ansatz(graph,2,np.random.default_rng(18),"P")
    fit = train_peaking(random,peaking,"101",backend=backend,maxiter=60)
    assert fit.final_loss < fit.initial_loss-.05
    assert peak_probability(random.then(fit.circuit),"101","dense") == pytest.approx(1-fit.final_loss,abs=1e-10)


def test_sweeping_masking_and_rejection():
    graph, circuit = example(3)
    old = unitary(circuit)
    swept,reports = transform_patches(circuit,graph,"sweep",1,np.random.default_rng(8),maxiter=120)
    assert reports[0]["accepted"]
    new = unitary(swept)
    phase = np.exp(-1j*np.angle(np.vdot(old,new)))
    assert np.linalg.norm(old-phase*new,2) <= reports[0]["operator_error"]+1e-10
    masked,mreports = transform_patches(circuit,graph,"mask",1,np.random.default_rng(9),maxiter=200)
    assert mreports[0]["accepted"]
    assert [(g.name,g.qubits) for g in masked.gates] != [(g.name,g.qubits) for g in circuit.gates]
    rejected,rreports = transform_patches(circuit,graph,"sweep",1,np.random.default_rng(9),maxiter=0,tolerance=1e-15)
    assert not rreports[0]["accepted"] and rejected == circuit


@pytest.mark.parametrize("seed",[1,7,29])
def test_swaps_and_routing_on_arbitrary_inputs(seed):
    graph,circuit = example(6,seed)
    edges = sorted(graph.edges())
    transformed = swap_transform(circuit,graph,[edges[0],edges[-1]])
    validate_connectivity(transformed,graph)
    assert np.allclose(unitary(circuit),unitary(transformed),atol=1e-11)
    # Adversarial non-local operation: endpoints chosen by graph distance.
    a,b = max(((a,b) for a in graph for b in graph),key=lambda p:nx.shortest_path_length(graph,*p))
    nonlocal_circuit = Circuit(6,(Gate("RZZ",(a,b),.817),Gate("SWAP",(b,a))))
    routed = route_to_graph(nonlocal_circuit,graph)
    assert np.allclose(unitary(nonlocal_circuit),unitary(routed),atol=1e-12)
    validate_connectivity(routed,graph)


def test_resources_rejected_before_dense_allocation():
    with pytest.raises(MemoryError):
        statevector(Circuit(30))
    with pytest.raises(MemoryError):
        unitary(Circuit(12))
    _,circuit = example()
    with pytest.raises(ContractionBudgetExceeded):
        AmplitudePlan(circuit,"0"*circuit.n,Budget(max_tensor_bytes=1))


def test_checkpoint_roundtrip(tmp_path):
    _,circuit = example()
    path=tmp_path/"circuit.json"
    circuit.save(path)
    assert Circuit.load(path) == circuit


def test_existing_modules():
    reports = compare_legacy(ROOT/"schon_bestehende_codes_für_BA")
    assert all(report["max_abs_error"]<1e-12 for report in reports.values())


def test_external_reference():
    report=inspect_external_qasm(ROOT/"references/peaked_circuit_heavy_hex_49x4020.qasm")
    assert report["n_qubits"] == 49 and report["window_gates"] > 2
    assert report["window_qiskit_error"] < 1e-11
    assert report["window_quimb_error"] < 1e-11
    assert not report["full_peak_verified"]


def test_full_protocol_checkpoints_bounds_and_seed():
    config=Config(max_qubits=3,train_steps=35,identity_depth=2,patch_count=1,patch_steps=100,swaps=1)
    called=[]
    result=build(config,callback=lambda name,c,m:called.append(name))
    report=result.report
    assert len(called)==7
    assert report["probability_lower_bound"]-1e-10 <= report["final_probability"] <= report["probability_upper_bound"]+1e-10
    for circuit in result.stages.values():
        validate_connectivity(circuit,result.graph)
    # Exakte Insertion vor jeglicher Approximation.
    assert np.allclose(unitary(result.stages["03_shallow_peaked"]),unitary(result.stages["04_identity_inserted"]),atol=1e-11)
    repeated=build(replace(config,patch_count=0,swaps=0))
    assert result.stages["01_random"] == repeated.stages["01_random"]
    assert report["shallow_probability"] == pytest.approx(repeated.report["shallow_probability"],abs=1e-12)


@pytest.mark.parametrize("factory",[lambda: heavy_hex(0),lambda: heavy_hex(1,25),
    lambda: Gate("RY",(0,),float("nan")),lambda: Gate("RZZ",(1,1)),lambda: Circuit(1,(Gate("X",(2,)),)),
    lambda: build(Config(max_qubits=3,target="1000")),lambda: build(Config(random_depth=0))])
def test_invalid_inputs(factory):
    with pytest.raises(ValueError):
        factory()




def test_tiny_probability_is_not_lost_in_one_minus_loss():
    random = Circuit(1,(Gate("RY",(0,),np.pi),))
    fit = train_peaking(random,Circuit(1),"0",backend="dense",maxiter=0)
    expected = np.cos(np.pi/2)**2
    assert fit.final_probability > 0
    assert fit.final_probability == pytest.approx(expected, rel=1e-10, abs=0)
    assert 1-fit.final_loss == 0  # zeigt, warum das separate Feld erforderlich ist



def test_rescaling_preserves_objective_and_escapes_tiny_initial_peak():
    random = Circuit(1,(Gate("RY",(0,),np.pi-.001),))
    peaking = Circuit(1,(Gate("RY",(0,),0),))
    fit = train_peaking(random,peaking,"0",backend="dense",maxiter=40)
    assert fit.rescalings >= 1
    assert fit.final_probability > .99999
    assert peak_probability(random.then(fit.circuit),"0","dense") == pytest.approx(fit.final_probability,abs=1e-12)

def test_qasm_export_roundtrip(tmp_path):
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector
    from hqap.references import export_qasm
    graph,circuit = example(5)
    circuit = swap_transform(circuit,graph,[sorted(graph.edges())[0]])
    path = tmp_path/"circuit.qasm"
    export_qasm(circuit,path)
    restored = qasm2.load(str(path))
    actual = Statevector.from_instruction(restored)
    expected = Statevector.from_instruction(to_qiskit(circuit))
    assert np.allclose(actual.data,expected.data,atol=1e-12)

