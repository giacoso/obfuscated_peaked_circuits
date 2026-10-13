"""Fünf Schritte aus Kapitel 3, mit Callbacks für alle Zwischenstände."""
from dataclasses import asdict, dataclass
import numpy as np
from .circuit import Circuit
from .optimization import train_peaking
from .simulation import Budget, ContractionBudgetExceeded, peak_probability
from .topology import ansatz, heavy_hex, validate_connectivity
from .transforms import swap_transform, transform_patches


@dataclass(frozen=True)
class Config:
    cells: int = 2
    max_qubits: int | None = 8
    random_depth: int = 3
    peaking_depth: int = 2
    identity_depth: int = 6
    seed: int = 7
    target: str | None = None
    random_angle_scale: float = 1.0
    training_backend: str = "tensor"
    rescale_training: bool = True
    train_steps: int = 100
    train_seconds: float = 120
    patch_count: int = 2
    patch_width: int = 2
    patch_steps: int = 150
    patch_seconds: float = 20
    patch_tolerance: float = 1e-5
    kick: float = 0.35
    swaps: int = 2
    min_peak: float = 0.01
    verify_final: bool = True


@dataclass
class Result:
    graph: object
    stages: dict[str, Circuit]
    report: dict


def build(config=Config(), callback=None, budget=Budget()):
    if min(config.random_depth, config.peaking_depth, config.identity_depth) < 1:
        raise ValueError("R-, P- und U-Tiefen müssen positiv sein")
    if config.swaps < 0 or not 0 <= config.min_peak <= 1:
        raise ValueError("Ungültige SWAP-Anzahl oder Peak-Schwelle")
    graph = heavy_hex(config.cells, config.max_qubits)
    target = config.target if config.target is not None else "0" * len(graph)
    if len(target) != len(graph) or set(target) - {"0", "1"}:
        raise ValueError("Zielbitstring muss genau ein Bit je Qubit enthalten")
    # Unabhängige RNG-Streams: Patch-Einstellungen ändern nicht R oder U.
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(5)]
    stages = {}

    def emit(name, circuit, metadata=None):
        validate_connectivity(circuit, graph)
        stages[name] = circuit
        if callback:
            callback(name, circuit, metadata or {})

    random = ansatz(graph, config.random_depth, streams[0], "R", config.random_angle_scale)
    peaking = ansatz(graph, config.peaking_depth, streams[1], "P", angle_scale=0.25)
    emit("01_random", random)
    emit("02_before_training", random.then(peaking))
    fit = train_peaking(random, peaking, target, backend=config.training_backend,
                        maxiter=config.train_steps, seconds=config.train_seconds, budget=budget, rescale=config.rescale_training)
    emit("03_shallow_peaked", random.then(fit.circuit), {"training_loss": fit.final_loss})
    identity_half = ansatz(graph, config.identity_depth, streams[2], "U")
    inverse = identity_half.inverse(block="U_DAGGER")
    prefix = random.then(identity_half)
    suffix = inverse.then(fit.circuit)
    emit("04_identity_inserted", prefix.then(suffix))
    common = dict(count=config.patch_count, width=config.patch_width, kick=config.kick,
                  tolerance=config.patch_tolerance, maxiter=config.patch_steps, seconds=config.patch_seconds)
    prefix, sweep = transform_patches(prefix, graph, "sweep", rng=streams[3], **common)
    emit("05_swept", prefix.then(suffix))
    prefix, mask = transform_patches(prefix, graph, "mask", rng=streams[3], **common)
    emit("06_masked", prefix.then(suffix))
    # §3 erlaubt verschiedene Reihenfolgen. Beide Rand-SWAPs und explizites
    # Routing gewährleisten die Unitärgleichheit auf dem physischen Graphen.
    edges = sorted(graph.edges())
    swaps = [edges[int(streams[4].integers(len(edges)))] for _ in range(config.swaps)] if edges else []
    final = swap_transform(prefix.then(suffix), graph, swaps)
    emit("07_final", final)
    shallow_probability = fit.final_probability
    # Teleskopabschätzung: phasenbereinigte Operatornormfehler addieren sich.
    # Die Schranke benötigt keine Simulation des tiefen fertigen Netzwerks.
    error_bound = sum(p["operator_error"] for p in sweep + mask if p.get("accepted"))
    lower = float(max(0, np.sqrt(max(shallow_probability, 0)) - error_bound) ** 2)
    upper = float(min(1, (np.sqrt(max(shallow_probability, 0)) + error_bound) ** 2))
    final_probability, verification = None, "not_requested"
    if config.verify_final:
        try:
            final_probability = peak_probability(final, target, budget=budget)
            verification = "exact_tensor_contraction"
        except ContractionBudgetExceeded as exc:
            verification = f"budget_exceeded: {exc}"
    checked_weight = final_probability if final_probability is not None else lower
    report = {"config": asdict(config), "n_qubits": len(graph), "target_big_endian": target,
              "training": {k: v for k, v in asdict(fit).items() if k != "circuit"},
              "shallow_probability": shallow_probability, "final_probability": final_probability,
              "final_verification": verification, "probability_lower_bound": lower,
              "probability_upper_bound": upper, "operator_error_bound": error_bound,
              "meets_requested_peak": checked_weight >= config.min_peak,
              "peak_status": "verified" if final_probability is not None else "bounded",
              "uniform_probability": float(2.0 ** (-len(graph))),
              "patches": sweep + mask, "swaps": swaps,
              "stage_gate_counts": {name: len(c.gates) for name, c in stages.items()},
              "note": "Numerische Konstruktion; kein Nachweis klassischer Härte oder eines Quantenvorteils."}
    return Result(graph, stages, report)

