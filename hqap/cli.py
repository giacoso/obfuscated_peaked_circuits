"""Kommandozeilenzugang: Demo, Konfiguration, Checkpoint-Plots und Skalierung."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import monotonic
import platform
from importlib.metadata import version
from . import Circuit, Config, build
from .simulation import Budget, ContractionBudgetExceeded
from .references import compare_legacy, compare_simulators, export_qasm, inspect_external_qasm
from .visualization import draw_circuit, draw_lattice, draw_summary, draw_tensor_network, draw_full_circuit


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in ("numpy", "scipy", "quimb", "qiskit", "networkx", "autograd")}}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment(config, output, budget=Budget(), plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", asdict(config))
    def checkpoint(name, circuit, metadata):
        circuit.save(output / (name + ".json"))
        print(f"{name}: {circuit.n} Qubits, {len(circuit.gates)} Gates", flush=True)
    start = monotonic()
    result = build(config, callback=checkpoint, budget=budget)
    result.report["elapsed_seconds"] = monotonic() - start
    result.report["environment"] = environment()
    import hashlib
    result.report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(Path(__file__).parent.glob("*.py"))}
    save_json(output / "coupling_map.json",
              {"nodes": [{"id": q, **data} for q, data in result.graph.nodes(data=True)],
               "edges": list(result.graph.edges()), "metadata": result.graph.graph})
    save_json(output / "report.json", result.report)
    export_qasm(result.stages["07_final"], output / "final.qasm")
    if plots:
        draw_lattice(result.graph, output / "heavy_hex")
        draw_summary(result, output / "overview")
        draw_full_circuit(result.stages["07_final"], output / "full_circuit")
        from .topology import heavy_hex
        draw_lattice(heavy_hex(config.cells), output / "full_lattice",
                     highlighted=[result.graph.nodes[q]["physical_id"] for q in result.graph])
        for name, circuit in result.stages.items():
            draw_circuit(circuit, output / (name+"_circuit"), name)
        draw_tensor_network(result.stages["07_final"], output / "tensor_network")
    print(json.dumps({key: result.report[key] for key in ("n_qubits", "shallow_probability", "final_probability",
          "probability_lower_bound", "meets_requested_peak", "elapsed_seconds")}, indent=2), flush=True)
    return result


def benchmark(cell_sizes, output, seconds=30, steps=60):
    """Vollständige Konstruktion je Größe, mit Status statt Erfolgssuggestion.

    Gleicher Seed, gleiche Tiefen, gleiche Ressourcenregeln; eine Stichprobe
    pro Größe ist ein Skalierungs-Smoketest, keine statistische Härteanalyse.
    Nach jedem Versuch wird der Bericht gespeichert (auch nach einem Abbruch).
    """
    rows = []
    output = Path(output)
    for cells in cell_sizes:
        config = Config(cells=cells, max_qubits=None, train_steps=steps,
                        train_seconds=seconds, patch_count=1, patch_steps=100,
                        identity_depth=6, verify_final=False)
        print(f"Skalierung: cells={cells}", flush=True)
        start = monotonic()
        try:
            result = run_experiment(config, output/f"cells{cells}", plots=False)
            row = {"cells": cells, "n": result.report["n_qubits"], "status": "constructed",
                   "lower_bound": result.report["probability_lower_bound"],
                   "shallow_probability": result.report["shallow_probability"],
                   "meets_peak": result.report["meets_requested_peak"],
                   "optimizer_success": result.report["training"]["success"],
                   "seconds": monotonic()-start}
        except (ContractionBudgetExceeded, MemoryError) as exc:
            row = {"cells": cells, "status": "resource_limit", "reason": str(exc),
                   "seconds": monotonic()-start}
        rows.append(row)
        save_json(output/"scaling.json", {"environment": environment(), "runs": rows})
    import matplotlib.pyplot as plt
    valid = [r for r in rows if "n" in r]
    if valid:
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        axes[0].plot([r["n"] for r in valid], [r["seconds"] for r in valid], "o-", color="#3478c8")
        axes[0].set(xlabel="Qubits", ylabel="Konstruktionszeit / s", title="Gemessene Skalierung (ein Seed)")
        axes[1].semilogy([r["n"] for r in valid], [max(1e-16,r["lower_bound"]) for r in valid], "o-", color="#2b9b83")
        axes[1].axhline(.01,color="#df665f",ls="--",label="Anforderung: 0.01")
        axes[1].set(xlabel="Qubits", ylabel="Untere Schranke für Peakgewicht", title="Nullwerte am Darstellungsminimum 1e-16")
        axes[1].legend()
        from .visualization import _save
        fig.tight_layout()
        _save(fig, output/"scaling")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path, default=Path("results/demo"))
    run.add_argument("--no-plots", action="store_true")
    viz = sub.add_parser("visualize")
    viz.add_argument("checkpoint", type=Path)
    viz.add_argument("--start", type=int, default=0)
    viz.add_argument("--gates", type=int, default=70)
    viz.add_argument("--output", type=Path, default=Path("results/custom_view"))
    scale = sub.add_parser("benchmark")
    scale.add_argument("--cells", nargs="+", type=int, default=[1,2,3,4,5])
    scale.add_argument("--seconds", type=float, default=30)
    scale.add_argument("--steps", type=int, default=60)
    scale.add_argument("--output", type=Path, default=Path("results/scaling"))
    compare = sub.add_parser("compare")
    compare.add_argument("--legacy", type=Path, default=Path("schon_bestehende_codes_für_BA"))
    compare.add_argument("--external", type=Path, default=Path("references/peaked_circuit_heavy_hex_49x4020.qasm"))
    compare.add_argument("--output", type=Path, default=Path("results/comparison.json"))
    args = parser.parse_args(argv)
    if args.command == "run":
        config = Config(**json.loads(args.config.read_text(encoding="utf-8-sig"))) if args.config else Config()
        result = run_experiment(config, args.output, plots=not args.no_plots)
        return 0 if result.report["meets_requested_peak"] else 2
    if args.command == "visualize":
        circuit = Circuit.load(args.checkpoint)
        draw_circuit(circuit, args.output, start=args.start, max_gates=args.gates)
        draw_tensor_network(circuit, str(args.output)+"_tn", start=args.start, max_gates=args.gates)
    elif args.command == "benchmark":
        benchmark(args.cells, args.output, args.seconds, args.steps)
    elif args.command == "compare":
        from .topology import heavy_hex, ansatz
        import numpy as np
        circuit = ansatz(heavy_hex(2,8),5,np.random.default_rng(21),"R")
        report = {"legacy": compare_legacy(args.legacy), "simulators": compare_simulators(circuit),
                  "external": inspect_external_qasm(args.external), "environment": environment()}
        save_json(args.output, report)
        print(json.dumps(report, indent=2))
    return 0

