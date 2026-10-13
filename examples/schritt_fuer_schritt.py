"""Schrittweiser Einstieg; vom Projektverzeichnis mit python -m examples.schritt_fuer_schritt.

Die Rechnung ist dieselbe wie in der CLI. Der Hook ist der Erweiterungspunkt:
Hier lassen sich später Messungen, andere Grafiken oder Checkpoints ergänzen.
"""
from pathlib import Path
from hqap import Config, build
from hqap.simulation import peak_probability
from hqap.visualization import draw_circuit, draw_tensor_network, draw_summary

output = Path("results/walkthrough")


def main():
    output.mkdir(parents=True, exist_ok=True)

    def inspect_stage(name, circuit, metadata):
        print(name, "Gates:", len(circuit.gates), metadata)
        circuit.save(output / (name + ".json"))
        # Nur der gewünschte Zwischenstand wird gezeichnet. Dafür muss der
        # Algorithmus nicht geändert und kein bereits trainiertes P neu erzeugt werden.
        if name == "06_masked":
            draw_circuit(circuit, output/"mitte", start=100, max_gates=50)
            draw_tensor_network(circuit, output/"mitte_tn", start=100, max_gates=50)

    config = Config(cells=2, max_qubits=8, seed=7)
    result = build(config, callback=inspect_stage)
    draw_summary(result, output/"uebersicht")
    final = result.stages["07_final"]
    print("Zielgewicht:", peak_probability(final, "0"*final.n))
    print("Absicherung:", result.report["probability_lower_bound"])


if __name__ == "__main__":
    main()

