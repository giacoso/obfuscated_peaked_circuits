# HQAP auf einem Heavy-Hex-Gitter

Kommentierte Forschungsimplementierung von Kapitel 3 aus **Gharibyan et al.,
„Heuristic Quantum Advantage with Peaked Circuits“, arXiv:2510.25838v1**.
Das lokale `Paper_für_Bachelorarbeit/paper.pdf` ist die hierfür geprüfte Fassung.

**Einstieg:** `hqap/protocol.py` zeigt den gesamten Ablauf.
`examples/schritt_fuer_schritt.py` zeigt, wie man Zwischenstände untersucht.
`results/demo/overview.png` fasst einen tatsächlich ausgeführten Lauf zusammen.

Die gemessenen Resultate und die Grenzen der Vergleiche stehen in
[ERGEBNISSE.md](ERGEBNISSE.md).

## Was genau reproduziert wird

Die fünf veröffentlichten Schritte sind implementiert:

1. Ein flacher Zufallsschaltkreis R und ein variationaler Peaking-Schaltkreis P.
   Trainiert wird das Gewicht eines vorgegebenen Bitstrings.
2. Zwischen R und P kommt ein tieferer Block U gefolgt von seiner **exakten**
   adjungierten Gate-Folge U†.
3. **Sweeping:** kleine lokale Patches erhalten veränderte Startwinkel und
   werden bei gleicher Gate-Architektur erneut optimiert.
4. **Masking:** lokale Patches werden durch eine **andere** trainierbare
   Gate-Folge ersetzt.
5. **SWAP-Transformationen:** Wire-Permutationen, Rand-SWAPs und physisches
   Routing werden konsistent zusammen ausgeführt.

**Das ist eine eigene Umsetzung der veröffentlichten Methode, keine
bitgenaue Reproduktion des internen Autorencodes.** Kapitel 3 enthält weder
einen vollständigen Generator noch alle Ansatz-, Initialisierungs-,
Patch-Auswahl- und Optimierereinstellungen. Unsere Entscheidungen werden
in [METHODIK.md](METHODIK.md) einzeln zugeordnet. Der Wechsel von der
All-to-all-Geometrie der Hauptexperimente zu Heavy Hex ist ausdrücklich eine
Anpassung an die Aufgabenstellung.

Ein erfolgreich konstruierter Peak beweist weder klassische Härte noch
Quantenvorteil. Die einfachen Standardeinstellungen sollen zuerst Korrektheit
und Änderbarkeit demonstrieren.

## Start unter Windows / PowerShell

Die projektlokale Umgebung `.venv` wurde für diese Arbeit bereits eingerichtet.
Alle Befehle im Projektverzeichnis ausführen:

```powershell
.\.venv\Scripts\python.exe -m hqap run --output results/demo
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m hqap compare
```

Neuinstallation auf einem anderen Rechner:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Die tatsächlich verwendeten Versionen stehen in `requirements-lock.txt`
und zusätzlich in den Ergebnisberichten. Für eine identische Umgebung:
`python -m pip install -r requirements-lock.txt`.
Die Bibliotheken liegen isoliert von euren ursprünglichen Skripten.
Das Einlesen der lokalen PDFs gehört nicht zur Laufzeit des Pakets.

## Eigene Experimente

```powershell
.\.venv\Scripts\python.exe -m hqap run --config configs/heavy_hex_35.json --output results/own35
.\.venv\Scripts\python.exe -m hqap benchmark --cells 1 2 3 4 5 --seconds 30 --steps 60
```

`cells` ist die Anzahl hexagonaler Zellen **pro Seite**. Das ist keine
Fehlerkorrektur-Code-Distanz. Beispiele vollständiger Gitter:

| cells | Qubits |
|---:|---:|
| 1 | 12 |
| 2 | 35 |
| 3 | 68 |
| 4 | 111 |
| 5 | 164 |
| 6 | 227 |
| 7 | 300 |
| 8 | 383 |

`max_qubits: null` verwendet das ganze Gitter. Eine Zahl schneidet per
Breitensuche einen verbundenen Teilgraphen aus. Solche kleinen Ausschnitte
können noch keine ganze Hexagonzelle enthalten.

Wichtige Parameter:

| Parameter | Bedeutung |
|---|---|
| random_depth | Zahl der R-Layer mit je einem Kanten-Matching |
| peaking_depth | Zahl der trainierbaren P-Layer |
| identity_depth | Tiefe von U; U† hat dieselbe Gate-Anzahl |
| random_angle_scale | Skalierung gleichverteilter Winkel aus [-π, π]; 1 ist Standard |
| train_steps / train_seconds | Iterations- und Zeitbudget des Peaking-Trainings |
| patch_count | Anzahl **Versuche je Art**, also Sweeping und Masking getrennt |
| patch_width | Höchstens 2–5 Qubits pro Patch; Standard 2 |
| patch_tolerance | Maximaler akzeptierter Verlust gemäß Gleichung (1) |
| patch_steps / patch_seconds | Budget **pro Patchversuch** |
| kick | Standardabweichung des Winkel-Kicks beim Sweeping |
| swaps | Anzahl global konsistenter Transpositionen |
| target | Bitstring in Reihenfolge q0, q1, ...; null bedeutet ausschließlich Nullen |
| min_peak | Eigene Anforderung an das finale Zielgewicht; Standard 0.01 |
| verify_final | Exakte finale Tensor-Kontraktion versuchen oder nur Fehlerschranke |
| training_backend | tensor für Skalierung, dense für kleine Kontrollsysteme |
| rescale_training | kontrollierte L-BFGS-B-Neustarts; false für die ursprüngliche Vergleichsvariante |

Für die Definition aller Standardwerte siehe `Config` in `hqap/protocol.py`.
Ein Zeitlimit ist **kooperativ**: Eine gerade laufende Kontraktion/BLAS-Operation
kann länger dauern. Die Kontraktionsbudgets begrenzen größte Zwischentensoren
und geschätzte Operationen, nicht den gesamten Prozessspeicher.

`run` gibt Exitcode **2** zurück, wenn die angeforderte Peak-Schwelle nicht
erreicht bzw. abgesichert wurde; die Diagnostik wird trotzdem gespeichert.
Ein Optimierungsstopp am Iterationslimit ist nicht dasselbe wie Konvergenz.

## Zwischenstände ansehen und verändern

Jeder Lauf speichert sieben JSON-Checkpoints. Der zusätzliche QASM-2-Export
zerlegt RZZ in CX-RZ-CX und SWAP in drei CX, damit er auch mit einem strikten
QASM-Importer funktioniert. Die Gate-Anzahl des Exports ist deshalb größer.

Die Checkpoints sind:

- 01: R
- 02: R gefolgt vom untrainierten P
- 03: trainiertes R ▷ P
- 04: R ▷ U ▷ U† ▷ P
- 05: nach Sweeping
- 06: nach Masking
- 07: nach SWAP-Transformationen und Routing

Beispiel: ein Fenster **mitten** im bereits gespeicherten Netzwerk:

```powershell
.\.venv\Scripts\python.exe -m hqap visualize results/demo/06_masked.json --start 180 --gates 50 --output results/middle
```

Das führt kein Training erneut aus. PNG und SVG zeigen jeweils genau das
beschriftete Gate-Fenster. Die Tensor-Grafik zeigt offene Schnittgrenzen und
behauptet nicht, dass der Zustand dort unverschränkt sei.

In Python:

```python
from dataclasses import replace
from hqap import Circuit
from hqap.visualization import draw_circuit, draw_tensor_network

circuit = Circuit.load("results/demo/06_masked.json")
gates = list(circuit.gates)
gates[0] = replace(gates[0], theta=gates[0].theta + 0.05)
changed = Circuit(circuit.n, tuple(gates))
draw_circuit(changed, "results/changed", start=180, max_gates=50)
draw_tensor_network(changed, "results/changed_tn", start=180, max_gates=50)
```

Nach eigenen Änderungen gelten die zuvor gespeicherten Peakwerte und
Fehlerschranken **nicht automatisch weiter**. Neu auswerten oder erneut bauen.

## Dateien und Erweiterungspunkte

| Datei | Aufgabe |
|---|---|
| hqap/circuit.py | Gate-Matrizen, Reihenfolge, inverse Schaltungen, Checkpoints |
| hqap/topology.py | Gitter, Matchings und Ansatzarchitektur |
| hqap/simulation.py | echte Tensor-Netzwerke, Kontraktionspfade, Ressourcenprüfung |
| hqap/optimization.py | Peaking-Ziel, Patch-Ziel, analytische/automatische Gradienten |
| hqap/transforms.py | sichere Patch-Auswahl, Sweeping, Masking, SWAP-Routing |
| hqap/protocol.py | kompletter Ablauf, Callbacks, globale Fehlerschranke |
| hqap/visualization.py | Gitter-, Circuit-, Tensor- und Diagnosegrafiken |
| hqap/references.py | NumPy/Quimb/Qiskit, Altcode und externe QASM-Referenz |
| hqap/cli.py | reproduzierbare Kommandozeilenläufe und Skalierung |
| tests/test_hqap.py | mathematische und integrationsbezogene Regressionen |

## Was die Vergleiche aussagen

- **NumPy ↔ Quimb ↔ Qiskit:** identische Gate-Folgen und vollständige kleine
  Zustände; zusätzlich Test nichttrivialer Bitstrings.
- **Eure circuits.py und functions.py:** identische vorgegebene Haar-U(4)-Gates
  und Brickwall-Reihenfolge. Der Altcode wird nicht verändert. Seine globale
  Zufallsinitialisierung wird durch eine explizite Gate-Liste umgangen.
- **Öffentliche Heavy-Hex-Referenz:** 49 Qubits, 4020 CZ-Gates und 8089
  Einzelqubitgates. Vollständiger Import und Strukturanalyse; ein begrenztes
  Zeitfenster wird mit drei Simulatoren verglichen. **Der vollständige
  Referenz-Peak wurde nicht klassisch verifiziert.**
- **Skalierung:** ein Seed je Größe und dokumentierte Budgets. Erzeugung,
  erfolgreiches Training und vollständige finale Simulation werden getrennt
  erfasst. Größere Qubitzahl allein ist kein Qualitätsnachweis.

Die Aqora-Datenquelle verweist auf veröffentlichte Schaltungen des Quantum
Advantage Trackers. Im gefundenen Datensatz steht kein vollständiger
Konstruktionsgenerator. Daher ist keine behauptete „Aqora-Generatorparität“
Teil dieses Projekts. Herkunft und unveränderte Datei-Prüfsumme stehen in
`references/provenance.json`.

## Quellen

- [Paper, Kapitel 3](https://arxiv.org/abs/2510.25838)
- [IBM: The IBM Quantum heavy hex lattice](https://www.ibm.com/quantum/blog/heavy-hex-lattice)
- [Aqora-Datensatz v0.0.0 mit Herkunftsbeschreibung](https://aqora.io/datasets/aqora/quantum-advantage-tracker-peaked-circuits/versions/0.0.0)
- [Öffentliche Benchmark-Schaltungen](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/tree/main/data/classically-verifiable-problems/circuit-models/peaked_circuit)

