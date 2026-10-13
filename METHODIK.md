# Methodische Zuordnung zur Bachelorarbeit

## Zeit- und Bitkonvention

Im Paper ist A ▷ B als B A definiert. Unsere Gate-Tupel sind **zeitlich**
geordnet: A.then(B) führt zuerst A, dann B aus. Eine Inverse muss deshalb
die Gate-Reihenfolge umkehren **und** jede Matrix adjungieren.

Intern ist q0 das höchstwertige Bit von |q0 q1 ...>. Qiskit nummeriert
Basiszustände entgegengesetzt. Der Adapter kehrt die Tensorachsen an der
Schnittstelle um, ohne physische Wire-Nummern zu ändern. In einem exportierten
QASM verwendet Qiskit daher den umgekehrten gedruckten Zielbitstring.

## Zuordnung: Veröffentlichung und konkrete Entscheidung

| Paper | Implementierung | Einordnung |
|---|---|---|
| §3: flaches R ▷ P | train_peaking | gleiche Zustandswahrscheinlichkeits-Zielfunktion |
| §3: U ▷ U† einfügen | Circuit.inverse | exakt bis auf numerische Rundung |
| §3.1 Gleichung (1) | trace_loss / patch_objective | exakt unquadrierter normierter Trace-Betrag |
| §3.1 Winkel-Sweeping | transform_patches(kind="sweep") | gleiche Architektur, Gauß-Kick, Nachoptimierung |
| §3.1 Masking | transform_patches(kind="mask") | neue Euler/RZZ-Architektur, volle Unitäranpassung |
| §3.2 Gleichung (2) | swap_transform | beide Rand-SWAPs, relabelte Schaltung, lokales Routing |
| Tensor-Netzwerk-Training | Quimb/Cotengra + Autograd | genaue Kontraktion ohne Bond-Trunkierung |
| Gitter und Paarung | unterteiltes Hexagongitter, Kanten-Matchings | Heavy-Hex-Anpassung der Aufgabenstellung |
| konkrete Gate-Ansätze | RZ-RY-RZ plus RZZ | eigene Wahl, im Kapitel nicht vollständig vorgegeben |
| Optimierung | L-BFGS-B mit Gradienten | eigene Wahl aus der gradientenbasierten Familie |
| Patch-Auswahl und -Budgets | kausal abgeschlossene lokale Abschnitte | eigene nachvollziehbare Heuristik |
| akzeptierter Patch-Loss | konfigurierbare obere Schranke | eigene Qualitätskontrolle |
| konservative Peak-Schranke | Summe lokaler Operatornormfehler | zusätzliche mathematische Kontrolle |

Die veröffentlichten Experimente verwenden insbesondere All-to-all-Konnektivität
auf Quantinuum. Unsere Konstruktion verändert diese Bedingung. Laufzeiten,
Härte und Peakgewichte des Papers sind deshalb keine Sollwerte für unsere
Heavy-Hex-Läufe.

## Gate-Ansatz und Tiefe

Eine Ebene enthält auf jedem Qubit RZ, RY und RZ, gefolgt von RZZ-Gates auf
einem Matching. Die Matchings decken zusammen alle Kanten ab; höchstens drei
Ebenen reichen für einen vollständigen Kantenzyklus. RZZ(theta) =
exp(-i theta Z⊗Z / 2). Der Gate-Satz ist universell bei ausreichender Tiefe.

Standard: R hat 3 Matching-Ebenen, P hat 2, U hat 6. Das ist eine
Demonstrationskonfiguration, kein rekonstruierter Autorenparametersatz.
R wird einmal zufällig erzeugt und danach festgehalten; P wird trainiert.
Die Phasenwinkel sind gleichverteilt auf [-π,π], also **kein Haar-Sampling**
der gesamten Schaltung oder beliebiger U(4)-Gates. Die Altcode-Regression
verwendet dagegen explizit vorgegebene Haar-U(4)-Matrizen.

Die physische Kopplung ist erfüllt, aber das ist noch kein IBM-ISA- bzw.
Pulsexport. RZZ muss für einen konkreten Prozessor in dessen native Gates
übersetzt werden; Richtung, Kalibrierung und Rauschen gehören zu einer
späteren Hardwarestudie.

## Zwei verschiedene Zielfunktionen

Peaking maximiert p_s = |<s| P(theta) R |0>|². Im Optimierer wird -p_s statt
1-p_s verwendet, um Auslöschung bei winzigen p_s zu vermeiden. Eine feste
positive Skalierung durch den initialen Wert verändert weder das Optimum
noch die Gradientenrichtung. Nach einer Verbesserung um mehr als Faktor 1000
wird L-BFGS-B mit einer neuen konstanten Skalierung und leerer Hessian-Historie
gestartet. Das verhindert numerische Line-Search-Probleme bei Änderungen über
viele Größenordnungen; Anzahl und Budgetverbrauch der Neustarts werden erfasst. Berichte zeigen weiterhin 1-p_s als Verlust.

Die Rückwärtsableitung der exakten Tensor-Kontraktion läuft über Autograd.
Ein unabhängiges dichtes Adjungiertenverfahren und Parameter-Shift-Tests
kontrollieren die Gradienten auf kleinen Registern.

Für einen k-Qubit-Patch mit d=2^k gilt genau die veröffentlichte Funktion:

L = 1 - |Tr(T† T_tilde)| / d.

Sie wird **nicht quadriert**. Ein zufälliger Eingangszustand oder nur |0>
reicht zur Prüfung eines inneren Patches nicht aus. Deshalb werden kleine
vollständige Unitärmatrizen verwendet. Die Komplexität hängt lokal von 4^k
ab, nicht von 4^N des gesamten Registers.

## Kausalität und SWAPs

Ein Patch darf nicht über ein Gate greifen, das seinen Support mit
Außenqubits verbindet. Der Selektor trennt an solchen Gates. Vollständig
disjunkte Gates innerhalb des Zeitintervalls kommutieren mit dem Patch;
deshalb kann die Ersetzung an seiner ersten Position eingefügt werden.

Für eine Transposition S ist X_sigma = S X S und S X_sigma S = X.
Beide Randoperationen bleiben im Code erhalten. Eine bloße Indexumbenennung
wäre auf einem festen Heavy-Hex-Gitter im Allgemeinen nicht physisch lokal.
Zusätzliche Routing-SWAPs verschieben deshalb Zustände entlang kürzester Wege
und stellen die Zuordnung unmittelbar danach wieder her.

Das Routing ist absichtlich einfach und überprüfbar. Es kann viel Overhead
erzeugen und leicht erkennbare Strukturen hinterlassen. Eine größere
Gate-Anzahl ist daher kein Beleg erfolgreicher Verschleierung.

## Peak-Schranke für große fertige Schaltkreise

Für jede akzeptierte Ersetzung wird eine globale Phase gewählt und der
Operatornormfehler ε_i = ||T_i - exp(-iφ_i) T_tilde_i||_2 berechnet.
Das Einbetten in weitere unitäre Operationen verändert diese Norm nicht.
Mit ε = Summe ε_i folgt durch Teleskopierung und Dreiecksungleichung:

max(0, sqrt(p_flach) - ε)² ≤ p_final ≤ min(1, (sqrt(p_flach) + ε)²).

Die Schranke setzt ein genau kontrahiertes flaches Training und unitäre
Gatefolgen voraus. Gleitkommarechnung macht sie zu einer numerisch
ausgewerteten Schranke, nicht zu einem mit Intervallarithmetik formal
zertifizierten Beweis. Tests prüfen ihre Gültigkeit auf kleinen Systemen.

Abgelehnte Patch-Vorschläge tragen keinen Fehler bei. SWAPs und Routing sind
exakte Identitäten der jeweiligen Unitärwirkung und tragen ebenfalls keinen
Approximationsfehler bei.

Das Zielbitstringgewicht muss nicht der größte Eintrag der Verteilung sein.
Ein Gewicht über 1/2 garantiert dies; für kleinere Gewichte wird hier nur die
angegebene Peakedness-Schwelle abgesichert. Die Chapter-4-Angriffsverfahren
(MPS/Belief Propagation/Pauli-Pfade) sind nicht als vollständige Härtestudie
implementiert.

## Quellen und Reproduzierbarkeit

Primärquelle der Methode ist die lokale Fassung vom 31. Oktober 2025
(arXiv:2510.25838v1). Das ebenfalls vorhandene arXiv:2404.14493v2 ist die
ältere Peaked-Circuit-Arbeit und ersetzt nicht Kapitel 3 des HQAP-Papers.

Die externe QASM-Datei stammt aus dem in provenance.json fixierten
Repository-Commit. Das Original beginnt mit OPENQASM 3.0, verwendet aber
den Alias u3 statt U. Beim Import wird dieser Alias nur im Speicher auf die
mathematisch identische U-Operation normalisiert. Die gespeicherte Datei
und ihre Prüfsumme bleiben unverändert.

Alle Laufparameter, RNG-Seed, Optimiererstatus, akzeptierten/abgelehnten
Patches und Bibliotheksversionen werden gespeichert. Der Altcode wird nicht
umgeschrieben; bekannte Unterschiede (z.B. ungenutztes seed_init und
unterschiedliche Quimb-Isometrisierungs-APIs) stehen bewusst außerhalb der
neuen Implementierung.

