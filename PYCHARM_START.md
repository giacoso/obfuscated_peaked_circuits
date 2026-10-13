# Start in PyCharm

Das eigentliche Python-Projekt liegt im Unterordner `Code für die Bachelorarbeit`.
PyCharm verwendet dafür Python 3.12 aus `Code für die Bachelorarbeit/.venv`.
Das Paket wurde am aktuellen Speicherort mit `pip install -e ".[test]"` installiert.
Eine Aktivierung der venv im Terminal ist für den grünen Startknopf nicht nötig.

## Programm starten

Oben in PyCharm die Startkonfiguration **HQAP Demo** auswählen und auf den grünen
Startknopf klicken (alternativ Shift+F10). Der Lauf verwendet `configs/demo.json`.
Neue Ergebnisse und Grafiken stehen in `results/pycharm_demo` im inneren Projektordner.
Die bisherigen Ergebnisse in `results/demo` bleiben erhalten.

Weitere Konfigurationen:
- **HQAP Schritt fuer Schritt**: führt `examples/schritt_fuer_schritt.py` als Modul aus;
  Ergebnisse unter `results/walkthrough`.
- **HQAP Tests**: führt die vorhandenen Tests mit `pytest -q` aus.

## Git und GitHub

Das Git-Repository liegt im eigentlichen Projektordner `Code für die Bachelorarbeit`.
Der Projektcode wird im Repository `https://github.com/giacoso/obfuscated_peaked_circuits`
versioniert. Die lokale Python-Umgebung, Cache-Dateien und lokale Editor-Einstellungen
sind über `.gitignore` ausgeschlossen.

Spätere Änderungen müssen erneut committed und gepusht werden, damit sie auf GitHub stehen.
Rote Dateinamen im Projektbaum zeigen nicht versionierte Dateien an und bedeuten
keinen Python-Fehler. Rote Unterstreichungen im Editor sind gesonderte Hinweise.

## Terminal (PowerShell)

Vom äußeren Ordner zuerst in das eigentliche Projekt wechseln:

```powershell
cd '.\Code für die Bachelorarbeit'
.\.venv\Scripts\python.exe -m hqap run --config configs/demo.json --output results/pycharm_demo
.\.venv\Scripts\python.exe -m pytest -q
```

Die Skripte unter `schon_bestehende_codes_für_BA` sind ältere, eigenständige
Referenzprogramme mit eigener `requirements.txt`; die Startkonfigurationen oben
beziehen sich auf das aktuelle HQAP-Paket.