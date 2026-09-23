# lossaver

`lossaver` ist eine kleine Sammlung eigenständiger Python-Skripte zum
Bereinigen, Modellieren und Analysieren von Mixed-Gamble-Loss-Aversion-
Daten. Es ist **kein** Python-Paket — es gibt nichts zu installieren außer
ein paar Bibliotheken, und jedes Skript wird direkt mit
`python scripts/0N_name.py [Optionen]` ausgeführt.

Die vollständige Pipeline-Beschreibung (Datenfluss, Datenformate in jedem
Schritt, warum sie so aufgebaut ist, und genaue Ausführungsanweisungen für
jedes Skript) steht in [PIPELINE.md](PIPELINE.md).

## Einrichtung

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` installiert die Laufzeitabhängigkeiten (`pandas`,
`numpy`, `matplotlib`, `scikit-learn`) sowie `pytest` zum Ausführen der
Tests. Wenn nur die Skripte ausgeführt werden sollen (ohne Tests), genügt
`pip install -r requirements.txt`.

## Schnellstart

Führe die vier Skripte in [scripts/](scripts/) der Reihe nach vom
Repository-Root aus aus:

```bash
python scripts/01_clean_and_sort.py --input-dir raw_data --out-dir processed_data
python scripts/02_logistic_regression.py --input-dir processed_data --summary-output logistic_regression_summary.csv --plots-dir plots
python scripts/03_participant_plots.py --input-dir processed_data --plots-dir plots
python scripts/04_merge_questionnaires.py --summary logistic_regression_summary.csv --quest-dir processed_data --output participant_summary.csv
```

Siehe [PIPELINE.md](PIPELINE.md) für Details, was jeder Schritt liest/schreibt
und warum, sowie alle optionalen Parameter.

## Die Plots der logistischen Regression verstehen (Schritt 2)

Jeder Plot `plots/<id>_logistic_regression.png` zeigt für einen Teilnehmer
die geschätzte logistische Annahmefunktion basierend auf `gain` und `loss`:

- x-Achse: der Gewinnwert (`gain`)
- y-Achse: die geschätzte Annahmewahrscheinlichkeit `P(accept)` zwischen 0 und 1
- blaue Kurve: die von der logistischen Regression geschätzte Annahmefunktion
- orange Punkte: die beobachteten Antworten des Teilnehmers (`0` = Ablehnung, `1` = Annahme)
- graue gestrichelte Linie bei `0.5`: die Schwelle, bei der das Modell Annahme
  und Ablehnung als gleich wahrscheinlich einschätzt

Eine steil ansteigende Kurve bedeutet, dass sich die Annahmewahrscheinlichkeit
mit steigendem Gewinn stark verändert. Eine flachere Kurve bedeutet, dass
Gewinne die Entscheidung nur schwach beeinflussen. Steigt die Kurve erst bei
hohen Gewinnen an, spricht das für stärkere Zurückhaltung gegenüber riskanten
Angeboten.

Die Koeffizienten lesen (`plots/<id>_logistic_regression_coefficients.csv`):

- `beta_gain`: wie stark mögliche Gewinne mit der Annahmeentscheidung
  zusammenhängen (bei konstantem `loss`)
- `beta_loss`: wie stark mögliche Verluste mit der Annahmeentscheidung
  zusammenhängen (bei konstantem `gain`)
- `beta_bias` (der Modell-Intercept): Grundtendenz zur Annahme oder Ablehnung,
  unabhängig von `gain` und `loss`

Vorzeichen-Interpretation:

- `beta_gain > 0`: höhere Gewinne gehen mit höherer Annahmewahrscheinlichkeit
  einher (erwartet)
- `beta_gain < 0`: höhere Gewinne gehen in diesen Daten mit *geringerer*
  Annahmewahrscheinlichkeit einher
- `beta_loss < 0`: höhere Verluste gehen mit geringerer
  Annahmewahrscheinlichkeit einher (erwartetes/typisches Muster)

Nicht-positive `beta_gain`-/`beta_loss`-Werte werden nicht herausgefiltert —
sie bleiben unverändert in `participant_summary.csv` erhalten. Sie können auf
Rauschen, geringe Antwortkonsistenz, ungewöhnliche Entscheidungsstrategien
oder echte Datenprobleme hindeuten und sind es wert, einzeln geprüft zu
werden.

Der Loss-Aversion-Parameter ist:

$$
\lambda = \frac{|\beta_{loss}|}{\beta_{gain}}
$$

- `lambda > 1`: Verluste werden stärker gewichtet als Gewinne (Loss Aversion)
- `lambda = 1`: Gewinne und Verluste werden ungefähr gleich stark gewichtet
- `lambda < 1`: Gewinne werden stärker gewichtet als Verluste

## Entwicklung

Tests ausführen:

```bash
python -m pytest -q
```

Die Tests liegen in [tests/](tests/) und importieren jedes nummerierte
Skript direkt aus [scripts/](scripts/) über den Helfer
[tests/script_loader.py](tests/script_loader.py) (da Dateinamen wie
`02_logistic_regression.py` keine gültigen Python-Modulnamen für eine
normale `import`-Anweisung sind).
