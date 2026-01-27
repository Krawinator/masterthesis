# Bestimmung netzkonformer Leistungsbänder für Batteriespeicher

## Überblick

Diese Arbeit entwickelt einen datengetriebenen Workflow zur **Ableitung netzkonformer Leistungsbänder** für Batteriespeicher in einem Hochspannungsnetz.

Ziel ist es, für jeden Zeitschritt ein zulässiges **Einspeise- bzw. Aufnahmeband** eines Batteriespeichers zu bestimmen, sodass:
- thermische Leitungsgrenzen eingehalten werden,
- reale Last- und Wetterdaten berücksichtigt werden,
- und der Ansatz reproduzierbar auf andere Netze übertragbar ist.

Die Lösung kombiniert:
- Zeitreihenaufbereitung,
- Lastprognosen,
- linearisierte Netzmodellierung (DC-Lastfluss, PTDF),
- und eine bandbasierte Nebenbedingungenanalyse.

---

## Projektstruktur (konzeptionell)

Das Projekt ist modular aufgebaut und trennt explizit zwischen:

- **Netzdefinition** (Graph)
- **Datenpipeline** (Messdaten, Wetterdaten, Prognosen)
- **Modelllogik** (Forecast, Lastfluss, Leistungsbänder)
- **Exploration / Showcase** (Notebooks)

Diese Trennung stellt sicher, dass:
- Änderungen am Netz ohne Codeänderungen möglich sind,
- Modelle austauschbar bleiben,
- und die Methodik nachvollziehbar dokumentiert ist.

---

## Netzmodellierung (Graph Builder)

### Motivation

Das betrachtete Netz wird explizit als **Graph** modelliert, um:
- Netzstruktur klar von der Rechenlogik zu trennen,
- reale Netztopologien abbilden zu können,
- und Änderungen reproduzierbar zu halten.

### Tool

Der Graph wird mit folgendem Tool erstellt:

src/raw/graph/graph_builder.py


### Knotentypen

Folgende Knotentypen werden unterstützt:

- **UW-Feld (`uw_field`)**  
  Netzknoten mit gemessener Wirkleistung

- **Batteriespeicher (`battery`)**  
  Speicher mit begrenzter maximaler Leistung

- **Sammelschiene (`busbar`)**  
  Elektrische Sammelpunkte

- **Leitungsknoten (`junction`)**  
  Reine Verbindungsknoten ohne eigene Messung

### Leitungen (Edges)

Knoten werden über Leitungen verbunden. Für jede Leitung müssen angegeben werden:
- `Strom_Limit_in_A` – thermisches Stromlimit
- `X_total_ohm` – Reaktanz der Leitung in Ohm

Zusätzlich gilt:
- jede Leitung benötigt eine **eindeutige ID**
- Start- und Zielknoten müssen definiert sein

### UW-Felder: Messdaten & Geokoordinaten

Für UW-Felder sind zusätzlich erforderlich:
- `P_Datapoint_ID` aus der e.IoT Console
- `Latitude_deg` und `Longitude_deg`

Diese Angaben werden verwendet für:
- den Abruf historischer Wirkleistungsdaten
- den Abruf von Wetterdaten über Geokoordinaten

### Derived Nodes

Für Knoten ohne direkten Messwert können sogenannte **derived nodes** definiert werden.

Dabei wird die Leistung aus anderen Knoten abgeleitet (z. B. Summe mehrerer Felder mit Koeffizienten).  
Die genaue Struktur ist in der Beispiel-Datei `whole_graph.json` dokumentiert.

### Batteriespeicher

Batteriespeicher besitzen zusätzlich:
- einen maximalen Leistungswert (`p_max_MW`)

Dieser Wert begrenzt später die berechneten Leistungsbänder.

### Import / Export

Im unteren Menü des Graph Builders können Graphen:
- importiert
- exportiert

werden.

### Ablage

Der finale Graph muss unter folgendem Pfad abgelegt werden:

src/data/raw/graph/whole_graph.json


---

## Datenpipeline

### Messdaten

- Historische Wirkleistungsdaten werden aus der e.IoT API geladen.
- Zeitauflösung: **15 Minuten**.
- Speicherung unter:
src/data/raw/timeseries/


### Wetterdaten

- Historische Wetterdaten und Wetterprognosen werden anhand der Knotenkoordinaten geladen.
- Wetterprognosen werden separat abgelegt unter:
src/data/raw/weather_forecast/


### Datenbereinigung

- Vereinheitlichung des Zeitrasters
- Behandlung kleiner Datenlücken
- Entfernung des Batteriespeicher-Einflusses aus historischen Zeitreihen

Ergebnis:
src/data/clean/timeseries_no_bess/


---

## Prognose

Für jedes relevante UW-Feld wird eine Lastprognose erzeugt.

Eigenschaften:
- Nutzung eines zuvor bestimmten Siegermodells
- Prognose ausschließlich auf Basis:
  - bereinigter historischer Zeitreihen
  - exogener Wetterprognosen
- Postprocessing der Prognose (Flip, Shift, Clipping), um
  konsistente Übergänge zur Historie sicherzustellen

Ausgabe:
src/data/pred_normalized/<node_id>_pred.csv


---

## Leistungsbandberechnung

Die Leistungsbandberechnung basiert auf:

- linearisierter DC-Lastflussrechnung
- komponentenweiser Slack-Node-Definition
- PTDF-Matrizen je Netzkomponente

Für jeden Zeitschritt wird berechnet:
- welche Einspeise- bzw. Aufnahmeleistung eines Batteriespeichers zulässig ist,
- sodass keine Leitung eine definierte Zielauslastung überschreitet.

Ausgabe:
- eine CSV pro Batteriespeicher mit zulässigem Leistungsband

---

## Showcase / Notebooks

Zur Veranschaulichung sind Notebooks enthalten, die exemplarisch zeigen:

1. Laden der vorbereiteten Daten  
2. Entfernen des Batteriespeicher-Einflusses  
3. Erzeugen einer Lastprognose  
4. Berechnung der Leistungsbänder  
5. Visualisierung eines resultierenden Leistungsbands  

Die vollständige Rechenlogik ist dabei bewusst **in Modulen gekapselt**, um Redundanz zu vermeiden.

---
## Authentisierung & Zugriff auf externe Datenquellen

Der Zugriff auf Messdaten aus der e.IoT-Plattform erfolgt über eine gesicherte REST-API.
Die Authentisierung basiert auf einem OAuth2-Client-Credentials-Flow.

Die hierfür benötigten Zugangsdaten werden nicht im Code hinterlegt, sondern über
Umgebungsvariablen (z. B. via `.env`-Datei) bereitgestellt.

Verwendete Umgebungsvariablen:

- `ARM_TENANT_ID` – Azure AD Tenant
- `ARM_CLIENT_ID` – Client-ID der registrierten Applikation
- `ARM_CLIENT_SECRET` – Client-Secret
- `SCOPE` – API-Scope für den Zugriff auf die e.IoT-Schnittstelle
- `SECRET` – internes Secret für Token-Handling

---

## Forecasting / Modelltraining `forecasting_experiments.ipynb`

Ergänzend zur produktiven Prognosepipeline existiert ein **Forecasting-Notebook** zur explorativen Modellierung.

In diesem Notebook können:
- verschiedene Prognosemodelle trainiert,
- alternative Feature-Sets und Hyperparameter getestet,
- und potenzielle **Siegermodelle** identifiziert werden.

Das Notebook dient ausschließlich der **Modellselektion und Bewertung**.  
Das final gewählte Modell wird anschließend in die modulare Forecasting-Pipeline integriert und dort reproduzierbar eingesetzt.

---

## How to run

Der vollständige End-to-End-Workflow wird über einen zentralen Einstiegspunkt ausgeführt:

```bash
python -m src.main
```

