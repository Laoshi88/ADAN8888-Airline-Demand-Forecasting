# Week 1 Supporting Reference: T-100 Selected Fields

These Fields Are Retained For The Week 2 Ingestion/Exploration Stage Of The Project.
The Week 1 Report Is For Defining Th Problem & The Dataset.
Week 2 Will Validate The Downloaded Schema And Begin Formal Data Exploration.

| Field | Planned role |
|---|---|
| Passengers | Outcome to shift forward one month |
| Seats | Capacity history / lagged predictor |
| DepScheduled | Scheduled frequency history |
| DepPerformed | Realized frequency history |
| Distance | Route feature |
| UniqueCarrier / AirlineID | Carrier identifiers |
| OriginAirportID / Origin | Origin identifiers |
| DestAirportID / Dest | Destination identifiers |
| Year / Quarter / Month | Time and seasonality |
| Class | Filter core analysis to F (scheduled passenger/cargo) |
| AircraftType / AircraftConfig | Aggregate aircraft mix; optional features |
| LoadFactor | Diagnostic; can also be recomputed |

**Planned analytical grain after processing:** carrier × origin × destination × month.

**Leakage guardrail:** realized month `t+1` seats or departures are not used to predict month `t+1` passengers unless a genuinely forward-looking schedule source is later added.
