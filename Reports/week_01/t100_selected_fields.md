# T-100 Selected Fields for Week 2

| Field | Role |
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

**Planned grain after processing:** carrier × origin × destination × month.
