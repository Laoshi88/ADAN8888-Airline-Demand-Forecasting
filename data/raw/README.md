# Raw Data — BTS T-100 Domestic Segment (All Carriers)

Download the raw extract from the official BTS TranStats T-100 Domestic Segment (All Carriers) table and place the untouched file in this folder.

Recommended initial scope for Week 2:
- Date range: January 2015 through the latest uniform month available at download time.
- Core service: filter `Class == F` (Scheduled Passenger/Cargo Service) during processing.
- Keep these fields at minimum: `Year`, `Quarter`, `Month`, `UniqueCarrier`, `AirlineID`, `UniqueCarrierName`, `OriginAirportID`, `Origin`, `OriginCityName`, `OriginState`, `DestAirportID`, `Dest`, `DestCityName`, `DestState`, `AircraftType`, `AircraftConfig`, `Class`, `Passengers`, `Seats`, `DepScheduled`, `DepPerformed`, `Distance`, `LoadFactor`.

Do **not** edit the raw downloaded file. Save cleaned/aggregated versions under `data/processed/`.

Source table: https://www.transtats.bts.gov/Tables.asp?QO_VQ=EEE
