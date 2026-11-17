# VoltPoint Firmware Release Notes

*Fictional sample data for the RAG demo.*

## Version 2.4.0 (2026-09-15)

- New: dynamic load balancing across up to 4 chargers on the same connection.
- New: ISO 15118 Plug & Charge (beta) for supported vehicles.
- Fixed: false E04 communication errors with some vehicle models.
- Known issue: in solar surplus mode, charging can pause for up to 2 minutes during cloudy intervals.

## Version 2.3.2 (2026-06-02)

- Security: fixed TLS certificate validation for the OCPP websocket connection.
- Improved: faster Wi-Fi reconnection after router restarts.

## Version 2.3.0 (2026-04-10)

- New: OCPP 2.0.1 support for connection to charge-point management systems.
- New: scheduled charging configurable in the app.
- Removed: legacy OCPP 1.5 support.

## How updates are installed

Firmware updates are installed automatically over the air at night between 02:00 and 04:00 local time. Owners can also start an update manually in the app under Settings > Firmware. An update takes about 10 minutes; do not switch off the power during an update.
