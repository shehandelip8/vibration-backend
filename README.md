# Vibration Monitoring Backend

Receives raw acceleration data from the ESP32 over HiveMQ, runs the
validated signal processing (detrend, integrate, FFT), stores results in
Postgres, and exposes a REST API for the dashboard frontend.

## IMPORTANT: single-worker deployment required

This service runs an MQTT listener in a background thread started at
import time. If deployed with multiple gunicorn workers, EACH worker
would open its own MQTT subscription, causing every message to be
processed and stored multiple times. The build/start commands below
are deliberately set to a single worker for this reason. If you later
need more HTTP capacity, split the MQTT listener into a separate Render
Background Worker service instead of raising the worker count here.

## Deploying to Render

1. Push this folder to a new GitHub repository.
2. On Render: create a PostgreSQL database first, and note its
   "Internal Database URL".
3. Create a new Web Service pointing at your repo, with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn --workers 1 app:app`
   - Environment variable: `DATABASE_URL` = the Postgres Internal
     Database URL from step 2.

## API endpoints

- `GET /` - health check
- `GET /devices/<device_id>/readings?limit=100` - recent readings
- `GET /devices/<device_id>/readings/<id>/waveform` - raw waveform
  (only available for readings flagged as anomalous)
- `GET /devices/<device_id>/status` - latest reading

## Local testing

```
pip install -r requirements.txt
export DATABASE_URL=postgresql://user:pass@localhost:5432/vibration
python app.py
```
