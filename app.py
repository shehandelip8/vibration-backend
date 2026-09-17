import os
import numpy as np
from flask import Flask, jsonify, request

from models import db, Device, Reading
from mqtt_listener import start_mqtt_listener

app = Flask(__name__)

# Render's Postgres connection string starts with "postgres://", but
# SQLAlchemy needs "postgresql://" - this handles that automatically.
database_url = os.environ["DATABASE_URL"].replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()

start_mqtt_listener(app, db, Device, Reading)


@app.route("/")
def health():
    return jsonify({"status": "ok"})


@app.route("/devices/<device_id>/readings")
def get_readings(device_id):
    """Recent readings for charting - lightweight, no raw waveform included."""
    limit = int(request.args.get("limit", 100))
    readings = (Reading.query
                .filter_by(device_id=device_id)
                .order_by(Reading.timestamp.desc())
                .limit(limit)
                .all())
    return jsonify([{
        "id": r.id,
        "timestamp": r.timestamp.isoformat(),
        "rms_velocity_mms": r.rms_velocity_mms,
        "peak_to_peak_velocity_mms": r.peak_to_peak_velocity_mms,
        "crest_factor": r.crest_factor,
        "dominant_freq_hz": r.dominant_freq_hz,
        "is_anomaly": r.is_anomaly,
    } for r in readings])


@app.route("/devices/<device_id>/readings/<int:reading_id>/waveform")
def get_waveform(device_id, reading_id):
    """Raw waveform for a specific reading - only available if it was flagged anomalous."""
    reading = Reading.query.filter_by(id=reading_id, device_id=device_id).first()
    if reading is None or reading.raw_waveform is None:
        return jsonify({"error": "No raw waveform stored for this reading (only kept for anomalies)."}), 404
    samples = np.frombuffer(reading.raw_waveform, dtype='<f4')
    return jsonify({"sample_rate": reading.sample_rate, "samples": samples.tolist()})


@app.route("/devices/<device_id>/readings/<int:reading_id>/chart")
def get_waveform_chart(device_id, reading_id):
    """
    A full webpage showing the raw waveform as a chart, all 16384 samples.
    No new Render service, no background worker, no separate hosting -
    this route lives in the same Flask app as everything else, and the
    page's JavaScript fetches /waveform (above) from the SAME origin, so
    no CORS setup is needed either. This is a quick way to see the chart
    yourself; your frontend developer's real dashboard will eventually
    replace this with a nicer version, but this is fully functional now.
    """
    return f"""
<!DOCTYPE html>
<html>
<head>
  <title>Waveform - Reading {reading_id}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    #status {{ color: #666; }}
  </style>
</head>
<body>
  <h2>Reading {reading_id} - Raw Waveform</h2>
  <p id="status">Loading...</p>
  <canvas id="waveformChart" width="1200" height="400"></canvas>

  <script>
    fetch('/devices/{device_id}/readings/{reading_id}/waveform')
      .then(res => res.json())
      .then(data => {{
        if (data.error) {{
          document.getElementById('status').innerText = 'Error: ' + data.error;
          return;
        }}
        const samples = data.samples;
        const sampleRate = data.sample_rate;
        document.getElementById('status').innerText =
          samples.length + ' samples @ ' + sampleRate + ' Hz (' +
          (samples.length / sampleRate * 1000).toFixed(1) + ' ms window)';

        // Time axis in milliseconds, one label per sample
        const timeLabels = samples.map((_, i) => (i / sampleRate * 1000).toFixed(2));

        new Chart(document.getElementById('waveformChart'), {{
          type: 'line',
          data: {{
            labels: timeLabels,
            datasets: [{{
              label: 'Acceleration (g)',
              data: samples,
              borderColor: 'rgb(59, 130, 246)',
              borderWidth: 1,
              pointRadius: 0,
            }}]
          }},
          options: {{
            animation: false,
            scales: {{
              x: {{ title: {{ display: true, text: 'Time (ms)' }}, ticks: {{ maxTicksLimit: 20 }} }},
              y: {{ title: {{ display: true, text: 'g' }} }}
            }}
          }}
        }});
      }})
      .catch(err => {{
        document.getElementById('status').innerText = 'Fetch failed: ' + err;
      }});
  </script>
</body>
</html>
"""


@app.route("/devices/<device_id>/status")
def get_status(device_id):
    """Latest reading - a quick 'is this machine OK right now' endpoint."""
    latest = (Reading.query
              .filter_by(device_id=device_id)
              .order_by(Reading.timestamp.desc())
              .first())
    if latest is None:
        return jsonify({"error": "No readings yet"}), 404
    return jsonify({
        "timestamp": latest.timestamp.isoformat(),
        "rms_velocity_mms": latest.rms_velocity_mms,
        "crest_factor": latest.crest_factor,
        "dominant_freq_hz": latest.dominant_freq_hz,
        "is_anomaly": latest.is_anomaly,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
