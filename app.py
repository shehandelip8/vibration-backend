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
