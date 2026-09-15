from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Device(db.Model):
    __tablename__ = "devices"
    id = db.Column(db.String, primary_key=True)  # e.g. "sensor1" - matches the MQTT topic for now
    name = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Reading(db.Model):
    __tablename__ = "readings"
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String, db.ForeignKey("devices.id"), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    sample_rate = db.Column(db.Float)
    num_samples = db.Column(db.Integer)

    rms_velocity_mms = db.Column(db.Float)
    peak_to_peak_velocity_mms = db.Column(db.Float)
    absolute_peak_velocity_mms = db.Column(db.Float)
    crest_factor = db.Column(db.Float)
    dominant_freq_hz = db.Column(db.Float)
    dominant_freq_mag = db.Column(db.Float)

    is_anomaly = db.Column(db.Boolean, default=False)

    # Only populated when is_anomaly=True - storing the full ~64KB raw
    # waveform for EVERY reading (every 15 min) would grow the database
    # unboundedly for no real benefit. This is the server-side equivalent
    # of the original "only upload raw data on anomaly" design, just
    # applied at the storage layer now that every capture gets uploaded.
    raw_waveform = db.Column(db.LargeBinary, nullable=True)
