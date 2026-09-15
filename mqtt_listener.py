import struct
import threading
from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt

from signal_processing import process_waveform

# --- HiveMQ Configuration - must match your ESP32 sketch exactly ---
MQTT_SERVER = "f46c830c84ad4653a77358ac1ae7778b.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "IoT_Project"
MQTT_PASS = "Shehan@1234"
MQTT_TOPIC = "vibration/sensor1/burst"
DEVICE_ID = "sensor1"  # TODO: derive from topic once you have multiple devices/topics

# Rolling-baseline anomaly detection: flag a reading if it deviates more
# than ANOMALY_K standard deviations from the device's own recent history.
# This is the mean/stddev approach discussed earlier - self-scales to each
# device's normal noise level instead of using one fixed threshold for
# every installation. Needs a minimum amount of history before it
# activates, same reasoning as the "run 4-5 baseline captures at install"
# plan - here it just accumulates that history automatically over the
# device's first several readings instead of a separate manual step.
ANOMALY_K = 3.0
MIN_HISTORY_FOR_BASELINE = 5
BASELINE_WINDOW = 50


def start_mqtt_listener(app, db, Device, Reading):
    """Starts the MQTT listener in a background thread. Call once at app startup.

    IMPORTANT: this must only run in a SINGLE process. If deployed behind
    gunicorn with multiple workers, each worker would open its own
    subscription and the same message would be processed (and stored)
    multiple times. The deploy command for this service is deliberately
    set to `gunicorn --workers 1 app:app` for this reason - see README.
    """

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[MQTT] Connected, subscribing to {MQTT_TOPIC}")
            client.subscribe(MQTT_TOPIC)
        else:
            print(f"[MQTT] Connection failed, rc={rc}")

    def on_message(client, userdata, msg):
        with app.app_context():
            try:
                handle_message(msg.payload, db, Device, Reading)
            except Exception as e:
                # Never let a bad/malformed message kill the listener thread
                print(f"[MQTT] Error handling message: {e}")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.tls_set()
    client.on_connect = on_connect
    client.on_message = on_message

    def run():
        while True:
            try:
                client.connect(MQTT_SERVER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                print(f"[MQTT] Connection error: {e}. Retrying in 10s...")
                import time
                time.sleep(10)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print("[MQTT] Listener thread started.")


def handle_message(payload: bytes, db, Device, Reading):
    if len(payload) < 8:
        print("[MQTT] Payload too short, ignoring.")
        return

    sample_count, sample_rate = struct.unpack('<II', payload[:8])
    expected_bytes = 8 + (sample_count * 4)
    if len(payload) != expected_bytes:
        print(f"[MQTT] Payload size mismatch: expected {expected_bytes}, got {len(payload)}. Discarding.")
        return

    samples = np.frombuffer(payload[8:], dtype='<f4')
    features = process_waveform(samples, sample_rate)

    if Device.query.get(DEVICE_ID) is None:
        db.session.add(Device(id=DEVICE_ID, name=DEVICE_ID))
        db.session.commit()

    history = (Reading.query
               .filter_by(device_id=DEVICE_ID)
               .order_by(Reading.timestamp.desc())
               .limit(BASELINE_WINDOW)
               .all())

    is_anomaly = False
    if len(history) >= MIN_HISTORY_FOR_BASELINE:
        hist_rms = np.array([r.rms_velocity_mms for r in history])
        hist_crest = np.array([r.crest_factor for r in history])
        rms_mean, rms_std = hist_rms.mean(), hist_rms.std()
        crest_mean, crest_std = hist_crest.mean(), hist_crest.std()

        rms_dev = abs(features["rms_velocity_mms"] - rms_mean) > ANOMALY_K * max(rms_std, 1e-6)
        crest_dev = abs(features["crest_factor"] - crest_mean) > ANOMALY_K * max(crest_std, 1e-6)
        is_anomaly = bool(rms_dev or crest_dev)
    else:
        print(f"[ANOMALY] Only {len(history)}/{MIN_HISTORY_FOR_BASELINE} readings so far - still baselining.")

    reading = Reading(
        device_id=DEVICE_ID,
        timestamp=datetime.utcnow(),
        sample_rate=sample_rate,
        num_samples=sample_count,
        rms_velocity_mms=features["rms_velocity_mms"],
        peak_to_peak_velocity_mms=features["peak_to_peak_velocity_mms"],
        absolute_peak_velocity_mms=features["absolute_peak_velocity_mms"],
        crest_factor=features["crest_factor"],
        dominant_freq_hz=features["dominant_freq_hz"],
        dominant_freq_mag=features["dominant_freq_mag"],
        is_anomaly=is_anomaly,
        raw_waveform=payload[8:] if is_anomaly else None,
    )
    db.session.add(reading)
    db.session.commit()

    print(f"[MQTT] Stored reading: RMS={features['rms_velocity_mms']:.4f}mm/s "
          f"Crest={features['crest_factor']:.2f} Anomaly={is_anomaly}")
