import struct
import threading
import time
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
ANOMALY_K = 3.0
MIN_HISTORY_FOR_BASELINE = 5
BASELINE_WINDOW = 50

# --- Chunk reassembly ---
# CHANGED: the ESP32 now splits each capture into several smaller MQTT
# messages (a single large streamed publish kept dying mid-transfer at a
# consistent ~5KB point, root-caused to a hard ceiling in the ESP32's
# network stack, not something fixable by retry-tuning alone - see
# firmware comments). Each message header is 16 bytes: capture_id(4),
# chunk_index(2), total_chunks(2), num_samples(4), sample_rate(4),
# followed by that chunk's raw float32 sample data.
#
# pending_captures holds partial captures in memory until all their
# chunks arrive: { capture_id: {"chunks": {index: bytes}, "total": N,
# "num_samples": X, "sample_rate": Y, "first_seen": timestamp} }
pending_captures = {}
CAPTURE_TIMEOUT_SECONDS = 120  # drop incomplete captures older than this


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
            print(f"[MQTT] Connected. Requesting subscription to '{MQTT_TOPIC}'...")
            result, mid = client.subscribe(MQTT_TOPIC)
            print(f"[MQTT] subscribe() call returned result={result}, mid={mid}")
        else:
            print(f"[MQTT] Connection failed, rc={rc}")

    def on_subscribe(client, userdata, mid, reason_codes, properties=None):
        print(f"[MQTT] Broker acknowledged subscription (mid={mid}): reason_codes={reason_codes}")

    def on_message(client, userdata, msg):
        with app.app_context():
            try:
                handle_chunk(msg.payload, db, Device, Reading)
            except Exception as e:
                # Never let a bad/malformed message kill the listener thread
                print(f"[MQTT] Error handling message: {e}")

    def on_log(client, userdata, level, buf):
        print(f"[MQTT LOG] {buf}")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.tls_set()
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.on_log = on_log

    def run():
        while True:
            try:
                client.connect(MQTT_SERVER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                print(f"[MQTT] Connection error: {e}. Retrying in 10s...")
                time.sleep(10)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print("[MQTT] Listener thread started.")


def cleanup_stale_captures():
    now = time.time()
    stale = [cid for cid, c in pending_captures.items()
             if now - c["first_seen"] > CAPTURE_TIMEOUT_SECONDS]
    for cid in stale:
        c = pending_captures.pop(cid)
        print(f"[MQTT] Dropping incomplete capture {cid}: "
              f"only got {len(c['chunks'])}/{c['total']} chunks before timeout.")


def handle_chunk(payload: bytes, db, Device, Reading):
    if len(payload) < 16:
        print(f"[MQTT] Chunk too short ({len(payload)} bytes), ignoring.")
        return

    capture_id, chunk_index, total_chunks, num_samples, sample_rate = struct.unpack('<IHHII', payload[:16])
    chunk_data = payload[16:]

    if capture_id not in pending_captures:
        pending_captures[capture_id] = {
            "chunks": {},
            "total": total_chunks,
            "num_samples": num_samples,
            "sample_rate": sample_rate,
            "first_seen": time.time(),
        }

    capture = pending_captures[capture_id]
    capture["chunks"][chunk_index] = chunk_data

    print(f"[MQTT] Capture {capture_id}: chunk {chunk_index + 1}/{total_chunks} received "
          f"({len(capture['chunks'])}/{total_chunks} total so far)")

    if len(capture["chunks"]) == capture["total"]:
        # All chunks in - reassemble in correct order and process.
        ordered = [capture["chunks"][i] for i in range(capture["total"])]
        full_payload = b"".join(ordered)

        expected_bytes = capture["num_samples"] * 4
        if len(full_payload) != expected_bytes:
            print(f"[MQTT] Capture {capture_id} reassembled size mismatch: "
                  f"expected {expected_bytes}, got {len(full_payload)}. Discarding.")
            del pending_captures[capture_id]
            return

        samples = np.frombuffer(full_payload, dtype='<f4')
        process_and_store(samples, capture["sample_rate"], full_payload, db, Device, Reading)
        del pending_captures[capture_id]

    cleanup_stale_captures()


def process_and_store(samples: np.ndarray, sample_rate: int, raw_payload: bytes, db, Device, Reading):
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

        # Per-band spectral fingerprint check: compares each frequency
        # band's energy against THIS device's own historical mean/stddev
        # for that band - not a fixed threshold. This is what makes
        # detection work regardless of which harmonic happens to dominate
        # a given machine/mounting (confirmed directly on this project's
        # own test motor: dominant energy sits at the 4th harmonic of
        # running speed, not the 1st - a fixed-frequency rule would miss
        # this entirely, but a per-band baseline doesn't care which band
        # is naturally loud, only whether THIS machine's own bands change).
        band_dev = False
        history_with_bands = [r for r in history if r.band_energies is not None]
        if len(history_with_bands) >= MIN_HISTORY_FOR_BASELINE and features.get("band_energies"):
            hist_bands = np.array([r.band_energies for r in history_with_bands])
            band_mean = hist_bands.mean(axis=0)
            band_std = hist_bands.std(axis=0)
            current_bands = np.array(features["band_energies"])
            deviations = np.abs(current_bands - band_mean) > ANOMALY_K * np.maximum(band_std, 1e-6)
            band_dev = bool(np.any(deviations))
            if band_dev:
                flagged = np.where(deviations)[0]
                band_ranges = [f"{i*100}-{(i+1)*100}Hz" for i in flagged]
                print(f"[ANOMALY] Band-energy deviation in: {', '.join(band_ranges)}")

        is_anomaly = bool(rms_dev or crest_dev or band_dev)
    else:
        print(f"[ANOMALY] Only {len(history)}/{MIN_HISTORY_FOR_BASELINE} readings so far - still baselining.")

    reading = Reading(
        device_id=DEVICE_ID,
        timestamp=datetime.utcnow(),
        sample_rate=sample_rate,
        num_samples=len(samples),
        rms_velocity_mms=features["rms_velocity_mms"],
        peak_to_peak_velocity_mms=features["peak_to_peak_velocity_mms"],
        absolute_peak_velocity_mms=features["absolute_peak_velocity_mms"],
        crest_factor=features["crest_factor"],
        dominant_freq_hz=features["dominant_freq_hz"],
        dominant_freq_mag=features["dominant_freq_mag"],
        band_energies=features.get("band_energies"),
        is_anomaly=is_anomaly,
        # CHANGED (temporary, for validation phase): was `raw_payload if
        # is_anomaly else None` - only kept raw data for flagged anomalies,
        # to avoid unbounded DB growth in production. Right now every
        # reading needs its raw waveform available for inspection/FFT
        # work, so this saves it unconditionally. Revert to the
        # is_anomaly-gated version before real deployment - a live fleet
        # of devices uploading every 15 minutes would otherwise grow the
        # database indefinitely.
        raw_waveform=raw_payload,
    )
    db.session.add(reading)
    db.session.commit()

    print(f"[MQTT] Stored reading: RMS={features['rms_velocity_mms']:.4f}mm/s "
          f"Crest={features['crest_factor']:.2f} Anomaly={is_anomaly}")
