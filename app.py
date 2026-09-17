import os
import numpy as np
from flask import Flask, jsonify, request

from models import db, Device, Reading
from mqtt_listener import start_mqtt_listener
from signal_processing import compute_spectrum, get_harmonic_magnitudes

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


@app.route("/devices/<device_id>/readings/<int:reading_id>/spectrum")
def get_spectrum(device_id, reading_id):
    """
    FFT magnitude spectrum for a reading, capped to 0-2000Hz - covers the
    range where motor running-speed harmonics actually live, and keeps the
    response small rather than returning all ~8000 bins up to Nyquist.
    """
    reading = Reading.query.filter_by(id=reading_id, device_id=device_id).first()
    if reading is None or reading.raw_waveform is None:
        return jsonify({"error": "No raw waveform stored for this reading."}), 404
    samples = np.frombuffer(reading.raw_waveform, dtype='<f4')
    freqs, magnitude = compute_spectrum(samples, reading.sample_rate)

    max_freq = 2000.0
    cutoff_idx = int(np.searchsorted(freqs, max_freq))
    return jsonify({
        "freqs": freqs[:cutoff_idx].tolist(),
        "magnitudes": magnitude[:cutoff_idx].tolist(),
    })


@app.route("/devices/<device_id>/readings/<int:reading_id>/harmonics")
def get_harmonics(device_id, reading_id):
    """
    Magnitude at 1x/2x/3x/4x of a given base frequency (?base_freq=24.87),
    e.g. a tachometer-measured RPM converted to Hz - lets you check whether
    the true running speed and its harmonics line up with real spectral
    energy, same idea as the manual ground-truth check done earlier.
    """
    base_freq = request.args.get("base_freq", type=float)
    if base_freq is None or base_freq <= 0:
        return jsonify({"error": "Provide ?base_freq=<Hz>, a positive number."}), 400

    reading = Reading.query.filter_by(id=reading_id, device_id=device_id).first()
    if reading is None or reading.raw_waveform is None:
        return jsonify({"error": "No raw waveform stored for this reading."}), 404
    samples = np.frombuffer(reading.raw_waveform, dtype='<f4')
    freqs, magnitude = compute_spectrum(samples, reading.sample_rate)

    harmonics = get_harmonic_magnitudes(freqs, magnitude, base_freq, num_harmonics=4)
    return jsonify({"base_freq_hz": base_freq, "harmonics": harmonics})


@app.route("/devices/<device_id>/readings/<int:reading_id>/chart")
def get_waveform_chart(device_id, reading_id):
    """
    A full webpage showing: RMS velocity/crest factor/dominant frequency
    (straight from the database, no extra fetch), the raw time-domain
    waveform, the FFT magnitude spectrum, and an input box to check 1x-4x
    harmonics of any expected frequency against the real spectrum - all in
    one place so this doesn't need to be pulled together by hand each time.
    """
    reading = Reading.query.filter_by(id=reading_id, device_id=device_id).first()
    if reading is None:
        return "Reading not found.", 404

    rms = f"{reading.rms_velocity_mms:.4f}" if reading.rms_velocity_mms is not None else "n/a"
    crest = f"{reading.crest_factor:.3f}" if reading.crest_factor is not None else "n/a"
    dom_freq = f"{reading.dominant_freq_hz:.2f}" if reading.dominant_freq_hz is not None else "n/a"

    return f"""
<!DOCTYPE html>
<html>
<head>
  <title>Waveform - Reading {reading_id}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 24px; max-width: 1200px; }}
    #status {{ color: #666; }}
    .stats {{ display: flex; gap: 32px; margin: 12px 0 24px 0; }}
    .stat {{ background: #f3f4f6; padding: 10px 16px; border-radius: 6px; }}
    .stat-label {{ font-size: 12px; color: #666; }}
    .stat-value {{ font-size: 20px; font-weight: bold; }}
    .harmonic-panel {{ margin-top: 24px; padding: 16px; background: #f9fafb; border-radius: 8px; }}
    table {{ border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: right; }}
    th {{ background: #eee; }}
    input {{ padding: 6px; width: 120px; }}
    button {{ padding: 6px 16px; margin-left: 8px; }}
  </style>
</head>
<body>
  <h2>Reading {reading_id} - Raw Waveform</h2>

  <div class="stats">
    <div class="stat"><div class="stat-label">RMS Velocity</div><div class="stat-value">{rms} mm/s</div></div>
    <div class="stat"><div class="stat-label">Crest Factor</div><div class="stat-value">{crest}</div></div>
    <div class="stat"><div class="stat-label">Dominant Freq</div><div class="stat-value">{dom_freq} Hz</div></div>
  </div>

  <p id="status">Loading waveform...</p>
  <canvas id="waveformChart" width="1200" height="350"></canvas>

  <h3>Frequency Spectrum (0-2000 Hz)</h3>
  <p id="specStatus">Loading spectrum...</p>
  <canvas id="spectrumChart" width="1200" height="350"></canvas>

  <div class="harmonic-panel">
    <strong>Check harmonics against an expected frequency</strong><br>
    <label>Expected base frequency (Hz), e.g. RPM/60: </label>
    <input type="number" id="baseFreqInput" step="0.01" placeholder="e.g. 24.87">
    <button onclick="checkHarmonics()">Show 1x-4x Harmonics</button>
    <div id="harmonicResult"></div>
  </div>

  <script>
    let spectrumChartInstance = null;

    fetch('/devices/{device_id}/readings/{reading_id}/waveform')
      .then(res => res.json())
      .then(data => {{
        if (data.error) {{ document.getElementById('status').innerText = 'Error: ' + data.error; return; }}
        const samples = data.samples;
        const sampleRate = data.sample_rate;
        document.getElementById('status').innerText =
          samples.length + ' samples @ ' + sampleRate + ' Hz (' +
          (samples.length / sampleRate * 1000).toFixed(1) + ' ms window)';
        const timeLabels = samples.map((_, i) => (i / sampleRate * 1000).toFixed(2));
        new Chart(document.getElementById('waveformChart'), {{
          type: 'line',
          data: {{ labels: timeLabels, datasets: [{{
            label: 'Acceleration (g)', data: samples,
            borderColor: 'rgb(59, 130, 246)', borderWidth: 1, pointRadius: 0,
          }}] }},
          options: {{ animation: false, scales: {{
            x: {{ title: {{ display: true, text: 'Time (ms)' }}, ticks: {{ maxTicksLimit: 20 }} }},
            y: {{ title: {{ display: true, text: 'g' }} }}
          }} }}
        }});
      }})
      .catch(err => {{ document.getElementById('status').innerText = 'Fetch failed: ' + err; }});

    fetch('/devices/{device_id}/readings/{reading_id}/spectrum')
      .then(res => res.json())
      .then(data => {{
        if (data.error) {{ document.getElementById('specStatus').innerText = 'Error: ' + data.error; return; }}
        document.getElementById('specStatus').innerText = data.freqs.length + ' bins shown (0-2000 Hz)';
        const freqLabels = data.freqs.map(f => f.toFixed(1));
        spectrumChartInstance = new Chart(document.getElementById('spectrumChart'), {{
          type: 'line',
          data: {{ labels: freqLabels, datasets: [{{
            label: 'Magnitude', data: data.magnitudes,
            borderColor: 'rgb(234, 88, 12)', borderWidth: 1, pointRadius: 0,
          }}] }},
          options: {{ animation: false, scales: {{
            x: {{ title: {{ display: true, text: 'Frequency (Hz)' }}, ticks: {{ maxTicksLimit: 25 }} }},
            y: {{ title: {{ display: true, text: 'Magnitude' }} }}
          }} }}
        }});
      }})
      .catch(err => {{ document.getElementById('specStatus').innerText = 'Fetch failed: ' + err; }});

    function checkHarmonics() {{
      const baseFreq = document.getElementById('baseFreqInput').value;
      if (!baseFreq || baseFreq <= 0) {{ alert('Enter a positive frequency in Hz.'); return; }}
      fetch('/devices/{device_id}/readings/{reading_id}/harmonics?base_freq=' + baseFreq)
        .then(res => res.json())
        .then(data => {{
          if (data.error) {{ document.getElementById('harmonicResult').innerText = 'Error: ' + data.error; return; }}
          let html = '<table><tr><th>Harmonic</th><th>Target (Hz)</th><th>Nearest Bin (Hz)</th><th>Magnitude</th></tr>';
          data.harmonics.forEach(h => {{
            html += `<tr><td>${{h.harmonic}}x</td><td>${{h.target_freq_hz.toFixed(2)}}</td>` +
                    `<td>${{h.bin_freq_hz.toFixed(2)}}</td><td>${{h.magnitude.toFixed(2)}}</td></tr>`;
          }});
          html += '</table>';
          document.getElementById('harmonicResult').innerHTML = html;

          // Overlay harmonic markers on the spectrum chart as a second dataset
          if (spectrumChartInstance) {{
            const markerData = data.harmonics.map(h => ({{ x: h.bin_freq_hz.toFixed(1), y: h.magnitude }}));
            spectrumChartInstance.data.datasets = spectrumChartInstance.data.datasets.filter(d => d.label !== 'Harmonic markers');
            spectrumChartInstance.data.datasets.push({{
              label: 'Harmonic markers', type: 'scatter', data: markerData,
              backgroundColor: 'red', pointRadius: 6, pointStyle: 'star',
            }});
            spectrumChartInstance.update();
          }}
        }})
        .catch(err => {{ document.getElementById('harmonicResult').innerText = 'Fetch failed: ' + err; }});
    }}
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
