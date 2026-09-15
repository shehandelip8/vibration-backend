import numpy as np

G_TO_MMS2 = 9806.65  # 1g = 9806.65 mm/s^2 - same constant used in the ESP32 firmware


def process_waveform(samples: np.ndarray, sample_rate: float):
    """
    Takes raw acceleration samples (in g) and returns time-domain and
    frequency-domain features. This ports the EXACT algorithm already
    validated on real hardware in the ESP32 firmware (linear detrend +
    trapezoidal integration + windowed FFT) - not a new implementation,
    the same math, just running server-side now that the ESP32 only
    uploads raw acceleration.
    """
    n = len(samples)
    dt = 1.0 / sample_rate

    # --- Time-domain: integrate acceleration -> velocity (trapezoidal) ---
    accel_mms2 = samples.astype(np.float64) * G_TO_MMS2
    velocity = np.zeros(n)
    velocity[1:] = np.cumsum(0.5 * (accel_mms2[:-1] + accel_mms2[1:]) * dt)

    # --- Linear detrend (least-squares) - removes integration drift ---
    # This is the fix that was empirically validated as necessary earlier:
    # a straight mean-subtraction wasn't enough, the wandering baseline
    # needed a real least-squares line fit removed.
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, velocity, 1)
    velocity_detrended = velocity - (slope * x + intercept)

    rms_velocity = float(np.sqrt(np.mean(velocity_detrended ** 2)))
    peak_to_peak = float(velocity_detrended.max() - velocity_detrended.min())
    absolute_peak = float(np.max(np.abs(velocity_detrended)))
    crest_factor = absolute_peak / rms_velocity if rms_velocity > 0.001 else 0.0

    # --- Frequency-domain: windowed FFT on raw acceleration ---
    windowed = samples * np.hanning(n)
    fft_result = np.fft.rfft(windowed)
    magnitude = np.abs(fft_result)
    freqs = np.fft.rfftfreq(n, d=dt)

    # Skip the first couple of bins (DC/near-DC) before finding the peak -
    # same reasoning as the firmware's peak search.
    skip_bins = 2
    peak_idx = skip_bins + int(np.argmax(magnitude[skip_bins:]))
    dominant_freq = float(freqs[peak_idx])
    dominant_mag = float(magnitude[peak_idx])

    return {
        "rms_velocity_mms": rms_velocity,
        "peak_to_peak_velocity_mms": peak_to_peak,
        "absolute_peak_velocity_mms": absolute_peak,
        "crest_factor": crest_factor,
        "dominant_freq_hz": dominant_freq,
        "dominant_freq_mag": dominant_mag,
    }
