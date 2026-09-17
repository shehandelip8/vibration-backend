import numpy as np

G_TO_MMS2 = 9806.65  # 1g = 9806.65 mm/s^2 - same constant used in the ESP32 firmware


def compute_spectrum(samples: np.ndarray, sample_rate: float):
    """
    Windowed FFT magnitude spectrum - shared by process_waveform's dominant-
    peak search and the new harmonic-lookup feature, so both use the exact
    same underlying math rather than two separate FFT implementations.
    """
    n = len(samples)
    windowed = samples * np.hanning(n)
    fft_result = np.fft.rfft(windowed)
    magnitude = np.abs(fft_result)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return freqs, magnitude


def compute_band_energies(freqs: np.ndarray, magnitude: np.ndarray, band_width_hz: float = 100.0, max_freq_hz: float = 2000.0):
    """
    Sums spectral energy into fixed-width frequency bands (default 100Hz
    wide, up to 2000Hz - 20 bands total). This is what makes anomaly
    detection work across DIFFERENT machines without hardcoding which
    harmonic matters: instead of checking a specific frequency (which
    varies by mounting/resonance/sensor orientation per installation),
    this captures the whole spectral "shape" as a fixed-length fingerprint,
    which then gets compared against THIS machine's own historical normal
    for each band - see mqtt_listener.py's anomaly check.
    """
    num_bands = int(max_freq_hz // band_width_hz)
    band_energies = []
    for i in range(num_bands):
        low = i * band_width_hz
        high = (i + 1) * band_width_hz
        mask = (freqs >= low) & (freqs < high)
        band_energies.append(float(np.sum(magnitude[mask] ** 2)))
    return band_energies


def get_harmonic_magnitudes(freqs: np.ndarray, magnitude: np.ndarray, base_freq: float, num_harmonics: int = 4):
    """
    Looks up the magnitude at 1x, 2x, 3x, 4x (etc) of a given base frequency
    (e.g. a tachometer-measured running speed converted to Hz). Since the
    true harmonic frequency won't usually land exactly on a bin center,
    this reports the actual bin frequency used alongside the target, so you
    can see how close it landed - same "close is the win, not exact"
    reasoning as the synthetic FFT validation test.
    """
    bin_resolution = freqs[1] - freqs[0]
    results = []
    for h in range(1, num_harmonics + 1):
        target_freq = base_freq * h
        idx = int(round(target_freq / bin_resolution))
        idx = min(idx, len(magnitude) - 1)
        results.append({
            "harmonic": h,
            "target_freq_hz": target_freq,
            "bin_freq_hz": float(freqs[idx]),
            "magnitude": float(magnitude[idx]),
        })
    return results


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
    freqs, magnitude = compute_spectrum(samples, sample_rate)

    # Skip the first couple of bins (DC/near-DC) before finding the peak -
    # same reasoning as the firmware's peak search.
    skip_bins = 2
    peak_idx = skip_bins + int(np.argmax(magnitude[skip_bins:]))
    dominant_freq = float(freqs[peak_idx])
    dominant_mag = float(magnitude[peak_idx])

    # Spectral "fingerprint" for machine-specific anomaly detection - see
    # compute_band_energies() docstring for why this matters.
    band_energies = compute_band_energies(freqs, magnitude)

    return {
        "rms_velocity_mms": rms_velocity,
        "peak_to_peak_velocity_mms": peak_to_peak,
        "absolute_peak_velocity_mms": absolute_peak,
        "crest_factor": crest_factor,
        "dominant_freq_hz": dominant_freq,
        "dominant_freq_mag": dominant_mag,
        "band_energies": band_energies,
    }
