#!/usr/bin/env python
"""
T2 Decay Measurement Notebook
=============================

This script compares T2 decay extracted from three pulse sequences:
Hahn Echo, Carr-Purcell (CP), and Carr-Purcell-Meiboom-Gill (CPMG).

Modifications:
1. Instead of automatically detecting large pulse artifacts and masking them,
   the echo time windows are provided manually. Only data within these ranges
   is processed. Each echo is treated individually.
2. For each individual echo, a 3rd order Butterworth lowpass filter (5000 Hz,
   sampling rate 250 kHz) is applied using filtfilt. Next, the Hilbert
   transform and Gaussian smoothing are used to compute a smooth envelope.
3. The echo peaks (the maximum envelope values within the echo windows) are
   then fitted with a decaying exponential model to extract T2 for each file.
   Analysis and plots are generated accordingly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import os
import re

# ---------------------------- USER SETTINGS ----------------------------
# Manually define echo time ranges (in seconds) to process.
# Only data falling within these windows will be used.
# Adjust these values as needed.
ECHO_RANGES = [
    # (0.008, 0.012),
    (0.018, 0.022),
    (0.028, 0.032),
    (0.038, 0.042),
    (0.048, 0.052),
    (0.058, 0.062),
    (0.068, 0.072),
    (0.078, 0.082),
]


# ---------------------------- DATA LOADING ----------------------------
def load_data(folder, prefix):
    """
    Load CSV files from the specified folder that match the given prefix.
    Files must be named like "PREFIX<number>.csv". Also performs some basic
    baseline correction.
    """
    data = []
    pulse_times = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".csv") and re.match(rf"^{prefix}\d+\.csv$", fname):
            # For demonstration, a constant pulse time is used; adjust if needed.
            pulse_time = 46
            df = pd.read_csv(
                os.path.join(folder, fname), header=None, names=["t", "CH1", "CH2"]
            ).dropna()
            # Use all data beyond 0.007 s (adjust as needed)
            df = df[df["t"] > 0.007].copy()
            # Simple baseline correction:
            df["CH1"] = -(df["CH1"] - df["CH1"].mean())
            df["CH2"] = df["CH2"] - df["CH2"].mean()

            data.append(df)
            pulse_times.append(pulse_time)
    return data, pulse_times


# ---------------------------- MODEL DEFINITION ----------------------------
def model_exp(t, A, T2, C):
    """
    Exponential decay model.
    """
    return A * np.exp(-t / T2) + C


def model_exp_zeroed(t, A, T2):
    """
    Exponential decay model.
    """
    return A * np.exp(-t / T2)


# ---------------------------- ECHO EXTRACTION ----------------------------
def extract_echo_amplitudes(
    df,
    echo_ranges,
    channel="CH1",
    filter_order=3,
    cutoff=np.array([1000, 5000]),
    fs=25000,
    smoothing_sigma=20,
):
    """
    For a given DataFrame and manually specified echo ranges, extract the echo
    peak times and amplitudes. For each echo window:
      - Only data within the provided range is used.
      - A 3rd order Butterworth lowpass filter (cutoff=5000 Hz) is applied.
      - The Hilbert transform is used to compute the signal envelope.
      - A Gaussian filter smooths the envelope.
      - The maximum of the smoothed envelope (and its corresponding time)
        is taken as the echo peak.

    Returns:
      echo_times: 1D numpy array of times (in seconds) where echo peaks occur.
      echo_amplitudes: 1D numpy array of peak amplitudes.
    """
    t_all = df["t"].values
    signal_all = df[channel].values

    # Design Butterworth lowpass filter.
    b, a = butter(filter_order, cutoff, fs=fs, btype="bandpass", analog=False)

    # # Plot the PSD of the signal for debugging.
    # plt.figure(figsize=(10, 5))
    # plt.psd(signal_all, NFFT=2048, Fs=fs, Fc=0, label="Raw Signal")
    # plt.title(f"{channel} Power Spectral Density")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Power/Frequency (dB/Hz)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Apply filter to the signal.
    filtered_signal_all = filtfilt(b, a, signal_all)

    # # Plot the filtered vs raw signal for debugging.
    # plt.figure(figsize=(10, 5))
    # plt.plot(t_all, signal_all, "k-", alpha=0.3, label="Raw Signal")
    # plt.plot(t_all, filtered_signal_all, "r-", label="Filtered Signal")
    # plt.title(f"{channel} Filtered Signal")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    echo_times = []
    echo_amplitudes = []

    for t_start, t_end in echo_ranges:
        # Extract data within the echo window.
        mask = (t_all >= t_start) & (t_all <= t_end)
        if np.sum(mask) < 10:
            continue
        t_echo = t_all[mask]
        signal_echo = filtered_signal_all[mask]

        # Compute analytic signal and envelope.
        analytic_signal = hilbert(signal_echo)
        envelope = np.abs(analytic_signal)
        # Gaussian smoothing.
        smoothed_env = gaussian_filter1d(envelope, sigma=smoothing_sigma)
        # Find the maximum envelope value in this echo.
        idx_max = np.argmax(smoothed_env)
        echo_peak_time = t_echo[idx_max]
        echo_peak_amp = smoothed_env[idx_max]

        # # Plot the smoothed envelope for debugging.
        # plt.figure(figsize=(10, 5))
        # plt.plot(t_echo, signal_echo, "k-", alpha=0.3, label="Filtered Signal")
        # plt.plot(t_echo, smoothed_env, "r-", label="Smoothed Envelope")
        # plt.axvline(echo_peak_time, color="b", linestyle="--", label="Echo Peak")
        # plt.title(f"{channel} Echo Peak")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        echo_times.append(echo_peak_time)
        echo_amplitudes.append(echo_peak_amp)

    return np.array(echo_times), np.array(echo_amplitudes)


def fit_decay_manual_echoes(
    df, channel="CH1", echo_ranges=ECHO_RANGES, smoothing_sigma=10
):
    """
    For a given DataFrame, extract echo peak times and amplitudes (for one channel)
    using the manually specified echo ranges and then fit these echo peaks to an
    exponential decay model:
         model_exp(t, A, T2, C) = A * exp(-t / T2) + C

    Returns: fit parameters (popt) along with the echo peak times and amplitudes.
    If fewer than three echo peaks are found, returns NaNs.
    """
    t_peaks, amp_peaks = extract_echo_amplitudes(
        df, echo_ranges, channel=channel, smoothing_sigma=smoothing_sigma
    )
    if len(t_peaks) < 3:
        return [np.nan, np.nan, np.nan], t_peaks, amp_peaks
    # Initial guess: use first amplitude as A, half the time span as T2, last peak as C.
    p0 = [amp_peaks[0], (t_peaks[-1] - t_peaks[0]) / 2]  # , amp_peaks[-1]]
    try:
        popt, _ = curve_fit(model_exp_zeroed, t_peaks, amp_peaks, p0=p0, maxfev=10000)
    except Exception:
        popt = [np.nan, np.nan]
    return popt, t_peaks, amp_peaks


# ---------------------------- ANALYSIS ----------------------------
def analyze_echo_decay(
    data, pulse_times, channel="CH1", echo_ranges=ECHO_RANGES, smoothing_sigma=10
):
    """
    For each dataset in the input 'data', extract echo peak amplitudes from the
    manually defined echo windows and perform an exponential fit to obtain T2.

    Returns:
      - A DataFrame summarizing the results (one row per dataset).
      - A list with detailed fit information for plotting.
    """
    results = []
    all_fits = []
    for idx, df in enumerate(data):
        popt, t_peaks, amp_peaks = fit_decay_manual_echoes(
            df,
            channel=channel,
            echo_ranges=echo_ranges,
            smoothing_sigma=smoothing_sigma,
        )
        results.append(
            {
                "Pulse Time (us)": pulse_times[idx],
                "T2 (ms)": popt[1],
                "A": popt[0],
                # "C": popt[2],
            }
        )
        all_fits.append(
            {
                "t": df["t"].values,
                channel: df[channel].values,
                "t_peaks": t_peaks,
                "amp_peaks": amp_peaks,
                "fit": popt,
            }
        )
    return pd.DataFrame(results), all_fits


# ---------------------------- MAIN SCRIPT ----------------------------
def main():
    # Load data. Adjust the folder and prefixes as needed.
    # Uncomment the sequences you wish to analyze.
    # hahn_data, hahn_times = load_data('327', 'HAHN')
    cp_data, cp_times = load_data("4-10-25/CP", "CP")
    cpmg_data, cpmg_times = load_data("4-10-25/CPMG", "CPMG")

    # Analyze echo decay for a given channel (here using CH1).
    cp_results, cp_fits = analyze_echo_decay(
        cp_data, cp_times, channel="CH2", echo_ranges=ECHO_RANGES, smoothing_sigma=5
    )
    cpmg_results, cpmg_fits = analyze_echo_decay(
        cpmg_data,
        cpmg_times,
        channel="CH2",
        echo_ranges=ECHO_RANGES,
        smoothing_sigma=5,
    )

    print("CP Results:")
    print(cp_results)
    print("CPMG Results:")
    print(cpmg_results)

    # Plot examples for CP
    for i, fit in enumerate(cp_fits):
        t = fit["t"]
        raw = fit["CH2"]
        t_peaks = fit["t_peaks"]
        amp_peaks = fit["amp_peaks"]
        popt = fit["fit"]

        if len(t_peaks) > 0:
            t_fit = np.linspace(t_peaks.min(), t_peaks.max(), 200)
            model_curve = model_exp_zeroed(t_fit, *popt)
        else:
            t_fit = np.array([])
            model_curve = np.array([])

        # Apply the same filter to the raw signal for consistency.
        b, a = butter(3, [1000, 5000], fs=25000, btype="bandpass", analog=False)
        filtered_signal_all = filtfilt(b, a, raw)

        plt.figure(figsize=(10, 5))
        plt.plot(t, raw, "k-", alpha=0.3, label="Raw CH2")
        plt.plot(t, filtered_signal_all, "b-", alpha=0.3, label="Filtered Signal")
        plt.plot(t_peaks, amp_peaks, "ro", label="Echo Peaks")
        if t_fit.size > 0:
            plt.plot(t_fit, model_curve, "b--", label="Exp Fit")
        plt.title(f"CP Dataset {i+1} - CH2 Echo Decay")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"cp_echo_decay_{i+1}.png")
        plt.close()

    # Plot examples for CPMG
    for i, fit in enumerate(cpmg_fits):
        t = fit["t"]
        raw = fit["CH2"]
        t_peaks = fit["t_peaks"]
        amp_peaks = fit["amp_peaks"]
        popt = fit["fit"]

        if len(t_peaks) > 0:
            t_fit = np.linspace(t_peaks.min(), t_peaks.max(), 200)
            model_curve = model_exp_zeroed(t_fit, *popt)
        else:
            t_fit = np.array([])
            model_curve = np.array([])

        # Apply the same filter to the raw signal for consistency.
        b, a = butter(3, [1000, 5000], fs=25000, btype="bandpass", analog=False)
        filtered_signal_all = filtfilt(b, a, raw)

        plt.figure(figsize=(10, 5))
        # plt.plot(t, raw, "k-", alpha=0.3, label="Raw CH2")
        plt.plot(t, filtered_signal_all, "b-", alpha=0.3, label="Filtered Signal")
        plt.plot(t_peaks, amp_peaks, "ro", label="Echo Peaks")
        if t_fit.size > 0:
            plt.plot(t_fit, model_curve, "b--", label="Exp Fit")
        plt.title(f"CPMG Dataset {i+1} - CH2 Echo Decay")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"cpmg_echo_decay_{i+1}.png")
        plt.close()

    # Summary statistics for T2
    def summarize(series, label):
        mean = series.mean()
        std = series.std()
        sem = std / np.sqrt(len(series))
        ci95 = 1.96 * sem
        print(
            f"{label}:\n  Mean T2 = {mean:.4f} ms\n  Std = {std:.4f}\n  SEM = {sem:.4f}\n  95% CI = ±{ci95:.4f}\n"
        )

    print("--- CP T2 Results ---")
    summarize(cp_results["T2 (ms)"], "CP CH2")

    print("--- CPMG T2 Results ---")
    summarize(cpmg_results["T2 (ms)"], "CPMG CH2")


if __name__ == "__main__":
    main()
