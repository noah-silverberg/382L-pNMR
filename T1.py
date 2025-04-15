#!/usr/bin/env python
"""
T1 Echo Measurement Script
===========================

Purpose
-------
Measure T1 using the pi – variable wait time (tau) – pi/2 – fixed wait – pi sequence.
In each file, the echo (formed after the second pi pulse) is analyzed. The file names
are of the form "tauXX_Y.csv" where XX is the wait time in ms, and Y is the repetition number.
For each tau value the user must provide a cropping window (in seconds) where the echo occurs.
Within that window, each file is lowpass filtered, Hilbert transformed, and smoothed.
The maximum of the smoothed envelope is taken as the echo amplitude.
Finally, the echo amplitudes versus tau are fitted to an exponential decay:
    f(tau) = A * exp(-tau/T1) + C
and summary plots and statistics are produced.

------------------- USER SETTINGS -------------------
- Define the folder containing the data files.
- Define a dictionary with cropping windows (in seconds) keyed by tau value in ms.
-----------------------------------------------------
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

# ------------------- USER SETTINGS -------------------
DATA_FOLDER = "4-15-25"  # folder where the CSV files are located
# Cropping windows for each tau (in seconds); adjust these values as needed.
# Example: For tau = 10 ms, the echo is expected in the time window 0.010-0.012 s.
CROPPING_WINDOWS = {
    50: (78e-3, 82e-3),
    75: (103e-3, 107e-3),
    100: (128e-3, 132e-3),
    175: (203e-3, 207e-3),
    200: (228e-3, 232e-3),
    250: (278e-3, 282e-3),
    300: (328e-3, 332e-3),
    375: (403e-3, 407e-3),
    500: (528e-3, 532e-3),
    750: (778e-3, 782e-3),
    1000: (1028e-3, 1032e-3),
    1500: (1528e-3, 1532e-3),
}
# ------------------- END USER SETTINGS -------------------


# ------------------- MODEL FUNCTION -------------------
def model_exp(tau, A, T1, C):
    """
    Exponential decay model for T1 measurement.
    f(tau) = A * exp(-tau/T1) + C
    tau is assumed to be in milliseconds.
    """
    return A * np.exp(-tau / T1) + C


# ------------------- DATA LOADING -------------------
def load_t1_data(folder):
    """
    Load CSV files from the given folder.
    Files are assumed to be named "XXms_Y", where XX is tau (ms)
    and Y is the repetition number.

    Returns a dictionary mapping each tau value (int, in ms) to a list of DataFrames.
    """
    pattern = re.compile(
        r"(?P<tau>\d+)ms_(?P<rep>\d+)\.csv", re.IGNORECASE
    )  # Regex to match the filenames.
    data_dict = {}
    for fname in sorted(os.listdir(folder)):
        m = pattern.match(fname)
        if m is None:
            continue
        tau = int(m.group("tau"))
        df = pd.read_csv(
            os.path.join(folder, fname), header=None, names=["t", "CH1", "CH2"]
        ).dropna()
        # Baseline correction (if needed)
        df["CH1"] = df["CH1"] - df["CH1"].mean()
        df["CH2"] = df["CH2"] - df["CH2"].mean()
        # Store each file under its corresponding tau value.
        data_dict.setdefault(tau, []).append(df)
    return data_dict


# ------------------- ECHO EXTRACTION -------------------
def process_echo(
    df,
    cropping_window,
    channel="CH1",
    filter_order=3,
    cutoff=np.array([1000, 5000]),
    fs=250_000,
    smoothing_sigma=50,
):
    """
    Given a DataFrame and a cropping window (a tuple (t_start, t_end) in seconds),
    extract the echo from the specified channel.

    Steps:
      1. Crop the data to the specified window.
      2. Apply a lowpass Butterworth filter (using filtfilt).
      3. Compute the analytic signal via the Hilbert transform.
      4. Smooth the envelope with a Gaussian filter.
      5. Return the time array (cropped), the filtered envelope, and the maximum peak value.
    """
    t = df["t"].values
    signal = df[channel].values
    t_start, t_end = cropping_window
    mask = (t >= t_start) & (t <= t_end)
    if np.sum(mask) < 10:
        return None, None, np.nan
    t_crop = t[mask]
    sig_crop = signal[mask]

    # Butterworth lowpass design.
    b, a = butter(filter_order, cutoff, fs=fs, btype="bandpass")
    filtered = filtfilt(b, a, sig_crop)

    # Hilbert transform and envelope extraction.
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)

    # Gaussian smoothing.
    smooth_envelope = gaussian_filter1d(envelope, sigma=smoothing_sigma)

    # Determine the maximum envelope value.
    peak_val = np.max(smooth_envelope)
    return t_crop, smooth_envelope, peak_val


# ------------------- ANALYSIS -------------------
def analyze_t1_echo(data_dict, cropping_windows, channel="CH1"):
    """
    For each tau value (in ms), process all files:
      - Crop the signal to the echo window specified for that tau.
      - Extract the echo peak amplitude using process_echo.
    Returns:
      - results_df: DataFrame with one row per tau value (with averaged amplitude and error).
      - individual_list: list of tuples (tau, echo amplitude) for all files.
    """
    tau_vals = []
    mean_amplitudes = []
    amplitude_errs = []

    # For fitting using all individual points.
    all_tau = []
    all_amp = []
    for tau, dfs in sorted(data_dict.items()):
        # Get the cropping window for this tau.
        if tau not in cropping_windows:
            print(f"Warning: No cropping window provided for tau={tau} ms. Skipping.")
            continue
        window = cropping_windows[tau]
        echo_amplitudes = []
        for df in dfs:
            _, _, peak = process_echo(df, window, channel=channel)
            if not np.isnan(peak):
                echo_amplitudes.append(peak)
                all_tau.append(tau)
                all_amp.append(peak)
        if len(echo_amplitudes) == 0:
            continue
        tau_vals.append(tau)
        mean_amp = np.mean(echo_amplitudes)
        std_amp = np.std(echo_amplitudes, ddof=1)
        err_amp = std_amp / np.sqrt(len(echo_amplitudes))
        mean_amplitudes.append(mean_amp)
        amplitude_errs.append(err_amp)
    results_df = pd.DataFrame(
        {"Tau (ms)": tau_vals, "Echo Amp": mean_amplitudes, "Amp Err": amplitude_errs}
    )
    individual_list = (np.array(all_tau), np.array(all_amp))
    return results_df, individual_list


# ------------------- PLOTTING & FITTING -------------------
def plot_and_fit(results_df, individual, fit_filename="t1_fit.png"):
    """
    Plot the averaged echo amplitudes (with error bars) vs. tau and fit the
    data to a decaying exponential model.
    Also plot the individual measurements for reference.
    """
    taus = results_df["Tau (ms)"].values
    amps = results_df["Echo Amp"].values
    errs = results_df["Amp Err"].values
    all_tau, all_amp = individual

    # Fit the model to the individual data points.
    p0 = [np.max(all_amp), 100.0, np.min(all_amp)]  # initial guess: A, T1 (ms), C
    try:
        popt, pcov = curve_fit(model_exp, all_tau, all_amp, p0=p0, maxfev=100000)
    except RuntimeError:
        print("Exponential fit failed.")
        popt = [np.nan, np.nan, np.nan]

    # Generate fitted curve over a dense tau range.
    tau_fit = np.linspace(np.min(taus), np.max(taus), 200)
    amp_fit = model_exp(tau_fit, *popt)

    plt.figure(figsize=(8, 6))
    # Plot individual points.
    # plt.plot(all_tau, all_amp, "ko", alpha=0.5, label="Individual Measurements")
    # Plot averaged points with error bars.
    # plt.errorbar(taus, amps, yerr=errs, fmt="ro", capsize=4, label="Averaged Data")
    plt.errorbar(
        taus,
        amps,
        yerr=np.std(all_amp.reshape(-1, 5), axis=1),
        fmt="o",
        capsize=4,
        label="Average Amplitude ± σ",
    )
    # Plot fit curve.
    plt.plot(
        tau_fit,
        amp_fit,
        "--",
        label=f"Exponential Fit",
    )
    plt.xlabel(r"$\tau$ (ms)")
    plt.ylabel("Echo Amplitude (a.u.)")
    plt.title(r"Echo Amplitude vs Wait Time ($\tau$)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fit_filename)
    plt.close()

    # Print fit parameters and reduced chi-square.
    fitted_all = model_exp(all_tau, *popt)
    residuals = all_amp - fitted_all
    variance = np.var(all_amp.reshape(-1, 5), axis=1)
    variance = np.repeat(variance, 5)
    chi_sq = np.sum((residuals) ** 2 / (variance + 1e-6))
    dof = len(all_amp) - len(popt)
    red_chi_sq = chi_sq / dof if dof > 0 else np.nan
    print("Exponential Fit Parameters (A, T1, C):", popt)
    print("Reduced Chi-square:", red_chi_sq)
    print("Chi-square:", chi_sq)
    print("Degrees of Freedom:", dof)

    # Print the T1 value and its uncertainty.
    t1_value = popt[1]
    t1_uncertainty = np.sqrt(np.diag(pcov))[1]
    print(f"T1 Value: {t1_value:.2f} ms ± {t1_uncertainty:.2f} ms")
    return popt


# ------------------- MAIN SCRIPT -------------------
def main():
    warnings.simplefilter("ignore")

    # Load data files from the folder.
    data_dict = load_t1_data(DATA_FOLDER)
    if not data_dict:
        print("No valid T1 files found. Exiting.")
        return

    fig, axs = plt.subplots(len(data_dict), 1, figsize=(10, 4 * len(data_dict)))
    for i, (tau, dfs) in enumerate(sorted(data_dict.items())):
        for df in [dfs[0]]:
            t = df["t"].values
            signal = df["CH1"].values
            axs[i].plot(t, signal, label="Signal")
            # Plot the filtered signal.
            t_crop, smooth_envelope, peak = process_echo(
                df, CROPPING_WINDOWS[tau], channel="CH1"
            )
            if t_crop is not None:
                axs[i].plot(
                    t_crop,
                    smooth_envelope,
                    label="Filtered Signal",
                    color="orange",
                    alpha=0.6,
                )
                axs[i].scatter(
                    t_crop[np.argmax(smooth_envelope)],
                    peak,
                    color="red",
                    label="Echo Peak",
                )
            axs[i].set_title(f"Signal for Tau={tau} ms")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("Amplitude")
            axs[i].set_xticks(np.arange(0, np.max(t), 0.025))
            axs[i].set_xlim(np.min(t), np.max(t))
            axs[i].grid()
            # Highlight the cropping window.
            t_start, t_end = CROPPING_WINDOWS[tau]
            axs[i].axvspan(
                t_start, t_end, color="red", alpha=0.5, label="Cropping Window"
            )
            axs[i].legend()
    plt.tight_layout()
    plt.savefig("all_signals.png")
    plt.close()

    # Analyze each file: extract echo amplitudes using the provided cropping windows.
    results_df, individual = analyze_t1_echo(data_dict, CROPPING_WINDOWS, channel="CH1")
    print("Averaged Echo Amplitudes by Tau (ms):")
    print(results_df)

    # Plot the data and fit to the exponential decay model.
    popt = plot_and_fit(results_df, individual, fit_filename="t1_exponential_fit.png")

    # Save the results to CSV.
    results_df.to_csv("t1_results.csv", index=False)


if __name__ == "__main__":
    main()
