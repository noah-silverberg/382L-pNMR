#!/usr/bin/env python
"""
pi_pulse_single.py
==================

Purpose
-------
Analyze only one of the two sequences: either "+π/2" or "–π/2",
based on the SIGN_MODE parameter below. It loads CSV files from
the folder '327/Pulse Calibration Redo' whose filenames follow the format:

    "<pulse_time>_<plus|minus>_<rep>.csv"

Only files matching the chosen SIGN_MODE are processed. For each unique
pulse time the repeated measurements are averaged into a complex signal.
A simultaneous composite fit is then performed, with uncertainties (sigma)
constructed by concatenating the standard deviations of the real and
imaginary parts.

Finally, a sinusoid-plus-linear model is fitted to the amplitude vs.
pulse time data. This script then prints the fitted parameters and
saves plots of the fit, the amplitude-profiling for each group, and the
sinusoid fit.

------------------- USER TOGGLE -------------------
Set SIGN_MODE to "plus" for +π/2 data, or "minus" for –π/2 data.
-----------------------------------------------------
"""

import os
import re
import copy
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------- USER TOGGLE HERE -------------------
SIGN_MODE = "minus"  # choose "plus" or "minus"
# --------------------------------------------------------


# --------------------------------------------------------
# 1) MODEL FUNCTIONS
# --------------------------------------------------------
def model_complex(t, A, T2star, f, phi, C_re, C_im):
    """
    Complex decaying sinusoid:
      A * exp(-t/T2star) * exp(i*(2π*f*t + phi)) + (C_re + i*C_im)
    """
    return A * np.exp(-t / T2star) * np.exp(1j * (2 * np.pi * f * t + phi)) + (
        C_re + 1j * C_im
    )


def model_simultaneous(dummy_x, *params, group_t_list):
    """
    Composite model for simultaneous fitting.
    For each group i we have parameters:
      [A_i, f_i, phi_i, C_re_i, C_im_i]
    and one global parameter T2star at the end.
    The output is a 1D array constructed by concatenating
    each group’s model real and imaginary parts.
    """
    N = len(group_t_list)
    T2star = params[-1]
    out = []
    for i in range(N):
        A, f, phi, C_re, C_im = params[5 * i : 5 * i + 5]
        comp = model_complex(group_t_list[i], A, T2star, f, phi, C_re, C_im)
        # Concatenate real and imag parts so that each complex
        # array of length M becomes a float array of length 2*M.
        out.append(np.concatenate((comp.real, comp.imag)))
    return np.concatenate(out)


def sinusoid_const(x, a, f, phi, c):
    """a·sin(2π f x + φ) + c   (no linear‑in‑x term)"""
    return a * np.sin(2 * np.pi * f * x + phi) + c


def find_peak_of_sinusoid(popt, x_rng, is_max=True):
    """Locate peak (is_max=True) or trough (False) on a dense grid."""
    x = np.linspace(x_rng.min(), x_rng.max(), 10_000)
    y = sinusoid_const(x, *popt)
    idx = np.argmax(y) if is_max else np.argmin(y)
    return x[idx], y[idx]


# --------------------------------------------------------
# 3) MAIN SCRIPT
# --------------------------------------------------------
def main():
    # Folder containing CSV data (use your working folder)
    folder_name = "4-8-25"
    # Regex to extract pulse time and sign from filename.
    pattern = re.compile(r"^(?P<pulse>\d+)_(?P<sign>plus|minus)_\d+\.csv$")
    # Only keep files matching the chosen SIGN_MODE.
    data_groups = defaultdict(list)
    for fname in os.listdir(folder_name):
        m = pattern.match(fname)
        if not m:
            continue
        if m.group("sign") != SIGN_MODE:
            continue
        pulse_time = int(m.group("pulse"))
        df = pd.read_csv(
            os.path.join(folder_name, fname), header=None, names=["t", "CH1", "CH2"]
        )
        df = df.dropna()
        # Use an appropriate time window (adjust as needed)
        if SIGN_MODE == "plus":
            df = df[(df["t"] > 0.007) & (df["t"] < 0.013)]
        else:
            df = df[(df["t"] > 0.018) & (df["t"] < 0.022)]
        df["CH1"] = df["CH1"] - df["CH1"].mean()
        df["CH2"] = df["CH2"] - df["CH2"].mean()
        if len(df) < 10:
            print(f"Skipping {fname}: not enough data points after filtering.")
            continue
        data_groups[pulse_time].append(df)

    if not data_groups:
        print(f"No data found for sign = {SIGN_MODE}. Exiting.")
        return

    # For each pulse time, average the complex signal.
    group_data_list = []
    for pt in sorted(data_groups.keys()):
        dfs = data_groups[pt]
        signals = []
        t_common = None
        # Use a Butterworth lowpass filter.
        b, a = butter(3, 5000, btype="lowpass", fs=250000)
        for df in dfs:
            t = df["t"].values
            if t_common is None:
                t_common = t
            raw = df["CH1"].values + 1j * df["CH2"].values
            try:
                filtered = filtfilt(b, a, raw)
            except Exception as e:
                print(f"Error filtering data for pulse {pt} in file {fname}: {e}")
                continue
            signals.append(filtered)
            if len(signals) == 0:
                print(f"No valid signals for pulse time {pt}. Skipping.")
                continue
        signals = np.array(signals)
        avg_signal = signals.mean(axis=0)
        # Compute standard deviation for real and imaginary parts separately.
        std_real = np.std(np.real(signals), axis=0, ddof=1)
        std_imag = np.std(np.imag(signals), axis=0, ddof=1)
        group_data_list.append(
            {
                "pulse_time": pt,
                "t": t_common,
                "avg_signal": avg_signal,
                "std_real": std_real,
                "std_imag": std_imag,
                "signals": signals,
            }
        )

    if not group_data_list:
        print("No groups with valid data found. Exiting.")
        return

    # ----- Compute envelope-based peak amplitudes for each pulse time group -----
    from scipy.signal import hilbert
    from scipy.ndimage import gaussian_filter1d

    amplitude_list = []
    amplitude_err_list = []
    pulse_times = []

    # Create lists to store all individual measurements for the fit
    all_pulse_time_vals = []
    all_A_vals = []
    all_A_sigma = []

    for group in group_data_list:
        # For each group, compute the amplitude for every individual signal.
        individual_As = []
        for signal in group["signals"]:
            # Compute envelopes of the individual signal (real and imaginary parts)
            env_real_ind = np.abs(hilbert(signal.real))
            env_imag_ind = np.abs(hilbert(signal.imag))

            # Smooth the envelopes.
            smooth_real_ind = gaussian_filter1d(env_real_ind, sigma=30)
            smooth_imag_ind = gaussian_filter1d(env_imag_ind, sigma=30)

            # Detect peak values.
            peak_real_ind = np.max(smooth_real_ind)
            peak_imag_ind = np.max(smooth_imag_ind)

            # Compute amplitude for this individual signal.
            A_ind = (peak_real_ind + peak_imag_ind) / 2.0
            individual_As.append(A_ind)

        # Compute the mean amplitude and error (using the standard error).
        amplitude = np.mean(individual_As)
        # Standard deviation divided by sqrt(n) gives the standard error:
        amplitude_err = np.std(individual_As, ddof=1) / np.sqrt(len(individual_As))

        n_points = len(individual_As)
        all_pulse_time_vals.extend([group["pulse_time"]] * n_points)
        all_A_vals.extend(individual_As)
        all_A_sigma.extend([amplitude_err] * n_points)

        amplitude_list.append(amplitude)
        amplitude_err_list.append(amplitude_err)
        pulse_times.append(group["pulse_time"])

    # Create a summary DataFrame including the amplitude error.
    results_df = pd.DataFrame(
        {
            "Pulse Time (us)": pulse_times,
            "A": amplitude_list,
            "A_err": amplitude_err_list,
        }
    )
    print(
        "Envelope amplitudes and their error estimates extracted for each pulse time group:"
    )
    print(results_df)

    # ----- Plot original averaged signals with their envelopes and detected peaks -----
    fig, axs = plt.subplots(
        nrows=len(group_data_list), ncols=2, figsize=(12, 3 * len(group_data_list))
    )

    # Ensure axs is 2D even if there is only one group.
    if len(group_data_list) == 1:
        axs = np.array([axs])

    # ----- Plot individual signals with their envelopes and detected peaks -----
    # Loop over each group in group_data_list
    for i, group in enumerate(group_data_list):
        n_signals = len(group["signals"])  # number of individual signals in this group
        # Create a subplot grid with one row per signal, two columns (Real and Imag)
        fig, axs = plt.subplots(n_signals, 2, figsize=(12, 3 * n_signals))

        # In case there is only one signal, force axs to be 2D.
        if n_signals == 1:
            axs = np.array([axs])

        # Get the common time axis for the current group.
        t = group["t"]

        # Loop over each individual signal in the current group
        for j, signal in enumerate(group["signals"]):
            # Compute the envelope of the real and imaginary parts using the Hilbert transform.
            env_real = np.abs(hilbert(signal.real))
            env_imag = np.abs(hilbert(signal.imag))

            # Smooth the envelopes using a Gaussian filter.
            smooth_env_real = gaussian_filter1d(env_real, sigma=30)
            smooth_env_imag = gaussian_filter1d(env_imag, sigma=30)

            # Find the peak indices (maximum value) in the smoothed envelopes.
            peak_idx_real = np.argmax(smooth_env_real)
            peak_idx_imag = np.argmax(smooth_env_imag)

            # Plot the Real part:
            axs[j, 0].plot(t, signal.real, "b-", label="Real Signal")
            axs[j, 0].plot(t, smooth_env_real, "r--", label="Envelope")
            axs[j, 0].plot(
                t[peak_idx_real],
                smooth_env_real[peak_idx_real],
                "ko",
                markersize=8,
                label="Peak",
            )
            axs[j, 0].set_title(f'Pulse {group["pulse_time"]} µs - Signal {j+1} (Real)')
            axs[j, 0].set_xlabel("Time (s)")
            axs[j, 0].set_ylabel("Amplitude")
            axs[j, 0].legend(fontsize="small")

            # Plot the Imaginary part:
            axs[j, 1].plot(t, signal.imag, "b-", label="Imag Signal")
            axs[j, 1].plot(t, smooth_env_imag, "r--", label="Envelope")
            axs[j, 1].plot(
                t[peak_idx_imag],
                smooth_env_imag[peak_idx_imag],
                "ko",
                markersize=8,
                label="Peak",
            )
            axs[j, 1].set_title(f'Pulse {group["pulse_time"]} µs - Signal {j+1} (Imag)')
            axs[j, 1].set_xlabel("Time (s)")
            axs[j, 1].set_ylabel("Amplitude")
            axs[j, 1].legend(fontsize="small")

        plt.tight_layout()
        # Save each group's figure with a filename that indicates the pulse time.
        plt.savefig(f"individual_envelopes_group_{group['pulse_time']}.png")
    plt.close()

    # ----- SINUSOID PLUS LINEAR FIT TO AMPLITUDE vs. PULSE TIME -----
    # Use the summary results from the simultaneous fit.
    # In this script we model A vs. pulse time with: a*sin(2π*f*x+phi) + (b*x+c)
    p0_sine = [
        results_df["A"].max(),  # a
        5e-3,  # f  (≈ 0.005 µs⁻¹  → 200 µs period)
        0.0,  # φ
        results_df["A"].mean(),
    ]  # c

    try:
        all_pulse_time_vals = np.asarray(all_pulse_time_vals)
        all_A_vals = np.asarray(all_A_vals)
        all_A_sigma = np.asarray(all_A_sigma)

        popt_sine, pcov_sine = curve_fit(
            sinusoid_const,
            all_pulse_time_vals,
            all_A_vals,
            p0=p0_sine,
            sigma=all_A_sigma,
            absolute_sigma=True,
            maxfev=100_000,
        )

    except RuntimeError:
        print("Sinusoid fit failed.")
        return

    print("Sinusoid fit parameters:", popt_sine)
    x_fit = np.linspace(
        results_df["Pulse Time (us)"].min(), results_df["Pulse Time (us)"].max(), 400
    )
    y_fit = sinusoid_const(x_fit, *popt_sine)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        results_df["Pulse Time (us)"],
        results_df["A"],
        yerr=results_df["A_err"],
        fmt="o",
        capsize=4,
        label="Data (±σ)",
    )
    plt.plot(x_fit, y_fit, "--", label="Sinusoid Fit")
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.xlabel("Pulse Time (μs)", fontsize=16)
    plt.ylabel("Amplitude (a.u.)", fontsize=16)
    plt.title("Amplitude vs Pulse Time", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("sinusoid_fit_SINGLE_{}.png".format(SIGN_MODE))
    plt.close()

    # Compute reduced chi-square for the sinusoid fit.
    fitted_all = sinusoid_const(all_pulse_time_vals, *popt_sine)
    residuals = all_A_vals - fitted_all
    chi_sq = np.sum((residuals / all_A_sigma) ** 2)
    dof = len(all_A_vals) - len(popt_sine)
    reduced_chi_sq = chi_sq / dof
    print(f"Reduced chi-square for sinusoid fit: {reduced_chi_sq:.4f}")

    # Find maximum and minimum.
    max_x, max_y = find_peak_of_sinusoid(popt_sine, x_fit)
    print("Sinusoid plus linear fit results:")
    print(f"Peak (Pi pulse) at: {max_x:.4f} us, amplitude = {max_y:.4f}")

    # ------------------------------------------------------------------
    #  Error on peak (π/2) pulse time via analytic error propagation
    # ------------------------------------------------------------------
    def x0_error(a, f, phi, c, pcov, branch=+1):
        """
        branch = +1 for peak  (π/2),  branch = -1 for trough (-π/2).
        Returns (x0 [µs], σ_x0 [µs]).
        """
        x0 = (branch * np.pi / 2 - phi) / (2 * np.pi * f)  # seconds
        # gradient w.r.t [a, f, phi, c]
        dfd = -(branch * np.pi / 2 - phi) / (2 * np.pi * f**2)
        dph = -1 / (2 * np.pi * f)
        grad = np.array([0.0, dfd, dph, 0.0])
        var = grad @ pcov @ grad
        sig = np.sqrt(var)
        return x0, sig  # → µs

    a, f, phi, c = popt_sine
    peak_x, peak_err = x0_error(a, f, phi, c, pcov_sine, branch=+1)

    print(f"π pulse  : {peak_x:.4f} ± {peak_err:.4f} µs (1 σ)")

    residuals = all_A_vals - sinusoid_const(all_pulse_time_vals, *popt_sine)
    chi_sq = np.sum((residuals / all_A_sigma) ** 2)
    dof = len(all_A_vals) - len(popt_sine)
    print(f"Reduced χ² = {chi_sq/dof:.3f}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
