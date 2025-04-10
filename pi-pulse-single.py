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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------- USER TOGGLE HERE -------------------
SIGN_MODE = "plus"  # choose "plus" or "minus"
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


def sinusoid_plus_linear(x, a, f, phi, b, c):
    """
    Model: a*sin(2π*f*x + phi) + (b*x + c)
    Used to fit amplitude vs. pulse time.
    """
    return a * np.sin(2 * np.pi * f * x + phi) + (b * x) + c


def find_max_sinusoid_plus_linear(popt, x_fit):
    """
    Finds the maximum of the sinusoid_plus_linear fit.
    """
    a, f, phi, b, c = popt
    x = np.linspace(np.min(x_fit), np.max(x_fit), 10000)
    y = sinusoid_plus_linear(x, a, f, phi, b, c)
    idx = np.argmax(y)
    return x[idx], y[idx]


def find_min_sinusoid_plus_linear(popt, x_fit):
    """
    Finds the minimum of the sinusoid_plus_linear fit.
    """
    a, f, phi, b, c = popt
    x = np.linspace(np.min(x_fit), np.max(x_fit), 10000)
    y = sinusoid_plus_linear(x, a, f, phi, b, c)
    idx = np.argmin(y)
    return x[idx], y[idx]


# --------------------------------------------------------
# 2) AMPLITUDE PROFILING HELPER
# --------------------------------------------------------
def profile_Ai(i_fixed, popt_best, perr, group_t, ydata, sigma_all, folder="profiles"):
    """
    Profiles the amplitude A_i (for group i_fixed) by scanning over a grid of A values,
    performing a local fit with A fixed, and computing the chi-square.
    Returns the best-fit A value and an estimated 1-sigma error.
    Saves a profile plot in the folder.
    """
    os.makedirs(folder, exist_ok=True)
    A_best = popt_best[5 * i_fixed]
    dA = 1.5 * abs(perr[5 * i_fixed])
    A_grid = np.linspace(A_best - dA, A_best + dA, 21)
    chi_vals = []

    def fixed_model(dummy, *free):
        N = len(group_t)
        full = []
        idx_free = 0
        for j in range(N):
            if j == i_fixed:
                full.append(Ai)  # Ai is set by the outer loop.
                full.extend(free[idx_free : idx_free + 4])
                idx_free += 4
            else:
                full.extend(free[idx_free : idx_free + 5])
                idx_free += 5
        full.append(free[-1])
        return model_simultaneous(None, *full, group_t_list=group_t)

    for Ai in A_grid:
        # Build initial guess (exclude amplitude for group i_fixed)
        p0_local = []
        N = len(group_t)
        for j in range(N):
            if j == i_fixed:
                p0_local.extend(popt_best[5 * j + 1 : 5 * j + 5])
            else:
                p0_local.extend(popt_best[5 * j : 5 * j + 5])
        p0_local.append(popt_best[-1])
        try:
            popt_local, _ = curve_fit(
                lambda d, *fp: fixed_model(d, *fp),
                None,
                ydata,
                p0=p0_local,
                sigma=sigma_all,
                absolute_sigma=True,
                maxfev=4000,
            )
            idx_free = 0
            full_params = []
            for j in range(N):
                if j == i_fixed:
                    full_params.append(Ai)
                    full_params.extend(popt_local[idx_free : idx_free + 4])
                    idx_free += 4
                else:
                    full_params.extend(popt_local[idx_free : idx_free + 5])
                    idx_free += 5
            full_params.append(popt_local[-1])
            resid = ydata - model_simultaneous(None, *full_params, group_t_list=group_t)
            chi_sq = np.sum((resid / sigma_all) ** 2)
            chi_vals.append(chi_sq)
        except RuntimeError:
            chi_vals.append(np.inf)
    chi_vals = np.array(chi_vals)
    best_idx = np.argmin(chi_vals)
    A_prof_best = A_grid[best_idx]
    chi_min = chi_vals[best_idx]
    try:
        # Estimate 1-sigma by linear interpolation on either side.
        left_interp = np.interp(
            chi_min + 1, chi_vals[:best_idx][::-1], A_grid[:best_idx][::-1]
        )
        right_interp = np.interp(chi_min + 1, chi_vals[best_idx:], A_grid[best_idx:])
        err = 0.5 * ((A_prof_best - left_interp) + (right_interp - A_prof_best))
    except Exception:
        err = np.nan
    plt.figure()
    plt.plot(A_grid, chi_vals, "o-")
    plt.axhline(chi_min + 1, color="r", ls="--")
    plt.axvline(A_prof_best, color="g", ls="--")
    plt.xlabel(f"A_{i_fixed}")
    plt.ylabel("Chi-square")
    plt.title(f"Profiling A_{i_fixed}")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"profile_A_{i_fixed}.png"))
    plt.close()
    return i_fixed, A_prof_best, err


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
    p0_sine = [3, 5e-3, 0, 1e-5, 1e-2]  # initial guess: [a, f, phi, b, c]
    try:
        # ----- Sinusoid-plus-linear fit using the extracted amplitudes -----
        p0_sine = [3, 5e-3, 0, 1e-5, 1e-2]  # initial guess: [a, f, phi, b, c]
        all_pulse_time_vals = np.array(all_pulse_time_vals)
        all_A_vals = np.array(all_A_vals)
        all_A_sigma = np.array(all_A_sigma)

        popt_sine, pcov_sine = curve_fit(
            sinusoid_plus_linear,
            all_pulse_time_vals,
            all_A_vals,
            p0=p0_sine,
            maxfev=100000,
            sigma=all_A_sigma,
            absolute_sigma=True,
        )

    except RuntimeError:
        print("Sinusoid fit failed.")
        return

    print("Sinusoid plus linear fit parameters:", popt_sine)
    x_fit = np.linspace(
        results_df["Pulse Time (us)"].min(), results_df["Pulse Time (us)"].max(), 200
    )
    y_fit = sinusoid_plus_linear(x_fit, *popt_sine)
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        results_df["Pulse Time (us)"],
        results_df["A"],
        yerr=results_df["A_err"],
        fmt="o",
        capsize=4,
        label="Data (with σ)",
    )

    plt.plot(x_fit, y_fit, "--", label="Sinusoid plus linear fit")
    plt.xlabel("Pulse Time (us)")
    plt.ylabel("Amplitude A")
    plt.title("Fit Parameter A vs. Pulse Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sinusoid_fit_SINGLE_{}.png".format(SIGN_MODE))
    plt.close()

    # Compute reduced chi-square for the sinusoid fit.
    fitted_all = sinusoid_plus_linear(all_pulse_time_vals, *popt_sine)
    residuals = all_A_vals - fitted_all
    chi_sq = np.sum((residuals / all_A_sigma) ** 2)
    dof = len(all_A_vals) - len(popt_sine)
    reduced_chi_sq = chi_sq / dof
    print(f"Reduced chi-square for sinusoid fit: {reduced_chi_sq:.4f}")

    # Find maximum and minimum.
    max_x, max_y = find_max_sinusoid_plus_linear(popt_sine, x_fit)
    print("Sinusoid plus linear fit results:")
    print(f"Peak (Pi/2 pulse) at: {max_x:.4f} us, amplitude = {max_y:.4f}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
