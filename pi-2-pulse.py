#!/usr/bin/env python
"""
pi_pulse_analysis.py

This script:
  1. Loads CSV files from the folder '327/Pulse Calibration Redo' (files matching "^\d+_\d+\.csv$")
  2. Groups the data by pulse time and computes the averaged complex signal.
  3. Performs a simultaneous fit of all groups using a composite model.
  4. Plots the data and fit for each group and saves the figure ("fit_results.png").
  5. Profiles the amplitude A_i for each group in parallel.
     – Each profile plot is saved as "profile_A_<i>.png".
  6. Fits a sinusoid to the fitted amplitudes (with profiled uncertainties),
     saves the sinusoid plot ("sinusoid_fit.png"), and prints the maximum pulse time.
"""

import os
import re
import copy
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend so plots can be saved
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# -------------------------------
# MODEL FUNCTIONS AND HELPERS
# -------------------------------


def model_exp(t, A, T2star, C):
    """Model for a decaying exponential: A * exp(-t/T2star) + C"""
    return A * np.exp(-t / T2star) + C


def model_complex(t, A, T2star, f, phi, C_real, C_imag):
    """
    Returns the complex signal:
      A * exp(-t/T2star) * exp(i*(2*pi*f*t + phi)) + (C_real + i*C_imag)
    """
    return A * np.exp(-t / T2star) * np.exp(1j * (2 * np.pi * f * t + phi)) + (
        C_real + 1j * C_imag
    )


def model_simultaneous(dummy_x, *params, group_t_list):
    """
    Composite model for simultaneous fitting.
    The parameter vector is organized as:
       For each group i: [A_i, f_i, phi_i, C_re_i, C_im_i],
       followed by one global parameter T2star.
    """
    N = len(group_t_list)
    T2star = params[-1]
    y_model = []
    for i in range(N):
        A = params[5 * i + 0]
        f = params[5 * i + 1]
        phi = params[5 * i + 2]
        C_re = params[5 * i + 3]
        C_im = params[5 * i + 4]
        t_arr = group_t_list[i]
        model_vals = model_complex(t_arr, A, T2star, f, phi, C_re, C_im)
        y_model.append(np.concatenate((model_vals.real, model_vals.imag)))
    return np.concatenate(y_model)


def compute_chi_square(params, ydata, sigma, group_t_list):
    """Compute chi-square for the simultaneous model given group_t_list."""
    y_model = model_simultaneous(None, *params, group_t_list=group_t_list)
    chi_sq = np.sum(((ydata - y_model) / sigma) ** 2)
    return chi_sq


def model_simultaneous_fixed_Ai(
    dummy_x, i_fixed, A_fixed_value, *free_params, group_t_list
):
    """
    Modified composite model that fixes amplitude A_i for group i_fixed.
    The free parameters (free_params) are used to reconstruct the full parameter vector.
    """
    full_params = []
    N = len(group_t_list)
    idx_free = 0
    for i in range(N):
        if i == i_fixed:
            full_params.append(A_fixed_value)
            full_params.extend(free_params[idx_free : idx_free + 4])
            idx_free += 4
        else:
            full_params.extend(free_params[idx_free : idx_free + 5])
            idx_free += 5
    full_params.append(free_params[-1])
    return model_simultaneous(dummy_x, *full_params, group_t_list=group_t_list)


def abs_sinusoid(x, a, f, phi):
    """Model for an absolute sinusoid: a * |sin(2*pi*f*x + phi)|"""
    return a * np.abs(np.sin(2 * np.pi * f * x + phi))


def abs_sinusoid_poly(x, a, f, phi, a2, b):
    # Returns: a * |sin(2*pi*f*x + phi)| multiplied by the polynomial (a2*x^2 + b*x)
    return a * np.abs(np.sin(2 * np.pi * f * x + phi)) * (a2 * x**2 + b * x)


def sinusoid_poly(x, a, f, phi, a2, b, c):
    # Returns: a * |sin(2*pi*f*x + phi)| multiplied by the polynomial (a2*x^2 + b*x)
    return a * np.sin(2 * np.pi * f * x + phi) * (a2 * x**2 + b * x) + c


def find_max_abs_sinusoid(popt, x_fit):
    """Find the x-value (pulse time) where the absolute sinusoid is maximized."""
    a, f, phi = popt
    max_x = (np.pi / 2 - phi) / (2 * np.pi * f)
    max_y = abs_sinusoid(max_x, a, f, phi)
    return max_x, max_y


def find_max_sinusoid_poly(popt, x_fit, range=(40, 60)):
    """Find the x-value (pulse time) where the sinusoid polynomial is maximized."""
    a, f, phi, a2, b, c = popt
    # Find the maximum of the entire function, in the range 40-60 us
    x = np.linspace(*range, 10000)
    y = sinusoid_poly(x, a, f, phi, a2, b, c)
    max_idx = np.argmax(y)
    max_x = x[max_idx]
    max_y = y[max_idx]
    return max_x, max_y


def find_min_sinusoid_poly(popt, x_fit, range=(80, 120)):
    """Find the x-value (pulse time) where the sinusoid polynomial is minimized."""
    a, f, phi, a2, b, c = popt
    # Find the minimum of the entire function, in the range 40-60 us
    x = np.linspace(*range, 10000)
    y = sinusoid_poly(x, a, f, phi, a2, b, c)
    min_idx = np.argmin(y)
    min_x = x[min_idx]
    min_y = y[min_idx]
    return min_x, min_y


# This function profiles the amplitude for one group (indexed by i_fixed)
# It scans a grid of A values, performs a local fit with A fixed, computes chi-square,
# saves the chi-square profile plot, and returns the best-fit value and error.
def profile_single_group(
    i_fixed, best_fit_params, perr, group_t_list_local, ydata_sim, sigma_all
):
    # Define a local composite model using the passed group_t_list_local.
    def local_model_simultaneous(dummy_x, *params):
        N = len(group_t_list_local)
        T2star = params[-1]
        y_model = []
        for i in range(N):
            A = params[5 * i + 0]
            f = params[5 * i + 1]
            phi = params[5 * i + 2]
            C_re = params[5 * i + 3]
            C_im = params[5 * i + 4]
            t_arr = group_t_list_local[i]
            model_vals = model_complex(t_arr, A, T2star, f, phi, C_re, C_im)
            y_model.append(np.concatenate((model_vals.real, model_vals.imag)))
        return np.concatenate(y_model)

    # Define a local fixed-A model that uses the local model above.
    def local_model_simultaneous_fixed_Ai(dummy_x, *free_params):
        full_params = []
        N = len(group_t_list_local)
        idx_free = 0
        for j in range(N):
            if j == i_fixed:
                full_params.append(A_fixed_value)
                full_params.extend(free_params[idx_free : idx_free + 4])
                idx_free += 4
            else:
                full_params.extend(free_params[idx_free : idx_free + 5])
                idx_free += 5
        full_params.append(free_params[-1])
        return local_model_simultaneous(dummy_x, *full_params)

    # Get best-fit amplitude and error for group i_fixed.
    A_best = best_fit_params[5 * i_fixed]
    naive_err = perr[5 * i_fixed]
    A_min = A_best - 1.5 * abs(naive_err)
    A_max = A_best + 1.5 * abs(naive_err)
    A_values = np.linspace(A_min, A_max, 21)
    chisq_values = []
    N = len(group_t_list_local)

    for A_fixed_value in A_values:
        # Build initial guess for free parameters (omit the fixed amplitude)
        init_guess_free = []
        for j in range(N):
            if j == i_fixed:
                init_guess_free.extend(best_fit_params[5 * j + 1 : 5 * j + 5])
            else:
                init_guess_free.extend(best_fit_params[5 * j : 5 * j + 5])
        init_guess_free.append(best_fit_params[-1])
        # Create a lambda that calls our local fixed-A model.
        func = lambda dummy_x, *free_params: local_model_simultaneous_fixed_Ai(
            dummy_x, *free_params
        )
        try:
            popt_local, _ = curve_fit(
                func,
                None,
                ydata_sim,
                p0=init_guess_free,
                sigma=sigma_all,
                absolute_sigma=True,
                maxfev=2000,
            )
            # Reconstruct the full parameter vector.
            full_param_vec = []
            idx_free = 0
            for j in range(N):
                if j == i_fixed:
                    full_param_vec.append(A_fixed_value)
                    full_param_vec.extend(popt_local[idx_free : idx_free + 4])
                    idx_free += 4
                else:
                    full_param_vec.extend(popt_local[idx_free : idx_free + 5])
                    idx_free += 5
            full_param_vec.append(popt_local[-1])
            chi_sq_local = compute_chi_square(
                full_param_vec, ydata_sim, sigma_all, group_t_list_local
            )
            chisq_values.append((A_fixed_value, chi_sq_local))
        except RuntimeError:
            chisq_values.append((A_fixed_value, 1e15))
    chisq_values = np.array(chisq_values)
    A_prof_vals = chisq_values[:, 0]
    chisq_prof_vals = chisq_values[:, 1]
    chi_sq_min = np.min(chisq_prof_vals)
    A_at_min = A_prof_vals[np.argmin(chisq_prof_vals)]

    def find_intersection(x, y, y0):
        for k in range(len(x) - 1):
            if (y[k] - y0) * (y[k + 1] - y0) < 0:
                slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
                return x[k] + (y0 - y[k]) / slope
        return None

    target = chi_sq_min + 1
    left_mask = A_prof_vals <= A_at_min
    right_mask = A_prof_vals >= A_at_min
    A_left = (
        find_intersection(
            A_prof_vals[left_mask][::-1], chisq_prof_vals[left_mask][::-1], target
        )
        if np.any(left_mask)
        else None
    )
    A_right = (
        find_intersection(A_prof_vals[right_mask], chisq_prof_vals[right_mask], target)
        if np.any(right_mask)
        else None
    )
    if (A_left is not None) and (A_right is not None):
        err_left = A_at_min - A_left
        err_right = A_right - A_at_min
        A_err = 0.5 * (err_left + err_right)
    else:
        err_left = err_right = np.nan
        A_err = np.nan

    plt.figure(figsize=(8, 4))
    plt.plot(A_prof_vals, chisq_prof_vals, "o-")
    plt.axhline(chi_sq_min + 1, color="red", linestyle="--", label="1-sigma")
    plt.axvline(A_at_min, color="green", linestyle="--", label="Best A_i")
    plt.xlabel(f"A_{i_fixed}")
    plt.ylabel("Chi-square")
    plt.title(f"Profiled Chi-square for A_{i_fixed}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"profile_A_{i_fixed}.png")
    plt.close()

    result = {
        "A_prof_best": A_at_min,
        "A_minus": err_left,
        "A_plus": err_right,
        "A_err_est": A_err,
        "chi_sq_min_profile": chi_sq_min,
    }
    return i_fixed, result


# -------------------------------
# MAIN SCRIPT
# -------------------------------


def main():
    # ----- Data Loading -----
    folder_name = "327/Pulse Calibration Redo"
    file_names = [f for f in os.listdir(folder_name) if re.match(r"^\d+_\d+\.csv$", f)]
    data = []
    pulse_times = []
    for file_name in file_names:
        # Assumes file name format: "<pulse_time>_<...>.csv"
        pulse_time = int(file_name.split("_")[0])
        df = pd.read_csv(
            os.path.join(folder_name, file_name), header=None, names=["t", "CH1", "CH2"]
        )
        df = df.dropna()
        df = df[(df["t"] > 0.001) & (df["t"] < 0.004)]
        data.append(df)
        pulse_times.append(pulse_time)

    # ----- Group Data by Pulse Time -----
    grouped_data = defaultdict(list)
    for df, pt in zip(data, pulse_times):
        grouped_data[pt].append(df)
    group_data_list = []
    pulse_times_sorted = sorted(grouped_data.keys())
    for pt in pulse_times_sorted:
        dfs = grouped_data[pt]
        signals = []
        t_common = None
        for df in dfs:
            t = df["t"].values
            if t_common is None:
                t_common = t
            # Create complex signal from CH1 and CH2 and smooth using lowpass Butterworth.
            b, a = butter(3, 5000, btype="lowpass", fs=500000)
            sig = filtfilt(b, a, df["CH1"].values + 1j * df["CH2"].values)
            signals.append(sig)
        signals = np.array(signals)
        avg_real = np.mean(np.real(signals), axis=0)
        avg_imag = np.mean(np.imag(signals), axis=0)
        std_real = (
            np.std(np.real(signals), axis=0, ddof=1)
            if signals.shape[0] > 1
            else np.zeros_like(avg_real)
        )
        std_imag = (
            np.std(np.imag(signals), axis=0, ddof=1)
            if signals.shape[0] > 1
            else np.zeros_like(avg_imag)
        )
        avg_signal = avg_real + 1j * avg_imag
        group_data_list.append(
            {
                "pulse_time": pt,
                "t": t_common,
                "avg_signal": avg_signal,
                "std_real": std_real,
                "std_imag": std_imag,
            }
        )

    # ----- Build Composite Data for Simultaneous Fitting -----
    ydata_list = []
    for group in group_data_list:
        y_group = np.concatenate((group["avg_signal"].real, group["avg_signal"].imag))
        ydata_list.append(y_group)
    ydata_sim = np.concatenate(ydata_list)
    # Build the list of time arrays (one per group)
    group_t_list = [group["t"] for group in group_data_list]

    # ----- Check if fit and profiling CSV files exist -----
    fit_results_csv = "fit_results.csv"
    profiled_results_csv = "profiled_results.csv"
    if os.path.exists(fit_results_csv) and os.path.exists(profiled_results_csv):
        results_df = pd.read_csv(fit_results_csv)
        profiled_df = pd.read_csv(profiled_results_csv)
        print(
            "Loaded previously saved fit and profiling results. Skipping heavy computations."
        )
        heavy_computations = False
    else:
        heavy_computations = True

    if heavy_computations:

        # ----- Simultaneous Fit -----
        # Build an initial guess p0.
        p0 = []
        for group in group_data_list:
            avg_signal = group["avg_signal"]
            # Guess: A=15, f=-2200 Hz, phi=phase of first point, C_real=-2, C_imag=0.
            p0.extend([15, -2200, np.angle(avg_signal[0]), -2, 0])
        p0.append(0.00063)  # Global T2star guess.
        sigma_all = np.concatenate(
            [group["std_real"] for group in group_data_list]
            + [group["std_imag"] for group in group_data_list]
        )
        # Use a lambda to pass group_t_list into the composite model.
        try:
            popt, pcov = curve_fit(
                lambda dummy_x, *params: model_simultaneous(
                    dummy_x, *params, group_t_list=group_t_list
                ),
                None,
                ydata_sim,
                p0=p0,
                sigma=sigma_all,
                absolute_sigma=True,
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))
            fit_success = True
        except RuntimeError:
            popt = [np.nan] * len(p0)
            perr = [np.nan] * len(p0)
            fit_success = False

        global_T2star = popt[-1]
        all_fits = []
        N_groups = len(group_t_list)
        for i, group in enumerate(group_data_list):
            A = popt[5 * i + 0]
            f_val = popt[5 * i + 1]
            phi = popt[5 * i + 2]
            C_re = popt[5 * i + 3]
            C_im = popt[5 * i + 4]
            t_arr = group["t"]
            fitted_curve = model_complex(
                t_arr, A, global_T2star, f_val, phi, C_re, C_im
            )
            group_params = popt[5 * i : 5 * i + 5]
            group_result = list(group_params) + [global_T2star]
            all_fits.append(
                {
                    "pulse_time": group["pulse_time"],
                    "t": t_arr,
                    "avg_real": np.real(group["avg_signal"]),
                    "avg_imag": np.imag(group["avg_signal"]),
                    "std_real": group["std_real"],
                    "std_imag": group["std_imag"],
                    "fit_real": fitted_curve.real,
                    "fit_imag": fitted_curve.imag,
                    "popt": group_result,
                    "perr": perr[5 * i : 5 * i + 5].tolist() + [perr[-1]],
                    "fit_success": fit_success,
                }
            )

        # ----- Plot and Save Fit Results -----
        num_fits = len(all_fits)
        fig, axs = plt.subplots(
            nrows=num_fits, ncols=2, figsize=(12, 3 * num_fits), sharex=True
        )
        if num_fits == 1:
            axs = np.array([axs])
        for i, fit in enumerate(all_fits):
            DOWNSAMPLE_RATE = 25
            axs[i, 0].errorbar(
                fit["t"][::DOWNSAMPLE_RATE],
                fit["avg_real"][::DOWNSAMPLE_RATE],
                yerr=fit["std_real"][::DOWNSAMPLE_RATE],
                fmt="o",
                alpha=0.6,
                label="Data Real",
            )
            if fit["fit_success"]:
                axs[i, 0].plot(
                    fit["t"], fit["fit_real"], "-", alpha=0.6, label="Fit Real"
                )
            else:
                axs[i, 0].text(
                    0.05, 0.9, "Fit Failed", transform=axs[i, 0].transAxes, color="red"
                )
            axs[i, 0].set_ylabel("Real Part")
            axs[i, 0].set_title(f'Pulse {fit["pulse_time"]} (Real)')
            axs[i, 0].legend(fontsize="small")
            axs[i, 0].grid(True)

            axs[i, 1].errorbar(
                fit["t"][::DOWNSAMPLE_RATE],
                fit["avg_imag"][::DOWNSAMPLE_RATE],
                yerr=fit["std_imag"][::DOWNSAMPLE_RATE],
                fmt="o",
                alpha=0.6,
                label="Data Imag",
            )
            if fit["fit_success"]:
                axs[i, 1].plot(
                    fit["t"], fit["fit_imag"], "-", alpha=0.6, label="Fit Imag"
                )
            else:
                axs[i, 1].text(
                    0.05, 0.9, "Fit Failed", transform=axs[i, 1].transAxes, color="red"
                )
            axs[i, 1].set_ylabel("Imaginary Part")
            axs[i, 1].set_title(f'Pulse {fit["pulse_time"]} (Imaginary)')
            axs[i, 1].legend(fontsize="small")
            axs[i, 1].grid(True)
        for ax in axs[-1, :]:
            ax.set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig("fit_results.png")
        plt.close()

        # Build summary DataFrame for amplitude A
        results_list = []
        for fit in all_fits:
            if fit["fit_success"]:
                A_val = np.abs(fit["popt"][0])
                A_err_val = fit["perr"][0]
                results_list.append(
                    {
                        "Pulse Time (us)": fit["pulse_time"],
                        "A": A_val,
                        "A_err": A_err_val,
                    }
                )
        results_df = pd.DataFrame(results_list).sort_values("Pulse Time (us)")
        if fit_success:
            print(f"Global T2star: {global_T2star*1e3:.4f} ± {perr[-1]*1e3:.4f} ms")
        for i, fit in enumerate(all_fits):
            if fit["fit_success"]:
                print(
                    f"Pulse Time {fit['pulse_time']} us: A={fit['popt'][0]:.4f} ± {fit['perr'][0]:.4f}, "
                    f"f={fit['popt'][1]:.4f} ± {fit['perr'][1]:.4f}, "
                    f"phi={fit['popt'][2]:.4f} ± {fit['perr'][2]:.4f}, "
                    f"C_real={fit['popt'][3]:.4f} ± {fit['perr'][3]:.4f}, "
                    f"C_imag={fit['popt'][4]:.4f} ± {fit['perr'][4]:.4f}, "
                    f"T2star={fit['popt'][-1]*1e3:.4f} ± {fit['perr'][-1]*1e3:.4f} ms"
                )
            else:
                print("Fit failed for some groups.")

        # ----- Save Fit Results to CSV -----
        fit_results_csv = "fit_results.csv"
        if not os.path.exists(fit_results_csv):
            results_df.to_csv(fit_results_csv, index=False)
        else:
            # If CSV exists, load it so that subsequent runs use the saved results.
            results_df = pd.read_csv(fit_results_csv)
            print("Loaded fit results from CSV.")

        # ----- Parallelized Profiling of A_i -----
        profiled_errors = {}
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    profile_single_group,
                    i,
                    copy.copy(popt),
                    perr,
                    group_t_list,  # pass list of time arrays
                    ydata_sim,
                    sigma_all,
                )
                for i in range(len(group_t_list))
            ]
            for future in as_completed(futures):
                i_fixed, res = future.result()
                profiled_errors[f"A_{i_fixed}"] = res
        print("Profiled 1-sigma errors for each A_i:")
        for key in sorted(profiled_errors.keys()):
            info = profiled_errors[key]
            print(
                f"{key}: best={info['A_prof_best']:.5g}, -err={info['A_minus']:.5g}, +err={info['A_plus']:.5g}, avg_err={info['A_err_est']:.5g}"
            )

        # Add profiled error to results_df
        results_df["A_err_profiled"] = np.nan
        for i, row in results_df.iterrows():
            key = f"A_{i}"
            if key in profiled_errors:
                results_df.at[i, "A_err_profiled"] = profiled_errors[key]["A_err_est"]

        # Convert the profiled_errors dictionary to a DataFrame
        profiled_list = []
        for key, val in profiled_errors.items():
            # Extract group index from key "A_i"
            group_index = int(key.split("_")[1])
            profiled_list.append(
                {
                    "Group": group_index,
                    "A_prof_best": val["A_prof_best"],
                    "A_minus": val["A_minus"],
                    "A_plus": val["A_plus"],
                    "A_err_est": val["A_err_est"],
                    "chi_sq_min_profile": val["chi_sq_min_profile"],
                }
            )
        profiled_df = pd.DataFrame(profiled_list).sort_values("Group")

        # ----- Save Profiled Results to CSV -----
        profiled_results_csv = "profiled_results.csv"
        if not os.path.exists(profiled_results_csv):
            profiled_df.to_csv(profiled_results_csv, index=False)
        else:
            profiled_df = pd.read_csv(profiled_results_csv)
            print("Loaded profiled results from CSV.")

        # ----- Load Previously Saved Fit and Profiled Results if available -----
        if os.path.exists(fit_results_csv) and os.path.exists(profiled_results_csv):
            results_df = pd.read_csv(fit_results_csv)
            profiled_df = pd.read_csv(profiled_results_csv)
            print("Loaded previously saved fit and profiling results.")
        else:
            print("No saved CSV found; using current computed results.")

    else:
        print(
            "Skipping heavy computations; loading fit and profiling results from CSV."
        )
        results_df = pd.read_csv(fit_results_csv)
        profiled_df = pd.read_csv(profiled_results_csv)

    # ----- Sinusoid Fit to A vs. Pulse Time -----
    # Parameters: a, f, phi, a2, b
    # Equation: a * |sin(2*pi*f*x + phi)| * (a2*x^2 + b*x + c)
    # p0_sin = [3e0, 5e-5, 0, 0, 1e-2, 10]
    p0_sin = [3e0, 5e-3, 0, 1e-5, 1e-2, 11]
    popt_sin, pcov_sin = curve_fit(
        sinusoid_poly,
        results_df["Pulse Time (us)"],
        results_df["A"],
        sigma=profiled_df["A_err_est"],
        absolute_sigma=True,
        p0=p0_sin,
        maxfev=100000,
    )
    print(popt_sin)
    x_fit = np.linspace(
        results_df["Pulse Time (us)"].min(), results_df["Pulse Time (us)"].max(), 200
    )
    y_fit = sinusoid_poly(x_fit, *popt_sin)
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        results_df["Pulse Time (us)"],
        results_df["A"],
        yerr=profiled_df["A_err_est"],
        fmt="o",
        capsize=4,
        label="Data (profiled σ)",
    )
    plt.plot(x_fit, y_fit, "--", label="Absolute Sinusoid Fit")
    plt.xlabel("Pulse Time (us)")
    plt.ylabel("Amplitude A")
    plt.title("Fit Parameter A vs. Pulse Time (±1σ from Profiled Errors)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sinusoid_fit.png")
    plt.close()

    # Reduced chi-square of the sinusoid fit
    residuals = results_df["A"] - sinusoid_poly(
        results_df["Pulse Time (us)"], *popt_sin
    )
    chi_square = np.sum((residuals / profiled_df["A_err_est"]) ** 2)
    dof = len(results_df) - len(popt_sin)
    reduced_chi_square = chi_square / dof
    print(f"Reduced Chi-square for sinusoid fit: {reduced_chi_square:.4f}")

    # Find maximum of the absolute sinusoid fit.
    max_x, max_y = find_max_sinusoid_poly(popt_sin, x_fit)
    print("Pi/2 Pulse Time:")
    print(f"Max x (Pulse Time): {max_x:.4f} us")
    print(f"Max y (Amplitude): {max_y:.4f}")

    # Find minimum of the sinusoid polynomial fit.
    min_x, min_y = find_min_sinusoid_poly(popt_sin, x_fit)
    print("Pi Pulse Time:")
    print(f"Min x (Pulse Time): {min_x:.4f} us")
    print(f"Min y (Amplitude): {min_y:.4f}")


if __name__ == "__main__":
    main()
