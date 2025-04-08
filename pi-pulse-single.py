#!/usr/bin/env python
"""
pi_pulse_single.py
==================

Purpose
-------
Analyze only one of the two sequences: either "+π/2" or "–π/2",
based on the SIGN_MODE parameter below. It loads all CSV files
in '327/Pi Pulse Calibration/' that match the naming:

    "<tau>_<plus|minus>_<rep>.csv"

...but then filters so that only those with `sign == SIGN_MODE`
(plus or minus) are used. The script then:

1. Groups the data by τ (the candidate π-pulse length).
2. Averages repeats for each τ.
3. Performs a simultaneous fit to a complex decay model with
   a *global* T₂* across all τ groups.
4. Profiles the amplitudes A_i for each group to get robust
   error bars from χ² profiling.
5. Fits amplitude vs. τ using a flexible "sinusoid_poly" shape.
6. Locates either the maximum (for plus) or the minimum (for minus).
7. Prints the final time for that extremum (peak or valley).

You can therefore run:

    SIGN_MODE = 'plus'    # to analyze the +π/2 dataset only

or

    SIGN_MODE = 'minus'   # to analyze the –π/2 dataset only

Implementation Details
----------------------
* Data location: '327/Pi Pulse Calibration/<tau>_<sign>_<rep>.csv'
* Plot naming, CSV output, etc., are analogous to pi-2-pulse.py.
* Heavy computations are skipped on subsequent runs if
  "fit_results_SINGLE_<sign>.csv" is already present.
* You get a single final time – either τ_peak(+π/2) or τ_valley(–π/2).

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
SIGN_MODE = "plus"  # or "minus"
# --------------------------------------------------------


# --------------------------------------------------------
# 1) MODELS (same as your pi-2-pulse or pi-pulse scripts)
# --------------------------------------------------------
def model_complex(t, A, T2star, f, phi, C_re, C_im):
    return A * np.exp(-t / T2star) * np.exp(1j * (2 * np.pi * f * t + phi)) + (
        C_re + 1j * C_im
    )


def model_simultaneous(dummy_x, *params, group_t_list):
    """
    Composite fit function.  For each group i, we have parameters:
       [A_i, f_i, phi_i, C_re_i, C_im_i],
    plus one global T2star at the end.
    """
    N = len(group_t_list)
    T2star = params[-1]
    out = []
    for i in range(N):
        A, f, phi, Cre, Cim = params[5 * i : 5 * i + 5]
        comp = model_complex(group_t_list[i], A, T2star, f, phi, Cre, Cim)
        # Flatten from complex to real-imag pairs:
        out.append(np.concatenate(comp.view(np.float64)))
    return np.concatenate(out)


def sinusoid_poly(x, a, f, phi, a2, b, c):
    """
    Flexible function to capture amplitude vs. tau:
      A(tau) = a * sin(2π f tau + phi) * (a2*tau^2 + b*tau) + c
    Enough freedom to fit minor distortions or offsets.
    """
    return a * np.sin(2 * np.pi * f * x + phi) * (a2 * x**2 + b * x) + c


def extremum_of_fit(popt, kind="max", search=(20, 200)):
    """
    Evaluate the sinusoid_poly over a grid (search range in μs),
    then find either the maximum or the minimum.
    """
    xx = np.linspace(*search, 5000)
    yy = sinusoid_poly(xx, *popt)
    idx = np.argmax(yy) if kind == "max" else np.argmin(yy)
    return xx[idx], yy[idx]


# --------------------------------------------------------
# 2) PROFILE HELPER – unchanged from older scripts
# --------------------------------------------------------
def profile_Ai(i_fixed, popt_best, perr, group_t, ydata, sigma_all, folder="profiles"):
    """
    χ²-profile amplitude A_i around the best-fit solution, to get
    a robust 1-σ bound.  Saves a profile plot in `folder/`.
    """
    os.makedirs(folder, exist_ok=True)
    A_best = popt_best[5 * i_fixed]
    dA = 1.5 * abs(perr[5 * i_fixed])
    A_grid = np.linspace(A_best - dA, A_best + dA, 21)
    chi_vals = []

    # We'll define a "fixed Ai" model by re-building the parameter vector
    def fixed_model(dummy, *free):
        N = len(group_t)
        full = []
        idx_free = 0
        for j in range(N):
            if j == i_fixed:
                full.append(Ai)
                # next 4 from free are [f, phi, C_re, C_im]
                full.extend(free[idx_free : idx_free + 4])
                idx_free += 4
            else:
                full.extend(free[idx_free : idx_free + 5])
                idx_free += 5
        full.append(free[-1])  # the global T2star
        return model_simultaneous(None, *full, group_t_list=group_t)

    for Ai in A_grid:
        # build p0 without the amplitude for group i_fixed
        p0 = []
        N = len(group_t)
        for j in range(N):
            if j == i_fixed:
                p0.extend(popt_best[5 * j + 1 : 5 * j + 5])
            else:
                p0.extend(popt_best[5 * j : 5 * j + 5])
        p0.append(popt_best[-1])  # T2*

        try:
            popt_local, _ = curve_fit(
                lambda d, *fp: fixed_model(d, *fp),
                None,
                ydata,
                p0=p0,
                sigma=sigma_all,
                absolute_sigma=True,
                maxfev=4000,
            )
            # Evaluate χ²
            # reconstruct the full param vector with Ai
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

    # Find 1-sigma crossing (Δχ² = 1)
    # We'll try simple linear interpolation on left and right sides
    try:
        # left side
        left_slice = slice(None, best_idx)
        A_left = np.interp(
            chi_min + 1, chi_vals[left_slice][::-1], A_grid[left_slice][::-1]
        )
        # right side
        right_slice = slice(best_idx, None)
        A_right = np.interp(chi_min + 1, chi_vals[right_slice], A_grid[right_slice])
        err = 0.5 * ((A_prof_best - A_left) + (A_right - A_prof_best))
    except ValueError:
        err = np.nan

    # Plot & save
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
# 3) MAIN
# --------------------------------------------------------
def main():
    base = "327/Pi Pulse Calibration"
    pattern = re.compile(r"^(?P<tau>\d+)_(?P<sign>plus|minus)_\d+\.csv$")
    data_groups = defaultdict(list)

    # Read all files but keep only those that match SIGN_MODE.
    for fname in os.listdir(base):
        m = pattern.match(fname)
        if not m:
            continue
        sign = m.group("sign")  # 'plus' or 'minus'
        if sign != SIGN_MODE:
            continue  # skip any file that doesn't match our chosen sign
        tau = int(m.group("tau"))
        df = pd.read_csv(
            os.path.join(base, fname), header=None, names=["t", "CH1", "CH2"]
        ).dropna()
        df = df[(df["t"] > 0.001) & (df["t"] < 0.004)]
        data_groups[tau].append(df)

    # Combine repeats (if any) for each tau
    group_list = []
    for tau in sorted(data_groups.keys()):
        dfs = data_groups[tau]
        t_vals = dfs[0]["t"].values
        # filter
        b, a = butter(3, 5000, fs=500000, btype="low")
        sigs = []
        for df in dfs:
            raw = df["CH1"].values + 1j * df["CH2"].values
            flt = filtfilt(b, a, raw)
            sigs.append(flt)
        sigs = np.array(sigs)  # shape: (#repeats, #points)
        avg = sigs.mean(axis=0)
        std = sigs.std(axis=0, ddof=1) if len(sigs) > 1 else np.zeros_like(avg)
        group_list.append(dict(tau=tau, t=t_vals, avg=avg, std=std))

    # Build composite arrays
    # Each group contributes real and imag data to the global fit
    ydata = np.concatenate([g["avg"].view(np.float64) for g in group_list])
    sigma_all = np.concatenate([g["std"].view(np.float64) for g in group_list])
    sigma_all[sigma_all == 0] = 1.0  # to avoid zero division
    t_lists = [g["t"] for g in group_list]

    # Decide on a filename based on plus/minus
    results_file = f"fit_results_SINGLE_{SIGN_MODE}.csv"
    needs_fit = not os.path.exists(results_file)

    if needs_fit:
        # ---- Simultaneous Fit ----
        p0 = []
        for g in group_list:
            # same guesses as usual
            # [A, f, phi, Cre, Cim]
            A_guess = 15
            freq_guess = -2000
            phi_guess = np.angle(g["avg"][0])
            Cre_guess = -2
            Cim_guess = 0
            p0.extend([A_guess, freq_guess, phi_guess, Cre_guess, Cim_guess])
        p0.append(0.0006)  # global T2*

        try:
            popt, pcov = curve_fit(
                lambda d, *pp: model_simultaneous(d, *pp, group_t_list=t_lists),
                None,
                ydata,
                p0=p0,
                sigma=sigma_all,
                absolute_sigma=True,
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError:
            popt = np.array([np.nan] * len(p0))
            perr = np.array([np.nan] * len(p0))

        # ---- Profile A_i ----
        results = []
        with ProcessPoolExecutor() as exe:
            futs = []
            for i in range(len(group_list)):
                futs.append(
                    exe.submit(profile_Ai, i, popt, perr, t_lists, ydata, sigma_all)
                )
            for fu in as_completed(futs):
                idx, val, err = fu.result()
                results.append(dict(group_idx=idx, A_prof_best=val, A_err=err))
        # Merge back with group info
        outrows = []
        for r in results:
            g = group_list[r["group_idx"]]
            outrows.append(dict(tau=g["tau"], A=r["A_prof_best"], A_err=r["A_err"]))
        outrows = sorted(outrows, key=lambda x: x["tau"])
        df_out = pd.DataFrame(outrows)
        df_out.to_csv(results_file, index=False)
    else:
        df_out = pd.read_csv(results_file)

    # Now fit amplitude vs. tau with sinusoid_poly
    if len(df_out) < 3:
        print(
            f"Not enough data points (only {len(df_out)}) for sign={SIGN_MODE}. Exiting."
        )
        return

    p0_sine = [3, 5e-3, 0, 1e-5, 1e-2, 10]
    popt_sine, _ = curve_fit(
        sinusoid_poly,
        df_out["tau"],
        df_out["A"],
        sigma=df_out["A_err"],
        absolute_sigma=True,
        p0=p0_sine,
        maxfev=20000,
    )

    # Depending on sign, we either want the maximum (plus) or minimum (minus).
    if SIGN_MODE == "plus":
        t_ext, A_ext = extremum_of_fit(popt_sine, kind="max")
        print(f"Sign=plus => Found peak at τ = {t_ext:.2f} us, amplitude = {A_ext:.3f}")
    else:
        t_ext, A_ext = extremum_of_fit(popt_sine, kind="min")
        print(
            f"Sign=minus => Found valley at τ = {t_ext:.2f} us, amplitude = {A_ext:.3f}"
        )

    # Plot
    xx = np.linspace(df_out["tau"].min(), df_out["tau"].max(), 300)
    yy = sinusoid_poly(xx, *popt_sine)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        df_out["tau"],
        df_out["A"],
        yerr=df_out["A_err"],
        fmt="o",
        label=f"{SIGN_MODE} data",
    )
    plt.plot(xx, yy, "--", label="sinusoid fit")
    plt.axvline(t_ext, color="k", ls=":")
    plt.title(f"{SIGN_MODE.upper()}: amplitude vs. τ  (extremum at {t_ext:.2f} µs)")
    plt.xlabel("Candidate π-pulse length (µs)")
    plt.ylabel("Fitted amplitude A")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"single_{SIGN_MODE}_summary.png")
    plt.close()

    print("Analysis complete. Fit results saved to:", results_file)
    print("Plot saved to: single_{SIGN_MODE}_summary.png")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
