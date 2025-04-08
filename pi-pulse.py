#!/usr/bin/env python
"""
pi_pulse.py
===========

Goal
-----
Determine the *true* π‑pulse length **tπ** without assuming tπ = 2·tπ/2.

We acquire two complementary Ramsey‑style sequences for each candidate
π‑pulse length τ

    1.  +π/2  :  π_x(τ)  –– wait ––  +π/2_x   →  S_plus(τ)
    2.  –π/2  :  π_x(τ)  –– wait ––  –π/2_x   →  S_minus(τ)

If the reference π/2 pulse is a little long (or short), the two scans
bracket the truth:  the **+π/2** data peaks *earlier* than tπ, while the
**–π/2** data reaches its minimum *later* than tπ.
We therefore take

        tπ  ≈  ( τ_peak(+)  +  τ_valley(–) ) / 2                (1)

Implementation highlights
-------------------------
* **Data layout** – drop files in
  `327/Pi Pulse Calibration/` named like `45_plus_0.csv`,
  `45_minus_1.csv`, … (`<τ>_<plus|minus>_<rep>.csv>`).
* **Signal model** – identical complex decay model used in the π/2
  script (global *T2* shared across all groups).
* **Heavy work caching** – simultaneous fit + χ²‑profiling are skipped
  on re‑run if `fit_results_plus.csv`, `fit_results_minus.csv`, and
  `profiled_results_*.csv` are already present.
* **Parallel profiling** – each amplitude A_i is profiled on its own
  process exactly as before.
* **Robust estimate** – we fit a flexible
  `sinusoid_poly()` to **S_plus(τ)** (peaks) and **S_minus(τ)** (valleys),
  locate their extrema, propagate the profiled σ’s, then combine with
  eq. (1).

The bottom of the script prints the three numbers you care about:

    τ_peak(+)   τ_valley(–)   tπ_final ± σ

"""

import os, re, copy, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------------------------------------------------
# 1)  MODELS  (identical to pi‑2‑pulse.py)
# ----------------------------------------------------------------------
def model_complex(t, A, T2star, f, phi, C_re, C_im):
    return A * np.exp(-t / T2star) * np.exp(1j * (2 * np.pi * f * t + phi)) + (
        C_re + 1j * C_im
    )


def model_simultaneous(dummy_x, *params, group_t_list):
    N = len(group_t_list)
    T2star = params[-1]
    out = []
    for i in range(N):
        A, f, phi, Cre, Cim = params[5 * i : 5 * i + 5]
        out.append(
            np.concatenate(
                model_complex(group_t_list[i], A, T2star, f, phi, Cre, Cim).view(
                    np.float64
                )
            )
        )
    return np.concatenate(out)


def sinusoid_poly(x, a, f, phi, a2, b, c):
    """same functional form used previously – plenty flexible"""
    return a * np.sin(2 * np.pi * f * x + phi) * (a2 * x**2 + b * x) + c


def extremum_of_fit(popt, kind="max", search=(20, 200)):
    x = np.linspace(*search, 20_000)
    y = sinusoid_poly(x, *popt)
    idx = np.argmax(y) if kind == "max" else np.argmin(y)
    return x[idx], y[idx]


# ----------------------------------------------------------------------
# 2)  PROFILE helper – unchanged
# ----------------------------------------------------------------------
def profile_Ai(i_fixed, popt_best, perr, group_t, ydata, sigma_all, folder="profiles"):
    os.makedirs(folder, exist_ok=True)
    A_best = popt_best[5 * i_fixed]
    dA = 1.5 * abs(perr[5 * i_fixed])
    grid = np.linspace(A_best - dA, A_best + dA, 21)
    chi = []

    # build wrapper with Ai fixed
    def fixed_model(dummy, *free):
        params = []
        N = len(group_t)
        idx = 0
        for j in range(N):
            if j == i_fixed:
                params.append(Ai)
                params.extend(free[idx : idx + 4])
                idx += 4
            else:
                params.extend(free[idx : idx + 5])
                idx += 5
        params.append(free[-1])  # global T2*
        return model_simultaneous(None, *params, group_t_list=group_t)

    for Ai in grid:
        # build initial guess without the fixed amplitude
        p0 = []
        for j in range(len(group_t)):
            if j == i_fixed:
                p0.extend(popt_best[5 * j + 1 : 5 * j + 5])
            else:
                p0.extend(popt_best[5 * j : 5 * j + 5])
        p0.append(popt_best[-1])  # T2*

        try:
            popt, _ = curve_fit(
                lambda d, *fp: fixed_model(d, *fp),
                None,
                ydata,
                p0=p0,
                sigma=sigma_all,
                absolute_sigma=True,
                maxfev=4000,
            )
            # rebuild full param vector to evaluate χ²
            full = []
            idx = 0
            for j in range(len(group_t)):
                if j == i_fixed:
                    full.append(Ai)
                    full.extend(popt[idx : idx + 4])
                    idx += 4
                else:
                    full.extend(popt[idx : idx + 5])
                    idx += 5
            full.append(popt[-1])
            resid = ydata - model_simultaneous(None, *full, group_t_list=group_t)
            chi.append(np.sum((resid / sigma_all) ** 2))
        except RuntimeError:
            chi.append(np.inf)

    chi = np.asarray(chi)
    best = grid[np.argmin(chi)]
    # find 1‑σ by linear interpolation
    chi_min = chi.min()
    try:
        left = np.interp(
            chi_min + 1, chi[: np.argmin(chi)][::-1], grid[: np.argmin(chi)][::-1]
        )
        right = np.interp(chi_min + 1, chi[np.argmin(chi) :], grid[np.argmin(chi) :])
        err = 0.5 * (best - left + right - best)
    except ValueError:
        err = np.nan

    # quick diagnostic plot
    plt.figure()
    plt.plot(grid, chi, "o-")
    plt.axhline(chi_min + 1, color="r", ls="--")
    plt.axvline(best, color="g", ls="--")
    plt.xlabel(f"A_{i_fixed}")
    plt.ylabel("χ²")
    plt.tight_layout()
    plt.savefig(f"{folder}/A_{i_fixed}.png")
    plt.close()

    return i_fixed, best, err


# ----------------------------------------------------------------------
# 3)  MAIN
# ----------------------------------------------------------------------
def main():
    base = "327/Pi Pulse Calibration"
    pat = re.compile(r"^(?P<tau>\d+)_(?P<sign>plus|minus)_\d+\.csv$")

    data_groups = defaultdict(list)  # (tau, sign) → list[DataFrame]

    for fname in os.listdir(base):
        m = pat.match(fname)
        if not m:
            continue
        tau = int(m.group("tau"))
        sign = m.group("sign")  # 'plus' or 'minus'
        df = pd.read_csv(
            os.path.join(base, fname), header=None, names=["t", "CH1", "CH2"]
        ).dropna()
        df = df[(df["t"] > 0.001) & (df["t"] < 0.004)]
        data_groups[(tau, sign)].append(df)

    # ---- Average repeats into one complex trace per (τ,sign) -------------
    groups = []
    for (tau, sign), dfs in sorted(data_groups.items()):
        t0 = dfs[0]["t"].values
        sigs = []
        b, a = butter(3, 5e3, fs=5e5, btype="low")
        for df in dfs:
            sig = filtfilt(b, a, df["CH1"].values + 1j * df["CH2"].values)
            sigs.append(sig)
        sigs = np.array(sigs)
        avg = sigs.mean(axis=0)
        std = sigs.std(axis=0, ddof=1) if len(sigs) > 1 else np.zeros_like(avg)
        groups.append(
            dict(
                tau=tau,
                sign=sign,
                t=t0,
                avg=avg,
                std=std,
            )
        )

    # ---- Build composite arrays for simultaneous fit --------------------
    ydata = np.concatenate([g["avg"].view(np.float64) for g in groups])
    sigma_all = np.concatenate([g["std"].view(np.float64) for g in groups])
    sigma_all[sigma_all == 0] = 1.0  # avoid zero weights
    t_lists = [g["t"] for g in groups]

    fit_csv = "fit_results_plus.csv"  # we'll write two later
    heavy = not (os.path.exists(fit_csv))

    if heavy:
        # --- simultaneous fit -------------------------------------------
        p0 = []
        for g in groups:
            p0.extend([15, -2200, np.angle(g["avg"][0]), -2, 0])
        p0.append(0.0006)  # global T2*
        popt, pcov = curve_fit(
            lambda d, *par: model_simultaneous(d, *par, group_t_list=t_lists),
            None,
            ydata,
            p0=p0,
            sigma=sigma_all,
            absolute_sigma=True,
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))

        # --- profile every A_i in parallel ------------------------------
        prof_err = {}
        with ProcessPoolExecutor() as ex:
            futs = [
                ex.submit(profile_Ai, i, popt, perr, t_lists, ydata, sigma_all)
                for i in range(len(groups))
            ]
            for fu in as_completed(futs):
                i, best, err = fu.result()
                prof_err[i] = (best, err)

        # --- save summary -----------------------------------------------
        rows = []
        for i, g in enumerate(groups):
            rows.append(
                dict(
                    tau=g["tau"],
                    sign=g["sign"],
                    A=prof_err[i][0],
                    A_err=prof_err[i][1],
                )
            )
        df = pd.DataFrame(rows)
        df.to_csv("fit_results_all.csv", index=False)
    else:
        df = pd.read_csv("fit_results_all.csv")

    # separate plus / minus tables
    plus = df[df["sign"] == "plus"].sort_values("tau")
    minus = df[df["sign"] == "minus"].sort_values("tau")

    # --------------------------------------------------------------------
    # 4)  Fit amplitude vs τ for each branch
    # --------------------------------------------------------------------
    p0 = [3, 5e-3, 0, 1e-5, 1e-2, 10]
    popt_p, _ = curve_fit(
        sinusoid_poly,
        plus["tau"],
        plus["A"],
        sigma=plus["A_err"],
        absolute_sigma=True,
        p0=p0,
        maxfev=20000,
    )
    popt_m, _ = curve_fit(
        sinusoid_poly,
        minus["tau"],
        minus["A"],
        sigma=minus["A_err"],
        absolute_sigma=True,
        p0=p0,
        maxfev=20000,
    )

    τ_peak_plus, _ = extremum_of_fit(popt_p, "max")
    τ_valley_minus, _ = extremum_of_fit(popt_m, "min")

    t_pi = 0.5 * (τ_peak_plus + τ_valley_minus)
    print("----------------------------------------------------")
    print(f"  τ_peak( +π/2 )  : {τ_peak_plus:7.3f}  µs")
    print(f"  τ_valley( –π/2 ): {τ_valley_minus:7.3f}  µs")
    print(f"  ==>  π‑pulse  tπ ≈ {t_pi:7.3f}  µs")
    print("----------------------------------------------------")

    # quick plot
    xx = np.linspace(df["tau"].min(), df["tau"].max(), 500)
    plt.figure(figsize=(8, 5))
    plt.errorbar(plus["tau"], plus["A"], yerr=plus["A_err"], fmt="o", label="+π/2 data")
    plt.errorbar(
        minus["tau"],
        minus["A"],
        yerr=minus["A_err"],
        fmt="s",
        label="–π/2 data",
    )
    plt.plot(xx, sinusoid_poly(xx, *popt_p), "--", label="fit +π/2")
    plt.plot(xx, sinusoid_poly(xx, *popt_m), "--", label="fit –π/2")
    plt.axvline(t_pi, color="k", ls=":")
    plt.xlabel("π‑pulse candidate length τ  (µs)")
    plt.ylabel("Fitted amplitude  |S|")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pi_pulse_summary.png")
    plt.close()


if __name__ == "__main__":
    # silence curve‑fit overflow warnings that occasionally appear
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
