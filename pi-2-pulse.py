#!/usr/bin/env python
"""
pi_pulse_analysis.py   – fast variant (T₂★ fixed per average‑group fit)
2025‑04‑15

Changes in this version
=======================
•  Stage‑1 (average‑group) fit is unchanged.
•  Stage‑2 now *fixes* T₂★ to the Stage‑1 value and fits every CSV file
   **independently** (no giant simultaneous fit).  Each fit therefore
   optimises only five parameters [A, f, φ, Cᵣ, Cᵢ].  The Stage‑1 parameters
   for that pulse‑time provide the initial guess.  σ arrays are copied from
   the Stage‑1 group as before.  This speeds the script up dramatically.
•  All subsequent statistics, plots, and sinusoid fit logic are the same as
   in the previous version.
"""

import os, re, itertools
from collections import defaultdict
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit

# ═════════════════════════════════════════════════════════════════════════════
#  Model helpers
# ═════════════════════════════════════════════════════════════════════════════


def model_complex(t, A, T2star, f, phi, C_real, C_imag):
    """Complex decaying sinusoid."""
    return A * np.exp(-t / T2star) * np.exp(1j * (2 * np.pi * f * t + phi)) + (
        C_real + 1j * C_imag
    )


def model_simultaneous(dummy_x, *params, group_t_list):
    """Composite model (used only for Stage‑1)."""
    N = len(group_t_list)
    T2star = params[-1]
    out = []
    for i in range(N):
        A, f, phi, Cre, Cim = params[5 * i : 5 * i + 5]
        sig = model_complex(group_t_list[i], A, T2star, f, phi, Cre, Cim)
        out.append(np.concatenate((sig.real, sig.imag)))
    return np.concatenate(out)


def sinusoid_plus_linear(x, a, f, phi, b, c):
    return a * np.sin(2 * np.pi * f * x + phi) + b * x + c


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    folder = "327/Pulse Calibration Redo"
    csv_files = [f for f in os.listdir(folder) if re.match(r"^\d+_\d+\.csv$", f)]

    # ------------------------------------------------------------------
    #  Load data
    # ------------------------------------------------------------------
    raw_data, pt_list = [], []
    for f in csv_files:
        pt = int(f.split("_")[0])
        if pt in [20, 25]:
            continue
        df = pd.read_csv(
            os.path.join(folder, f), header=None, names=["t", "CH1", "CH2"]
        )
        df = df.dropna()
        df = df[(df["t"] > 0.001) & (df["t"] < 0.004)]
        raw_data.append(df)
        pt_list.append(pt)

    # ------------------------------------------------------------------
    #  Stage‑1  (average‑group simultaneous fit)
    # ------------------------------------------------------------------
    grouped = defaultdict(list)
    for df, pt in zip(raw_data, pt_list):
        grouped[pt].append(df)

    b_lp, a_lp = butter(3, 5000, btype="lowpass", fs=500000)

    stage1_groups, stage1_t, stage1_sigma = [], [], []
    for pt in sorted(grouped.keys()):
        sigs = [
            filtfilt(b_lp, a_lp, df["CH1"].values + 1j * df["CH2"].values)
            for df in grouped[pt]
        ]
        sigs = np.asarray(sigs)
        avg_sig = sigs.mean(axis=0)
        std_sig = (
            sigs.std(axis=0, ddof=1) if sigs.shape[0] > 1 else np.zeros_like(avg_sig)
        )
        stage1_groups.append(
            dict(pulse_time=pt, t=grouped[pt][0]["t"].values, sig=avg_sig, std=std_sig)
        )
        stage1_t.append(grouped[pt][0]["t"].values)
        stage1_sigma.append(np.tile(std_sig, 2))

    y_stage1 = np.concatenate(
        [np.concatenate((g["sig"].real, g["sig"].imag)) for g in stage1_groups]
    )
    sigma_stage1 = np.concatenate(stage1_sigma)

    p0 = list(
        itertools.chain.from_iterable([[15, -2100, 0, -1.5, 0] for _ in stage1_groups])
    ) + [6.3e-4]

    popt1, pcov1 = curve_fit(
        lambda x, *p: model_simultaneous(x, *p, group_t_list=stage1_t),
        None,
        y_stage1,
        p0=p0,
        sigma=sigma_stage1,
        absolute_sigma=True,
        maxfev=20000,
    )
    perr1 = np.sqrt(np.diag(pcov1))
    T2_fixed = popt1[-1]
    T2_fixed_err = perr1[-1]

    # map pulse_time → Stage‑1 parameter vector
    guess_by_pt = {
        g["pulse_time"]: popt1[5 * i : 5 * i + 5] for i, g in enumerate(stage1_groups)
    }

    # ------------------------------------------------------------------
    #  Stage‑2  (independent fits with T₂★ fixed)
    # ------------------------------------------------------------------
    results_rows = []
    for df, pt in zip(raw_data, pt_list):
        t = df["t"].values
        sig = filtfilt(b_lp, a_lp, df["CH1"].values + 1j * df["CH2"].values)

        # σ: copy from Stage‑1 std for this pulse‑time
        std_grp = stage1_groups[sorted(grouped.keys()).index(pt)]["std"]
        sigma_vec = np.tile(std_grp, 2)
        sigma_vec[sigma_vec == 0] = 1.0  # avoid zeros

        # initial guess from Stage‑1
        A0, f0, phi0, Cre0, Cim0 = guess_by_pt[pt]

        # model with fixed T₂★
        def model_fixed_T2(t_arr, A, f, phi, Cre, Cim):
            s = model_complex(t_arr, A, T2_fixed, f, phi, Cre, Cim)
            return np.concatenate((s.real, s.imag))

        ydata = np.concatenate((sig.real, sig.imag))
        popt, pcov = curve_fit(
            model_fixed_T2,
            t,
            ydata,
            p0=[A0, f0, phi0, Cre0, Cim0],
            sigma=sigma_vec,
            absolute_sigma=True,
            maxfev=15000,
        )
        perr = np.sqrt(np.diag(pcov))

        A, fval, phi, Cre, Cim = popt
        results_rows.append(
            dict(
                pulse_time=pt,
                A=np.abs(A),
                A_sign=A,
                f=fval,
                phi=phi,
                C_real=Cre,
                C_imag=Cim,
                T2star=T2_fixed,
                A_err=perr[0],
            )
        )

    results_df = pd.DataFrame(results_rows)

    # ------------------------------------------------------------------
    #  Group statistics for A
    # ------------------------------------------------------------------
    group_stats = (
        results_df.groupby("pulse_time")["A"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "A_mean", "std": "A_std", "count": "N"})
        .reset_index()
    )
    results_df = results_df.merge(group_stats[["pulse_time", "A_std"]], on="pulse_time")

    # ------------------------------------------------------------------
    #  Plot fits  (one figure per pulse‑time)
    # ------------------------------------------------------------------
    # ── Replace the old plotting loop with this block ──────────────────────────
    # for pt in sorted(grouped.keys()):
    #     idxs = results_df.index[results_df["pulse_time"] == pt].tolist()
    #     nrep = len(idxs)

    #     # one row per repeat, two columns (Real | Imag)
    #     fig, axs = plt.subplots(
    #         nrep, 2, figsize=(10, 3 * nrep), sharex=True, sharey=False
    #     )
    #     if nrep == 1:  # ensure 2‑D indexing even for a single repeat
    #         axs = np.array([axs])

    #     colors = plt.cm.tab10.colors
    #     for row, idx in enumerate(idxs):
    #         df = raw_data[idx]
    #         t = df["t"].values
    #         sig = filtfilt(b_lp, a_lp, df["CH1"].values + 1j * df["CH2"].values)

    #         A, fval, phi, Cre, Cim = results_df.loc[
    #             idx, ["A_sign", "f", "phi", "C_real", "C_imag"]
    #         ].values
    #         fit_curve = model_complex(t, A, T2_fixed, fval, phi, Cre, Cim)

    #         # Real component
    #         axs[row, 0].plot(t, sig.real, "o", ms=2, color=colors[row % 10], alpha=0.6)
    #         axs[row, 0].plot(t, fit_curve.real, "-", color=colors[row % 10])
    #         axs[row, 0].set_ylabel("Signal (a.u.)")
    #         axs[row, 0].set_title(f"Rep {row} – Real") if row == 0 else None

    #         # Imag component
    #         axs[row, 1].plot(t, sig.imag, "o", ms=2, color=colors[row % 10], alpha=0.6)
    #         axs[row, 1].plot(t, fit_curve.imag, "-", color=colors[row % 10])
    #         axs[row, 1].set_title(f"Rep {row} – Imag") if row == 0 else None

    #     for col in range(2):
    #         axs[-1, col].set_xlabel("Time (s)")

    #     plt.suptitle(f"Pulse {pt} µs", fontsize=14, y=1.02)
    #     plt.tight_layout()
    #     plt.savefig(f"fit_group_{pt}.png")
    #     plt.close()

    # ------------------------------------------------------------------
    #  Sinusoid + linear fit to all individual A values
    # ------------------------------------------------------------------
    x_all = results_df["pulse_time"].values
    y_all = results_df["A"].values
    sigma_A = results_df["A_std"].values
    sigma_A[sigma_A == 0] = sigma_A[sigma_A != 0].min()  # protect zeros

    p0_sin = [y_all.max(), 5e-3, 0.0, 0.0, 0.0]
    popt_sin, pcov_sin = curve_fit(
        sinusoid_plus_linear,
        x_all,
        y_all,
        sigma=sigma_A,
        absolute_sigma=True,
        p0=p0_sin,
        maxfev=100000,
    )
    residuals = y_all - sinusoid_plus_linear(x_all, *popt_sin)
    chi2 = np.sum((residuals / sigma_A) ** 2)
    dof = len(y_all) - len(popt_sin)
    red_chi2 = chi2 / dof

    # plot means
    x_plot = np.linspace(
        group_stats["pulse_time"].min(), group_stats["pulse_time"].max(), 400
    )
    y_plot = sinusoid_plus_linear(x_plot, *popt_sin)
    plt.figure(figsize=(7, 4))
    plt.errorbar(
        group_stats["pulse_time"] / 2,
        group_stats["A_mean"],
        yerr=group_stats["A_std"],
        fmt="o",
        capsize=4,
        label="Average Amplitude ± σ",
    )
    plt.plot(x_plot / 2, y_plot, "--", label="Sinusoid + Linear Fit")
    plt.xlabel("Pulse time (µs)")
    plt.ylabel("Amplitude  (a.u.)")
    plt.title("Amplitude vs Pulse Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sinusoid_fit.png")
    plt.close()

    # ------------------------------------------------------------------
    #  Summary and sinusoid peak/trough extraction
    # ------------------------------------------------------------------
    print(f"Fixed global T2* = {T2_fixed*1e3:.4g} ± {T2_fixed_err*1e3:.4g} ms")
    print("Sinusoid parameters [a, f, φ, b, c] =")
    print("  ", ", ".join(f"{v:.4g}" for v in popt_sin))
    print(f"Reduced χ² of sinusoid fit (all repeats): {red_chi2:.3f}")

    # Create a full x grid over the complete pulse time range
    pulse_grid = np.linspace(
        group_stats["pulse_time"].min(), group_stats["pulse_time"].max(), 400
    )
    y_full = sinusoid_plus_linear(pulse_grid, *popt_sin)

    # Find the peak (maximum) in the region 40-60 µs
    peak_mask = (pulse_grid >= 40) & (pulse_grid <= 60)
    if np.any(peak_mask):
        peak_region = y_full[peak_mask]
        peak_indices = np.where(peak_mask)[0]
        peak_idx = peak_indices[np.argmax(peak_region)]
        peak_pulse_time = pulse_grid[peak_idx]
        print(
            f"Pulse time corresponding to peak in range (40, 60): {peak_pulse_time / 2:.4f} µs"
        )
    else:
        print("No pulse time in the range (40, 60) µs found for peak.")

    # Find the trough (minimum) in the region 80-100 µs
    trough_mask = (pulse_grid >= 80) & (pulse_grid <= 100)
    if np.any(trough_mask):
        trough_region = y_full[trough_mask]
        trough_indices = np.where(trough_mask)[0]
        trough_idx = trough_indices[np.argmin(trough_region)]
        trough_pulse_time = pulse_grid[trough_idx]
        print(
            f"Pulse time corresponding to trough in range (80, 100): {trough_pulse_time / 2:.4f} µs"
        )
    else:
        print("No pulse time in the range (80, 100) µs found for trough.")

    # save CSVs
    results_df.to_csv("fit_results_individual.csv", index=False)
    group_stats.to_csv("group_stats.csv", index=False)


if __name__ == "__main__":
    main()
