#!/usr/bin/env python
"""
sample_mass_T2star_peaks.py      – rev‑2
* plots now show **raw trace (grey)** + **filtered trace (colour)**
* rest of the workflow unchanged
2025‑04‑17
"""

import os, re
from collections import defaultdict
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


# ═════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═════════════════════════════════════════════════════════════════════════════
def exp_decay(t, A, T2):
    return A * np.exp(-t / T2)


def composite_peaks(dummy_x, *params, peak_t_list):
    n_tr = len(peak_t_list)
    T2 = params[-1]
    return np.concatenate(
        [exp_decay(peak_t_list[i], params[i], T2) for i in range(n_tr)]
    )


def detect_peaks(y, prominence):
    return find_peaks(y, prominence=prominence)[0]


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    folder = "4-17-25/rapid"
    pat = re.compile(r"^(\d+)_(\d+)\.csv$")
    files = [f for f in os.listdir(folder) if pat.match(f)]
    if not files:
        raise RuntimeError("No CSV files found")

    grouped = defaultdict(list)
    for fn in files:
        mc = int(pat.match(fn).group(1))
        df = pd.read_csv(
            os.path.join(folder, fn), header=None, names=["t", "CH1", "CH2"]
        ).dropna()
        df = df[(df.t < 0.0028)]
        df["CH1"] -= df["CH1"].mean()
        df["CH2"] -= df["CH2"].mean()
        grouped[mc].append(df)

    b_bp, a_bp = butter(3, [5_000, 15_000], btype="bandpass", fs=1_000_000)

    results = []
    for mc in sorted(grouped):
        peak_t_list, peak_y_list = [], []
        trace_labels, diagnostics = [], []
        # diagnostics tuples: (t, raw, filt, peak_t, peak_y, env)

        for rep, df in enumerate(grouped[mc]):
            t = df.t.values
            raw_sig = df.CH1.values + 1j * df.CH2.values
            filt_sig = filtfilt(b_bp, a_bp, raw_sig)
            for comp_raw, comp_filt, name in zip(
                (raw_sig.real, raw_sig.imag),
                (filt_sig.real, filt_sig.imag),
                ("Real", "Imag"),
            ):
                sm = gaussian_filter1d(comp_filt, sigma=5)
                prom = 0.02 * (sm.max() - sm.min())
                # Only consider time points in the inner 70% of the trace
                valid = np.logical_and(
                    t > t[0] + 0.15 * (t[-1] - t[0]), t < t[-1] - 0.15 * (t[-1] - t[0])
                )
                if not valid.any():
                    continue
                sub_t = t[valid]
                sub_sm = sm[valid]
                p_idx_rel = detect_peaks(sub_sm, prominence=prom)
                if len(p_idx_rel) < 3:
                    continue
                p_idx = np.where(valid)[0][p_idx_rel]
                peak_t, peak_y = t[p_idx], np.abs(comp_filt[p_idx])

                peak_t_list.append(peak_t)
                peak_y_list.append(peak_y)
                trace_labels.append(f"{name} rep{rep}")
                diagnostics.append((t, comp_raw, comp_filt, peak_t, peak_y, None))

        if not peak_t_list:
            print(f"{mc/100:.2f} g: no usable peaks, skipping.")
            continue

        # global fit
        ydata = np.concatenate(peak_y_list)
        p0 = [py[0] for py in peak_y_list] + [5e-4]
        popt, pcov = curve_fit(
            lambda x, *p: composite_peaks(x, *p, peak_t_list=peak_t_list),
            None,
            ydata,
            p0=p0,
            maxfev=40000,
        )
        T2star, T2err = popt[-1], np.sqrt(np.diag(pcov))[-1]
        results.append(dict(mass_g=mc / 100, T2star=T2star, T2star_err=T2err))

        # prepare envelopes for plotting
        for i, d in enumerate(diagnostics):
            t_vec, r_vec, f_vec, p_t, p_y, _ = d
            diagnostics[i] = d[:-1] + (exp_decay(t_vec, popt[i], T2star),)

        # plotting
        fig, axs = plt.subplots(
            len(diagnostics), 1, figsize=(10, 2.7 * len(diagnostics)), sharex=True
        )
        if len(diagnostics) == 1:
            axs = [axs]
        cols = plt.cm.tab10.colors
        for ax, diag, lbl, col in zip(axs, diagnostics, trace_labels, cols * 100):
            t_vec, raw_v, filt_v, p_t, p_y, env = diag
            ax.plot(t_vec, raw_v, lw=0.6, color="0.70", label="Raw")
            ax.plot(t_vec, filt_v, lw=0.9, color=col, label="Filtered")
            ax.plot(p_t, p_y, "o", ms=4, label="Peaks")
            ax.plot(t_vec, env, "--", lw=1.2, color=col, label="Fit env.")
            ax.set_ylabel("Signal (a.u.)")
            ax.set_title(lbl)
            ax.legend(fontsize=7)
        axs[-1].set_xlabel("Time (s)")
        plt.suptitle(
            rf"Mass {mc/100:.2f} g – $T_2^*={T2star*1e3:.3f}\pm{T2err*1e3:.3f}$ ms",
            y=1.02,
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(f"peaks_fit_mass_{mc:02d}.png")
        plt.close()

        print(f"{mc/100:.2f} g:  T2* = {T2star*1e3:.3f} ± {T2err*1e3:.3f} ms")

    # summary plot
    if results:
        res_df = pd.DataFrame(results).sort_values("mass_g")
        res_df.to_csv("t2star_vs_mass.csv", index=False)
        plt.figure(figsize=(7, 4))
        plt.errorbar(
            res_df.mass_g,
            res_df.T2star * 1e3,
            yerr=res_df.T2star_err * 1e3,
            fmt="o",
            capsize=4,
        )
        plt.xlabel("Sample mass (g)")
        plt.ylabel(r"$T_2^{\!*}$ (ms)")
        plt.title(r"$T_2^{\!*}$ vs Sample Mass")
        plt.tight_layout()
        plt.savefig("T2star_vs_sample_mass.png")
        plt.close()
    else:
        print("No valid fits produced.")


if __name__ == "__main__":
    main()
