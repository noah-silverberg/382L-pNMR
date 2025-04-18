#!/usr/bin/env python
"""
sample_mass_T2star_peaks.py      – rev‑1
Fit *only the peak envelope* of each (Real / Imag) trace to a shared
exponential decay:

    peak(t) ≃ A_trace · exp(−t / T2*)

* band‑pass (1–5 kHz) → Gaussian smooth → peak detection
* simultaneous fit of all peak trains (own A, common T2*)
* diagnostics: filtered trace + peaks + fitted envelope
* summary: T2* versus sample mass

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
    """
    Concatenate the model for every peak train.
    params = [A0, A1, ..., A_(n‑1), T2]
    """
    n_tr = len(peak_t_list)
    T2 = params[-1]
    out = []
    for i in range(n_tr):
        A = params[i]
        out.append(exp_decay(peak_t_list[i], A, T2))
    return np.concatenate(out)


def detect_peaks(y, prominence):
    idx, _ = find_peaks(y, prominence=prominence)
    return idx


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    folder = "4-17-25/rapid"  # data directory
    pat = re.compile(r"^(\d+)_(\d+)\.csv$")  # 37_0.csv → mass 0.37 g
    files = [f for f in os.listdir(folder) if pat.match(f)]
    if not files:
        raise RuntimeError("No CSV files found")

    # group files by sample mass code
    grouped = defaultdict(list)
    for fn in files:
        mc = int(pat.match(fn).group(1))
        df = pd.read_csv(
            os.path.join(folder, fn), header=None, names=["t", "CH1", "CH2"]
        ).dropna()
        df = df[(df.t > 0.0008) & (df.t < 0.0028)]  # crop trailing junk
        grouped[mc].append(df)

    # 1–5 kHz Butterworth band‑pass
    b_bp, a_bp = butter(3, [5_000, 20_000], btype="bandpass", fs=1_000_000)

    results = []
    for mc in sorted(grouped):
        peak_t_list, peak_y_list = [], []  # per‑trace peak data
        trace_labels = []  # for plotting
        diagnostics = []  # (t, filt_sig, peaks_t, peaks_y, fit_env)

        # ----------- preprocess each trace (Real & Imag) -----------
        for rep, df in enumerate(grouped[mc]):
            t = df.t.values
            complex_sig = filtfilt(b_bp, a_bp, df.CH1.values + 1j * df.CH2.values)

            for comp, comp_name in zip(
                (complex_sig.real, complex_sig.imag), ("Real", "Imag")
            ):
                # smooth to suppress noise before peak finding
                sm = gaussian_filter1d(comp, sigma=5)
                # dynamic prominence: 20 % of smoothed p‑to‑p
                prom = 0.02 * (sm.max() - sm.min())
                peak_idx = detect_peaks(sm, prominence=prom)
                if len(peak_idx) < 3:  # skip traces with too few peaks
                    continue
                peak_t = t[peak_idx]
                peak_y = np.abs(comp[peak_idx])

                peak_t_list.append(peak_t)
                peak_y_list.append(peak_y)
                trace_labels.append(f"{comp_name} rep{rep}")

                diagnostics.append((t, comp, peak_t, peak_y, None))  # fit_env later

        n_tr = len(peak_t_list)
        if n_tr == 0:
            print(f"{mc/100:.2f} g: no usable peaks, skipping.")
            continue

        # ----------- build global fit arrays -----------
        ydata = np.concatenate(peak_y_list)
        sigma = None  # un‑weighted fit; could supply np.sqrt(y) as Poisson weights
        p0 = []  # amplitude guesses
        for py in peak_y_list:
            p0.append(py[0])  # first peak amplitude
        p0.append(5e-4)  # T2* initial 0.5 ms

        # ----------- simultaneous fit -----------
        popt, pcov = curve_fit(
            lambda x, *p: composite_peaks(x, *p, peak_t_list=peak_t_list),
            None,
            ydata,
            p0=p0,
            sigma=sigma,
            maxfev=40000,
        )
        T2star = popt[-1]
        T2err = np.sqrt(np.diag(pcov))[-1] if np.isfinite(pcov).all() else np.nan
        results.append(dict(mass_g=mc / 100, T2star=T2star, T2star_err=T2err))

        # ----------- diagnostics plots -----------
        colors = plt.cm.tab10.colors
        for i, (t_vec, comp_sig, p_t, p_y, _) in enumerate(diagnostics):
            A_i = popt[i]
            fit_env = exp_decay(t_vec, A_i, T2star)
            diagnostics[i] = (t_vec, comp_sig, p_t, p_y, fit_env)

        rows = len(diagnostics)
        fig, axs = plt.subplots(rows, 1, figsize=(10, 2.5 * rows), sharex=True)
        if rows == 1:
            axs = [axs]
        for ax, diag, lbl in zip(axs, diagnostics, trace_labels):
            t_vec, sig_vec, p_t, p_y, env = diag
            ax.plot(t_vec, sig_vec, lw=0.8, color="0.3")
            ax.plot(p_t, p_y, "o", ms=4, label="Peaks")
            ax.plot(t_vec, env, "--", lw=1.2, label="Fit envelope")
            ax.set_ylabel("Signal (a.u.)")
            ax.set_title(lbl)
            ax.legend(fontsize=8)
        axs[-1].set_xlabel("Time (s)")
        plt.suptitle(
            rf"Mass {mc/100:.2f} g – $T_2^*={T2star*1e3:.3f}\pm"
            f"{(T2err*1e3 if np.isfinite(T2err) else np.nan):.3g}$ ms",
            y=1.02,
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(f"peaks_fit_mass_{mc:02d}.png")
        plt.close()

        print(
            f"{mc/100:.2f} g:  T2* = {T2star*1e3:.3f} ± "
            f"{(T2err*1e3 if np.isfinite(T2err) else np.nan):.3g} ms"
        )

    # ----------- mass‑dependence summary -----------
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
