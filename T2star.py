#!/usr/bin/env python
"""
sample_mass_T2star.py          – rev‑6
• No constant offsets (C_real, C_imag) are fit – filtered data are zero‑mean
• Per‑repeat parameters are now [A, f, φ] only
• Correlation‑matrix heat‑map adjusted to new parameter set
2025‑04‑17
"""

import os, re
from collections import defaultdict
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

from scipy.signal import butter, filtfilt, hilbert
from scipy.optimize import curve_fit


# ═════════════════════════════════════════════════════════════════════════════
#  Model helpers (no constant offsets)
# ═════════════════════════════════════════════════════════════════════════════
def model_complex(t, A, T2star, f, phi):
    """A · exp(−t/T2*) · exp(i(2πft+φ))"""
    return A * np.exp(-t / T2star) * np.exp(1j * (2 * np.pi * f * t + phi))


def composite_model(dummy_x, *params, group_t_list):
    """
    Concatenate real & imag parts of all repeats.
    Parameter vector = [A0, f0, φ0,  A1, f1, φ1, …,  T2star]
    """
    n_rep = len(group_t_list)
    T2star = params[-1]
    out = []
    for i in range(n_rep):
        A, f, phi = params[3 * i : 3 * i + 3]
        sig = model_complex(group_t_list[i], A, T2star, f, phi)
        out.append(np.concatenate((sig.real, sig.imag)))
    return np.concatenate(out)


# ═════════════════════════════════════════════════════════════════════════════
#  Helper functions for initial guesses
# ═════════════════════════════════════════════════════════════════════════════
def guess_frequency(sig, t):
    dt = np.median(np.diff(t))
    f_axis = np.fft.rfftfreq(len(sig), d=dt)
    spectrum = np.abs(np.fft.rfft(sig.real - sig.real.mean()))
    return -f_axis[np.argmax(spectrum[1:]) + 1]


def guess_T2star(envelope, t):
    mask = envelope > 0
    env, t_sel = envelope[mask], t[mask]
    n = max(10, int(0.3 * len(env)))
    if len(env[:n]) < 2:
        return None
    slope, _ = np.polyfit(t_sel[:n], np.log(env[:n]), 1)
    return -1 / slope if slope < 0 else None


# ═════════════════════════════════════════════════════════════════════════════
#  Main analysis
# ═════════════════════════════════════════════════════════════════════════════
def main():
    folder = "4-17-25"
    pat = re.compile(r"^(\d+)_(\d+)\.csv$")
    files = [f for f in os.listdir(folder) if pat.match(f)]
    if not files:
        raise RuntimeError("No CSV files found")

    grouped = defaultdict(list)
    for fn in files:
        mc = int(pat.match(fn).group(1))
        df = pd.read_csv(
            os.path.join(folder, fn), header=None, names=["t", "CH1", "CH2"]
        )
        df = df.dropna()
        df = df[df.t < 0.0023]
        grouped[mc].append(df)

    # 1–5 kHz Butterworth band‑pass
    b_bp, a_bp = butter(3, [1_000, 5_000], btype="bandpass", fs=1_000_000)

    results = []
    for mc in sorted(grouped):
        # ---------- filter signals ----------
        filt_sigs, t_list = [], []
        for df in grouped[mc]:
            t = df.t.values
            filt = filtfilt(b_bp, a_bp, df.CH1.values + 1j * df.CH2.values)
            t_list.append(t)
            filt_sigs.append(filt)
        filt_sigs = np.asarray(filt_sigs)
        n_rep = filt_sigs.shape[0]

        # ---------- build y‑vector & σ ----------
        std_real = filt_sigs.real.std(axis=0, ddof=1)
        std_imag = filt_sigs.imag.std(axis=0, ddof=1)
        sigma_vec = np.tile(np.concatenate((std_real, std_imag)), n_rep)
        sigma_vec[sigma_vec == 0] = sigma_vec[sigma_vec > 0].min()
        ydata = np.concatenate([np.concatenate((s.real, s.imag)) for s in filt_sigs])

        # ---------- initial guess ----------
        p0, t2_guesses = [], []
        for s, t in zip(filt_sigs, t_list):
            amp0 = 0.5 * (np.abs(s).max() - np.abs(s).min())
            f0 = -2200.0
            phi0 = 3 * np.pi / 4
            env = np.abs(hilbert(s.real))
            t2_est = 5e-4
            if t2_est:
                t2_guesses.append(t2_est)
            p0.extend([amp0, f0, phi0])
        p0.append(np.median(t2_guesses) if t2_guesses else 5e-4)

        # ---------- fit ----------
        popt, pcov = curve_fit(
            lambda x, *p: composite_model(x, *p, group_t_list=t_list),
            None,
            ydata,
            p0=p0,
            # sigma=sigma_vec,
            # absolute_sigma=True,
            maxfev=60_000,
        )
        T2star, T2err = popt[-1], np.sqrt(np.diag(pcov))[-1]
        results.append(dict(mass_g=mc / 100, T2star=T2star, T2star_err=T2err))

        # ---------- correlation‑matrix heat‑map ----------
        labels = [lbl for i in range(n_rep) for lbl in (f"A{i}", f"f{i}", f"φ{i}")]
        labels.append("T2*")
        sigs = np.sqrt(np.diag(pcov))
        corr = pcov / np.outer(sigs, sigs)
        corr[np.isnan(corr)] = 0.0
        fig_corr, ax_corr = plt.subplots(figsize=(5.5, 4.5))
        im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax_corr.set_xticks(range(len(labels)))
        ax_corr.set_yticks(range(len(labels)))
        ax_corr.set_xticklabels(labels, rotation=90, fontsize=7)
        ax_corr.set_yticklabels(labels, fontsize=7)
        ax_corr.set_title(f"Correlation matrix – mass {mc/100:.2f} g")
        plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04).ax.set_ylabel(
            "ρ", rotation=0, labelpad=10
        )
        plt.tight_layout()
        plt.savefig(f"corrmat_mass_{mc:02d}.png")
        plt.close(fig_corr)

        # ---------- diagnostic plots ----------
        colors = plt.cm.tab10.colors
        fig, axs = plt.subplots(n_rep, 2, figsize=(10, 3 * n_rep), sharex=True)
        axs = np.atleast_2d(axs)
        for i in range(n_rep):
            A, f, phi = popt[3 * i : 3 * i + 3]
            fit = model_complex(t_list[i], A, T2star, f, phi)
            # real
            axs[i, 0].fill_between(
                t_list[i],
                filt_sigs[i].real - std_real,
                filt_sigs[i].real + std_real,
                color=colors[i % 10],
                alpha=0.15,
                label="±σ",
            )
            axs[i, 0].plot(
                t_list[i],
                filt_sigs[i].real,
                ".",
                ms=2,
                color=colors[i % 10],
                label="Data",
            )
            axs[i, 0].plot(
                t_list[i], fit.real, "--", lw=1.2, color=colors[i % 10], label="Fit"
            )
            axs[i, 0].set_ylabel("Signal (a.u.)")
            axs[i, 0].legend(fontsize=7, loc="upper right")
            # imag
            axs[i, 1].fill_between(
                t_list[i],
                filt_sigs[i].imag - std_imag,
                filt_sigs[i].imag + std_imag,
                color=colors[i % 10],
                alpha=0.15,
                label="±σ",
            )
            axs[i, 1].plot(
                t_list[i],
                filt_sigs[i].imag,
                ".",
                ms=2,
                color=colors[i % 10],
                label="Data",
            )
            axs[i, 1].plot(
                t_list[i], fit.imag, "--", lw=1.2, color=colors[i % 10], label="Fit"
            )
            axs[i, 1].legend(fontsize=7, loc="upper right")
        axs[-1, 0].set_xlabel("Time (s)")
        axs[-1, 1].set_xlabel("Time (s)")
        plt.suptitle(
            rf"Mass {mc/100:.2f} g  –  $T_2^* = {T2star*1e3:.3f}\pm{T2err*1e3:.3f}$ ms",
            y=1.02,
            fontsize=14,
        )
        plt.tight_layout()
        plt.savefig(f"t2star_fit_mass_{mc:02d}.png")
        plt.close()

        print(
            f"{mc/100:.2f} g:  T2* = {T2star*1e3:.3f} ± {T2err*1e3:.3f} ms  (n={n_rep})"
        )

    # ---------- mass‑dependence ----------
    res = pd.DataFrame(results).sort_values("mass_g")
    res.to_csv("t2star_vs_mass.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.errorbar(
        res.mass_g, res.T2star * 1e3, yerr=res.T2star_err * 1e3, fmt="o", capsize=4
    )
    plt.xlabel("Sample mass (g)")
    plt.ylabel(r"$T_2^{\!*}$ (ms)")
    plt.title(r"$T_2^{\!*}$ vs Sample Mass")
    plt.tight_layout()
    plt.savefig("T2star_vs_sample_mass.png")
    plt.close()


if __name__ == "__main__":
    main()
