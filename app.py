"""
app.py — Turntable Speed & Wow/Flutter Analyzer (Streamlit web app)

Web-app wrapper around the Phyphox gyroscope turntable speed analyzer.
Upload a Phyphox gyroscope CSV export (phone laid flat on the platter,
spindle-aligned axis) and get absolute speed (RPM) and wow/flutter
(unweighted, DIN/IEC/AES6-2008 weighted, and JIS-style weighted RMS).

Deploy on Streamlit Community Cloud:
    1. Push this file + requirements.txt to a GitHub repo
    2. share.streamlit.io -> New app -> point at this file
"""

from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Turntable Speed / Wow & Flutter Analyzer",
                    layout="wide")
st.title("Turntable Speed & Wow/Flutter Analyzer")
st.markdown(
    "Upload a **Phyphox gyroscope CSV** export (phone laid flat on the "
    "platter, spindle-aligned axis) to measure absolute turntable speed "
    "(RPM) and speed variation (wow & flutter) — unweighted, "
    "DIN 45507 / IEC 60386 / AES6-2008 weighted, and JIS-style weighted RMS."
)

# ----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------------
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader("Phyphox gyroscope CSV", type=["csv"])

nominal_rpm_options = "33.333, 45.0, 78.0"
nominal_rpm_text = st.sidebar.text_input(
    "Nominal RPM candidates (comma-separated)", nominal_rpm_options)
try:
    NOMINAL_RPM_CANDIDATES = [float(x.strip()) for x in nominal_rpm_text.split(",") if x.strip()]
except ValueError:
    st.sidebar.error("Could not parse nominal RPM candidates — using defaults.")
    NOMINAL_RPM_CANDIDATES = [33.333, 45.0, 78.0]

st.sidebar.subheader("Steady-state detection")
STEADY_ROLL_WINDOW = st.sidebar.number_input(
    "Rolling-std window (samples)", min_value=5, max_value=200, value=20, step=1)
STEADY_STD_THRESHOLD = st.sidebar.number_input(
    "Rolling-std threshold (rad/s)", min_value=0.001, max_value=1.0,
    value=0.02, step=0.001, format="%.3f")
STEADY_MIN_ABS_OMEGA = st.sidebar.number_input(
    "Minimum |ω| to count as spinning (rad/s)", min_value=0.01, max_value=50.0,
    value=1.0, step=0.1)
STEADY_EDGE_TRIM_S = st.sidebar.number_input(
    "Edge trim (s, motor settling tail)", min_value=0.0, max_value=10.0,
    value=1.0, step=0.1)

st.sidebar.subheader("Filtering")
VISUAL_LOWPASS_CUTOFF_HZ = st.sidebar.number_input(
    "Visual smoothing cutoff (Hz, plot only)", min_value=0.5, max_value=50.0,
    value=5.0, step=0.5)

analysis_cutoff_enabled = st.sidebar.checkbox(
    "Apply analysis low-pass cutoff", value=True,
    help="Uncheck to disable analysis filtering entirely (raw instantaneous "
         "RPM used for FFT/wow-flutter, limited only by Nyquist).")
ANALYSIS_LOWPASS_CUTOFF_HZ = None
if analysis_cutoff_enabled:
    ANALYSIS_LOWPASS_CUTOFF_HZ = st.sidebar.number_input(
        "Analysis low-pass cutoff (Hz)", min_value=1.0, max_value=200.0,
        value=40.0, step=1.0,
        help="Set well above any plausible mechanical component (motor "
             "cogging, belt/pulley harmonics, bearing defects) and only "
             "below the MEMS gyro noise floor. Lowering this to 'clean up' "
             "the FFT will throw away real flutter/cogging content above "
             "the cutoff.")

VISUAL_LOWPASS_ORDER = 3
ANALYSIS_LOWPASS_ORDER = 3

# ----------------------------------------------------------------------------
# AES6-2008 / DIN 45507 / IEC 60386 WEIGHTING FILTER
#
# Analog transfer function and pole/zero locations taken from the
# FidelisAnalog AES6-Wow-and-Flutter project
# (github.com/FidelisAnalog/AES6-Wow-and-Flutter), optimized against all
# 17 AES6 Table 1 spec points (reported 1.01 dB peak-to-peak error).
# Verified against the published Table 1 nominal curve to within ~1.3 dB
# across 0.1-200 Hz.
#
#   H(s) = G * s^3 * (s + 2*pi*227.9)
#          / [ (s + 2*pi*0.6265)^3 * (s + 2*pi*11.32) ]
#
# DIN/IEC/AES6 traditionally report a weighted PEAK-like statistic (here:
# weighted 2-sigma and weighted peak-to-peak), while JIS traditionally
# reports a weighted RMS statistic — same filter, different convention.
# This distinction is taken from the source project's naming convention
# ("weighted RMS (JIS)"), not independently verified against the JIS
# C 5521 document itself.
# ----------------------------------------------------------------------------
_WEIGHT_ZERO_HZ = 227.9
_WEIGHT_POLE1_HZ = 0.6265   # triple pole
_WEIGHT_POLE2_HZ = 11.32


def _aes6_transfer_function(f):
    f = np.asarray(f, dtype=float)
    s = 1j * 2 * np.pi * f
    wz = 2 * np.pi * _WEIGHT_ZERO_HZ
    wp1 = 2 * np.pi * _WEIGHT_POLE1_HZ
    wp2 = 2 * np.pi * _WEIGHT_POLE2_HZ
    with np.errstate(invalid="ignore", divide="ignore"):
        num = s ** 3 * (s + wz)
        den = (s + wp1) ** 3 * (s + wp2)
        H = num / den
    return np.nan_to_num(H)


@st.cache_data
def _aes6_weighting_peak_mag():
    f_ref = np.logspace(-2, 3, 20000)
    return np.abs(_aes6_transfer_function(f_ref)).max()


def aes6_weighting(f):
    return _aes6_transfer_function(f) / _aes6_weighting_peak_mag()


def weighted_deviation(dev_pct, fs):
    n = len(dev_pct)
    spec = np.fft.rfft(dev_pct - dev_pct.mean())
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    weight = aes6_weighting(freqs)
    spec_weighted = spec * weight
    return np.fft.irfft(spec_weighted, n=n)


def centroid_peak(freqs, mags, f_lo, f_hi):
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return None, None
    f_band = freqs[band]
    m_band = mags[band]
    if m_band.sum() == 0:
        return None, None
    f_c = np.sum(f_band * m_band) / np.sum(m_band)
    a_c = m_band.max()
    return f_c, a_c


# ----------------------------------------------------------------------------
# MAIN ANALYSIS
# ----------------------------------------------------------------------------
if uploaded_file is None:
    st.info("Upload a Phyphox gyroscope CSV export to begin. Expected columns: "
            "\"Time (s)\", \"Gyroscope x (rad/s)\", \"Gyroscope y (rad/s)\", "
            "\"Gyroscope z (rad/s)\".")
    st.stop()

try:
    raw_bytes = uploaded_file.getvalue().decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(raw_bytes))
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["Time (s)", "Gyroscope x (rad/s)", "Gyroscope y (rad/s)",
                      "Gyroscope z (rad/s)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing expected column(s): {missing}. "
                 f"Columns found: {list(df.columns)}")
        st.stop()

    t = df["Time (s)"].values
    gx = df["Gyroscope x (rad/s)"].values
    gy = df["Gyroscope y (rad/s)"].values
    gz = df["Gyroscope z (rad/s)"].values

    dt = np.diff(t)
    fs = 1.0 / np.median(dt)

    # Identify spin axis
    axes = {"x": gx, "y": gy, "z": gz}
    means = {k: np.mean(v) for k, v in axes.items()}
    spin_axis_name = max(means, key=lambda k: abs(means[k]))
    omega_raw = axes[spin_axis_name]
    if np.median(omega_raw[np.abs(omega_raw) > STEADY_MIN_ABS_OMEGA]) < 0:
        omega_raw = -omega_raw

    # Steady-state segment detection
    roll_std = pd.Series(omega_raw).rolling(
        int(STEADY_ROLL_WINDOW), center=True).std().values
    steady_mask = (roll_std < STEADY_STD_THRESHOLD) & (np.abs(omega_raw) > STEADY_MIN_ABS_OMEGA)
    steady_idx = np.where(steady_mask)[0]

    if len(steady_idx) < 50:
        st.error("Could not find a stable steady-state rotation segment. "
                 "Try adjusting the rolling-std threshold or minimum |ω| in the sidebar.")
        st.stop()

    gaps = np.where(np.diff(steady_idx) > 1)[0]
    run_starts = np.concatenate(([0], gaps + 1))
    run_ends = np.concatenate((gaps, [len(steady_idx) - 1]))
    run_lengths = run_ends - run_starts
    best_run = np.argmax(run_lengths)
    i0 = steady_idx[run_starts[best_run]]
    i1 = steady_idx[run_ends[best_run]]

    edge_trim_samples = int(round(STEADY_EDGE_TRIM_S * fs))
    i0 = i0 + edge_trim_samples
    i1 = i1 - edge_trim_samples
    if i1 <= i0:
        st.error("Edge trim too large for the detected steady segment — reduce it in the sidebar.")
        st.stop()

    t_steady = t[i0:i1 + 1]
    omega_steady = omega_raw[i0:i1 + 1]

    # RPM calculation
    rpm_inst = omega_steady * 60.0 / (2 * np.pi)
    nyq = fs / 2.0

    b_vis, a_vis = butter(VISUAL_LOWPASS_ORDER, VISUAL_LOWPASS_CUTOFF_HZ / nyq, btype="low")
    rpm_visual_smooth = filtfilt(b_vis, a_vis, rpm_inst)

    if ANALYSIS_LOWPASS_CUTOFF_HZ is None:
        rpm_analysis = rpm_inst
    else:
        cutoff = ANALYSIS_LOWPASS_CUTOFF_HZ
        if cutoff >= nyq:
            st.warning(f"Analysis cutoff ({cutoff:.1f} Hz) is at/above Nyquist "
                       f"({nyq:.1f} Hz) for this recording's sample rate "
                       f"({fs:.2f} Hz). Clipping to {0.95 * nyq:.1f} Hz.")
            cutoff = 0.95 * nyq
        b_an, a_an = butter(ANALYSIS_LOWPASS_ORDER, cutoff / nyq, btype="low")
        rpm_analysis = filtfilt(b_an, a_an, rpm_inst)

    rpm_mean = np.mean(rpm_analysis)
    nominal_rpm = min(NOMINAL_RPM_CANDIDATES, key=lambda n: abs(n - rpm_mean))
    speed_error_pct = 100.0 * (rpm_mean - nominal_rpm) / nominal_rpm

    # Wow & flutter (unweighted)
    rpm_dev_pct = 100.0 * (rpm_analysis - rpm_mean) / rpm_mean
    wf_rms_pct = np.sqrt(np.mean(rpm_dev_pct ** 2))
    wf_pp_pct = rpm_dev_pct.max() - rpm_dev_pct.min()
    wf_2sigma_pct = 2.0 * np.std(rpm_dev_pct)

    # Weighted (DIN/IEC/AES6 + JIS-style)
    rpm_dev_weighted = weighted_deviation(rpm_dev_pct, fs)
    wf_weighted_pp_pct = rpm_dev_weighted.max() - rpm_dev_weighted.min()
    wf_weighted_2sigma_pct = 2.0 * np.std(rpm_dev_weighted)
    wf_wrms_pct_jis = np.sqrt(np.mean(rpm_dev_weighted ** 2))

    # FFT of deviation
    n = len(rpm_dev_pct)
    window = np.hanning(n)
    dev_windowed = (rpm_dev_pct - rpm_dev_pct.mean()) * window
    fft_vals = np.fft.rfft(dev_windowed)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(fft_vals) / (n / 2)

    valid = fft_freqs > 0.05
    rev_freq = rpm_mean / 60.0
    f1, a1 = centroid_peak(fft_freqs[valid], fft_mag[valid], rev_freq * 0.7, rev_freq * 1.3)
    f2, a2 = centroid_peak(fft_freqs[valid], fft_mag[valid], rev_freq * 1.7, rev_freq * 2.3)

    band_mask = (fft_freqs > 0.05) & (fft_freqs < nyq)
    dom_i = np.argmax(fft_mag[band_mask])
    dom_freq = fft_freqs[band_mask][dom_i]
    dom_amp = fft_mag[band_mask][dom_i]

    # ------------------------------------------------------------------------
    # PLOT
    # ------------------------------------------------------------------------
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(17, 12.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

    fig.suptitle(f"Turntable Speed Analysis — {uploaded_file.name}\n"
                 f"(spin axis: gyroscope {spin_axis_name})",
                 fontsize=13, color="white")

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, omega_raw, color="0.5", linewidth=0.6, label="raw ω (all data)")
    ax1.plot(t_steady, omega_steady, color="#00d0ff", linewidth=0.8, label="steady-state segment")
    ax1.axvline(t_steady[0], color="lime", linestyle="--", linewidth=0.8)
    ax1.axvline(t_steady[-1], color="lime", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("ω (rad/s)")
    ax1.set_title("Raw gyro trace — spin-up / steady rotation / spin-down")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_steady, rpm_inst, color="0.5", linewidth=0.5, alpha=0.6, label="instantaneous")
    ax2.plot(t_steady, rpm_visual_smooth, color="#ffb000", linewidth=1.2,
              label=f"visual smoothing (LP {VISUAL_LOWPASS_CUTOFF_HZ:.0f} Hz)")
    ax2.axhline(nominal_rpm, color="lime", linestyle="--", linewidth=0.8,
                 label=f"nominal {nominal_rpm:.3f} RPM")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RPM")
    ax2.set_title("RPM vs time (steady-state segment)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.2)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_steady, rpm_dev_pct, color="#ff5577", linewidth=1.0, label="unweighted deviation")
    ax3.axhline(0, color="0.6", linewidth=0.6)
    ax3.axhline(wf_2sigma_pct, color="#00d0ff", linestyle="--", linewidth=1.0,
                 label=f"±2S (95%, unweighted) = ±{wf_2sigma_pct:.3f} %")
    ax3.axhline(-wf_2sigma_pct, color="#00d0ff", linestyle="--", linewidth=1.0)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Speed deviation (%)")
    ax3.set_title("Wow/flutter — speed deviation from mean (UNWEIGHTED)")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(alpha=0.2)

    ax4 = fig.add_subplot(gs[2, 0])
    plot_mask = fft_freqs > 0
    ax4.plot(fft_freqs[plot_mask], fft_mag[plot_mask], color="#00e0a0", linewidth=1.0)
    if f1 is not None:
        ax4.axvline(f1, color="yellow", linestyle="--", linewidth=0.8, label=f"1×rev ≈ {f1:.3f} Hz")
    if f2 is not None:
        ax4.axvline(f2, color="orange", linestyle="--", linewidth=0.8, label=f"2×rev ≈ {f2:.3f} Hz")
    ax4.set_xscale("log")
    ax4.set_xlim(fft_freqs[fft_freqs > 0].min(), nyq)
    tick_vals = [v for v in [0.1, 1, 10] if v < nyq] + [nyq]
    ax4.set_xticks(tick_vals)
    ax4.set_xticklabels([f"{v:g}" if v != nyq else f"{np.floor(v * 10) / 10:.1f}" for v in tick_vals])
    ax4.xaxis.set_minor_formatter(plt.NullFormatter())
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Deviation amplitude (%)")
    ax4.set_title("FFT of speed deviation")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(alpha=0.2, which="both")

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    summary_lines = [
        f"Sample rate: {fs:.2f} Hz",
        f"Spin axis: gyroscope {spin_axis_name}",
        f"Analysis LP cutoff: {ANALYSIS_LOWPASS_CUTOFF_HZ:.1f} Hz"
        if ANALYSIS_LOWPASS_CUTOFF_HZ is not None else "Analysis LP cutoff: none (raw)",
        f"Steady segment: {t_steady[0]:.2f} s → {t_steady[-1]:.2f} s "
        f"({t_steady[-1] - t_steady[0]:.2f} s, "
        f"{(t_steady[-1] - t_steady[0]) * rpm_mean / 60:.1f} revs)",
        "",
        f"Nominal speed: {nominal_rpm:.3f} RPM",
        f"Mean measured speed: {rpm_mean:.4f} RPM",
        f"Speed error: {speed_error_pct:+.3f} %",
        "",
        "--- UNWEIGHTED (raw speed deviation, full analysis band) ---",
        f"W/F RMS (unweighted): {wf_rms_pct:.3f} %",
        f"W/F peak-to-peak (unweighted): {wf_pp_pct:.3f} %",
        f"W/F 2S 95% (unweighted): ±{wf_2sigma_pct:.3f} %",
        "",
        "--- DIN 45507 / IEC 60386 / AES6-2008 weighted (verified filter) ---",
        f"W/F peak-to-peak (weighted): {wf_weighted_pp_pct:.3f} %",
        f"W/F 2S 95% (weighted): ±{wf_weighted_2sigma_pct:.3f} %",
        "",
        "--- JIS-style weighted RMS (same filter; RMS convention per JIS) ---",
        f"W/F WRMS (JIS-style): {wf_wrms_pct_jis:.3f} %",
        "",
        f"Once-per-rev component: {f1:.3f} Hz, {a1:.3f} %" if f1 is not None else "Once-per-rev: n/a",
        f"Twice-per-rev component: {f2:.3f} Hz, {a2:.3f} %" if f2 is not None else "Twice-per-rev: n/a",
        f"Dominant peak (0.05 Hz-Nyquist): {dom_freq:.3f} Hz, {dom_amp:.3f} %",
    ]
    ax5.text(0.02, 0.98, "\n".join(summary_lines), transform=ax5.transAxes,
             fontsize=12, va="top", ha="left", family="monospace", color="#ffffff",
             fontweight="bold")
    ax5.set_title("Summary", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)

    st.caption(
        "DIN/IEC/AES6 weighting filter: analog pole/zero transfer function from the "
        "FidelisAnalog AES6-Wow-and-Flutter project, verified against AES6-2008 Table 1 "
        "nominal points. JIS column reuses the same filter with an RMS statistic — this "
        "specific DIN-vs-JIS split is taken from that project's naming convention, not "
        "independently verified against the JIS C 5521 document."
    )

except Exception as e:
    st.error(f"Error processing file: {e}")
    st.exception(e)
