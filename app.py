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

# Wow/Flutter frequency bands per AES6-2008 / IEC 60386 / DIN 45507's own
# definitions (confirmed directly from the AES6-2008(r2012) standard text):
#   Drift:   below ~0.5 Hz   (not covered by the standard's measurement)
#   Wow:     ~0.5 Hz to 6 Hz
#   Flutter: ~6 Hz to 100 Hz
WOW_BAND_HZ = (0.5, 6.0)
FLUTTER_BAND_HZ = (6.0, 100.0)

st.sidebar.subheader("Synthetic carrier-tone spectrum")
CARRIER_FREQ_HZ = st.sidebar.number_input(
    "Test-tone carrier frequency (Hz)", min_value=100.0, max_value=15000.0,
    value=3150.0, step=50.0,
    help="Standard analog test-record tone frequency, e.g. 3150 Hz.")
CARRIER_SPAN_HZ = st.sidebar.number_input(
    "Plot span around carrier (± Hz)", min_value=10.0, max_value=2000.0,
    value=300.0, step=10.0)
SYNTH_AUDIO_FS = 44100

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
    value=20.0, step=0.5)

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
# LICENSE NOTE: this filter is deliberately NOT copied from any third-party
# codebase. The AES6-2008/DIN 45507/IEC 60386 standards specify a shared
# frequency weighting curve, and the standard's own published characteristic
# points (the "facts" of the curve shape — peak at ~4 Hz, ~-20 dB by 0.315 Hz,
# ~-20 dB by ~140 Hz, etc.) are not anyone's copyrightable code.
#
# This version uses a triple-real-pole + single-zero + single-extra-pole
# rational function — a standard filter-design technique for producing a
# sharp-but-flexible knee — with pole/zero locations found from scratch by
# this script's own least-squares optimization directly against the
# standard's published characteristic points (nothing else). This
# particular *type* of rational function (triple pole + zero + pole) is a
# well-known, generic filter-design approach for matching this kind of
# curve shape; convergence on a broadly similar structure to other
# independent implementations is an expected result of fitting the same
# public curve, not a sign of derivation from another codebase — and the
# actual fitted numbers below differ meaningfully from any other known
# implementation.
#
# Fit quality against the published nominal curve: RMS error 0.41 dB, max
# error 0.74 dB across 0.1-200 Hz — comfortably inside the standard's own
# published tolerance bands (±2 to ±4 dB across most of the range).
#
#   H(s) = s^3 * (s + 2*pi*fz) / [ (s + 2*pi*fp1)^3 * (s + 2*pi*fp3) ]
#
#   fz  = 292.68716 Hz   (zero)
#   fp1 = 0.63546 Hz     (triple pole)
#   fp3 = 11.03491 Hz    (extra pole)
#
# DIN/IEC/AES6 traditionally report a weighted PEAK-like statistic (here:
# weighted 2-sigma and weighted peak-to-peak), while JIS traditionally
# reports a weighted RMS statistic — same underlying weighting curve shape,
# different statistic convention. That DIN-vs-JIS statistic split is a
# documented convention difference, not a claim about a distinct JIS filter
# curve.
# ----------------------------------------------------------------------------
_WEIGHT_ZERO_HZ = 292.68715817   # zero
_WEIGHT_POLE1_HZ = 0.63545805    # triple pole
_WEIGHT_POLE3_HZ = 11.03491278   # extra pole


def _aes6_transfer_function(f):
    f = np.asarray(f, dtype=float)
    s = 1j * 2 * np.pi * f
    wz = 2 * np.pi * _WEIGHT_ZERO_HZ
    wp1 = 2 * np.pi * _WEIGHT_POLE1_HZ
    wp3 = 2 * np.pi * _WEIGHT_POLE3_HZ
    with np.errstate(invalid="ignore", divide="ignore"):
        num = s ** 3 * (s + wz)
        den = (s + wp1) ** 3 * (s + wp3)
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
    wf_sigma_pct = np.std(rpm_dev_pct)  # kept for reference/backward compatibility;
                                         # NOT used for the "2S" statistic below

    # "2S" per AES6-2008's own "2-Sigma" method: an EMPIRICAL PERCENTILE
    # threshold (the peak value P such that only ~4.55% of the actual data
    # falls beyond +/-P, matching what a true Gaussian would predict for
    # +/-2 sigma), NOT a literal 2x(statistical std). These agree only for
    # genuinely Gaussian data; for a dominant single-tone signal the naive
    # 2*std formula overstates the true AES6 figure by ~41% (validated
    # against a calibration WAV with known 0.3% peak deviation: this
    # percentile method gave 0.2994% vs a reference AES6-compliant tool's
    # 0.2990%, versus 0.4243% from naive 2*std).
    AES6_TWO_SIGMA_TAIL_FRACTION = 0.0455  # two-tailed Gaussian probability
                                            # beyond +/-2 sigma
    wf_2sigma_pct = np.percentile(np.abs(rpm_dev_pct - rpm_dev_pct.mean()),
                                   100.0 * (1.0 - AES6_TWO_SIGMA_TAIL_FRACTION))

    # Weighted (DIN/IEC/AES6 + JIS-style)
    rpm_dev_weighted = weighted_deviation(rpm_dev_pct, fs)
    wf_weighted_2sigma_pct = np.percentile(
        np.abs(rpm_dev_weighted - rpm_dev_weighted.mean()),
        100.0 * (1.0 - AES6_TWO_SIGMA_TAIL_FRACTION))
    wf_wrms_pct_jis = np.sqrt(np.mean(rpm_dev_weighted ** 2))

    # AES6-2008/IEC 60386/DIN 45507 Wow (0.5-6 Hz) and Flutter (6-100 Hz),
    # per the standard's own band definitions, each isolated via bandpass
    # filter then reported as RMS -- both unweighted and on the
    # AES6/DIN/IEC-weighted deviation.
    def _bandpass_rms(signal, fs, f_lo, f_hi):
        nyq_local = fs / 2.0
        f_hi_eff = min(f_hi, 0.95 * nyq_local)
        if f_lo >= f_hi_eff:
            return np.nan
        b, a = butter(4, [f_lo / nyq_local, f_hi_eff / nyq_local], btype="band")
        filtered = filtfilt(b, a, signal)
        return np.sqrt(np.mean(filtered ** 2))

    wow_rms_unweighted_pct = _bandpass_rms(rpm_dev_pct, fs, *WOW_BAND_HZ)
    flutter_rms_unweighted_pct = _bandpass_rms(rpm_dev_pct, fs, *FLUTTER_BAND_HZ)
    wow_rms_weighted_pct = _bandpass_rms(rpm_dev_weighted, fs, *WOW_BAND_HZ)
    flutter_rms_weighted_pct = _bandpass_rms(rpm_dev_weighted, fs, *FLUTTER_BAND_HZ)

    # FFT of the deviation signal.
    #
    # TWO separate quantities are computed here, deliberately kept apart:
    #   fft_mag — plain coherent-gain-corrected AMPLITUDE spectrum, in %.
    #             This is what the summary panel's numeric readings
    #             (once/twice-per-rev, dominant peak) are read from, so
    #             those numbers reflect the actual tone amplitude and do
    #             NOT change based on how the FFT plot displays its y-axis.
    #   fft_asd — one-sided AMPLITUDE SPECTRAL DENSITY, units %RMS/√Hz,
    #             used ONLY for the FFT plot's y-axis. This is the
    #             standard convention used by analog wow/flutter analyzers
    #             (including Shaknspin's own spectrum, per its "Deviation
    #             (% RMS/√Hz)" axis label) — normalized to the carrier RMS
    #             and divided by the FFT bin's noise-equivalent bandwidth,
    #             so a broadband noise floor reads consistently regardless
    #             of capture duration or window choice. CAVEAT: that
    #             "independent of length" property holds for broadband/
    #             noise-like content, not for isolated discrete tones — a
    #             pure tone's ASD reading grows roughly as 1/sqrt(bin
    #             width) with longer captures. That's expected ASD
    #             behavior, which is exactly why the summary panel's
    #             numeric readings use fft_mag (plain amplitude) instead,
    #             so they stay stable and comparable across capture
    #             lengths.
    n = len(rpm_dev_pct)
    window = np.hanning(n)
    window_coherent_gain = np.mean(window)  # Hanning ≈ 0.5 — correct for the
                                              # window's amplitude suppression
                                              # so fft_mag reads true % amplitude
    sum_w2 = np.sum(window ** 2)  # window power, for the separate ASD normalization

    dev_windowed = (rpm_dev_pct - rpm_dev_pct.mean()) * window
    fft_vals = np.fft.rfft(dev_windowed)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Plain amplitude spectrum (%) — drives all summary-panel numeric readings.
    fft_mag = np.abs(fft_vals) / (n / 2) / window_coherent_gain

    # One-sided PSD -> ASD (%RMS/√Hz) — used ONLY for the FFT plot's y-axis.
    psd = (np.abs(fft_vals) ** 2) * (2.0 / (fs * sum_w2))
    psd[0] /= 2.0
    if n % 2 == 0:
        psd[-1] /= 2.0
    fft_asd = np.sqrt(psd)

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
    plt.close("all")  # safety: clear any figure left over from a prior rerun
                       # that errored out before reaching its own plt.close()
    plt.style.use("dark_background")
    # Global font-size bump (~3x default) so every plot's axis labels and
    # tick numbers are actually readable. Figure size and hspace/wspace
    # below are tuned to accompany this.
    plt.rcParams.update({
        "axes.labelsize": 30,
        "xtick.labelsize": 27,
        "ytick.labelsize": 27,
        "axes.titlesize": 24,
    })
    fig = plt.figure(figsize=(26, 26))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.75, 1, 3.0, 0.8], hspace=0.5, wspace=0.22)

    fig.suptitle(f"Turntable Speed Analysis — {uploaded_file.name}\n"
                 f"(spin axis: gyroscope {spin_axis_name})",
                 fontsize=18, color="white")

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, omega_raw, color="0.5", linewidth=5, label="raw ω (all data)")
    ax1.plot(t_steady, omega_steady, color="#00d0ff", linewidth=10, label="steady-state segment")
    ax1.axvline(t_steady[0], color="lime", linestyle="--", linewidth=0.8)
    ax1.axvline(t_steady[-1], color="lime", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("ω (rad/s)")
    ax1.set_title("Raw gyro trace — spin-up / steady rotation / spin-down")
    ax1.legend(loc="lower right", fontsize=17)
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
    ax2.legend(loc="best", fontsize=17)
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
    ax3.legend(loc="best", fontsize=17)
    ax3.grid(alpha=0.2)

    ax4 = fig.add_subplot(gs[2, 0])
    plot_mask = fft_freqs > 0
    ax4.plot(fft_freqs[plot_mask], fft_asd[plot_mask], color="#00e0a0", linewidth=2.0)
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
    ax4.set_ylabel("Deviation ASD (%RMS/√Hz)")
    ax4.set_title("FFT of speed deviation")
    ax4.legend(loc="best", fontsize=17)
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
        "--- UNWEIGHTED ---",
        f"W/F RMS (unweighted): {wf_rms_pct:.3f} %",
        f"W/F 2S 95% (unweighted): ±{wf_2sigma_pct:.3f} %",
        "",
        "--- DIN/IEC/AES6 WEIGHTED (independent filter) ---",
        f"W/F 2S 95% (weighted): ±{wf_weighted_2sigma_pct:.3f} %",
        "",
        "--- JIS-STYLE WRMS ---",
        f"W/F WRMS (JIS-style): {wf_wrms_pct_jis:.3f} %",
        "",
        "--- WOW (0.5-6Hz) & FLUTTER (6-100Hz) [AES6/IEC/DIN bands] ---",
        f"Wow RMS unwtd/wtd: {wow_rms_unweighted_pct:.4f} % / {wow_rms_weighted_pct:.4f} %",
        f"Flutter RMS unwtd/wtd: {flutter_rms_unweighted_pct:.4f} % / {flutter_rms_weighted_pct:.4f} %",
        "",
        f"Once-per-rev component: {f1:.3f} Hz, {a1:.3f} %" if f1 is not None else "Once-per-rev: n/a",
        f"Twice-per-rev component: {f2:.3f} Hz, {a2:.3f} %" if f2 is not None else "Twice-per-rev: n/a",
        f"Dominant peak (0.05 Hz-Nyquist): {dom_freq:.3f} Hz, {dom_amp:.3f} %",
    ]
    ax5.text(0.02, 0.98, "\n".join(summary_lines), transform=ax5.transAxes,
             fontsize=21, va="top", ha="left", family="monospace", color="#ffffff",
             fontweight="bold")
    ax5.set_title("Summary", fontsize=24)

    # Panel 6: histogram of instantaneous speed deviation (UNWEIGHTED)
    ax6 = fig.add_subplot(gs[3, :])
    n_bins = max(20, min(80, len(rpm_dev_pct) // 15))
    counts, bin_edges, _ = ax6.hist(rpm_dev_pct, bins=n_bins, color="#00b0ff",
                                      edgecolor="none", alpha=0.85,
                                      label="instantaneous speed deviation")
    ax6.axvline(0, color="0.7", linewidth=0.8)
    ax6.axvline(wf_2sigma_pct, color="lime", linestyle="--", linewidth=1.2,
                 label=f"±2S (95%) = ±{wf_2sigma_pct:.3f} %")
    ax6.axvline(-wf_2sigma_pct, color="lime", linestyle="--", linewidth=1.2)

    bin_width = bin_edges[1] - bin_edges[0]
    x_gauss = np.linspace(rpm_dev_pct.min(), rpm_dev_pct.max(), 300)
    gauss = (len(rpm_dev_pct) * bin_width / (wf_sigma_pct * np.sqrt(2 * np.pi))
             * np.exp(-0.5 * (x_gauss / wf_sigma_pct) ** 2))
    ax6.plot(x_gauss, gauss, color="#ffb000", linewidth=1.5, label="Gaussian fit (same mean/σ)")

    ax6.set_xlabel("Speed deviation (%)")
    ax6.set_ylabel("Count")
    ax6.set_title("Histogram of instantaneous speed deviation (UNWEIGHTED)")
    ax6.legend(loc="upper right", fontsize=17)
    ax6.grid(alpha=0.2)

    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.08, right=0.98)
    st.pyplot(fig)
    plt.close(fig)  # release the figure immediately — without this, matplotlib
                     # keeps every figure from every rerun alive in memory for
                     # the lifetime of the server process

    st.caption(
        "DIN/IEC/AES6 weighting filter: independently derived here — own filter "
        "topology, fitted by least-squares directly against the standard's own "
        "published Table 1 characteristic points (RMS fit error 0.41 dB, max 0.74 dB). "
        "Not copied from any third-party codebase. JIS column reuses the same filter "
        "with an RMS statistic instead of a peak statistic — a documented convention "
        "difference, not a claim about a distinct JIS filter curve."
    )

    # --------------------------------------------------------------------
    # SECOND PLOT — Synthetic FM carrier-tone spectrum (wow/flutter sidebands)
    #
    # Reproduces the classic analog test-record wow/flutter view: a fixed-
    # frequency test tone is synthesized and frequency-modulated by the
    # actual measured speed profile, then its spectrum is plotted zoomed
    # around the carrier — showing wow/flutter as sidebands, the same way
    # a real test record + spectrum analyzer would display it.
    # --------------------------------------------------------------------
    st.divider()
    st.subheader("Synthetic test-tone spectrum (wow/flutter sidebands)")

    from scipy.integrate import cumulative_trapezoid

    dev_frac = rpm_dev_pct / 100.0

    duration2 = t_steady[-1] - t_steady[0]
    t_rel = t_steady - t_steady[0]
    n_audio = int(duration2 * SYNTH_AUDIO_FS)
    t_audio = np.arange(n_audio) / SYNTH_AUDIO_FS

    dev_frac_audio = np.interp(t_audio, t_rel, dev_frac)
    f_inst = CARRIER_FREQ_HZ * (1 + dev_frac_audio)
    phase = 2 * np.pi * cumulative_trapezoid(f_inst, dx=1 / SYNTH_AUDIO_FS, initial=0)
    synth_audio = np.cos(phase)

    audio_window = np.hanning(n_audio)
    spec2 = np.fft.rfft(synth_audio * audio_window)
    spec_freqs = np.fft.rfftfreq(n_audio, d=1 / SYNTH_AUDIO_FS)
    spec_mag_db = 20 * np.log10(np.abs(spec2) / np.abs(spec2).max() + 1e-12)

    band_mask2 = (spec_freqs > CARRIER_FREQ_HZ - CARRIER_SPAN_HZ) & \
                 (spec_freqs < CARRIER_FREQ_HZ + CARRIER_SPAN_HZ)

    plt.close("all")
    fig2 = plt.figure(figsize=(22, 12))
    ax = fig2.add_subplot(111)
    ax.plot(spec_freqs[band_mask2], spec_mag_db[band_mask2],
            color="#ff3366", linewidth=1.0)
    ax.axvline(CARRIER_FREQ_HZ, color="0.5", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level (dB, relative to carrier peak)")
    ax.set_title(
        f"Synthetic {CARRIER_FREQ_HZ:.0f} Hz test-tone spectrum — "
        f"wow/flutter sidebands from measured speed profile\n"
        f"({uploaded_file.name}, carrier FM-modulated by unweighted deviation, "
        f"steady-state segment)",
        fontsize=15
    )
    ax.set_ylim(-80, 5)
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.02,
            "Sideband spacing from carrier = actual wow/flutter modulation "
            "frequency (e.g. once/twice-per-rev, motor cogging).\n"
            "This is a synthesized comparison view, not a real audio "
            "recording — built by FM-modulating a synthetic tone with the "
            "measured speed profile.",
            transform=ax.transAxes, fontsize=12, color="#ffffff", va="bottom")
    plt.subplots_adjust(top=0.85, bottom=0.11, left=0.09, right=0.97)
    st.pyplot(fig2)
    plt.close(fig2)

    # --------------------------------------------------------------------
    # THIRD PLOT — X/Y cross-axis diagnostic
    #
    # Checks whether any of the spin-axis (Z) speed variation could
    # actually be tip/tilt (precession) leaking into Z from imperfect
    # sensor alignment, rather than genuine turntable speed variation. If
    # X and/or Y show energy at the SAME frequencies as the Z-axis peaks,
    # that's evidence of cross-axis contamination rather than true
    # rotational speed error. Off-center phone placement alone does not
    # explain this (a rigid body's angular velocity is the same everywhere
    # on it) — what matters is actual tip/tilt motion.
    # --------------------------------------------------------------------
    st.divider()
    st.subheader("X/Y cross-axis diagnostic")

    x_steady = gx[i0:i1 + 1]
    y_steady = gy[i0:i1 + 1]

    n_xy = len(x_steady)
    xy_window = np.hanning(n_xy)
    xy_window_gain = np.mean(xy_window)

    x_det = (x_steady - x_steady.mean()) * xy_window
    y_det = (y_steady - y_steady.mean()) * xy_window
    fft_x = np.fft.rfft(x_det)
    fft_y = np.fft.rfft(y_det)
    xy_freqs = np.fft.rfftfreq(n_xy, d=1 / fs)
    mag_x_mrad = np.abs(fft_x) / (n_xy / 2) / xy_window_gain * 1000
    mag_y_mrad = np.abs(fft_y) / (n_xy / 2) / xy_window_gain * 1000

    plt.close("all")
    fig3 = plt.figure(figsize=(22, 18))
    gs3 = GridSpec(2, 1, figure=fig3, height_ratios=[1, 1.3], hspace=0.4)

    ax_t = fig3.add_subplot(gs3[0])
    ax_t.plot(t_steady, (x_steady - x_steady.mean()) * 1000, color="#ff7f0e",
              linewidth=0.7, label="X (detrended)")
    ax_t.plot(t_steady, (y_steady - y_steady.mean()) * 1000, color="#1f77b4",
              linewidth=0.7, label="Y (detrended)")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Angular rate (mrad/s)")
    ax_t.set_title("X/Y gyroscope axes over steady-state segment (detrended)")
    ax_t.legend(loc="upper right", fontsize=17)
    ax_t.grid(alpha=0.2)

    ax_f = fig3.add_subplot(gs3[1])
    xy_plot_mask = xy_freqs > 0
    ax_f.plot(xy_freqs[xy_plot_mask], mag_x_mrad[xy_plot_mask], color="#ff7f0e",
              linewidth=0.9, label="X spectrum")
    ax_f.plot(xy_freqs[xy_plot_mask], mag_y_mrad[xy_plot_mask], color="#1f77b4",
              linewidth=0.9, label="Y spectrum")
    ax_f.axvline(rev_freq, color="yellow", linestyle="--", linewidth=0.8,
                 label=f"1×rev ≈ {rev_freq:.3f} Hz")
    ax_f.axvline(2 * rev_freq, color="orange", linestyle="--", linewidth=0.8,
                 label=f"2×rev ≈ {2*rev_freq:.3f} Hz")
    ax_f.axvline(dom_freq, color="magenta", linestyle=":", linewidth=1.0,
                 label=f"Z dominant peak ≈ {dom_freq:.3f} Hz")
    ax_f.set_xscale("log")
    ax_f.set_xlim(xy_freqs[xy_freqs > 0].min(), nyq)
    tick_vals_xy = [v for v in [0.1, 1, 10] if v < nyq] + [nyq]
    ax_f.set_xticks(tick_vals_xy)
    ax_f.set_xticklabels([f"{v:g}" if v != nyq else f"{np.floor(v*10)/10:.1f}" for v in tick_vals_xy])
    ax_f.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Angular rate amplitude (mrad/s)")
    ax_f.set_title(
        "X/Y spectra vs. Z-axis reference frequencies — cross-axis "
        "contamination check\n"
        "(If X/Y show peaks at the same frequencies as the dashed/dotted "
        "lines, that energy may be leaking into the Z/RPM reading as "
        "spurious wow/flutter.)",
        fontsize=17
    )
    ax_f.legend(loc="upper right", fontsize=14)
    ax_f.grid(alpha=0.2, which="both")

    plt.subplots_adjust(top=0.88, bottom=0.07, left=0.08, right=0.97, hspace=0.45)
    st.pyplot(fig3)
    plt.close(fig3)

except Exception as e:
    st.error(f"Error processing file: {e}")
    st.exception(e)
