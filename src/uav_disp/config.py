"""Scene/measurement constants shared across the pipeline."""

# Single decode-pass crop of the 4K frame covering both targets (x, y, w, h)
ROI = (1600, 600, 1344, 1216)

# Approximate target centers in ROI-local coordinates (refined by detect.find_board)
CABLE_APPROX = (300, 290)   # checkerboard on the vibrating cable
REF_APPROX = (907, 738)     # stationary reference checkerboard on tripod

APPROX_SQUARE_PX = 30       # rough checker square size for detection

TARGET_MM = 55.0            # full 2x2 board side -> one square is 27.5 mm
SQUARE_MM = TARGET_MM / 2

FPS = 50.0
LDS_FS = 10_000.0
MAX_LAG_S = 1.0             # LDS/video clock offset is < 1 s (data_notes.txt)

# Band of real cable motion: VIV modes at ~3.9 and ~11.7 Hz; LDS has <0.2% of its
# power below 2 Hz and almost none above 14 Hz. The band-pass removes the
# parallax drift left after translation compensation and high-freq tracker noise.
HIGHPASS_HZ = 3.0
LOWPASS_HZ = 14.0

# Fusion weight for the ZNCC signal when combining with CoTracker3 (~inverse-
# variance weighting of the two trackers' independent noise components)
FUSION_W_ZNCC = 0.8

# --- phase-based branch (Chen 2015 / Yang 2024; see reference_paper/IMPLEMENTATION_PLAN.md) ---
PHASE_SPEC        = dict(n_scales=4, n_orients=2, half_octave=True, twidth=1.0)
PHASE_PATCH       = 192          # target patch side, px
PHASE_PAD         = 32           # reflect pad per side before the FFT (256^2 spectra)
PHASE_COARSE_STEP = 1.0          # re-cut threshold, px (piecewise-constant integer cut)
PHASE_COARSE_K    = 51           # median filter length on the coarse track (frames)
PHASE_COND_MAX    = 50.0         # per-frame cond(M) flag threshold
PHASE_DECIMATE    = 1            # Chen-style spatial decimation before filtering (ablation: 1, 2, 4)
PHASE_BLOCK_W     = 0.25         # H.264 macroblock-boundary weight
PHASE_BAND        = None         # default output band for phase methods: None = full band (0-25 Hz)
EGO_TILE, EGO_N_TILES, EGO_HUBER_K, EGO_NEAR_PX = 128, 40, 1.345, 300
MAGNIFY_BANDS     = [(3.7, 4.1, 20.0), (11.5, 11.9, 50.0)]   # (f_lo, f_hi, alpha)
GDGIF_H, GDGIF_MU = 16, 0.022
ROW_TIME_S        = None         # rolling-shutter tau (s/row); filled by run_egomotion.py
SCORE_BANDS       = ((0.0, 3.0), (3.0, 14.0), (14.0, 25.0))  # per-band scoring vs LDS
