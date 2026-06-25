# ISRO BAH 2026 — Problem Statement 7
# AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves
## Complete Master Reference: Strategy, Implementation & Submission Guide

> **Compiled from:** `exoplanet_hackathon_guide.md` · `AstroNet_Transit_ISRO_BAH2026.docx` · `Exoplanet_Hackathon_Guide.pdf` · BAH 2026 Official Website · Internet Research (ExoMiner++, SHERLOCK, TRICERATOPS)
>
> *Document prepared for ISRO Bharatiya Antariksh Hackathon 2026 · Grand Finale: 6–7 August 2026*

---

## Table of Contents

1. [Hackathon Context](#1-hackathon-context)
2. [Problem Analysis](#2-problem-analysis)
3. [Signal Classes & Discriminators](#3-signal-classes--discriminators)
4. [Full Pipeline Architecture](#4-full-pipeline-architecture)
5. [Data Sources & Acquisition](#5-data-sources--acquisition)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Period Search — BLS & TLS](#7-period-search--bls--tls)
8. [SNR & Statistical Significance](#8-snr--statistical-significance)
9. [AI Classification Framework](#9-ai-classification-framework)
10. [Parameter Estimation](#10-parameter-estimation)
11. [Uncertainty Estimation](#11-uncertainty-estimation)
12. [Evaluation Strategy & Metrics](#12-evaluation-strategy--metrics)
13. [Visualization Requirements](#13-visualization-requirements)
14. [Full Tech Stack & Libraries](#14-full-tech-stack--libraries)
15. [Professional-Level Tools (External Research)](#15-professional-level-tools-external-research)
16. [Professional-Level Datasets (External Research)](#16-professional-level-datasets-external-research)
17. [Validation Targets](#17-validation-targets)
18. [Implementation Timeline](#18-implementation-timeline)
19. [Report Writing Strategy](#19-report-writing-strategy)
20. [Demo Pitch Structure](#20-demo-pitch-structure)
21. [Key Formulas & Quick Reference](#21-key-formulas--quick-reference)
22. [Key References](#22-key-references)
23. [Quick-Start Code Block](#23-quick-start-code-block)
24. [Additional / Supplementary Information](#24-additional--supplementary-information)

---

## 1. Hackathon Context

**Event:** Bharatiya Antariksh Hackathon 2026 (BAH 2026) — Third Edition
**Organiser:** Indian Space Research Organisation (ISRO), powered by Hack2skill
**Problem Statement:** PS-07 — AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves
**Format:** 30-hour grand finale hackathon (6–7 August 2026)
**Team Size:** 3–4 members
**Eligibility:** Undergraduate, Postgraduate, and PhD students enrolled in Indian institutions. Working professionals are **not** eligible.
**Cost:** Free to participate

### Timeline

| Milestone | Date |
|---|---|
| Registration & Idea Submission Opens | 10 June 2026 |
| Problem Statement Explainer Session 1 | 15 June 2026 |
| Problem Statement Explainer Session 2 | 16 June 2026 |
| Registration & Idea Submission Ends | 1 July 2026 |
| Final Shortlist Announcement | 20 July 2026 |
| Induction Session | 21 July 2026 |
| **Grand Finale** | **6–7 August 2026** |

### Rewards & Benefits

- Recognition by ISRO and opportunity to be considered for an **internship at ISRO**
- II AC travel fare reimbursed for all finalists
- Mentorship from ISRO scientists and domain experts throughout the hackathon
- National-level recognition on a prestigious platform

### Contact

- **Email:** support+isrobah2026@hack2skill.com
- **Discord:** https://discord.gg/KMKtbxBJpW
- **Hackathon Page:** https://hack2skill.com/event/bah2026/

---

## 2. Problem Analysis

### What Are We Detecting?

This hackathon demands a complete ML pipeline for astronomical signal detection. The core challenge is a **multi-class classification + parameter estimation problem on time-series data**.

When a planet passes in front of its host star (a **transit**), it blocks a small fraction of the star's light, creating a periodic, tiny dip in the **light curve** (brightness vs. time). TESS (Transiting Exoplanet Survey Satellite) captures these events at 2-minute cadence, observing approximately 20,000 stars per sector.

Transit depth for an Earth-sized planet around a Sun-like star is approximately **84 ppm (0.008%)** — smaller than most noise sources. The problem statement explicitly requires:

- Detection of transit signals
- Multi-class classification (planet vs. non-planet signals)
- Estimation of physical parameters (period, duration, depth) with uncertainty
- Calibrated confidence / SNR scores

### Why Is This Hard?

- Transit depth for an Earth-sized planet is ~84 ppm (0.008%) — smaller than most noise sources
- TESS full-frame images contain crowded fields with blending from nearby stars
- Each TESS sector = ~27 days of continuous data per star
- A single sector may capture only 2–3 transit events for long-period planets
- Instrumental systematics (momentum dumps, scattered light) mimic transit-like features
- Periodic transit signals buried in correlated (red) noise and instrumental systematics
- Stars with TESS magnitude < 6 are prone to saturation artefacts
- Light curves with fewer than ~500 valid cadences post-clipping have insufficient phase coverage

### Specific Challenges

| Challenge | Description |
|---|---|
| **Challenge 1** | Periodic transit signals buried in correlated (red) noise and instrumental systematics |
| **Challenge 2** | Distinguishing true planetary transits from eclipsing binaries (EBs), blends, and starspots |
| **Challenge 3** | Estimating physical parameters (period, duration, depth) accurately from sparse photometry |
| **Challenge 4** | Providing calibrated confidence/SNR so mission planners can triage candidates reliably |
| **Challenge 5** | Running the full pipeline efficiently without expensive HPC infrastructure |

> **Prize Tip (from PDF guide):** Most teams only do detection. Win by delivering all four: detection + classification + parameter estimation + confidence intervals. The evaluation criteria explicitly rewards classification accuracy AND parameter accuracy.

---

## 3. Signal Classes & Discriminators

### The Four Signal Classes to Classify

| Signal Type | Cause | Key Distinguishing Features |
|---|---|---|
| **Exoplanet Transit (PC)** | Planet crossing the stellar disk | Small, flat-bottomed dip; depth <3%; no secondary eclipse (or tiny); odd/even depths equal; centroid stable |
| **Eclipsing Binary (EB)** | Two stars orbiting each other | Deep primary + secondary eclipse (depth often >1%); secondary eclipse at phase 0.5; odd/even depth difference; V-shaped or grazing profile; ellipsoidal variations |
| **Background Blend / False Positive** | Contaminating background eclipsing binary diluted by target star flux | Centroid shifts during eclipse; aperture-dependent depth; common in crowded TESS fields |
| **Stellar Variability / Other FP** | Stellar magnetic activity (starspots, pulsations) | Quasi-periodic or sinusoidal modulation; not strictly periodic; no flat-bottom feature; often longer timescale or irregular period; asymmetric dips |

A fifth class sometimes distinguished explicitly:

| Signal Type | Cause | Notes |
|---|---|---|
| **Instrumental / Systematic** | Detector noise, satellite roll, momentum dumps | Non-astrophysical; correlated with TESS quality flags |

### Key Vetting / Discriminating Tests

| Test | What It Detects | Threshold |
|---|---|---|
| **Odd/Even depth difference** | Eclipsing binary | > 3σ difference → EB |
| **Secondary eclipse depth** | EB or hot Jupiter | Depth > 200 ppm at phase 0.5 → EB |
| **Centroid shift** | Background blend | Shift > 3σ during transit → Blend |
| **V-shape vs. flat bottom** | Grazing EB vs. planet | V-shaped ingress+egress → EB |
| **Transit depth threshold** | Planet vs. EB | Depth > 1% → likely EB |
| **χ² improvement** | Real signal vs. noise | Significant χ² improvement → real signal |
| **CROWDSAP metric** | Dilution by neighbours | < 0.9 → investigate blending |

---

## 4. Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       FULL PIPELINE OVERVIEW                            │
│                                                                         │
│  [1] DATA INGESTION      [2] PREPROCESSING     [3] PERIOD SEARCH        │
│  ─────────────────       ─────────────────     ─────────────────        │
│  lightkurve download  →  Quality masking    →  BLS / TLS algorithm      │
│  MAST/TESS Archive       Detrending (Wotan)    SDE & SNR calculation    │
│  CBV correction          Outlier removal        Period vetting           │
│  Async HTTP bulk DL      Normalization          Multi-planet iteration   │
│                          Gap filling/masking                             │
│                                                                         │
│  [4] FEATURE EXTRACTION  [5] CLASSIFICATION    [6] PARAMETER FIT        │
│  ─────────────────────   ─────────────────     ─────────────────        │
│  Transit shape features  CNN (AstroNet-style)  batman transit model     │
│  Centroid analysis    →  + XGBoost ensemble →  MCMC (emcee)            │
│  Secondary eclipse       OR                     Uncertainty estimation  │
│  Odd/even depth diff     Bi-LSTM + Transformer  Posterior plots         │
│  Duration/period ratio   4-class softmax        Corner plots            │
│                          + confidence score                             │
│                                                                         │
│  [7] OUTPUT & REPORTING                                                 │
│  ─────────────────────                                                  │
│  Phase-folded LC plot · Classification label · Transit parameters       │
│  SNR / confidence · 3-page PDF report · Candidate catalogue CSV         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage-by-Stage Description

| Stage | Actions & Outputs |
|---|---|
| **Stage 0: Data Ingest** | Download TESS 2-min LC via Lightkurve/astroquery from MAST. Store as HDF5. Process all ~20k targets in the sector. Async HTTP parallel download for 4× speed improvement. |
| **Stage 1: Preprocessing** | Normalize → sigma-clip → flatten (biweight/SG filter) → quality mask → gap interpolation → store cleaned flux array. Discard LCs with <500 valid cadences. |
| **Stage 2: Periodicity Screen** | Run BLS/TLS on all light curves. Keep targets with SDE > 5.0 (loose threshold). Compute SDE, SR, period, t0, duration for each. |
| **Stage 3: Feature Extraction** | For each candidate: compute all 8+ engineered features (odd/even depth, secondary eclipse, centroid, v-shape, CROWDSAP, etc.). |
| **Stage 4: Phase Folding** | Fold at BLS/TLS period. Create 201-pt global view + 61-pt local view (zoomed on transit center). These are CNN inputs. |
| **Stage 5: ML Classification** | Run dual-view CNN. Get 4-class softmax probabilities. Run XGBoost on features. Ensemble: weighted average of both. Output: class label + confidence score. |
| **Stage 6: Parameter Fit** | For planet-class detections (probability > 0.70): run batman + MCMC. Extract P, T14, δ with uncertainties. |
| **Stage 7: Visualisation** | Generate: raw LC plot, detrended LC, BLS/TLS power spectrum, phase-folded LC + model, parameter corner plot, classification confidence bar chart. |

---

## 5. Data Sources & Acquisition

### Primary Dataset (Required by Problem Statement)

| Source | URL | Description |
|---|---|---|
| **TESS Input Catalog (TIC) + Bulk Download** | https://archive.stsci.edu/tess/tic_ctl.html | ~20–30k light curves per sector; 2-min cadence; required by problem statement |
| **MAST Portal** | https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html | Full query interface for all TESS products |
| **TESScut** | https://mast.stsci.edu/tesscut/ | Pixel-level cutouts for centroid analysis (TPF data) |

**Recommended Sector:** Sector 1 (Southern sky, Jul–Aug 2018) — well-studied, best validation data available.

### Training / Labeled Datasets

| Source | URL | What It Contains |
|---|---|---|
| **ExoFOP-TESS** | https://exofop.ipac.caltech.edu/tess/ | TESS Objects of Interest (TOIs): confirmed planets, false positives, EBs. ~5,000 confirmed planets, ~8,000 FPs, ~4,000 EBs |
| **NASA Exoplanet Archive** | https://exoplanetarchive.ipac.caltech.edu/ | Confirmed exoplanets + false positives with labels |
| **TESS TOI Catalog (MIT)** | https://tess.mit.edu/toi-releases/ | MIT TOI list with disposition labels (PC, FP, EB, etc.) |
| **AstroNet Training Data (Google)** | https://github.com/google-research/exoplanet-ml | Kepler labeled dataset (34,032 TCEs) used to train Google's AstroNet |
| **Kepler DR24 TCE Catalog** | https://archive.stsci.edu/kepler/ | Thompson et al. 2018 — 34,032 labeled transit candidates |
| **Kaggle — NASA Kepler** | https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results | Pre-labeled Kepler candidates (CONFIRMED / FALSE POSITIVE) |
| **TESS Eclipsing Binary Catalog** | https://archive.stsci.edu/tess/tic_ctl.html | Known EBs for binary classification training |

### Accessing Data via Python

```python
import lightkurve as lk
from astroquery.mast import Catalogs, Tesscut, Observations

# Search and download a single star's light curve (2-min cadence)
search_result = lk.search_lightcurve("TIC 261136679", mission="TESS", cadence="short")
lc = search_result[0].download()

# Sector-wide bulk approach — query Observations table
obs = Observations.query_criteria(
    obs_collection='TESS',
    dataproduct_type='timeseries',
    sequence_number=1  # Sector 1
)
# Filter for 2-min cadence (short cadence) products

# Query TIC for stars in a region
tic_catalog = Catalogs.query_region("83.82 -5.39", radius=1, catalog="TIC")

# Bulk download sector light curves
search = lk.search_lightcurve("Sector 1", mission="TESS", cadence="short")
lc_collection = search.download_all(quality_bitmask="default")
```

---

## 6. Preprocessing Pipeline

### Step 1: Quality Masking & Normalization

```python
import lightkurve as lk
import numpy as np

lc = lk.search_lightcurve("TIC 261136679", mission="TESS",
                            cadence="short", sector=1)[0].download()

# Remove NaNs and quality-flagged cadences
lc = lc.remove_nans().remove_outliers(sigma=5)  # 5σ sigma-clipping

# Normalize flux (median = 1.0)
lc = lc.normalize()
```

Key quality flags to remove: bit 1 (AttitudeTweak), bit 2 (SafeMode), bit 8 (CosmicRay), bit 16 (ManualExclude).

Stars with TESS magnitude < 6 should be excluded to avoid saturation artefacts.

### Step 2: Detrending (Flattening Stellar Variability)

Stellar activity, instrumental trends, and long-term systematics must be removed before searching for transits. **Critical: Don't over-flatten or you will erase the transit signal!**

```python
from wotan import flatten

# Biweight detrending — most robust against transits (recommended default)
flat_flux, trend = flatten(
    lc.time.value,
    lc.flux.value,
    method='biweight',
    window_length=0.75,   # days; tune based on expected transit duration
    return_trend=True
)
```

#### Detrending Strategies to Compare

| Method | Notes |
|---|---|
| `biweight` (Wotan) | Robust, recommended default — resists outlier influence |
| `cofiam` | Cointegrated Frequency Analysis — good for TESS |
| `gp` / `celerite2` | Gaussian Process with Matérn-3/2 kernel — most accurate, slower |
| **CBV Corrector** | Cotrending Basis Vectors via `lightkurve`'s `CBVCorrector` — removes spacecraft systematics |
| Savitzky-Golay | Classic flatten method; window_length=401 cadences |

Also from the PDF guide:
- Gap Filling: TESS has ~13-day data gaps per sector. Interpolate or mask. Mark gap-edge points as unreliable.
- Phase Folding: Once period found, fold at period P, plot vs. phase. Transit appears as dip at phase 0.

### Step 3: Additional Preprocessing Notes (from Submission Doc)

- Outlier sigma-clipping at 3σ (iterative) removes cosmic-ray hits and single-cadence anomalies
- Biweight mid-correlation detrending removes long-term stellar variability whilst preserving transit shapes
- Gaussian Process (GP) noise modelling with Matérn-3/2 kernel estimates correlated noise structure
- Data augmentation (phase-folding jitter, flux rescaling, noise injection, transit time jitter) expands training set 5–10× at no extra cost
- Light curves with fewer than 500 valid cadences post-clipping are discarded as insufficient
- The contamination ratio (CROWDSAP) from the TESS Input Catalog is used to flag high-blend targets (contamination > 0.3) and weight the blend classifier output

---

## 7. Period Search — BLS & TLS

### Box Least Squares (BLS) — The Standard Algorithm

BLS is the gold-standard algorithm for detecting periodic box-shaped transit signals. It searches over a grid of trial periods, transit durations, and phases, finding the combination that best fits a box-shaped dip. **Always run BLS first — it gives you candidate periods for ML input features.**

```python
from astropy.timeseries import BoxLeastSquares
import numpy as np

model = BoxLeastSquares(lc.time.value, lc.flux.value, dy=lc.flux_err.value)
periods = np.linspace(0.5, 13, 10000)  # Search 0.5–13 day periods
durations = np.linspace(0.01, 0.5, 50)  # Transit durations in days
result = model.power(periods, durations)

best_period = result.period[np.argmax(result.power)]
best_t0 = result.transit_time[np.argmax(result.power)]
best_dur = result.duration[np.argmax(result.power)]

# Signal Detection Efficiency (SDE) — your SNR metric!
SDE = (result.power.max() - result.power.mean()) / result.power.std()
# SDE > 7 is conventionally significant
```

The submission document also uses a GPU-accelerated BLS via a CuPy wrapper, scanning from 0.5 to 30 days at 50,000 frequency steps. BLS outputs:
- Best-fit period, epoch, transit duration, and depth
- BLS Signal Residue (SR) and Signal Detection Efficiency (SDE) as pre-filter metrics
- Phase-folded light curve centred on the deepest transit for downstream classification

### Transit Least Squares (TLS) — Recommended Over BLS

TLS is specifically designed for transit shapes and uses a physically motivated model (realistic ingress/egress) instead of a box. It detects **10–15% more planets, especially small ones.**

```python
from transitleastsquares import transitleastsquares

model = transitleastsquares(lc.time.value, flat_flux)
results = model.power(
    oversampling_factor=5,
    duration_grid_step=1.05,
    use_threads=4
)

print(f"Period: {results.period:.5f} days")
print(f"Transit depth: {results.depth:.6f}")
print(f"Duration: {results.duration:.5f} days")
print(f"SNR: {results.snr:.2f}")
print(f"SDE: {results.SDE:.2f}")
```

**Vetting thresholds:**
- SDE > 7 → strong signal candidate
- SNR > 7 → significant detection
- Transit depth < 1% → planet candidate (> 1% likely EB)

### Multi-Planet Search Strategy

After finding the first signal, **subtract/mask it and re-run BLS/TLS to find additional planets**. This iterative residual search is how multi-planet systems are detected. Implement at least 3 iterations to demonstrate thoroughness.

---

## 8. SNR & Statistical Significance

```python
import scipy.stats as stats

# Compute CDPP (Combined Differential Photometric Precision)
cdpp = lc.estimate_cdpp(transit_duration=results.duration * 24)

# Transit SNR
n_transits = int((lc.time.value[-1] - lc.time.value[0]) / results.period)
depth_ppm = results.depth * 1e6
snr = depth_ppm / (cdpp.value / np.sqrt(n_transits))

# Statistical significance (bootstrap or analytic)
z_score = results.SDE  # approximation
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

# Signal Detection Efficiency (from BLS)
SDE = (P_max - mean(P)) / std(P)  # > 7.0 = significant detection
```

Reporting strategy:
- Report SDE (Signal Detection Efficiency) from BLS/TLS. **SDE > 7.0 = significant detection.**
- Transit SNR = depth / (noise_per_point × sqrt(n_in_transit))
- Per-event SNR = δ / σ_per_transit
- Classifier confidence = max softmax probability (calibrated via temperature scaling; ECE < 0.04 target)
- CDPP (Combined Differential Photometric Precision) at 1-hour timescale as photometric noise floor
- Colour-coded confidence tiers: **Gold (> 0.90)**, **Silver (0.70–0.90)**, **Bronze (< 0.70)**

---

## 9. AI Classification Framework

### 4-Class Taxonomy

```
INPUT: Vetting metrics + phase-folded light curve
                    ↓
        ┌───────────────────────┐
        │   CLASSIFIER MODULE   │
        └───────────────────────┘
                    ↓
    ┌───────────────────────────────────┐
    │  Class 0: PLANET CANDIDATE (PC)   │
    │  Class 1: ECLIPSING BINARY (EB)   │
    │  Class 2: BACKGROUND BLEND (BG)   │
    │  Class 3: OTHER / FALSE POSITIVE  │
    └───────────────────────────────────┘
                    ↓
        Confidence score [0.0 – 1.0]
```

---

### Approach A: Dual-View CNN (AstroNet-Style) — Primary Approach

Inspired by **AstroNet (Shallue & Vanderburg 2018)** and **AstroNet-Triage (Yu et al. 2019)**, this is the **state-of-the-art approach** — any judge familiar with exoplanet science will immediately recognise this.

Use **two CNN inputs**:
- **Global view:** Full phase-folded curve (2001 points)
- **Local view:** Zoomed 201-point transit center

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_transit_cnn(input_length=201):
    """Stage 1: 1D CNN on Phase-Folded Light Curve"""
    inp = keras.Input(shape=(input_length, 1))
    # Global view
    x = layers.Conv1D(16, 5, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    # Dense head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(4, activation='softmax')(x)  # 4 classes
    return keras.Model(inp, out)

# Classes: 0=Planet, 1=EB, 2=Blend/FP, 3=Stellar Var

# Phase-fold and bin to fixed-length array
def phase_fold(time, flux, period, t0, nbins=201):
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    idx = np.argsort(phase)
    binned = np.interp(np.linspace(-0.5, 0.5, nbins),
                       phase[idx], flux[idx])
    return binned
```

**Training strategy:**
- Pre-train on Kepler labeled data (AstroNet dataset — 34,032 TCEs)
- Fine-tune on TESS-specific TOI catalog from ExoFOP
- Data augmentation: noise injection, transit time jitter, flux scaling, phase-folding jitter

---

### Approach B: Bi-LSTM + Transformer Ensemble (AstroNet-Transit, Novel Approach)

*Source: AstroNet_Transit_ISRO_BAH2026.docx — Submission Document*

This is the architecture proposed in the ISRO submission. Uses **PyTorch** instead of TensorFlow/Keras.

**Two-stage deep learning classifier:**

- **Stage A — Bi-LSTM (64 hidden units, 2 layers):** Captures temporal dependencies in the flux time series; sensitive to asymmetric ingress/egress shapes that differentiate EBs from planets; halves parameter count (~1.2M vs. ~2.5M in AstroNet) while improving sensitivity to asymmetric signals.
- **Stage B — Transformer Encoder (4 heads, 2 layers, d_model=128):** Attends to the global shape and secondary eclipse phase; vital for blend detection.
- **Late Fusion:** Outputs of both branches concatenated and passed through a 3-layer MLP → 4-class softmax.

| Framework Parameter | Value |
|---|---|
| Framework | PyTorch 2.x; single NVIDIA T4 GPU (~14 GB VRAM) |
| Loss Function | Weighted cross-entropy (class weights inversely proportional to label frequency) |
| Optimiser | AdamW, lr=1×10⁻³, weight decay=1×10⁻⁴; cosine annealing LR schedule over 50 epochs |
| Regularisation | Dropout (p=0.3) after each LSTM layer; label smoothing (ε=0.1) |
| Validation | Stratified k-fold (k=5); final model selected by F1-macro on validation set |
| Classification Threshold | Softmax probability > 0.70 for transit class triggers parameter estimation (tunable) |
| Experiment Tracking | MLflow — model versioning, metric logging, reproducibility |
| Training Dataset | ~85,000 augmented samples (5,000 confirmed planets + 8,000 FPs + 4,000 EBs × augmentation) |
| Train/Val/Test Split | 80/10/10 stratified |

---

### Approach C: XGBoost on Engineered Features (Ensemble Stage 2)

```python
import xgboost as xgb

def extract_features(results, lc, flat_flux):
    return {
        # Transit geometry
        'transit_depth': results.depth,
        'transit_duration': results.duration,
        'period': results.period,
        'rp_rs': results.rp_rs,
        'b': results.b,

        # Statistical discriminators
        'sde': results.SDE,
        'snr': results.snr,
        'odd_even_mismatch': compute_odd_even_diff(lc, results),  # Key EB test
        'secondary_depth': find_secondary_eclipse(flat_flux, results.period),
        'duration_ratio': results.duration / results.period,

        # Centroid motion (blend indicator)
        'centroid_shift': compute_centroid_shift(lc, results),    # From TPF pixels
        'contamination_ratio': lc.meta.get('CROWDSAP', 1.0),

        # Light curve shape
        'ingress_egress_ratio': results.duration_partial / results.duration,
        'chi2_flat': results.chi2_min / results.chi2,

        # Period coherence
        'period_snr': results.period / results.period_uncertainty if hasattr(results, 'period_uncertainty') else 0,
    }

# Odd/even depth difference — strongest EB discriminator
def compute_odd_even_diff(lc, results):
    """EB transits alternate depth (odd shallower than even)"""
    odd_transits = [i for i in range(len(results.transit_times)) if i % 2 == 1]
    even_transits = [i for i in range(len(results.transit_times)) if i % 2 == 0]
    return abs(odd_depth - even_depth) / ((odd_depth + even_depth) / 2)
```

#### Engineered Features Table

| Feature Name | Physical Meaning & Why It Matters |
|---|---|
| `odd_even_depth_diff` | Difference between odd and even transit depths — key EB discriminator |
| `secondary_eclipse_depth` | Depth at phase 0.5 — present in EBs, absent in planet transits |
| `centroid_shift` | Pixel centroid offset during transit — non-zero means blend/background EB |
| `duration_over_period` | Transit duration / orbital period — diagnostic of impact parameter |
| `depth_ratio` | Transit depth / photometric noise — low = likely FP |
| `v_shape_metric` | Ingress+egress duration / total duration — V-shape = EB, flat = planet |
| `SDE` | Signal Detection Efficiency from BLS — detection significance |
| `period_snr` | Period coherence metric — spurious periods have lower coherence |

---

### Ensemble Combination

```python
# Soft voting ensemble
p_cnn = cnn_model.predict(phase_folded_lc)     # shape: (4,)
p_xgb = xgb_model.predict_proba(features_df)   # shape: (4,)

final_proba = 0.6 * p_cnn + 0.4 * p_xgb       # weighted average
predicted_class = np.argmax(final_proba)
confidence = final_proba[predicted_class]
```

---

### Explainability — SHAP Values

Use `shap` to explain which features drove each classification. This is a significant differentiator that judges will appreciate:

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(features_df)
shap.summary_plot(shap_values, features_df)
```

---

## 10. Parameter Estimation

### batman — Mandel & Agol (2002) Transit Model

`batman` (Bad-Ass Transit Model cAlculatioN) computes the Mandel & Agol (2002) transit model in milliseconds. Use it to fit the phase-folded light curve and extract the three required parameters: orbital period P, transit duration T₁₄, and transit depth δ.

```python
import batman
import scipy.optimize as opt

params = batman.TransitParams()
params.t0 = 0.             # Time of central transit
params.per = best_period   # Orbital period [days]
params.rp = 0.1            # Planet/star radius ratio sqrt(depth)
params.a = 10.             # Semi-major axis [stellar radii]
params.inc = 89.           # Orbital inclination [degrees]
params.ecc = 0.            # Eccentricity (0 for circular)
params.w = 90.             # Longitude of periastron
params.u = [0.3, 0.1]     # Limb darkening coefficients
params.limb_dark = 'quadratic'

m = batman.TransitModel(params, phase_times)
flux_model = m.light_curve(params)
```

The submission doc also uses `scipy.optimize.minimize` (Nelder-Mead) initialised from BLS priors as a fast analytic fit step before MCMC, which delivers orbital parameters in the same run with no separate post-processing step.

---

### batman + emcee — Full MCMC Parameter Estimation

Use `emcee` for MCMC sampling to get proper Bayesian uncertainty estimates on all parameters. This satisfies the 'confidence level' and 'uncertainty estimation' requirement in the evaluation criteria. **Report median ± 1σ from the posterior distribution for each parameter.**

```python
import emcee

def batman_model(params_array, t):
    p = batman.TransitParams()
    p.t0  = params_array[0]   # Mid-transit time
    p.per = params_array[1]   # Orbital period (days)
    p.rp  = params_array[2]   # Rp/Rs (planet-to-star radius ratio)
    p.a   = params_array[3]   # Semi-major axis / stellar radius
    p.inc = params_array[4]   # Orbital inclination (degrees)
    p.ecc = 0                 # Circular orbit (simplification)
    p.w   = 90
    p.u   = [0.3, 0.3]        # Limb darkening (quadratic)
    p.limb_dark = "quadratic"
    m = batman.TransitModel(p, t)
    return m.light_curve(p)

def log_likelihood(params, t, flux, flux_err):
    model = batman_model(params, t)
    return -0.5 * np.sum(((flux - model) / flux_err) ** 2)

def log_prior(params):
    t0, per, rp, a, inc = params
    if (0 < rp < 0.5 and 0 < per < 100 and
            0 < a < 200 and 50 < inc < 90):
        return 0.0
    return -np.inf

def log_probability(params, t, flux, flux_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t, flux, flux_err)

# MCMC sampling
nwalkers, ndim = 32, 5
initial = [t0_guess, period_guess, rp_guess, a_guess, inc_guess]
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                  args=(t_transit, flux_transit, flux_err))
sampler.run_mcmc(pos, 5000, progress=True)

# Extract median + 16th/84th percentile uncertainties
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
params_median = np.median(flat_samples, axis=0)
params_err_lo = params_median - np.percentile(flat_samples, 16, axis=0)
params_err_hi = np.percentile(flat_samples, 84, axis=0) - params_median
```

**Validation check:** MCMC chains converged when acceptance fraction is **0.2–0.5**.

### Output Parameters Per Transit Candidate

| Parameter | Symbol | Unit | How Derived |
|---|---|---|---|
| Orbital period | P | days | TLS + MCMC refinement |
| Transit duration | T₁₄ | hours | TLS + batman fit |
| Transit depth | δ | ppm | batman — (Rp/Rs)² |
| Planet radius ratio | Rp/Rs | — | batman core fit parameter |
| Impact parameter | b | — | batman — (a/Rs) × cos(i) |
| Orbital inclination | i | degrees | batman |
| Semi-major axis | a/Rs | stellar radii | batman |
| SNR | — | — | Depth / RMS in phase |
| False positive prob. | FPP | % | Classifier confidence |

---

## 11. Uncertainty Estimation

### Comprehensive Uncertainty Strategy (from Submission Doc)

| Uncertainty Type | Method |
|---|---|
| **Photometric noise floor** | CDPP metric at 1-hour timescale per light curve; used to set detection floor |
| **Period uncertainty** | Half-width of BLS/TLS peak above noise level (formal period error); confirmed via phase-fold residual minimisation |
| **Transit parameter uncertainties** | 1σ credible intervals from 100-walker MCMC bootstrap (emcee); run only for high-confidence transits (score > 0.85) to preserve throughput |
| **Classification confidence** | Multi-class softmax probability; calibrated via **temperature scaling** on held-out validation set to ensure well-calibrated probabilities (ECE < 0.04) |
| **Blend contamination** | Propagated using TIC contamination ratio (CROWDSAP); reported as additive depth correction with uncertainty |

Report median ± 1σ from the posterior distribution. Use the `corner` package for visualisation.

---

## 12. Evaluation Strategy & Metrics

### Training / Validation Split

| Dataset | Purpose | Source |
|---|---|---|
| Kepler confirmed PCs + FPs | Pre-training CNN | AstroNet dataset (Google, 34,032 samples) |
| TESS TOI labeled catalog | Fine-tuning + validation | ExoFOP-TESS |
| Held-out TESS sector | Blind test set | Random 20% of downloaded LCs |

### Classification Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(classification_report(y_true, y_pred,
      target_names=['Planet', 'EB', 'Blend', 'Other']))

# ROC-AUC per class (one-vs-rest)
auc = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovr')
```

Key metrics to report:
- **Precision / Recall / F1** for each class
- **False Positive Rate for planets** — most critical; don't want false planet detections
- **Confusion matrix** (visualised as heatmap)
- **ROC-AUC** (one-vs-rest per class)

### Evaluation Alignment with Hackathon Criteria

| Criterion | Target | Validation Method |
|---|---|---|
| Accuracy of Detection & Classification | > 93% accuracy on ExoFOP-labelled TESS test set | Precision/Recall/F1 per class |
| Accuracy of Parameters Estimated | Period within 0.1% for SNR>10; depth within 5%; duration within 10% | Validated against published ExoFOP values |
| Methods / Approach | Hybrid classical + deep learning; lightweight; automated; reproducible | MLflow logging |
| Visualisation & Clarity | 4-panel diagnostic per candidate; colour-coded confidence tiers; interactive HTML | Plotly + Matplotlib |

### Parameter Estimation Metrics

For known transiting planets in the validation set:
- Period recovery rate (% within 1% of true period)
- Depth accuracy: mean absolute error in ppm
- Duration accuracy: mean absolute error in hours
- Coverage probability of MCMC 68% credible interval

### Validation Checklist

- Period recovery within 1% of published value
- Transit depth within 10% of published value
- Classification accuracy > 90% on curated dataset
- SDE > 7 for all known planets, < 7 for known non-detections
- MCMC chains converged (acceptance fraction 0.2–0.5)

---

## 13. Visualization Requirements

Judges reward strong, clear visualisations. Create these panels for maximum impact:

### Figure 1: Raw Light Curve

Full sector raw flux vs. time. Mark quality-flagged cadences in red. Show the trend line overlaid. Demonstrates data understanding.

### Figure 2: Detrended Light Curve

Flattened flux with transit dips visible. Overplot BLS/TLS-predicted transit times as vertical dashed lines.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 8))
axes[0].plot(lc.time.value, lc.flux.value, '.k', ms=1, label='Raw flux')
axes[1].plot(lc.time.value, flat_flux, '.k', ms=1, label='Detrended')
axes[2].plot(results.folded_phase, results.folded_y, '.k', ms=1)
axes[2].plot(results.model_folded_phase, results.model_folded_model, 'r-', lw=2)
axes[2].set_xlabel('Phase')
```

### Figure 3: BLS/TLS Power Spectrum

Period vs. power. Mark the peak period and show 3σ, 5σ threshold lines. Visually convincing detection significance.

### Figure 4: Phase-Folded LC + Model

Phase vs. flux with batman model overlaid in orange/red. Binned data points. Show both **global** and **zoomed local** view side-by-side.

### Figure 5: Classification Dashboard

Bar chart of softmax probabilities for the 4 classes. Clean, colour-coded.

```python
classes = ['Planet Candidate', 'Eclipsing Binary', 'Background Blend', 'Other/FP']
colors = ['#2196F3', '#F44336', '#FF9800', '#9E9E9E']
plt.bar(classes, confidence_scores, color=colors)
plt.ylabel('Classifier Probability')
plt.title(f'TIC {tic_id} — Predicted: {predicted_class} ({confidence:.1%} confidence)')
```

### Figure 6: MCMC Corner Plot

2D posterior distributions for all parameters (P, Rp/Rs, inc, a). Shows medians + 1σ, 2σ contours.

```python
import corner
labels = ['t₀', 'Period (d)', 'Rp/Rs', 'a/Rs', 'Inclination (°)']
corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
              show_titles=True, title_kwargs={"fontsize": 12})
```

### Standardised 4-Panel Diagnostic Plot (per candidate)

Proposed in the submission document — every detected signal produces a standardised 4-panel diagnostic plot (PNG + interactive HTML via Plotly):
- **Panel 1:** Full raw and detrended light curve with detected transit epochs marked
- **Panel 2:** BLS/TLS periodogram with peak annotated and false-alarm probability contours
- **Panel 3:** Phase-folded light curve with Mandel-Agol best-fit model overlaid and residuals
- **Panel 4:** Classifier softmax bar chart showing 4-class probability breakdown with confidence interval

### Interactive Dashboard (Bonus)

Build a Streamlit or Plotly Dash app showing:
- Any TIC ID on demand
- Detrended LC + TLS result + classification in one view
- Manual disposition override capability
- Detections plotted on a star map using Astropy SkyCoord + folium

---

## 14. Full Tech Stack & Libraries

### Core Language

**Python 3.10+** — entire astronomy ecosystem is Python-first; fastest iteration.

### Essential Astronomy Libraries

| Library | Role | Install |
|---|---|---|
| `lightkurve` | NASA's official TESS/Kepler Python library. LC download, fold, BLS, plot. | `pip install lightkurve` |
| `astropy` | FITS handling, BLS periodogram, coordinate transforms, units, FITS I/O | `pip install astropy` |
| `astroquery` | Programmatic access to MAST, NASA archives; bulk downloads | `pip install astroquery` |
| `wotan` | State-of-the-art detrending (better than Savitzky-Golay for transit preservation) | `pip install wotan` |
| `transitleastsquares` (TLS) | Superior transit period search; beats BLS for small planets | `pip install transitleastsquares` |
| `batman-package` | Mandel & Agol (2002) transit light curve model. Fast, accurate. | `pip install batman-package` |
| `emcee` | MCMC ensemble sampler for parameter estimation with uncertainty posteriors | `pip install emcee` |
| `corner` | Corner plot visualisation for MCMC posteriors; publication-quality | `pip install corner` |
| `eleanor` | Extract light curves from TESS Full Frame Images (FFIs) | `pip install eleanor` |
| `celerite2` | Gaussian Process red noise modelling for correlated noise | `pip install celerite2` |

### ML / AI Stack

| Library | Role |
|---|---|
| `tensorflow` / `keras` | CNN classifier (AstroNet-style dual-view 1D CNN) |
| `torch` (PyTorch 2.x) | Bi-LSTM + Transformer architecture (Colab T4-compatible) |
| `scikit-learn` | XGBoost/Random Forest; train/test split; cross-validation; metrics |
| `xgboost` | Gradient boosted trees for feature-based classification |
| `shap` | Explainability — which features drove each classification |
| `imbalanced-learn` | Handle class imbalance (planets are rare!) |
| `mlflow` | Model versioning, metric logging, reproducibility |
| `huggingface accelerate` | Training acceleration for PyTorch models |

### Visualization

| Library | Role |
|---|---|
| `matplotlib` | Primary plotting (light curves, phase-folded plots, confusion matrices) |
| `plotly` | Interactive, web-ready visualizations; interactive HTML exports |
| `bokeh` | Dashboard for bulk inspection |
| `corner` | MCMC corner plots for parameter posteriors |
| `seaborn` | Heatmaps for confusion matrices |

### Utilities

| Library | Role |
|---|---|
| `numpy`, `scipy` | Numerical computing, curve fitting, signal processing, statistics |
| `pandas` | Catalog management, feature tables, candidate catalogue CSV |
| `h5py` / `tables` | Efficient storage of large light curve arrays |
| `tqdm` | Progress bars for bulk processing |
| `fpdf2` / `reportlab` | PDF report generation |
| `jupyter` | Interactive development and visualization |
| `cupy` | GPU-accelerated BLS via CuPy wrapper (optional, big speedup) |

### Full pip install command

```bash
pip install lightkurve astropy astroquery batman-package emcee corner \
    tensorflow xgboost scikit-learn wotan transitleastsquares \
    eleanor celerite2 plotly shap imbalanced-learn mlflow \
    torch huggingface-hub pandas h5py tqdm fpdf2
```

### What to Avoid

- `PyRAF` / `IRAF` — legacy, complex setup, not needed
- Training large transformers from scratch — use pre-trained CNN weights instead
- Local GPU training during the hackathon — use Kaggle/Colab GPU if needed

---

## 15. Professional-Level Tools (External Research)

The following tools are used in published peer-reviewed exoplanet papers and represent the current state of the art in the field (2024–2026).

### SHERLOCK Pipeline

**Full name:** Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-based seeKers
**What it is:** End-to-end open-source Python pipeline for TESS/Kepler exoplanet search, vetting, validation, Bayesian fitting, and observing window computation.
**Performance:** Recovered **98% of TOIs** in a benchmark test on the confirmed TESS planet catalogue.
**Repository:** https://github.com/mrbarrientosg/SHERLOCK
**Paper:** Dévora-Pajares et al. (2024), MNRAS 532, 4752

**Six integrated modules:**
1. Automated data acquisition from MAST
2. Transit signal search
3. Candidate vetting
4. Statistical validation
5. Bayesian modeling for precise ephemerides
6. Computation of observational windows for ground-based follow-up

**Use in hackathon:** Wrap around your classifier as a sanity check or cite it in your report as a benchmark comparison.

---

### ExoMiner++ (NASA Ames)

**What it is:** NASA's production-grade deep learning classifier for TESS transit signal vetting. Published October 2025, Astronomical Journal Vol. 170, No. 5 (Valizadegan et al. 2025, DOI: 10.3847/1538-3881/ae03a4).
**Repository:** https://github.com/nasa/ExoMiner
**Key stats:** Among 147,568 unlabelled TCEs, ExoMiner++ identified **7,330 planet candidates** from 2-minute TESS data. At 95% precision, achieves ~80% recall.
**Architecture:** Multi-branch deep learning with inputs: periodogram, flux trend, difference image, unfolded flux, spacecraft attitude control data. Combines Kepler + TESS training data via transfer learning.
**ExoMiner++ 2.0:** Extended to TESS Full Frame Image (FFI) transit signals (published January 2026).
**Use:** Compare your classifier's output against ExoMiner++ predictions on the same targets as a validation sanity check.

---

### TRICERATOPS / TRICERATOPS+

**Full name:** Tool for Rating Interesting Candidate Exoplanets and Reliability Analysis of Transits Originating from Proximate Stars
**What it is:** Python package for computing False Positive Probability (FPP) for TESS planet candidates using Bayesian framework. **Recommended replacement for VESPA** (VESPA is now retired and unmaintained as of 2023).
**Repository:** https://github.com/stevenstetzler/triceratops
**Paper:** Giacalone et al. (2021); TRICERATOPS+ — Gomez Barrientos et al. (2025), AJ 170:148

**Key inputs:** Phase-folded TESS light curve, aperture mask, Gaia stellar parameters, high-contrast imaging contrast curves.
**Key outputs:**
- **FPP (False Positive Probability)** — total probability the signal is not a planet
- **NFPP (Nearby False Positive Probability)** — probability from a resolved nearby star

**Validation thresholds:**
- Validated planet (VP): FPP < 1.5% and NFPP < 0.1%
- Run 20 independent MCMC realizations and report mean ± std for stability

**Use in hackathon:** After classification, run TRICERATOPS on your top planet candidates to compute FPP and NFPP. This is the standard observational astronomy approach and will strongly impress judges.

---

### vespa (Legacy — Now Retired)

Statistical validation tool for Kepler/K2-era planet candidates. Morton (2012, 2015). Uses TRILEGAL Galaxy model.
**Note:** As of 2023, VESPA is no longer maintained. The community recommends TRICERATOPS for TESS targets. Still cited in papers for historical context or K2 data.

---

### eleanor

**What it is:** Python package to extract light curves from TESS Full Frame Images (FFIs) for stars not observed at 2-minute cadence.
**Repository:** https://github.com/afeinstein20/eleanor
**Use:** Access the full ~400,000 star catalogue observed at 30-minute FFI cadence (vs. only ~20,000 stars at 2-minute cadence per sector).

---

### TESS SPOC Pipeline (Science Processing Operations Center)

**What it is:** NASA's official TESS data reduction pipeline operated at MIT. Produces PDCSAP (Pre-search Data Conditioning Simple Aperture Photometry) flux used by lightkurve by default.
**Access:** Data available via MAST. SPOC TCE (Threshold Crossing Events) catalogue available at https://archive.stsci.edu/tess/

---

### Quick-Look Pipeline (QLP)

**What it is:** MIT's fast pipeline for TESS FFI data. Produces LightCurves for ~13 million stars at 10-minute and 30-minute cadence.
**Access:** MAST archive. Useful for broader surveys beyond the 2-min cadence sample.

---

### ExoplanetArchive Python Client

```bash
pip install pyvo
```

Query the NASA Exoplanet Archive programmatically for confirmed planets, stellar parameters, and TOI dispositions.

---

## 16. Professional-Level Datasets (External Research)

### Primary Mission Data

| Dataset | Stars Covered | Cadence | Access |
|---|---|---|---|
| **TESS 2-min SPOC LCs** | ~20,000/sector | 2 min | MAST via lightkurve |
| **TESS 10-min QLP LCs** | ~1M/sector | 10 min | MAST |
| **TESS FFIs (30-min)** | ~400,000/sector | 30 min | MAST via eleanor / TESScut |
| **Kepler Long Cadence** | 197,096 stars | 30 min | MAST archive |
| **Kepler Short Cadence** | ~512 stars/quarter | 1 min | MAST archive |

### Labeled Training Datasets

| Dataset | Size | URL | Notes |
|---|---|---|---|
| **AstroNet (Shallue & Vanderburg 2018)** | 34,032 TCEs | https://github.com/google-research/exoplanet-ml | Kepler DR24 — gold standard; labels: PC, AFP, NTP |
| **ExoFOP-TESS TOI Catalog** | ~5,000 PC + ~12,000 FP/EB | https://exofop.ipac.caltech.edu/tess/ | Community-vetted labels for TESS targets |
| **ExoMiner++ Vetting Catalog** | 147,568 TCEs labelled | arxiv:2502.09790 | State-of-the-art TESS labels (2025) |
| **Kepler Confirmed Planet Catalog** | 2,662 planets | https://exoplanetarchive.ipac.caltech.edu/ | Positive class — all confirmed |
| **Kepler False Positive Catalog** | 5,100+ FPs | https://exoplanetarchive.ipac.caltech.edu/ | Negative class — labelled FPs |
| **Thompson et al. 2018 (DR24)** | 34,032 TCEs | https://archive.stsci.edu/kepler/ | Kepler DR24 TCE catalog with labels |
| **Kaggle NASA Kepler Dataset** | 9,564 samples | https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results | Easy to use; pre-labelled CONFIRMED/FP |
| **TESS Eclipsing Binary Catalog** | ~4,000+ EBs | https://archive.stsci.edu/ | Known EBs for binary classification |

### Stellar Parameter Databases (for Feature Engineering)

| Dataset | Content | URL |
|---|---|---|
| **TESS Input Catalog v8 (TICv8)** | Stellar parameters (Teff, logg, Rs, Ms, TESS mag) for ~1.7 billion objects | https://tess.mit.edu/science/tess-input-catalog/ |
| **Gaia DR3** | Precise parallaxes, proper motions, stellar parameters | https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=I/355/gaiadr3 |
| **2MASS** | JHK photometry for blend analysis | https://www.ipac.caltech.edu/2mass/ |

### Access Patterns

```python
# ExoFOP labels — download disposition table
import pandas as pd
toi_df = pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
# Columns: TOI, Period, Depth, Duration, TFOP WG Disposition, Comments

# NASA Exoplanet Archive — confirmed planets
import pyvo
service = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
results = service.search("SELECT * FROM ps WHERE default_flag=1")
```

---

## 17. Validation Targets

Use these well-known TESS targets to validate your pipeline **before scaling up**. If your pipeline doesn't find WASP-121b, something is wrong.

| Target / TIC ID | System Description | Period(s) [days] | Notes |
|---|---|---|---|
| **WASP-121b** (TIC 22529346) | Hot Jupiter; deep transit ~1.4%; Sector 7 | 1.27 | Easy detection — sanity check first |
| **TOI-270** (TIC 259377017) | 3 confirmed super-Earths; Sector 3 | 3.36, 5.66, 11.38 | Validate full multi-planet pipeline |
| **55 Cancri e** (TIC 332064670) | Ultra-hot super-Earth; Sector 21 | 0.74 | Very shallow transit — stress test |
| **TIC 349488688** | Known eclipsing binary | — | EB classification validation |
| **L 98-59** (TIC 307210830) | 3 small planets; Sector 2 | 2.25, 3.69, 7.45 | Benchmark multi-planet system |

### Expected Pipeline Accuracy on Validation Set

- **Classification accuracy:** > 90% on curated dataset
- **Period recovery:** Within 1% of published value
- **Transit depth:** Within 10% of published value
- **SDE:** > 7 for all known planets, < 7 for known non-detections
- **Parameter accuracies (submission doc targets):** Period within 0.1% for SNR > 10; depth within 5%; duration within 10%

---

## 18. Implementation Timeline

### Recommended 30-Hour Build Order (from PDF Guide)

| Timeframe | Tasks |
|---|---|
| **Hour 0–3** | Environment setup. Download TESS Sector 1 data for 100 test targets. Run basic lightkurve pipeline. Validate on WASP-121b (easy detection). |
| **Hour 3–8** | Full preprocessing pipeline. BLS on all targets. SDE threshold filter. Bulk download complete sector. Engineer all 8+ classification features. |
| **Hour 8–16** | CNN architecture. Train on curated dataset. Add XGBoost stage. Ensemble. Validate: target > 90% accuracy on held-out test set. |
| **Hour 16–22** | batman fitting on top planet candidates. MCMC with emcee. Generate all 6 required visualisations. Parameter table. |
| **Hour 22–26** | Centroid analysis (bonus). TLS instead of BLS (bonus). Streamlit dashboard (bonus). Polish all plots. |
| **Hour 26–30** | Write 3-page report. Final pipeline run on all science targets. QA: check all required outputs are present. Package submission. |

### Hour-by-Hour Plan — 24h Hackathon (from Markdown Guide)

| Time Block | Task | Owner |
|---|---|---|
| **0–2h** | Repo setup, install all dependencies, download one TESS sector (~20k LCs), verify data pipeline works end-to-end on one star | Everyone |
| **2–6h** | Build preprocessing pipeline: detrend all LCs, run TLS on each, extract features + SNR → save to CSV | ML lead |
| **2–6h** | Download ExoFOP labels, merge with TLS features, prepare training/validation split | Data lead |
| **6–10h** | Train CNN (AstroNet-style) on Kepler, fine-tune on TESS TOIs; train XGBoost on features | ML lead |
| **6–10h** | Implement batman + emcee for parameter estimation on top candidates | Astro lead |
| **10–16h** | Apply full pipeline to all 20k LCs; extract candidates with SDE > 7; classify all | Everyone |
| **16–20h** | Build visualisations (phase-folded plots, corner plots, confidence bars); interactive dashboard | Viz lead |
| **20–22h** | Write 3-page report; validate top candidates against known TOI list | All |
| **22–23h** | Practice demo run; make slide deck; final polish | All |
| **23–24h** | Buffer — nothing new, only bug fixes | — |

### Scope Contract

**IN SCOPE (must ship):**
- End-to-end pipeline on ≥ 1 full TESS sector
- 4-class classifier with confusion matrix + AUC
- Parameter estimation for top 5 planet candidates
- Phase-folded visualisation for each classified signal
- Confidence/SNR score per detection

**STRETCH (if time allows):**
- Interactive Plotly/Dash inspection dashboard
- Multi-sector search for longer-period planets
- Centroid pixel analysis for blend validation
- Comparison of BLS vs. TLS recovery rates
- TRICERATOPS FPP computation on top candidates
- MLflow logging for reproducibility

**OUT OF SCOPE:**
- Ground-truth follow-up observations
- Full atmospheric characterisation
- Real-time streaming data processing
- GPU model training during hackathon (use Colab/Kaggle)

---

## 19. Report Writing Strategy

### 3-Page Report Structure (from PDF Guide)

| Section | Exact Content to Include |
|---|---|
| **Page 1 — Methodology** | Intro (2 sentences) + Pipeline diagram (flowchart) + BLS/TLS algorithm + ML architecture. Mention Shallue & Vanderburg 2018 as inspiration. Include confusion matrix from curated dataset validation. |
| **Page 2 — Results** | Table of all detected signals with: TIC ID, period, depth, duration, SDE, classification, confidence. Include Figure 4 (phase-fold + model) for your best planet detection. |
| **Page 3 — Uncertainties & Assumptions** | MCMC posterior plots + parameter table with uncertainties. List assumptions: circular orbit, quadratic limb darkening, Gaussian noise, single planet per system. Discuss limitations. |

### Key Assumptions to Declare

- Circular orbit (eccentricity = 0)
- Quadratic limb darkening with fixed coefficients
- Gaussian photometric noise (ignoring red noise in simple models)
- Single planet per system for MCMC (extend for multi-planet)
- TESS 2-minute cadence data from a single sector
- TICv8 contamination ratio used for blend flagging

---

## 20. Demo Pitch Structure

### The 10-Second Hook

*"One in ten thousand TESS stars is orbited by a planet — but finding it means untangling signals that look almost identical to 4 other completely different phenomena. We built an AI pipeline that does in seconds what takes astronomers weeks."*

### Demo Flow (show, don't tell)

1. Show a raw TESS light curve — it looks like noise
2. Show the detrended + TLS periodogram — a clean period peak emerges
3. Show the phase-folded light curve — the transit is undeniable
4. Show the classifier output — *"Planet Candidate, 94.2% confidence"*
5. Show the MCMC corner plot — *"Period: 3.47 ± 0.001 days, Depth: 1820 ± 65 ppm"*
6. Show the grid of all 20,000 stars processed — *"We screened an entire TESS sector in 45 minutes"*

### The Tech (30 seconds)

Python · lightkurve + TLS for signal detection · AstroNet-style CNN + XGBoost ensemble for classification · batman + emcee for parameter fitting · Runs on free Google Colab T4 GPU

### The Vision

Scalable to all 26 TESS sectors (~500k stars) and extendable to Roman Space Telescope data (2026+).

### The Differentiators

| What We Did | Why It Wins |
|---|---|
| 4-class classifier (not binary) | Enables direct triage without additional follow-up tools |
| Bi-LSTM + Transformer (or AstroNet CNN) | State-of-the-art architecture; lower FP rate than BLS-only |
| batman + MCMC inline | No post-processing step needed; parameters from same run |
| Colour-coded confidence tiers | Gold/Silver/Bronze intuitive for mission scientists |
| Runs on free Colab T4 GPU | 70% lower cost vs. existing GPU-cluster solutions |
| MLflow tracking | Fully reproducible and version-controlled |

---

## 21. Key Formulas & Quick Reference

| Quantity | Formula |
|---|---|
| **Transit Depth** | δ = (Rp / Rs)² [dimensionless, ~ppm to %] |
| **Transit Duration (T₁₄)** | T₁₄ = (P/π) × arcsin(Rs/a × √((1+Rp/Rs)² − b²) / sin(i)) |
| **Impact Parameter** | b = (a/Rs) × cos(i) [0 = central transit, 1 = grazing] |
| **SDE (BLS/TLS)** | SDE = (P_max − mean(P)) / std(P) [Signal Detection Efficiency, > 7 = detection] |
| **Transit SNR** | SNR = δ / σ_CDPP × √(N_transit_points) |
| **Stellar Density (from transit)** | ρ_star = (3π) / (G × P²) × (a/Rs)³ |
| **Rp in Earth radii** | Rp = (Rp/Rs) × Rs_solar_radii × 109.0 [if Rs known from TICv8] |
| **Log-likelihood (MCMC)** | lnL = −0.5 × Σ((data − model)² / σ² + ln(2πσ²)) |
| **Earth-transit depth (Sun-like)** | ~84 ppm (0.008%) |
| **Normalised flux** | f(t) = 1 − δ during transit, 1.0 otherwise |

---

## 22. Key References

### Papers

| Paper | Relevance |
|---|---|
| **Shallue & Vanderburg (2018)**, AJ 155, 94 | AstroNet: CNN architecture for transit classification from Kepler (96% AUC) |
| **Yu et al. (2019)** | AstroNet-Triage: TESS-adapted triage classifier |
| **Valizadegan et al. (2021)**, ApJ | ExoMiner: Validated 301 Kepler exoplanets using deep learning |
| **Valizadegan et al. (2025)**, AJ 170, No.5 | ExoMiner++: Enhanced TESS 2-min classification; identified 7,330 planet candidates |
| **Dévora-Pajares et al. (2024)**, MNRAS 532, 4752 | SHERLOCK pipeline; recovered 98% of TOIs |
| **Hippke & Heller (2019)**, A&A 623, A39 | Transit Least Squares (TLS) algorithm |
| **Kovacs, Zucker & Mazeh (2002)**, A&A 391, L369 | Original Box Least Squares (BLS) algorithm |
| **Mandel & Agol (2002)**, ApJ 580, L171 | Analytic transit light curve model (batman) |
| **Foreman-Mackey et al. (2013)** | emcee MCMC sampler |
| **Giacalone et al. (2021)** | TRICERATOPS: FPP calculation for TESS planet candidates |
| **Gomez Barrientos et al. (2025)**, AJ 170:148 | TRICERATOPS+: Updated multicolor validation |
| **Kruse et al. (2019)**, ApJS 244, 11 | TESS photometry systematics |
| **Thompson et al. (2018)**, ApJS 235, 38 | Kepler DR24 TCE catalog (training data) |
| **Ricker et al. (2015/2016)** | TESS mission overview |
| **Lightkurve Collaboration (2018)** | lightkurve package documentation |

### Software & Repositories

| Resource | URL |
|---|---|
| **lightkurve docs** | https://docs.lightkurve.org |
| **TLS GitHub** | https://github.com/hippke/TLS |
| **AstroNet GitHub (Google)** | https://github.com/google-research/exoplanet-ml |
| **ExoMiner GitHub (NASA)** | https://github.com/nasa/ExoMiner |
| **SHERLOCK GitHub** | https://github.com/mrbarrientosg/SHERLOCK |
| **batman docs** | https://lweb.cfa.harvard.edu/~lkreidberg/batman/ |
| **emcee docs** | https://emcee.readthedocs.io |
| **Wotan GitHub** | https://github.com/hippke/wotan |

### Data Archives

| Resource | URL |
|---|---|
| **TESS bulk download** | https://archive.stsci.edu/tess/tic_ctl.html |
| **ExoFOP-TESS** | https://exofop.ipac.caltech.edu/tess/ |
| **NASA Exoplanet Archive** | https://exoplanetarchive.ipac.caltech.edu/ |
| **MAST Portal** | https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html |
| **MIT TOI Releases** | https://tess.mit.edu/toi-releases/ |
| **TESScut** | https://mast.stsci.edu/tesscut/ |

---

## 23. Quick-Start Code Block

```python
# Full pipeline in ~50 lines — run this first to validate your setup
import lightkurve as lk
import numpy as np
from wotan import flatten
from transitleastsquares import transitleastsquares
import matplotlib.pyplot as plt

# 1. Download (use a known planet for validation)
lc = lk.search_lightcurve("TOI 700", mission="TESS",
                            cadence="short")[0].download()
lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

# 2. Detrend
flat_flux, trend = flatten(lc.time.value, lc.flux.value,
                            method='biweight', window_length=0.75,
                            return_trend=True)

# 3. Period search
model = transitleastsquares(lc.time.value, flat_flux)
results = model.power(oversampling_factor=5)

# 4. Print summary
print(f"✅ Period:   {results.period:.4f} ± {results.period_uncertainty:.4f} d")
print(f"✅ Depth:    {results.depth*1e6:.0f} ppm")
print(f"✅ Duration: {results.duration*24:.2f} h")
print(f"✅ SNR:      {results.snr:.1f}")
print(f"✅ SDE:      {results.SDE:.1f}")

# 5. Phase-fold plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(results.folded_phase, results.folded_y, '.k', ms=1, alpha=0.4)
ax.plot(results.model_folded_phase, results.model_folded_model,
        'r-', lw=2, label=f'Period={results.period:.4f}d')
ax.set(xlabel='Phase', ylabel='Relative Flux', title='Phase-Folded Transit')
ax.legend()
plt.tight_layout()
plt.savefig('transit_candidate.png', dpi=150)
plt.show()
```

### Complete BLS Code Block (from PDF Guide)

```python
from astropy.timeseries import BoxLeastSquares

model = BoxLeastSquares(lc.time.value, lc.flux.value, dy=lc.flux_err.value)
periods = np.linspace(0.5, 13, 10000)
durations = np.linspace(0.01, 0.5, 50)
result = model.power(periods, durations)
best_period = result.period[np.argmax(result.power)]
best_t0 = result.transit_time[np.argmax(result.power)]
best_dur = result.duration[np.argmax(result.power)]
SDE = (result.power.max() - result.power.mean()) / result.power.std()
```

### batman MCMC Code Block (from PDF Guide)

```python
import batman
params = batman.TransitParams()
params.t0 = 0.
params.per = best_period
params.rp = 0.1
params.a = 10.
params.inc = 89.
params.ecc = 0.
params.w = 90.
params.u = [0.3, 0.1]
params.limb_dark = 'quadratic'
m = batman.TransitModel(params, phase_times)
flux_model = m.light_curve(params)
# Use emcee for MCMC uncertainty estimation
# pip install emcee corner
```

---

## 24. Additional / Supplementary Information

This section contains information present in the source documents that does not directly apply to the hackathon execution strategy but is included in full for completeness.

---

### 24.1 AstroNet-Transit Submission Document — Full Comparison Table

*Source: AstroNet_Transit_ISRO_BAH2026.docx*

This is the specific architectural proposal and differentiator analysis from the submission document, expanded for reference:

| Aspect | Existing Approaches | AstroNet-Transit Pipeline |
|---|---|---|
| **Detection Method** | BLS (Box Least Squares) only | BLS pre-filter + Bi-LSTM + Transformer ensemble |
| **False Positive Rate** | ~25–35% | Target < 8% via multi-class classifier |
| **Data Pre-processing** | Manual or semi-automated | Fully automated: detrending, normalisation, inpainting |
| **Compute Cost** | GPU cluster required | Runs on single T4 GPU / Colab (free tier) |
| **Classification** | Binary (planet/non-planet) | 4-class: transit, eclipse, blend, other |
| **Parameter Estimation** | Separate MCMC tool needed | Inline Mandel-Agol fitting, fast analytic priors |
| **SNR Reporting** | Post-hoc or absent | Real-time per-event confidence score |
| **Training time** | Multi-GPU hours | < 2 hours on single T4 GPU |
| **Inference time (25k LCs)** | Variable | < 15 minutes |

**Cost Reduction Details:** AstroNet and ExoMiner require multi-GPU training runs (V100/A100) costing hundreds of USD on cloud. The Bi-LSTM + small Transformer trains on a single T4 GPU in < 2 hours — Colab-free-tier compatible. Inference over 25,000 light curves completes in < 15 minutes (~70% cost reduction claimed).

**Parameter Count Comparison:** 1.2M parameters (Bi-LSTM + Transformer) vs. ~2.5M in AstroNet (CNN).

---

### 24.2 Deliverables Table from Submission Document

*Source: AstroNet_Transit_ISRO_BAH2026.docx*

| Deliverable | Description |
|---|---|
| **AI Detection Pipeline** | End-to-end Python package: ingest TESS FITS → output classified CSV with parameters. Single command execution. |
| **Classification Report** | Per-star classification into transit / eclipsing binary / blend / other with softmax confidence scores |
| **Parameter Table** | Orbital period (days), transit duration (hours), transit depth (ppm) with 1σ uncertainties for all transit candidates |
| **Diagnostic Plots** | 4-panel publication-quality plot per candidate; exportable as PNG and interactive HTML |
| **SNR / Confidence** | Per-event SNR (δ/σ) and classifier confidence score; colour-coded tier: Gold (> 0.90), Silver (0.70–0.90), Bronze (< 0.70) |
| **Benchmark Report** | Precision, Recall, F1, AUC-ROC per class on held-out test set; confusion matrix; runtime profiling |

---

### 24.3 Prize-Winning Extras (from PDF Guide)

Differentiators that win hackathons, listed here as a separate record:

| Extra Feature | Implementation & Why It Wins |
|---|---|
| **Transit Least Squares (TLS)** | Use TLS instead of BLS. TLS uses a physically motivated model (realistic ingress/egress) instead of a box. Detects 10–15% more planets, especially small ones. |
| **Centroid Analysis** | Check if the flux centroid shifts during transit. Non-zero shift = background eclipsing binary (blend). Implement with pixel-level data (TESS TPF = Target Pixel Files). Dramatically reduces false positives. |
| **SHERLOCK Integration** | Open-source automated vetting pipeline. Wrap around your classifier as a sanity check. Cite it in your report. |
| **Synthetic Transit Injection** | Inject batman-generated transits into real light curves at known depths/periods. Measure recovery fraction. This gives you a **completeness map** — extremely impressive to present. |
| **Interactive Dashboard** | Build a Streamlit or Plotly Dash app showing detections on a star map (Astropy SkyCoord + folium). Judges can click stars to see light curves. |
| **Comparison Metrics** | Report precision, recall, F1-score, and ROC-AUC on the curated test set. Use a confusion matrix heatmap. Quantitative rigor wins points. |
| **TRICERATOPS FPP** | Compute False Positive Probability for top planet candidates — the current community-standard validation step |

---

### 24.4 Data Pipeline Assumptions (from Submission Document)

*Listed here as a reference for the report's assumptions section:*

- TESS 2-minute (high-cadence) data from a single sector (~27 days) is used; 20-second cadence sectors are supported but not required
- Light curves with fewer than 500 valid cadences post-clipping are discarded (insufficient phase coverage)
- Stars with TESS magnitude < 6 are excluded to avoid saturation artefacts
- The contamination ratio from the TESS Input Catalog is used to flag high-blend targets (contamination > 0.3) and weight the blend classifier output accordingly
- Circular orbit assumed (eccentricity = 0) for MCMC simplification
- Quadratic limb darkening with coefficients from Claret & Bloemen (2011)
- Gaussian photometric noise model for log-likelihood calculation

---

### 24.5 Hackathon FAQ (from Official Website)

*Source: https://hack2skill.com/event/bah2026/*

| Question | Answer |
|---|---|
| Who can participate? | Undergraduate, Graduate/Postgraduate, and PhD students enrolled in Indian institutions |
| Ideal team size? | 3 to 4 members, possibly from different colleges/universities |
| Cost? | Entirely free. No registration or submission fee. |
| Prior space research experience required? | No. Anyone with passion for innovation and problem-solving can participate. |
| Mentors available? | Yes, selected teams receive technical mentorship from ISRO scientists throughout. |
| Prototype needed in idea submission phase? | No. Initial phase requires a detailed concept/idea proposal only, using provided templates. |
| Travel expenses covered for finalists? | Yes. II AC travel fare reimbursed for all finalists. |

---

### 24.6 Other BAH 2026 Problem Statements

*Source: https://hack2skill.com/event/bah2026/ — Listed for context; PS-07 is the relevant one for this guide.*

| PS | Title |
|---|---|
| PS-01 | Optimising Urban Heat Mitigation and cooling strategies via AIML |
| PS-02 | Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery |
| PS-03 | Development of Surface AQI & Identification of HCHO Hotspots over India using Satellite Data |
| PS-04 | Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility |
| PS-05 | AI-Powered Digital Twin of India's Climate using India's National Data |
| PS-06 | AI-Driven Automated Crop type, Moisture Stress Detection using Spectral Signatures |
| **PS-07** | **AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves** |
| PS-08 | Detection and Characterization of Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 Data |
| PS-09 | Algorithms for Wavefront reconstruction using Shack-Hartmann Wavefront Sensor time-series data |
| PS-10 | Infrared image colorisation and enhancement for improved object interpretation |
| PS-11 | Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data |
| PS-12 | Enhancing Temporal Resolution of Satellite Imagery using AI/ML based on Optical Flow |
| PS-13 | Air-Gapped Predictive Copilot for Secure MPLS Operations |
| PS-14 | Forecasting Energetic Particle Radiation Environment for ISRO's Geostationary Satellites |
| PS-15 | Forecasting and/or Nowcasting of Solar Flares using Aditya-L1 data |

---

*End of Document*

*Master guide compiled from: `exoplanet_hackathon_guide.md` (detailed strategy + code), `AstroNet_Transit_ISRO_BAH2026.docx` (submission proposal with Bi-LSTM + Transformer architecture), `Exoplanet_Hackathon_Guide.pdf` (PDF strategy guide with prize tips and execution timeline), BAH 2026 official website (https://hack2skill.com/event/bah2026/), and internet research on ExoMiner++, SHERLOCK, TRICERATOPS, and professional datasets (2024–2026).*
