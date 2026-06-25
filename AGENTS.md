# AGENTS.md

## What This Repo Is

Planning and documentation for **ISRO BAH 2026 PS-07** — an AI-enabled exoplanet detection pipeline. No code exists yet. All decisions live in `.planning/` and `docs/adr/`.

## Architecture (Source of Truth)

**The current architecture is defined by `.planning/PROJECT.md` and `docs/adr/0001-0004`**, NOT by the legacy documents:

| Legacy doc (IGNORED) | Current decision |
|---|---|
| `claude-prod.md` — Bi-LSTM + Transformer, FastAPI, PostgreSQL, HDF5, live backend | **Dual-View CNN** (TensorFlow/Keras), **Next.js read-only** over pre-rendered static files, `.npz` + **Parquet** |
| `claude-roadmap.md` — 1 sector, Bi-LSTM | **3 sectors** (1,2,3), CNN ensemble |

`claude-prod.md`, `claude-roadmap.md`, and `main.md` are gitignored. Do not use them as architectural sources.

## Quick Orientation

```
.planning/PROJECT.md       ← project context, 18 key decisions, constraints
.planning/REQUIREMENTS.md  ← 55 v1 requirements (DATA-*, PREP-*, DET-*, etc.)
.planning/ROADMAP.md       ← 4 phases: Foundation → Intelligence → Characterization → Presentation
.planning/STATE.md         ← current position, blockers, session continuity
.planning/research/        ← STACK, FEATURES, ARCHITECTURE, PITFALLS, SUMMARY
.planning/config.json      ← YOLO mode, standard granularity, parallel execution
CONTEXT.md                 ← domain glossary — 35 terms, use these exact words
docs/adr/0001-0004         ← architecture decisions (CNN, 3 sectors, 7× augmentation, Kepler pre-training)
```

## GSD Workflow

Always check `.planning/STATE.md` for current phase before any action.

```
/gsd-plan-phase N     ← create plan for phase N
/gsd-execute-phase    ← execute current plan
/gsd-verify-work      ← validate after execution
/gsd-progress         ← check overall progress
```

## Key Decisions An Agent Must Know

- **TLS primary, BLS validation** — not BLS-first (detects 10–15% more small planets)
- **4-class classifier** (PC, EB, Blend, Stellar Var) — instrumentals are a preprocessing gate, NOT a 5th class
- **Two-gate MCMC**: Nelder-Mead (SDE>7 + p>0.70) gates full emcee (SDE>7 + p>0.85, top 15 only)
- **3-tier SDE**: <5 discard, 5–7 sub-threshold, ≥7 full pipeline
- **2001 global + 201 local** phase-folded views (AstroNet spec)
- **Mask** 13-day TESS data gaps, do NOT interpolate
- **Per-star limb darkening** from TICv8, no hard-coded `[0.3, 0.3]`
- **Confidence = temperature-scaled ensemble softmax** (0.6×CNN + 0.4×XGBoost), ECE < 0.04 target
- **Dashboard**: Next.js reads pre-rendered PNGs/JSON from `/outputs/` — no live Python backend
- **Centroid analysis is core** (not stretch) — TPF download for shift > 3σ blend flag

## Stack (When Writing Code)

- Python 3.12, TensorFlow 2.21 (Keras), XGBoost 3.3, scikit-learn 1.7
- lightkurve, astropy, astroquery, wotan, transitleastsquares (TLS), batman-package, emcee 3.1, celerite2 0.3, corner, shap, mlflow 3.14
- Next.js 15 (App Router), shadcn/ui, Tailwind 4, Plotly.js, Leaflet.js
- Storage: `.npz` per light curve + Parquet master catalogue (NOT HDF5, NOT PostgreSQL)
- GPU: Colab T4 free tier for training; CPU for inference

## Pre-Hackathon Prerequisites

- CNN Kepler DR24 pre-training must complete during 7-day prep window (not during 30-hour hackathon)
- Pre-download TESS Sector 1 data before event (MAST rate-limiting risk with multiple teams)
