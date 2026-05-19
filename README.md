
# High-Background Single Particle Tracking Pipeline

A two-stage workflow for robust single-particle tracking in difficult fluorescence microscopy datasets with strong background signal, static contaminants, particle overlap, and transient signal loss.

This pipeline combines:

- **ImageJ/Fiji preprocessing** for particle detection
- A **custom Python identity-first tracking framework** for robust trajectory assignment in noisy microscopy data

---

## Overview

Real-world single-particle tracking experiments are often complicated by:

- strong diffuse background fluorescence
- static bright contaminants
- touching or merged particles
- transient disappearance from low signal-to-noise
- particle crossing events
- unstable identity assignment

This workflow addresses these challenges by separating:

1. **Particle detection**
2. **Identity-aware tracking**

rather than relying solely on conventional nearest-neighbour tracking.

---

# Workflow

```text
Raw microscopy movie
        ↓
Fiji preprocessing macro
        ↓
Binary detection mask
        ↓
Python tracker
        ↓
Anchor detection
        ↓
SUM-core subtraction
        ↓
Rolling support generation
        ↓
Identity-first assignment
        ↓
Track recovery / reclaim
        ↓
CSV outputs + overlay movies
````

---

# Stage 1 — Fiji Preprocessing

**File:** `high_background_particle_detection.ijm`

This Fiji macro prepares fluorescence image stacks for tracking by enhancing particle-like structures while suppressing broad background fluorescence.

## Main Approach

The macro uses **Difference of Gaussian (DoG)** filtering:

```math
\mathrm{DoG}=G(\sigma=1)-G(\sigma=4)
```

Where:

* small Gaussian blur: `σ = 1 px`
* large Gaussian blur: `σ = 4 px`

This selectively enhances particle-sized features while removing slowly varying background structures.

---

## Processing Steps

The preprocessing pipeline performs:

1. Duplicate source stack
2. Contrast enhancement
3. 8-bit conversion
4. Small Gaussian blur
5. Large Gaussian blur
6. DoG subtraction
7. Multiplication with original image
8. Local smoothing
9. MaxEntropy thresholding
10. Binary mask generation

---

## Output

* Binary particle mask stack suitable for downstream tracking

---

## Best Suited For

* Single fluorophore tracking
* Membrane particle tracking
* High-background fluorescence movies
* Noisy punctate microscopy data

---

# Stage 2 — Python Identity-First Tracking

**File:** `identity_first_particle_tracking_sum_subtracted_support_v3_ready.py`

A custom tracking framework designed specifically for difficult microscopy datasets where conventional tracking approaches perform poorly.

Unlike standard nearest-neighbour trackers, this system prioritises **particle identity preservation** over simple frame-to-frame positional matching.

---

# Core Features

## Identity-First Tracking

Tracks are treated as persistent identities rather than temporary positional assignments.

This improves robustness during:

* missed detections
* temporary signal loss
* particle crossing events
* ambiguous assignments

---

## SUM Projection Anchor Detection

Persistent bright immobile regions are identified using a full-stack SUM projection.

These regions are treated as **anchors**, not moving particles.

### Benefits

* prevents static contaminants from hijacking tracking
* preserves true moving particle identities
* enables anchor-specific reclaim logic

---

## SUM-Core Subtraction

Only the brightest core pixels from anchor regions are retained and subtracted from frames prior to support generation.

### Purpose

* removes static contamination
* preserves moving particle detection
* avoids false trajectory support from immobile structures

---

## Rolling Support Projections

Sliding maximum projections are generated from cleaned image frames to build motion support maps.

This allows temporarily disappearing particles to retain detectable motion evidence.

---

## Projection-Aware Assignment Scoring

Particle assignments are scored using multiple criteria:

* spatial distance
* object area similarity
* bounding box overlap
* pixel mask overlap
* motion prediction
* projection support consistency

This provides substantially more robust assignment than standard nearest-neighbour approaches.

---

## Lost-Track Recovery

Tracks can survive temporary disappearance and later be reclaimed.

Supported behaviours include:

* short missed detections
* reclaim after signal recovery
* anchor re-identification

---

## Automatic Parameter Calibration

Tracking parameters can be automatically tuned using anchor behaviour.

Adaptive calibration includes:

* assignment distance
* reclaim distance
* missed frame tolerance
* scoring weights

---

## Interactive GUI

The Python pipeline includes a graphical interface with:

* TIFF stack selection
* overlay stack selection
* parameter editor
* progress tracking

No command-line usage is required.

---

# Outputs

The Python pipeline generates:

* tracked particle assignments
* trajectory data
* MSD curves
* step-distance analysis
* anchor validation tables
* quality-control outputs
* overlay movies
* support visualisations
* tracking event logs

---

# Dependencies

## Fiji / ImageJ

Required for preprocessing:

* Fiji
* Gaussian Blur
* Image Calculator
* Thresholding tools

---

## Python

Required packages:

```bash
numpy
pandas
matplotlib
scipy
scikit-image
tifffile
tkinter
```

Install using:

```bash
pip install numpy pandas matplotlib scipy scikit-image tifffile
```

---

# Intended Applications

Designed for:

* single-particle tracking
* membrane diffusion analysis
* supported lipid bilayer experiments
* fluorophore motility analysis
* noisy live-cell microscopy
* high-background fluorescence imaging

---

# Why This Pipeline?

Conventional tracking tools often fail in difficult microscopy datasets because they assume:

* clean detections
* simple nearest-neighbour motion
* no persistent contaminants
* minimal signal loss

This pipeline was designed specifically for real experimental datasets where those assumptions break down.

```
```
