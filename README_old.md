High-Background Single Particle Tracking Pipeline

A two-stage workflow for robust single-particle tracking in challenging fluorescence microscopy datasets with strong background signal, static contaminants, and transient particle loss.

This pipeline combines ImageJ/Fiji preprocessing for particle detection with a custom Python identity-first tracking framework designed for noisy microscopy data where conventional nearest-neighbour tracking performs poorly.

Overview

Single-particle tracking in real microscopy data is often complicated by:

strong diffuse background fluorescence
static bright contaminants
touching or merged particles
transient disappearance due to low signal-to-noise
particle crossing events
unstable identity assignment

This workflow addresses these challenges by separating particle detection from identity-aware tracking.

Workflow
Stage 1 — Fiji preprocessing

File: high_background_particle_detection.ijm

This macro prepares fluorescence image stacks for tracking by enhancing small particle-like signals while suppressing broad background.

Main approach

Uses Difference of Gaussian (DoG) filtering:

Small Gaussian blur:

σ = 1 px

Large Gaussian blur:

σ = 4 px

Difference:

DoG=G(σ=1)−G(σ=4)

This enhances particle-sized structures while removing slowly varying background.

Processing steps
duplicate source stack
contrast enhancement
8-bit conversion
small Gaussian blur
large Gaussian blur
DoG subtraction
multiplication with original image
local smoothing
MaxEntropy thresholding
binary mask generation
Output

Binary particle mask stack for tracking.

Intended use

Best suited for:

single fluorophore tracking
membrane particle tracking
high-background fluorescence movies
noisy punctate microscopy data
Stage 2 — Python identity-first tracking

File: identity_first_particle_tracking_sum_subtracted_support_v3_ready.py

Custom particle tracking framework built for difficult microscopy datasets.

Unlike conventional trackers, this system prioritises particle identity preservation over simple frame-to-frame nearest-neighbour matching.

Core features
Identity-first tracking

Tracks are treated as persistent identities rather than temporary positional matches.

This improves robustness during:

missed detections
temporary signal loss
particle crossing
ambiguous assignments
SUM projection anchor detection

Persistent bright immobile regions are detected using a full-stack SUM projection.

These are treated as anchors, not moving particles.

Benefits:

prevents static contaminants from hijacking tracking
preserves true moving particle identities
enables anchor-specific reclaim logic
SUM-core subtraction

Only the brightest core pixels from anchor regions are retained.

These pixels are subtracted from frames before support generation.

Purpose:

removes static contamination
preserves moving particle detection
avoids false trajectory support from immobile structures
Rolling support projections

Sliding maximum projections are generated from cleaned frames.

Used to build motion support maps.

Purpose:

weak particles that disappear temporarily may still leave detectable motion evidence.

Projection-aware assignment scoring

Particle assignments are scored using multiple criteria:

spatial distance
object area similarity
bounding box overlap
pixel mask overlap
motion prediction
projection support consistency

This is substantially more robust than nearest-neighbour assignment.

Lost-track recovery

Tracks can survive temporary disappearance.

Supports:

short missed detections
reclaim after signal recovery
anchor re-identification
Automatic parameter calibration

Tracking parameters can be adjusted automatically using anchor behaviour.

Adaptive tuning includes:

assignment distance
reclaim distance
missed frame tolerance
scoring weights
Interactive GUI

Includes:

TIFF stack selection
overlay stack selection
parameter editor
progress tracking

No command-line usage required.

Example workflow

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

Outputs

Python pipeline generates:

tracked particle assignments
trajectory data
MSD curves
step distance analysis
anchor validation tables
quality control outputs
overlay movies
support visualisations
tracking event logs
Dependencies
Fiji / ImageJ

Required for preprocessing:

Fiji
Gaussian Blur
Image Calculator
Thresholding tools
Python

Required packages:

numpy
pandas
matplotlib
scipy
scikit-image
tifffile
tkinter

Install:

pip install numpy pandas matplotlib scipy scikit-image tifffile
Intended applications

Designed for:

single particle tracking
membrane diffusion analysis
supported lipid bilayer experiments
fluorophore motility analysis
noisy live-cell microscopy
high-background fluorescence imaging
Why this pipeline?

Conventional tracking tools often fail in difficult microscopy datasets because they assume:

clean detections
simple nearest-neighbour motion
no persistent contaminants
minimal signal loss

This pipeline was designed specifically for real experimental datasets where those assumptions break down.
