import sys
import csv
import json
import os
import re
import zipfile

from datetime import datetime

from glob import glob

from os import system

import matplotlib
try:
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# REFERENCE
# ========================================================================
#
# Curvature Polarization Transport Gravity:
# A Variational Framework for Galaxies and Cluster Mergers
#
# Section 11: Galaxy Limit (Effective Reduction)
# Section 15: Structural Mode N
#
#
# - Author: Carter L Glass Jr
# - E-mail: carterglass@bellsouth.net
# - Orchid: https://orcid.org/0009-0005-7538-543X
#
# ========================================================================

# ========================================================================
# CPTG SPARC ROTATION-CURVE BENCHMARK (AUDITED)
# ========================================================================
#
# Audit goals:
# - preserve benchmarks, summaries, and plots
# - prevent hidden Newtonian mixing after CPTG solve
# - use component velocities ONLY to build baryonic source fields
# - keep CPTG and MOND baryonic baselines explicit and separate
# - keep SPARC discovery restricted to rotmod files only
#
# Important:
# - vgas, vdisk, vbul are used once to build g_bar
# - after that, ONLY total fields g_cptg and g_mond are evolved / compared
# - no post-CPTG component velocity recombination is performed
# ========================================================================

# ========================================================================
# HOW MOND AND CPTG ARE COMPUTED IN THIS SCRIPT
# ========================================================================
#
# Both theories are evaluated on the SAME SPARC galaxy data file.
#
# For each galaxy, the script reads the following measured columns:
#
#   r_kpc        : radius [kpc]
#   vobs_km_s    : observed circular velocity [km/s]
#   errv_km_s    : observational velocity uncertainty [km/s]
#   vgas_km_s    : gas component velocity contribution [km/s]
#   vdisk_km_s   : stellar disk component velocity contribution [km/s]
#   vbul_km_s    : bulge component velocity contribution [km/s]
#
# The same raw galaxy data are supplied to BOTH theories.
# The theories differ in:
#
#   (1) how they map baryonic component data into a source field
#   (2) what physical law they apply to that source field
#
# ========================================================================

# ========================================================================
# 1. COMMON PREPROCESSING
# ========================================================================
#
# Radii and velocities are converted to SI units:
#
#   r_m   = r_kpc * KPC_TO_M
#   vobs  = vobs_km_s  * 1000
#   errv  = errv_km_s  * 1000
#   vgas  = vgas_km_s  * 1000
#   vdisk = vdisk_km_s * 1000
#   vbul  = vbul_km_s  * 1000
#
# The observed centripetal acceleration is:
#
#   g_obs = vobs^2 / r_m
#
# This observed quantity is used only for comparison against the
# theory predictions and for reporting RAR residuals.
#
# ========================================================================

# ========================================================================
# 2. MOND: INPUT DATA AND COMPUTATION
# ========================================================================
#
# MOND uses the same SPARC baryonic component data, but it treats them
# in the conventional MOND way: gas, disk, and bulge all enter with
# unit weight.
#
# MOND baryonic velocity field:
#
#   vbar_mond^2 = 1.0 * vgas^2
#               + 1.0 * vdisk^2
#               + 1.0 * vbul^2
#
# MOND baryonic acceleration:
#
#   gbar_mond = vbar_mond^2 / r_m
#
# MOND then applies its standard acceleration law with a fixed universal
# MOND acceleration scale A0_MOND:
#
#   x = gbar_mond / A0_MOND
#   g_mond = gbar_mond / (1 - exp(-sqrt(x)))
#
# The MOND circular velocity prediction is then:
#
#   v_mond = sqrt(g_mond * r_m)
#
# Important MOND features in this script:
#
#   - Uses only the MOND baryonic field gbar_mond
#   - Uses the MOND law directly
#   - Uses fixed A0_MOND
#   - Does NOT use CPTG weights
#   - Does NOT use CPTG transport, Rc, or a_star
#   - Does NOT enter the CPTG fitting loop
#
# ========================================================================

# ========================================================================
# 3. CPTG: INPUT DATA AND COMPUTATION
# ========================================================================
#
# CPTG uses the same SPARC baryonic component data, but maps them into
# a different source field according to the locked CPTG geometric weights:
#
#   gas weight   = 27/50
#   disk weight  = 23/50
#   bulge weight = 16/25
#
# CPTG baryonic velocity field:
#
#   vbar_cptg^2 = (27/50) * vgas^2
#               + (23/50) * vdisk^2
#               + (16/25) * vbul^2
#
# CPTG baryonic acceleration:
#
#   gbar_cptg = vbar_cptg^2 / r_m
#
# CPTG does NOT use MOND's A0_MOND and does NOT use MOND's interpolation law.
#
# ========================================================================

# ========================================================================
# 4. CPTG FITTING / INFERENCE STAGE
# ========================================================================
#
# CPTG is solved by inferring a single galaxy-specific structural scale a_star.
# This quantity represents the curvature-polarization acceleration scale of the system.
#
# Instead of a discrete grid search, a_star is determined using a two-stage
# continuous optimization in log-space:
#
#   x = log10(a_star)
#
# Stage 1: bounded scalar minimization
#   - chi^2(x) is minimized over x in [-12, -9]
#
# Stage 2: local refinement
#   - a fine scan is performed in a small window around the optimum:
#       x_best +/- dx   (typically dx ~= 0.02)
#
# For each candidate a_star:
#
#   Rc is derived directly from a_star:
#       Rc = 48*pi*c^2 / a_star
#
# Therefore, in the fully constrained CPTG setup:
#
#   a_star  is inferred from the galaxy structure
#   Rc      is NOT fitted independently (derived quantity)
#
# The locked CPTG total-field acceleration equation is:
#
#   x = r_m / Rc
#   g_geom = ETA * a_star * x^(7/5) / (1 + x^(12/5))
#
#   g_new = gbar_cptg
#         + [ sqrt(a_star * gbar_cptg) + g_geom ]
#           / [ 1 + (g / a_star)^(123/50) ]^(1/4)
#
# This equation is solved iteratively until convergence.
#
# After convergence:
#
#   v_cptg = sqrt(g_cptg * r_m)
#
# The optimal a_star is the value that minimizes:
#
#   chi2_cptg = sum( ((vobs - v_cptg) / errv)^2 )
#
# Important CPTG features in this script:
#
#   - Uses only the CPTG baryonic field gbar_cptg
#   - Uses CPTG-specific geometric component weights
#   - Uses the nonlinear CPTG field equation
#   - Includes the transport/geometric term g_geom
#   - Infers a_star per galaxy via continuous optimization
#   - Derives Rc directly from a_star (fully constrained scaling)
#   - Does NOT use MOND A0_MOND
#   - Does NOT use the MOND acceleration law
#
# In summary:
#
#   The CPTG solution is obtained by solving a constrained nonlinear field
#   equation with a single inferred structural scale, using continuous
#   optimization rather than discrete grid sampling.
#
# ========================================================================

# ========================================================================
# 5. THEORY SEPARATION
# ========================================================================
#
# Theories are kept strictly separate as follows:
#
#   MOND side:
#       data -> gbar_mond -> g_mond -> v_mond
#
#   CPTG side:
#       data -> gbar_cptg -> fit a_star -> derive Rc
#            -> solve g_cptg -> v_cptg
#
# They share:
#
#   - the same observed SPARC data file
#   - the same radii r
#   - the same measured gas/disk/bulge component inputs
#   - the same observed velocity vobs for comparison
#
# They do NOT share:
#
#   - the same baryonic source mapping
#   - the same acceleration law
#   - the same characteristic scale
#   - the same solver
#
# In other words:
#
#   SAME GALAXY DATA
#   DIFFERENT PHYSICAL INTERPRETATION
#   DIFFERENT SOURCE FIELD
#   DIFFERENT EQUATIONS
#   DIFFERENT PREDICTED ROTATION CURVES
#
# ========================================================================

# ========================================================================
# 6. DIAGNOSTICS / REPORTING
# ========================================================================
#
# After both theories are computed, the script compares them using:
#
#   - total rotation-curve chi^2
#   - velocity RMS error
#   - mean absolute velocity error
#   - RAR scatter
#   - bandwise performance versus structural mode number N
#
# Additional structural diagnostics are computed from the
# CPTG solution only, for CPTG interpretation and reporting.
# These diagnostics are NOT used by MOND.
#
# ========================================================================

# ========================================================================
# DATA / THEORY CITATIONS
# ========================================================================
#
# SPARC data source:
# Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016),
# "SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry
# and Accurate Rotation Curves,"
# The Astronomical Journal, 152, 157.
#
# MOND reference:
# Milgrom, M. (1983),
# "A Modification of the Newtonian Dynamics as a Possible Alternative
# to the Hidden Mass Hypothesis,"
# The Astrophysical Journal, 270, 365-370.
#
# MOND interpolation function used here:
# The benchmark uses the MOND form
#     g = g_bar / (1 - exp(-sqrt(g_bar / A0_MOND)))
# which corresponds to the interpolation function discussed by
# McGaugh, S. S. (2008),
# "Milky Way Mass Models and MOND,"
# The Astrophysical Journal, 683, 137-148.
#
# ========================================================================

# ========================================================================
# OPTIONAL METADATA FILES: WHAT THEY DO AND HOW TO USE THEM
# ========================================================================
#
# This benchmark is designed to run in TWO modes:
#
#   1) FILE-ONLY MODE
#      - uses only the galaxy data files found in EXTRACT_DIR
#      - no metadata file is required
#      - all galaxies found are benchmarked
#      - no database-specific sample labels are applied
#
#   2) METADATA-AWARE MODE
#      - uses the same galaxy data files
#      - ALSO loads a companion metadata file when one is available
#      - metadata is used only for OPTIONAL sample logic and reporting
#      - the core CPTG/MOND benchmark still comes from the galaxy files
#
#
# WHAT METADATA IS USED FOR
#
# When a usable metadata file is found, the script may use it to:
#
#   - determine whether a galaxy belongs to a PRIMARY science sample
#   - mark excluded galaxies with "!"
#   - compute inclination-aware error tracking when inclination data exist
#   - report both FULL-SAMPLE and PRIMARY-SAMPLE benchmark summaries
#
# Metadata does NOT replace the galaxy data files.
# Metadata only adds extra information about each galaxy.
#
#
# HOW THE SCRIPT FINDS METADATA
#
# The script looks for a companion metadata file automatically.
#
# It checks likely locations near:
#   - the script folder
#   - the ZIP file folder
#   - the extracted data folder
#
# It also checks common filenames such as:
#   SPARC_Lelli2016c.mrt
#   metadata.json
#   metadata.csv
#   metadata.tsv
#   sample_metadata.json
#   sample_metadata.csv
#   sample_metadata.tsv
#
# It may also scan for files ending in:
#   *.mrt
#   *.json
#   *.csv
#   *.tsv
#
#
# SUPPORTED METADATA FORMATS
#
# 1) SPARC .mrt
#    - intended for official SPARC master metadata
#    - lets the script derive the primary 153-galaxy science sample
#    - uses SPARC-style metadata such as quality flag and inclination
#
# 2) CSV / TSV
#    - generic tabular metadata
#    - one row per galaxy
#    - must include a galaxy-name column
#
# 3) JSON
#    - generic metadata object
#    - one entry per galaxy
#
#
# REQUIRED MATCHING RULE
#
# Metadata must identify galaxies by name in a way that matches the
# galaxy data filenames.
#
# Example:
#   data file:   NGC2403_rotmod.dat
#   metadata:    NGC2403
#
# The script normalizes names internally, so minor formatting differences
# are usually fine, but the galaxy identity still has to match.
#
#
# USEFUL GENERIC METADATA FIELDS
#
# If you are creating your own metadata file, these are the most useful
# fields to provide:
#
#   galaxy / name
#       The galaxy identifier
#
#   inc_deg
#       Inclination in degrees
#
#   e_inc_deg
#       Inclination uncertainty in degrees
#
#   q_flag
#       Quality flag, if your database uses one
#
#   primary_sample
#       True / False flag saying whether the galaxy belongs to the
#       preferred science sample
#
#
# GENERIC CSV / TSV EXAMPLE
#
#   galaxy,inc_deg,e_inc_deg,q_flag,primary_sample
#   NGC2403,62.9,2.0,1,True
#   DDO154,65.0,3.0,1,True
#   SomeGalaxy,24.0,5.0,2,False
#
#
# GENERIC JSON EXAMPLE
#
#   {
#     "NGC2403": {
#       "inc_deg": 62.9,
#       "e_inc_deg": 2.0,
#       "q_flag": 1,
#       "primary_sample": true
#     },
#     "DDO154": {
#       "inc_deg": 65.0,
#       "e_inc_deg": 3.0,
#       "q_flag": 1,
#       "primary_sample": true
#     }
#   }
#
#
# HOW PRIMARY-SAMPLE LOGIC WORKS
#
# If metadata includes an explicit field such as:
#   primary_sample = True / False
# then the script uses that directly.
#
# If explicit primary-sample flags are NOT present, the script may derive
# the primary sample from available metadata rules.
#
# For SPARC metadata, the script can derive the primary science sample
# using:
#   - quality flag
#   - inclination
#
# If no metadata is available, or no primary-sample information can be
# derived, the script simply treats the run as a FULL-SAMPLE benchmark.
#
#
# HOW INCLINATION SYSTEMATICS WORK
#
# If BOTH of these metadata fields exist:
#   inc_deg
#   e_inc_deg
#
# then the script can compute an inclination-systematic benchmark track
# in addition to the ordinary random-error benchmark track.
#
# If those fields are missing, the script skips that extra reporting and
# still runs normally.
#
#
# IMPORTANT METADATA FILTER RULE
#
# Generic CSV / TSV / JSON files are ONLY treated as metadata if they
# contain at least one real metadata field used by the benchmark, such as:
#
#   inc_deg
#   e_inc_deg
#   q_flag
#   primary_sample
#
# Files that only contain galaxy names or unrelated output columns are
# ignored and will NOT be misidentified as metadata.
#
#
# IMPORTANT DESIGN RULE
#
# The benchmark is metadata-OPTIONAL.
#
# That means:
#   - galaxy files are always the core input
#   - metadata only adds sample labeling and extra reporting
#   - the script should still work on other galaxy databases even when
#     no metadata file exists
#
#
# PRACTICAL USAGE
#
# To use metadata:
#
#   1) Place your metadata file near the script, ZIP file, or extracted
#      galaxy-data folder
#
#   2) Use one of the supported formats:
#      .mrt, .json, .csv, or .tsv
#
#   3) Make sure galaxy names in metadata match the galaxy data files
#
#   4) Include useful fields such as:
#      galaxy, inc_deg, e_inc_deg, q_flag, primary_sample
#
# If the script finds usable metadata, it will use it automatically.
# If it does not, it will still run in file-only mode.
# ========================================================================

# ========================================================================
# QUICK SETUP TUTORIAL
# ========================================================================
# This benchmark can:
# - load galaxy data from a ZIP archive or extracted folder
# - auto-detect galaxy Mode Type from the solved CPTG mode
# - print benchmark results to the screen
# - save the full printed benchmark output to a timestamped text file
# - optionally use metadata files for sample filtering and reporting
# ========================================================================
#
# ------------------------------------------------------------------------
# SECTION 1) INPUT DATABASE FILES
# ------------------------------------------------------------------------
#
# ZIP_PATH
#   This is the ZIP archive that contains your galaxy data files.
#
#   Example:
#   ZIP_PATH = os.path.join(SCRIPT_DIR, "sparc_database.zip")
#
#   That means:
#   - look in the same folder as this script
#   - find a ZIP file named "sparc_database.zip"
#
#   If your ZIP file has a different name:
#   ZIP_PATH = os.path.join(SCRIPT_DIR, "my_galaxy_data.zip")
#
#   If your ZIP file is somewhere else:
#   ZIP_PATH = r"D:\\Data\\Galaxies\\my_galaxy_data.zip"
#
#
# EXTRACT_DIR
#   This is the folder where the ZIP contents will be unpacked.
#
#   Example:
#   EXTRACT_DIR = os.path.join(SCRIPT_DIR, "sparc_database")
#
#   That means:
#   - create/use a folder named "sparc_database"
#   - inside the same folder as this script
#
#   If you want a different extracted folder name:
#   EXTRACT_DIR = os.path.join(SCRIPT_DIR, "my_galaxy_data")
#
#   If you want extraction somewhere else:
#   EXTRACT_DIR = r"D:\\Data\\Galaxies\\my_galaxy_data"
#
#   Important:
#   - ZIP_PATH is the ZIP archive
#   - EXTRACT_DIR is the unpacked folder
#   - they do NOT have to use the same name
#
#
# GALAXY_FILE_EXTENSION
#   This tells the script which galaxy files to load after extraction.
#
#   Example:
#   GALAXY_FILE_EXTENSION = ".dat"
#
#   Use:
#   ".dat" for files like DDO154.dat
#   ".txt" for files like DDO154.txt
#   ".csv" for files like DDO154.csv
#
#   Important:
#   - include the leading dot
#   - all galaxy files should use the same extension
#
#
# HOW THE SCRIPT FINDS FILES
#   After extraction, the script searches inside EXTRACT_DIR and all
#   subfolders for files matching GALAXY_FILE_EXTENSION.
#
#   Example:
#   my_galaxy_data.zip
#     -> galaxy_set/
#        -> DDO154.dat
#        -> DDO168.dat
#        -> NGC2366.dat
#
#   Example:
#   my_galaxy_data.zip
#     -> folderA/
#        -> dwarfs/
#           -> DDO154.dat
#        -> irregulars/
#           -> NGC2366.dat
#
#   The script deduplicates files by basename so duplicate extraction trees
#   do not double count the same galaxy.
#
#
# ------------------------------------------------------------------------
# SECTION 2) REQUIRED GALAXY DATA FORMAT
# ------------------------------------------------------------------------
#
# Each galaxy file must be a plain text numeric table with at least
# 6 whitespace-separated columns per usable row.
#
# Required columns, in this exact order:
#   col 1 = radius_kpc
#   col 2 = observed_velocity_km_s
#   col 3 = velocity_error_km_s
#   col 4 = gas_velocity_km_s
#   col 5 = disk_velocity_km_s
#   col 6 = bulge_velocity_km_s
#
# The script reads the FIRST 6 columns as:
#   r_kpc, vobs_km_s, errv_km_s, vgas_km_s, vdisk_km_s, vbul_km_s
#
# Extra columns are allowed, but ignored by the loader.
#
# Delimiter rules:
# - use spaces or tabs between numbers
# - do NOT rely on commas unless you rewrite the loader
#
# Comment rules:
# - blank lines are ignored
# - lines starting with # are ignored
#
# Minimum valid row example:
#   0.500  32.100  2.400  10.500  25.300  0.000
#
# Valid row with extra columns:
#   1.000  40.200  2.100  12.000  28.500  0.000  99.0  123.0
#
# Invalid row examples:
#   0.500,32.100,2.400,10.500,25.300,0.000   <- comma-separated
#   0.500 32.100 2.400 10.500 25.300         <- only 5 columns
#
#
# ------------------------------------------------------------------------
# SECTION 3) SAVED BENCHMARK OUTPUT FILE
# ------------------------------------------------------------------------
#
# In addition to printing benchmark results to the screen, the script can
# also save the full printed output to a text file.
#
# Output filename format:
#   CPTG_Benchmark_<extract_dir_name>_<month_day_year_hour_minute_second>.txt
#
# Important:
# - only the final folder name from EXTRACT_DIR is used
# - the directory portion of EXTRACT_DIR is removed
#
# Example:
#   EXTRACT_DIR = os.path.join(SCRIPT_DIR, "sparc_database")
#
#   Saved output file:
#   CPTG_Benchmark_sparc_database_04_23_2026_18_42_07.txt
#
# This output file is saved in SCRIPT_DIR unless you change the output path.
#
#
# ------------------------------------------------------------------------
# SECTION 4) MODE-ONLY AUTO GALAXY TYPE
# ------------------------------------------------------------------------
#
# This benchmark computes structural labels from the solved CPTG field.
#
# The script:
#   1) solves CPTG
#   2) computes the structural mode N
#   3) assigns:
#
#      mode_type = detect_galaxy_type_from_mode(N)
#
# This mode_type:
# - is derived ONLY from the mode value
# - uses no galaxy-name lookup
# - uses no catalog morphology
# - can be applied to SPARC and to custom galaxy databases
#
#
# MODE-TYPE THRESHOLDS
#
# LOW-MODE DWARF / IRREGULAR REGIME
#   N <= 1.45              : Dwarf Irregular
#   1.45 < N <= 1.75       : Magellanic Irregular
#   1.75 < N <= 1.95       : LSB Dwarf Disk
#   1.95 < N <= 2.15       : Transition Dwarf
#
# MID-MODE LSB / LATE-SPIRAL REGIME
#   2.15 < N <= 2.35       : LSB Spiral
#   2.35 < N <= 2.55       : Very Late Spiral
#   2.55 < N <= 2.80       : Late Spiral
#
# HIGHER-MODE SPIRAL / EARLY-DISK REGIME
#   2.80 < N <= 3.05       : Intermediate Spiral
#   3.05 < N <= 3.30       : Early Spiral
#   3.30 < N <= 3.55       : Bulged Spiral
#   3.55 < N <= 3.72       : Lenticular/Early Disk
#   N  > 3.72              : High-Mode Outlier
#
# Important:
# - Mode Type is a mode-based auto-detection layer
# - it is not a direct replacement for catalog morphology
# - it can be applied to SPARC and to custom galaxy databases equally
#
#
# ------------------------------------------------------------------------
# SECTION 5) CSMI PERFORMANCE TABLE
# ------------------------------------------------------------------------
#
# The CSMI PERFORMANCE table groups results by Mode Type.
#
# It reports CPTG / MOND performance by:
#   Dwarf Irregular
#   Magellanic Irregular
#   LSB Dwarf Disk
#   Transition Dwarf
#   LSB Spiral
#   Very Late Spiral
#   Late Spiral
#   Intermediate Spiral
#   Early Spiral
#   Bulged Spiral
#   Lenticular/Early Disk
#   High-Mode Outlier
#
# The print order follows MODE_TYPE_THRESHOLDS so the table appears in
# structural mode order rather than alphabetical order.
#
#
# ------------------------------------------------------------------------
# SECTION 6) ALL GALAXY-LEVEL DERIVED VALUES TABLE
# ------------------------------------------------------------------------
#
# The per-galaxy summary table prints:
#
#   Galaxy Name
#   Mode
#   Mode Type
#   Accel Rate
#   R_c[m]
#   gbar_cptg
#   gbar_mond
#   chi2(CPTG)
#   chi2(MOND)
#
# In this table only, Mode Type may be shown as:
#
#   Current
#   Current <- Lower
#   Current -> Higher
#
# This lets the benchmark report:
# - the continuous structural mode
# - the base mode type
# - whether the galaxy lies near a neighboring mode-type boundary
#
#
# MODE_TYPE_TRANSITION_WIDTH
#   This sets how close a mode value must be to a neighboring mode-type
#   boundary before the script labels that galaxy as transitioning rather
#   than stable.
#
# Example:
# - if MODE_TYPE_TRANSITION_WIDTH = 0.03
# - and a boundary is at N = 3.03
# - then values within 0.03 of that boundary are treated as transition cases
#
# Important:
# - this affects only the All Galaxy-Level Derived Values Mode Type display
# - it does not change the CSMI section grouping
#
#
# ------------------------------------------------------------------------
# SECTION 7) OPTIONAL METADATA FILES
# ------------------------------------------------------------------------
#
# This benchmark is metadata-optional.
#
# Metadata may be used for:
# - primary sample selection
# - excluded-sample marking
# - inclination-aware systematics
# - optional reporting
#
# Supported metadata formats:
#   *.mrt
#   *.json
#   *.csv
#   *.tsv
#
# Important:
# - generic CSV / TSV / JSON files are only treated as metadata if they
#   contain at least one usable metadata field such as:
#       inc_deg
#       e_inc_deg
#       q_flag
#       primary_sample
# - files that only contain galaxy names or unrelated output columns are
#   ignored and will NOT be misidentified as metadata
#
# This keeps custom-database runs clean when benchmark output files or
# other helper CSV files are present in the same folder.
#
#
# ------------------------------------------------------------------------
# SECTION 8) QUICK CHECKLIST
# ------------------------------------------------------------------------
#
# - Put the ZIP file where ZIP_PATH points
# - Set EXTRACT_DIR to the folder you want to use
# - Set GALAXY_FILE_EXTENSION to match your galaxy files
# - Make sure every galaxy file has at least 6 whitespace-separated
#   numeric columns in the required order
# - If you provide metadata, include at least one real metadata field
#   such as inc_deg, e_inc_deg, q_flag, or primary_sample
#
# The script will then:
#   - extract the database if needed
#   - find galaxy files
#   - deduplicate files by basename
#   - solve CPTG and MOND
#   - compute structural mode N
#   - assign Mode Type automatically from N
#   - print the per-galaxy table with Mode Type
#   - print the CSMI PERFORMANCE table by Mode Type
#   - save the full benchmark output to the timestamped text file
#
# ========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(SCRIPT_DIR, "sparc_database.zip")
EXTRACT_DIR = os.path.join(SCRIPT_DIR, "sparc_database")
GALAXY_FILE_EXTENSION = ".dat"
#=========================================================================

C_LIGHT = 299792458.0
KPC_TO_M = 3.085677581491367e19

A0_MOND = 1.2e-10   # MOND benchmark only

# ETA = 1/2
# Derived from the transport sector of the CPTG master action under galaxy-scale
# coarse-graining. In the quasi-static, axisymmetric disk limit, the transport
# tensor S^i S^j averages over the galaxy plane as:
#     <S^i S^j> = (1/2) P_disk^{ij}
# where P_disk^{ij} is the planar projector. This reduces the transport operator
# div(S^i S^j grad Phi) -> (1/2) div(grad Phi), yielding a factor of 1/2 in the effective
# geometric/transport term. With transport normalization absorbed into the
# reduced galaxy equation, this gives eta = 1/2 (no free parameter).
ETA = 1.0 / 2.0

# Locked CPTG component weights (exact fractions)
# CPTG uses its own geometric source mapping and is not forced to share MOND weights.
CPTG_GAS_WEIGHT = 27.0 / 50.0
CPTG_DISK_WEIGHT = 23.0 / 50.0
CPTG_BULGE_WEIGHT = 16.0 / 25.0

# Locked CPTG nonlinear / geometric exponents
CPTG_INNER_SUPPRESSION_POWER = 123.0 / 50.0
CPTG_OUTER_SUPPRESSION_POWER = 11.0 / 50.0
CPTG_GEOM_NUMERATOR_POWER = 7.0 / 5.0
CPTG_GEOM_DENOMINATOR_POWER = 12.0 / 5.0

# Fixed MOND baseline: standard baryonic decomposition
# MOND uses its own conventional baryonic field with unit component weights.
MOND_GAS_WEIGHT = 1.0
MOND_DISK_WEIGHT = 1.0
MOND_BULGE_WEIGHT = 1.0

# Structure / band diagnostics
KAPPA_STRUCT = KPC_TO_M
MIN_POINTS_FOR_STRUCTURE = 1

# Mode-only auto galaxy typing thresholds
# These empirical thresholds convert the continuous CPTG structural mode
# into an auto-detected galaxy type without using galaxy names, catalog labels,
# or database-specific metadata. This allows the benchmark to assign a
# structural galaxy type on any custom database once the CPTG mode has been
# computed from the solved field.
MODE_TYPE_THRESHOLDS = [
    (1.45, "Dwarf Irregular"),
    (1.75, "Magellanic Irregular"),
    (1.95, "LSB Dwarf Disk"),
    (2.15, "Transition Dwarf"),
    (2.35, "LSB Spiral"),
    (2.55, "Very Late Spiral"),
    (2.80, "Late Spiral"),
    (3.05, "Intermediate Spiral"),
    (3.30, "Early Spiral"),
    (3.55, "Bulged Spiral"),
    (3.72, "Lenticular/Early Disk"),
    (float("inf"), "High-Mode Outlier"),
]

MODE_TYPE_SHORT_LABELS = {
    "Dwarf Irregular": "Dwarf Irr",
    "Magellanic Irregular": "Magellanic Irr",
    "LSB Dwarf Disk": "LSB Dwarf Disk",
    "Transition Dwarf": "Transition Dwarf",
    "LSB Spiral": "LSB Spiral",
    "Very Late Spiral": "Very Late Spiral",
    "Late Spiral": "Late Spiral",
    "Intermediate Spiral": "Int Spiral",
    "Early Spiral": "Early Spiral",
    "Bulged Spiral": "Bulged Spiral",
    "Lenticular/Early Disk": "Lenticular/Early",
    "High-Mode Outlier": "High-Mode Outlier",
}

MODE_TYPE_INTERVALS = [
    (None, 1.45, "Dwarf Irregular"),
    (1.45, 1.75, "Magellanic Irregular"),
    (1.75, 1.95, "LSB Dwarf Disk"),
    (1.95, 2.15, "Transition Dwarf"),
    (2.15, 2.35, "LSB Spiral"),
    (2.35, 2.55, "Very Late Spiral"),
    (2.55, 2.80, "Late Spiral"),
    (2.80, 3.05, "Intermediate Spiral"),
    (3.05, 3.30, "Early Spiral"),
    (3.30, 3.55, "Bulged Spiral"),
    (3.55, 3.72, "Lenticular/Early Disk"),
    (3.72, None, "High-Mode Outlier"),
]

# Transition-zone half-width around each mode-type boundary.
# If a mode falls within 0.04 of a neighboring threshold, label it as transitioning.
MODE_TYPE_TRANSITION_WIDTH = 0.03

# Optional metadata-aware sample rules
# The core benchmark runs on galaxy files alone.
# When a companion metadata table is available, the script can derive
# database-specific subsets (such as the SPARC primary science sample)
# without hardcoded galaxy-name arrays.
SCIENCE_SAMPLE_MIN_INCLINATION_DEG = 30.0
SCIENCE_SAMPLE_EXCLUDED_Q_FLAGS = {3}
PRIMARY_SAMPLE_MARKER = "!"

# SPARC mass-model semantics
# - Vgas already includes the standard 1.33 helium factor in the published SPARC mass models.
# - Vdisk and Vbul are tabulated for unit stellar mass-to-light ratio at 3.6 micron.
# - The benchmark uses those supplied columns directly and does not rebuild or re-scale them elsewhere.
# - Signed component handling is enabled by default for gas, and extended to disk/bulge if negative values appear.
SPARC_SIGNED_GAS_BY_DEFAULT = True
SPARC_SIGNED_DISK_IF_NEGATIVE = True
SPARC_SIGNED_BULGE_IF_NEGATIVE = True

# Optional companion metadata table:
# - leave as None for automatic discovery
# - set to a file path to force a specific metadata table
METADATA_PATH = None
METADATA_REQUIRED = False

# Convert a signed velocity-like component into a signed contribution to V^2.
# This preserves inward/outward force information when SPARC-style component
# tables encode negative values in the inner regions.
def signed_square(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    return arr * np.abs(arr)

# Return the velocity component term used in the baryonic source field.
# When signed handling is enabled, preserve the sign convention of the mass
# model; otherwise use the ordinary squared-magnitude contribution.
def velocity_component_term(v: np.ndarray, use_signed: bool) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    return signed_square(arr) if use_signed else arr**2

# Extract the canonical galaxy name from a data-file path.
# This removes the standard rotmod suffix so filenames can be matched against
# metadata tables and used consistently in reporting.
def galaxy_name_from_path(path: str) -> str:
    name = os.path.basename(path)
    suffix = f"_rotmod{GALAXY_FILE_EXTENSION}"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    root, _ = os.path.splitext(name)
    return root

# Normalize a galaxy identifier into a robust lookup key.
# The key removes spacing and filename decoration so the same galaxy can be
# matched across data files, metadata tables, and report outputs.
def canonical_galaxy_key(name: str) -> str:
    return galaxy_name_from_path(str(name)).strip().replace(" ", "").upper()

# Parse a metadata value as a finite float when possible.
# Invalid, blank, or non-finite entries are converted to None so downstream
# sample-selection logic can distinguish missing metadata cleanly.
def parse_float_or_none(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        val = float(s)
    except Exception:
        return None
    return val if np.isfinite(val) else None

# Parse a metadata value as an integer quality/sample flag when possible.
# This is used for fields such as SPARC Q flags after first validating the
# source value as a finite numeric entry.
def parse_int_or_none(value) -> int | None:
    f = parse_float_or_none(value)
    return None if f is None else int(round(f))

# Parse a metadata value as a boolean sample-membership flag.
# Common textual and numeric truthy/falsy encodings are accepted, while
# unrecognized entries are left as None to indicate missing information.
def parse_bool_or_none(value) -> bool | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None

# Determine whether metadata place a galaxy in the primary science sample.
# Explicit primary-sample flags are used when present; otherwise the function
# derives the decision from quality-flag and inclination rules when available.
def metadata_primary_sample_flag(metadata: dict | None) -> bool | None:
    if not metadata:
        return None

    explicit = metadata.get("primary_sample")
    if explicit is not None:
        return bool(explicit)

    q_flag = metadata.get("q_flag")
    inc_deg = metadata.get("inc_deg")

    if q_flag is None and inc_deg is None:
        return None

    if q_flag in SCIENCE_SAMPLE_EXCLUDED_Q_FLAGS:
        return False
    if inc_deg is not None and inc_deg < SCIENCE_SAMPLE_MIN_INCLINATION_DEG:
        return False
    return True

# Convert inclination uncertainty into a velocity-space systematic term.
# This lets the benchmark report an inclination-aware chi^2 track alongside
# the random-error-only benchmark when metadata provide i and e_i values.
def inclination_velocity_systematic(v_m_s: np.ndarray, inc_deg: float, e_inc_deg: float) -> np.ndarray:
    v_m_s = np.asarray(v_m_s, dtype=float)
    inc_rad = np.deg2rad(float(inc_deg))
    e_inc_rad = np.deg2rad(max(float(e_inc_deg), 0.0))

    sin_i = np.sin(inc_rad)
    if abs(sin_i) < 1e-12:
        return np.full_like(v_m_s, np.inf, dtype=float)

    cot_i = np.cos(inc_rad) / sin_i
    return np.abs(v_m_s) * abs(cot_i) * e_inc_rad

# Build the ordered list of metadata files to try loading automatically.
# The search covers the script directory, the ZIP directory, and the extracted
# data directory so the benchmark can remain portable across databases.
def metadata_candidates(script_dir: str, zip_path: str, extract_dir: str) -> list[str]:
    candidates: list[str] = []

    if METADATA_PATH:
        candidates.append(os.path.abspath(METADATA_PATH))

    search_dirs = []
    for path in [script_dir, os.path.dirname(os.path.abspath(zip_path)), os.path.abspath(extract_dir)]:
        if path and os.path.isdir(path) and path not in search_dirs:
            search_dirs.append(path)

    preferred_names = [
        "SPARC_Lelli2016c.mrt",
        "Table1.mrt",
        "metadata.json",
        "metadata.csv",
        "metadata.tsv",
        "sample_metadata.json",
        "sample_metadata.csv",
        "sample_metadata.tsv",
    ]

    for directory in search_dirs:
        for name in preferred_names:
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                candidates.append(os.path.abspath(path))

        for pattern in ("*.mrt", "*.json", "*.csv", "*.tsv"):
            for path in sorted(glob(os.path.join(directory, pattern))):
                if os.path.isfile(path):
                    candidates.append(os.path.abspath(path))

    seen = set()
    unique = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique

# Load SPARC master metadata from an MRT table into normalized per-galaxy rows.
# The resulting dictionary provides inclination and quality information used
# for optional primary-sample labeling and inclination-aware reporting.
def load_sparc_mrt_metadata(path: str) -> dict[str, dict]:
    metadata_by_key: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 18:
                continue

            galaxy = parts[0].strip()
            inc_deg = parse_float_or_none(parts[5])
            e_inc_deg = parse_float_or_none(parts[6])
            q_flag = parse_int_or_none(parts[17])

            if not galaxy:
                continue
            if inc_deg is None and e_inc_deg is None and q_flag is None:
                continue

            metadata_by_key[canonical_galaxy_key(galaxy)] = {
                "name": galaxy,
                "inc_deg": inc_deg,
                "e_inc_deg": e_inc_deg,
                "q_flag": q_flag,
                "primary_sample": None,
            }
    return metadata_by_key

# Match a metadata-table column against a set of acceptable aliases.
# Column names are normalized before comparison so CSV/TSV metadata can use
# flexible naming while still mapping onto the benchmark's expected fields.
def best_matching_column(fieldnames: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {name: ''.join(ch for ch in name.lower() if ch.isalnum()) for name in fieldnames}
    alias_norms = {''.join(ch for ch in alias.lower() if ch.isalnum()) for alias in aliases}
    for original, norm in normalized.items():
        if norm in alias_norms:
            return original
    return None

# Load generic CSV/TSV metadata into the benchmark's normalized schema.
# This supports portable use on non-SPARC databases as long as the table
# contains a galaxy identifier and any optional sample-selection fields.
def load_tabular_metadata(path: str) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",	;|")
        except Exception:
            dialect = csv.excel
            dialect.delimiter = "\t" if path.lower().endswith(".tsv") else ","

        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            return {}

        galaxy_col = best_matching_column(reader.fieldnames, ("galaxy", "name", "object", "id", "file", "filename"))
        inc_col = best_matching_column(reader.fieldnames, ("inc", "inclination", "i"))
        e_inc_col = best_matching_column(reader.fieldnames, ("einc", "e_inc", "incerr", "incerror", "inclinationerror", "einclination"))
        q_col = best_matching_column(reader.fieldnames, ("q", "quality", "qualityflag", "qflag"))
        primary_col = best_matching_column(reader.fieldnames, ("primary", "isprimary", "science_sample", "primarysample", "in_primary_sample"))

        if galaxy_col is None:
            return {}

        # Ignore generic CSV/TSV files that only contain object names but do not
        # supply any actual metadata fields used by the benchmark.
        if inc_col is None and e_inc_col is None and q_col is None and primary_col is None:
            return {}

        metadata_by_key: dict[str, dict] = {}
        useful_rows = 0
        for row in reader:
            galaxy = (row.get(galaxy_col) or "").strip()
            if not galaxy:
                continue

            inc_deg_val = parse_float_or_none(row.get(inc_col)) if inc_col else None
            e_inc_deg_val = parse_float_or_none(row.get(e_inc_col)) if e_inc_col else None
            q_flag_val = parse_int_or_none(row.get(q_col)) if q_col else None
            primary_sample_val = parse_bool_or_none(row.get(primary_col)) if primary_col else None

            if inc_deg_val is not None or e_inc_deg_val is not None or q_flag_val is not None or primary_sample_val is not None:
                useful_rows += 1

            metadata_by_key[canonical_galaxy_key(galaxy)] = {
                "name": galaxy,
                "inc_deg": inc_deg_val,
                "e_inc_deg": e_inc_deg_val,
                "q_flag": q_flag_val,
                "primary_sample": primary_sample_val,
            }

        return metadata_by_key if useful_rows > 0 else {}

# Load generic JSON metadata into the benchmark's normalized schema.
# JSON support lets external databases provide per-galaxy sample and
# inclination metadata without depending on a tabular file format.
# Supported JSON shapes:
#   1) a list of row dictionaries, each with a galaxy/name/object/file field
#   2) an object with a "rows" list using the same row-dictionary format
#   3) a galaxy-keyed object such as {"NGC2403": {"inc_deg": 62.9, ...}}
def load_json_metadata(path: str) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload.get("rows", [])
    elif isinstance(payload, dict):
        rows = []
        for galaxy_key, value in payload.items():
            if not isinstance(value, dict):
                continue
            row = dict(value)
            row.setdefault("galaxy", galaxy_key)
            rows.append(row)
    else:
        rows = []

    metadata_by_key: dict[str, dict] = {}
    useful_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        galaxy = row.get("galaxy") or row.get("name") or row.get("object") or row.get("file") or row.get("filename")
        if not galaxy:
            continue

        inc_deg_val = parse_float_or_none(row.get("inc_deg", row.get("inc", row.get("inclination"))))
        e_inc_deg_val = parse_float_or_none(row.get("e_inc_deg", row.get("e_inc", row.get("inc_error"))))
        q_flag_val = parse_int_or_none(row.get("q_flag", row.get("q", row.get("quality_flag"))))
        primary_sample_val = parse_bool_or_none(row.get("primary_sample", row.get("is_primary")))

        if inc_deg_val is not None or e_inc_deg_val is not None or q_flag_val is not None or primary_sample_val is not None:
            useful_rows += 1

        metadata_by_key[canonical_galaxy_key(str(galaxy))] = {
            "name": str(galaxy),
            "inc_deg": inc_deg_val,
            "e_inc_deg": e_inc_deg_val,
            "q_flag": q_flag_val,
            "primary_sample": primary_sample_val,
        }
    return metadata_by_key if useful_rows > 0 else {}

# Attempt to load a usable companion metadata table, if one is available.
# The benchmark falls back to file-only mode when no metadata can be loaded
# unless metadata have been marked as required by configuration.
def load_optional_metadata(script_dir: str, zip_path: str, extract_dir: str) -> tuple[dict[str, dict], str | None]:
    for path in metadata_candidates(script_dir, zip_path, extract_dir):
        try:
            lower = path.lower()
            if lower.endswith(".mrt"):
                metadata = load_sparc_mrt_metadata(path)
            elif lower.endswith(".json"):
                metadata = load_json_metadata(path)
            elif lower.endswith(".csv") or lower.endswith(".tsv"):
                metadata = load_tabular_metadata(path)
            else:
                continue
        except Exception:
            metadata = {}

        if metadata:
            return metadata, path

    if METADATA_REQUIRED:
        raise FileNotFoundError("No usable companion metadata file was found.")
    return {}, None

# Ensure the SPARC database is extracted and available locally.
def ensure_database_available(zip_path: str, extract_dir: str) -> None:
    """Use existing extracted folder when present; otherwise extract once."""
    if os.path.isdir(extract_dir):
        return

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Could not find '{zip_path}'. Put sparc_database.zip in the same folder as this script."
        )

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

# Find and deduplicate SPARC rotmod data files by basename.
def find_unique_dat_files(extract_dir: str) -> list[str]:
    """
    Find unique SPARC rotmod files only.

    Deduplicate by basename so duplicate nested extraction trees do not
    double count the same galaxy file.
    """
    pattern = os.path.join(extract_dir, "**", f"*{GALAXY_FILE_EXTENSION}")
    found = [os.path.abspath(f) for f in glob(pattern, recursive=True) if os.path.isfile(f)]

    unique_by_name: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for path in sorted(found):
        name = os.path.basename(path)
        if name not in unique_by_name:
            unique_by_name[name] = path
        else:
            collisions.setdefault(name, [unique_by_name[name]]).append(path)

    if collisions:
        print(f"Duplicate filenames detected: {len(collisions)}")
        print("Using the first occurrence of each basename to prevent double counting.\n")

    return sorted(unique_by_name.values())

# Load one SPARC rotmod file and return the required data columns.
def load_sparc_file(path: str):
    """
    Load SPARC rotmod file.
    Accepts files with at least 6 columns.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                vals = [float(x) for x in parts]
                rows.append(vals)
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No usable numeric rows in {path}")

    width = max(len(r) for r in rows)
    arr = np.full((len(rows), width), np.nan, dtype=float)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r

    r_kpc = arr[:, 0]
    vobs_km_s = arr[:, 1]
    errv_km_s = arr[:, 2]
    vgas_km_s = arr[:, 3]
    vdisk_km_s = arr[:, 4]
    vbul_km_s = arr[:, 5]

    return r_kpc, vobs_km_s, errv_km_s, vgas_km_s, vdisk_km_s, vbul_km_s

# Build a baryonic source field from weighted component velocities.
def build_baryonic_field(
    r_m: np.ndarray,
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    gas_weight: float,
    disk_weight: float,
    bulge_weight: float,
) -> np.ndarray:
    """
    Build the baryonic source field once from SPARC component velocities.

    This is the ONLY place component velocities are combined.
    After this step, the model evolves only the total baryonic field g_bar.
    """
    use_signed_gas = SPARC_SIGNED_GAS_BY_DEFAULT or np.any(vgas < 0.0)
    use_signed_disk = SPARC_SIGNED_DISK_IF_NEGATIVE and np.any(vdisk < 0.0)
    use_signed_bulge = SPARC_SIGNED_BULGE_IF_NEGATIVE and np.any(vbul < 0.0)

    gas_term = velocity_component_term(vgas, use_signed_gas)
    disk_term = velocity_component_term(vdisk, use_signed_disk)
    bulge_term = velocity_component_term(vbul, use_signed_bulge)

    vb2_signed = gas_weight * gas_term + disk_weight * disk_term + bulge_weight * bulge_term
    vb2 = np.maximum(vb2_signed, 0.0)
    gbar = vb2 / np.maximum(r_m, 1e-30)
    return gbar

# Compute the MOND total acceleration from the MOND baryonic field.
def mond_acceleration(gbar: np.ndarray) -> np.ndarray:
    """Standard MOND acceleration law applied to the MOND baryonic field only."""
    x = np.maximum(gbar / A0_MOND, 1e-30)
    return gbar / (1.0 - np.exp(-np.sqrt(x)))

# Derive the CPTG characteristic scale Rc directly from a_star.
def rc_from_a_star(a_star: float) -> float:
    return (48.0 * np.pi * C_LIGHT**2) / np.maximum(a_star, 1e-30)

# Iteratively solve the locked CPTG total-field acceleration equation.
def cptg_acceleration(
    gbar: np.ndarray,
    r_m: np.ndarray,
    a_star: float,
    rc: float,
    n_iter: int = 300,
    tol: float = 1e-13,
) -> np.ndarray:
    """
    Solve the locked CPTG total-field acceleration:

        g = gbar + (sqrt(a_* gbar) + g_geom)
                  / [1 + (g / a_*)^(123/50)]^(1/4)

    with

        x = r / Rc
        g_geom = ETA * a_star * x^(7/5) / (1 + x^(12/5))

    No component fields are evolved separately here.
    """
    gbar = np.maximum(np.asarray(gbar, dtype=float), 1e-30)
    r_m = np.maximum(np.asarray(r_m, dtype=float), 1e-30)
    g = gbar.copy()

    x = r_m / np.maximum(rc, 1e-30)
    x_num = x ** CPTG_GEOM_NUMERATOR_POWER
    x_den = x ** CPTG_GEOM_DENOMINATOR_POWER

    for _ in range(n_iter):
        g_geom = ETA * a_star * x_num / (1.0 + x_den)
        numerator = np.sqrt(np.maximum(a_star * gbar, 0.0)) + g_geom
        denominator = (1.0 + (g / np.maximum(a_star, 1e-30)) ** CPTG_INNER_SUPPRESSION_POWER) ** CPTG_OUTER_SUPPRESSION_POWER
        g_new = gbar + numerator / denominator

        rel = np.max(np.abs(g_new - g) / np.maximum(np.abs(g_new), 1e-30))
        g = g_new
        if rel < tol:
            break

    return g

# Fit the galaxy-specific CPTG scale a_star and derive Rc directly.
def fit_a_star_and_rc_for_galaxy(
    g_bar: np.ndarray,
    r_m: np.ndarray,
    vobs_m_s: np.ndarray,
    errv_m_s: np.ndarray,
) -> dict:
    """
    CPTG-only inference of a_star using:
      1) bounded scalar minimization in log10(a_star)
      2) local refinement scan around the optimum
      3) symmetric edge-triggered bound extension when the optimum is clipped

    Extension rule:
      - if the refined optimum lands near a bound and the next outward scan
        improves chi^2 at 6-decimal precision, extend by 1 dex and refit
      - stop when the next outward scan reproduces the same chi^2 at
        6-decimal precision, treating that value as the floor/ceiling

    Rc is derived directly from a_star.
    """

    from scipy.optimize import minimize_scalar

    err_safe = np.maximum(errv_m_s, 1e-30)

    base_lo = -12.0
    base_hi = -9.0
    edge_tol = 0.005
    extension_step = 1.0
    stage1_xatol = 1e-8
    half_width = 0.02
    floor_decimals = 6
    max_extensions_per_side = 12

    # Evaluate total CPTG velocity chi^2 for a trial log10(a_star) during fitting.
    def chi2_of_log_a(x: float) -> float:
        a_star = 10.0 ** x
        rc = rc_from_a_star(a_star)

        g_cptg = cptg_acceleration(
            g_bar,
            r_m,
            a_star=a_star,
            rc=rc,
        )
        v_cptg = np.sqrt(np.maximum(g_cptg, 0.0) * r_m)
        chi2 = np.sum(((vobs_m_s - v_cptg) / err_safe) ** 2)
        return float(chi2)

    # Run the two-stage local CPTG scale search inside the current log10(a_star) bounds.
    def refine_over_bounds(x_lo_bound: float, x_hi_bound: float) -> tuple[float, float]:
        result = minimize_scalar(
            chi2_of_log_a,
            bounds=(x_lo_bound, x_hi_bound),
            method="bounded",
            options={"xatol": stage1_xatol},
        )

        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("a_star optimization failed")

        x0 = float(result.x)

        x_lo = max(x_lo_bound, x0 - half_width)
        x_hi = min(x_hi_bound, x0 + half_width)

        x_refine = np.linspace(x_lo, x_hi, 161)
        chi2_refine = np.array([chi2_of_log_a(x) for x in x_refine], dtype=float)

        idx_best = int(np.argmin(chi2_refine))
        best_log_a = float(x_refine[idx_best])
        best_chi2 = float(chi2_refine[idx_best])
        return best_log_a, best_chi2

    lo = base_lo
    hi = base_hi
    best_log_a, best_chi2 = refine_over_bounds(lo, hi)

    lower_extensions = 0
    upper_extensions = 0

    while lower_extensions < max_extensions_per_side and best_log_a <= lo + edge_tol:
        trial_lo = lo - extension_step
        trial_best_log_a, trial_best_chi2 = refine_over_bounds(trial_lo, hi)

        if round(trial_best_chi2, floor_decimals) < round(best_chi2, floor_decimals):
            lo = trial_lo
            best_log_a = trial_best_log_a
            best_chi2 = trial_best_chi2
            lower_extensions += 1
        else:
            break

    while upper_extensions < max_extensions_per_side and best_log_a >= hi - edge_tol:
        trial_hi = hi + extension_step
        trial_best_log_a, trial_best_chi2 = refine_over_bounds(lo, trial_hi)

        if round(trial_best_chi2, floor_decimals) < round(best_chi2, floor_decimals):
            hi = trial_hi
            best_log_a = trial_best_log_a
            best_chi2 = trial_best_chi2
            upper_extensions += 1
        else:
            break

    a_star = 10.0 ** best_log_a
    rc = rc_from_a_star(a_star)

    return {
        "a_star": a_star,
        "rc": rc,
    }

# Compute the root-mean-square of an array.
def rms(x) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2))) if arr.size else float("nan")

# Compute the standard deviation of an array.
def scatter(x) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.std(arr)) if arr.size else float("nan")

# Compute percent improvement of a model relative to a reference.
def improvement(reference: float, model: float) -> float:
    if reference == 0 or not np.isfinite(reference) or not np.isfinite(model):
        return float("nan")
    return float((reference - model) / reference * 100.0)

# Estimate the CPTG structural scale and mode diagnostics from the solved field.
def estimate_structure_scale_and_mode(r_m: np.ndarray, g_field: np.ndarray, kappa: float = KAPPA_STRUCT) -> float:
    """Estimate the curvature-based structural mode number N = R/lambda from the supplied field."""
    r_m = np.asarray(r_m, dtype=float)
    g_field = np.asarray(g_field, dtype=float)

    if r_m.size < MIN_POINTS_FOR_STRUCTURE:
        return float("nan")

    order = np.argsort(r_m)
    r_sorted = r_m[order]
    g_sorted = np.maximum(g_field[order], 1e-30)

    dg = np.gradient(g_sorted, r_sorted)
    C = g_sorted**2 + (kappa * dg)**2
    dC = np.gradient(C, r_sorted)

    finite = np.isfinite(r_sorted) & np.isfinite(dC) & (r_sorted > 0)
    if np.count_nonzero(finite) < MIN_POINTS_FOR_STRUCTURE:
        return float("nan")

    R_m = float(np.max(r_sorted))
    mask_outer = (r_sorted > 1 / 5 * R_m) & finite

    if np.count_nonzero(mask_outer) < MIN_POINTS_FOR_STRUCTURE:
        return float("nan")

    r_outer = r_sorted[mask_outer]
    dC_outer = np.abs(dC[mask_outer])

    window = 5
    if len(dC_outer) > window:
        kernel = np.ones(window) / window
        dC_outer = np.convolve(dC_outer, kernel, mode="same")

    weight_sum = np.sum(dC_outer)
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        return float("nan")

    lambda_m = float(np.sum(r_outer * dC_outer) / weight_sum)
    N = float(R_m / lambda_m) if lambda_m > 0 else float("nan")
    return N

# Assign an empirical auto-detected galaxy type from the CPTG mode only.
# This classifier is mode-based, not catalog-based, so it can be applied
# consistently to SPARC and to custom galaxy databases once mode has been
# computed from the solved CPTG field.
def detect_galaxy_type_from_mode(N: float) -> str:
    if not np.isfinite(N):
        return "Unknown"

    for upper_bound, label in MODE_TYPE_THRESHOLDS:
        if N <= upper_bound:
            return label

    return "Unknown"

# Determine the edge margin used to mark a transition case in the
# All Galaxy-Level Derived Values table.
def mode_type_transition_width(lower_bound: float | None, upper_bound: float | None) -> float:
    if lower_bound is None or upper_bound is None:
        return MODE_TYPE_TRANSITION_WIDTH
    return min(MODE_TYPE_TRANSITION_WIDTH, (upper_bound - lower_bound) / 3.0)

# Build the 38-character transition-aware label used only in the
# All Galaxy-Level Derived Values table.
def mode_type_display_from_mode(N: float) -> str:
    if not np.isfinite(N):
        return "Unknown"

    for idx, (lower_bound, upper_bound, full_label) in enumerate(MODE_TYPE_INTERVALS):
        in_interval = (
            (lower_bound is None or N > lower_bound) and
            (upper_bound is None or N <= upper_bound)
        )
        if not in_interval:
            continue

        current_label = MODE_TYPE_SHORT_LABELS.get(full_label, full_label)
        width = mode_type_transition_width(lower_bound, upper_bound)

        if lower_bound is not None and idx > 0 and N <= lower_bound + width:
            prev_full = MODE_TYPE_INTERVALS[idx - 1][2]
            prev_label = MODE_TYPE_SHORT_LABELS.get(prev_full, prev_full)
            return f"{current_label} <- {prev_label}"

        if upper_bound is not None and idx < len(MODE_TYPE_INTERVALS) - 1 and N >= upper_bound - width:
            next_full = MODE_TYPE_INTERVALS[idx + 1][2]
            next_label = MODE_TYPE_SHORT_LABELS.get(next_full, next_full)
            return f"{current_label} -> {next_label}"

        return current_label

    return "Unknown"

# Resolve a user-entered file, partial filename, directory, or comma-separated
# galaxy list into benchmark data files. Directory input loads all matching
# galaxy files inside it. Partial filename input searches available benchmark
# folders and uses the first sorted match if multiple files match. For comma-
# separated input, whitespace around each galaxy name is ignored and each entry
# is resolved independently.
def resolve_user_file_selection(user_input: str) -> tuple[list[str], str]:
    query = user_input.strip().strip('"')
    if not query:
        return [], ""

    search_roots = []
    for root in [EXTRACT_DIR, SCRIPT_DIR, os.getcwd()]:
        root_abs = os.path.abspath(root)
        if os.path.isdir(root_abs) and root_abs not in search_roots:
            search_roots.append(root_abs)

    def resolve_single_query(single_query: str) -> tuple[list[str], str]:
        single_query = single_query.strip().strip('"')
        if not single_query:
            return [], ""

        expanded = os.path.abspath(os.path.expanduser(single_query))

        if os.path.isdir(expanded):
            files = find_unique_dat_files(expanded)
            return files, f"directory: {expanded}"

        if os.path.isfile(expanded):
            return [expanded], f"file: {expanded}"

        matches = []
        query_lower = single_query.lower()
        for root in search_roots:
            pattern = os.path.join(root, "**", f"*{GALAXY_FILE_EXTENSION}")
            for path in glob(pattern, recursive=True):
                if os.path.isfile(path) and query_lower in os.path.basename(path).lower():
                    matches.append(os.path.abspath(path))

        matches = sorted(dict.fromkeys(matches))
        if not matches:
            return [], f"no match for: {single_query}"

        if len(matches) > 1:
            print(f"Multiple matches found for '{single_query}'. Loading first sorted match:")
            print(matches[0])

        return [matches[0]], f"partial match: {single_query} -> {matches[0]}"

    if "," in query:
        tokens = [part.strip() for part in query.split(",") if part.strip()]
        selected_files = []
        selected_labels = []
        missing_entries = []

        for token in tokens:
            token_files, token_label = resolve_single_query(token)
            if token_files:
                selected_files.extend(token_files)
                selected_labels.append(token_label)
            else:
                missing_entries.append(token)

        selected_files = sorted(dict.fromkeys(selected_files))

        if missing_entries:
            print("No usable galaxy files found for comma-separated entries:")
            for missing in missing_entries:
                print(f"  {missing}")

        if not selected_files:
            return [], f"no matches for: {query}"

        return selected_files, "comma-separated selection: " + "; ".join(selected_labels)

    return resolve_single_query(query)

# Parse a mode-filter command from the interactive search field.
# Accepted forms are m=2, M=2.1, m=2.15, or M=2.151.
# The entered value is treated as a prefix pattern against the 3-decimal
# printed Mode column. For example, m=2 matches 2.xxx only, m=2.1 matches
# 2.1xx only, m=2.15 matches 2.15x only, and m=2.151 matches 2.151 only.
def parse_mode_filter_input(user_input: str) -> tuple[dict | None, str | None]:
    query = user_input.strip()
    match = re.fullmatch(r"[mM]\s*=\s*([0-9]+(?:\.[0-9]{0,3})?)", query)
    if not match:
        if re.match(r"[mM]\s*=", query):
            return None, "Mode filter format is m=<mode>, with up to 3 decimals; examples: m=2, m=2.1, m=2.15, m=2.151"
        return None, None

    raw_mode = match.group(1)
    if raw_mode.endswith("."):
        raw_mode = raw_mode[:-1]

    if not raw_mode:
        return None, "Mode filter is missing a numeric mode value."

    decimals = len(raw_mode.split(".", 1)[1]) if "." in raw_mode else 0
    target_value = float(raw_mode)
    target_text = f"{target_value:.{decimals}f}"

    return {
        "target_value": target_value,
        "decimals": decimals,
        "target_text": target_text,
        "label": f"mode prefix match: Mode starts with {target_text}",
    }, None

# Return True when a computed CSMI mode matches the interactive m=<mode> filter.
# The comparison uses the printed 3-decimal Mode value as a prefix string so
# m=2 includes only 2.xxx modes, not 1.xxx values rounded up to 2.
def mode_number_matches_filter(mode_n: float, mode_filter: dict | None) -> bool:
    if mode_filter is None or not np.isfinite(mode_n):
        return False
    target_text = str(mode_filter.get("target_text"))
    return f"{mode_n:.3f}".startswith(target_text)

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# Run the full audited SPARC CPTG vs MOND benchmark pipeline.
def main(files_override: list[str] | None = None, run_label: str | None = None, mode_filter: dict | None = None) -> None:
    system("clear||cls")
    print()
    print('*' * 60)
    print("* CPTG / MOND Galaxy Benchmark Utility                     *")
    print('*' * 60)
    print("* Research Paper Reference:                                *")
    print("*                                                          *")
    print("* Curvature Polarization Transport Gravity                 *")
    print("*                                                          *")
    print("* A Variational Framework for Galaxies and Cluster Mergers *")
    print("*                                                          *")
    print("* Section 11: Galaxy Limit (Effective Reduction)           *")
    print("* Section 15: Structural Mode N                            *")
    print("*                                                          *")
    print("* - Author: Carter L Glass Jr                              *")
    print("* - E-mail: carterglass@bellsouth.net                      *")
    print("* - Orchid ID: https://orcid.org/0009-0005-7538-543X       *")
    print('*' * 60)
    print()

    os.chdir(SCRIPT_DIR)

    print(f"Script directory: {SCRIPT_DIR}")
    print(f"ZIP path        : {ZIP_PATH}")
    print(f"Extract dir     : {EXTRACT_DIR}\n")

    ensure_database_available(ZIP_PATH, EXTRACT_DIR)
    metadata_by_key, metadata_source = load_optional_metadata(SCRIPT_DIR, ZIP_PATH, EXTRACT_DIR)
    metadata_loaded = metadata_source is not None

    if files_override is None:
        files = find_unique_dat_files(EXTRACT_DIR)
        active_label = run_label or "full database"
    else:
        files = sorted(os.path.abspath(f) for f in files_override)
        active_label = run_label or "selected file set"

    if not files:
        print(f"No {GALAXY_FILE_EXTENSION} files found for: {active_label}")
        print("Provide a valid file, partial filename, directory, or place sparc_database.zip / extracted data beside this script.")
        return

    print(f"Active benchmark set: {active_label}")
    if mode_filter is not None:
        print(f"Mode filter         : {mode_filter['label']}")
        print("Mode filter behavior: process all database files, include only matching modes")
    print(f"Unique files found: {len(files)}\n")
    print("Please wait. Processing data files. After processing, save and close plot to search again.\n\n")

    gbar_all = []
    gbar_mond_all = []
    gobs_all = []
    gcptg_all = []
    gmond_all = []

    radius_all = []
    vcptg_all = []
    vmond_all = []
    vobs_all = []

    chi2_cptg_total = 0.0
    chi2_mond_total = 0.0

    abs_err_cptg = []
    abs_err_mond = []
    vel_err_cptg = []
    vel_err_mond = []

    rar_resid_cptg = []
    rar_resid_mond = []

    r_norm_all = []
    rar_resid_cptg_all = []
    rar_resid_mond_all = []
    rar_resid_newtonian_all = []

    avg_curve_r = []
    avg_curve_obs = []
    avg_curve_cptg = []
    avg_curve_mond = []

    selected_curve_data = []

    galaxy_summaries = []

    used_files = 0
    skipped_files = 0
    mode_filter_processed_files = 0
    mode_filter_excluded_files = 0

    primary_sample_files = 0
    excluded_sample_files = 0
    primary_sample_known_files = 0
    inclination_sample_files = 0

    chi2_cptg_total_primary = 0.0
    chi2_mond_total_primary = 0.0
    chi2_cptg_total_incl = 0.0
    chi2_mond_total_incl = 0.0
    chi2_cptg_total_primary_incl = 0.0
    chi2_mond_total_primary_incl = 0.0

    abs_err_cptg_primary = []
    abs_err_mond_primary = []
    vel_err_cptg_primary = []
    vel_err_mond_primary = []
    rar_resid_cptg_primary = []
    rar_resid_mond_primary = []
    primary_points = 0

    for file in files:
        try:
            r_kpc, vobs_km_s, errv_km_s, vgas_km_s, vdisk_km_s, vbul_km_s = load_sparc_file(file)

            r_m = r_kpc * KPC_TO_M
            vobs = vobs_km_s * 1000.0
            errv = np.maximum(errv_km_s * 1000.0, 1e-30)
            vgas = vgas_km_s * 1000.0
            vdisk = vdisk_km_s * 1000.0
            vbul = vbul_km_s * 1000.0

            galaxy_name = galaxy_name_from_path(file)
            metadata = metadata_by_key.get(canonical_galaxy_key(galaxy_name), None)

            primary_sample_flag = metadata_primary_sample_flag(metadata)
            primary_sample_known = primary_sample_flag is not None
            primary_sample = bool(primary_sample_flag) if primary_sample_known else True

            inc_deg = metadata.get("inc_deg") if metadata else None
            e_inc_deg = metadata.get("e_inc_deg") if metadata else None
            inclination_known = inc_deg is not None and e_inc_deg is not None

            if inclination_known:
                errv_total = np.sqrt(errv**2 + inclination_velocity_systematic(vobs, inc_deg=float(inc_deg), e_inc_deg=float(e_inc_deg))**2)
            else:
                errv_total = errv

            # Build baryonic source fields ONCE from component velocities.
            g_bar_cptg = build_baryonic_field(
                r_m=r_m,
                vgas=vgas,
                vdisk=vdisk,
                vbul=vbul,
                gas_weight=CPTG_GAS_WEIGHT,
                disk_weight=CPTG_DISK_WEIGHT,
                bulge_weight=CPTG_BULGE_WEIGHT,
            )
            g_bar_mond = build_baryonic_field(
                r_m=r_m,
                vgas=vgas,
                vdisk=vdisk,
                vbul=vbul,
                gas_weight=MOND_GAS_WEIGHT,
                disk_weight=MOND_DISK_WEIGHT,
                bulge_weight=MOND_BULGE_WEIGHT,
            )

            g_obs = vobs**2 / np.maximum(r_m, 1e-30)

            fit_scale = fit_a_star_and_rc_for_galaxy(
                g_bar=g_bar_cptg,
                r_m=r_m,
                vobs_m_s=vobs,
                errv_m_s=errv,
            )

            a_star_g = fit_scale["a_star"]
            rc_g = fit_scale["rc"]

            # Evolve ONLY total fields.
            g_cptg = cptg_acceleration(g_bar_cptg, r_m, a_star=a_star_g, rc=rc_g)
            g_mond = mond_acceleration(g_bar_mond)

            v_cptg = np.sqrt(np.maximum(g_cptg, 0.0) * r_m)
            v_mond = np.sqrt(np.maximum(g_mond, 0.0) * r_m)

            mode_n = estimate_structure_scale_and_mode(r_m=r_m, g_field=g_cptg)

            if mode_filter is not None:
                mode_filter_processed_files += 1
                if not mode_number_matches_filter(mode_n, mode_filter):
                    mode_filter_excluded_files += 1
                    continue

            chi2_cptg = np.sum(((vobs - v_cptg) / errv) ** 2)
            chi2_mond = np.sum(((vobs - v_mond) / errv) ** 2)
            chi2_cptg_incl = np.sum(((vobs - v_cptg) / np.maximum(errv_total, 1e-30)) ** 2)
            chi2_mond_incl = np.sum(((vobs - v_mond) / np.maximum(errv_total, 1e-30)) ** 2)

            chi2_cptg_total += chi2_cptg
            chi2_mond_total += chi2_mond
            if inclination_known:
                chi2_cptg_total_incl += chi2_cptg_incl
                chi2_mond_total_incl += chi2_mond_incl
                inclination_sample_files += 1

            if primary_sample_known:
                primary_sample_known_files += 1

            vel_err_cptg.extend((vobs - v_cptg).tolist())
            vel_err_mond.extend((vobs - v_mond).tolist())
            abs_err_cptg.extend(np.abs(vobs - v_cptg).tolist())
            abs_err_mond.extend(np.abs(vobs - v_mond).tolist())

            rar_resid_cptg_gal_arr = np.log10(np.maximum(g_obs / np.maximum(g_cptg, 1e-30), 1e-300))
            rar_resid_mond_gal_arr = np.log10(np.maximum(g_obs / np.maximum(g_mond, 1e-30), 1e-300))
            rar_resid_newtonian_gal_arr = np.log10(np.maximum(g_obs / np.maximum(g_bar_cptg, 1e-30), 1e-300))
            rar_resid_cptg.extend(rar_resid_cptg_gal_arr.tolist())
            rar_resid_mond.extend(rar_resid_mond_gal_arr.tolist())

            gbar_all.extend(g_bar_cptg.tolist())
            gbar_mond_all.extend(g_bar_mond.tolist())
            gobs_all.extend(g_obs.tolist())
            gcptg_all.extend(g_cptg.tolist())
            gmond_all.extend(g_mond.tolist())

            radius_all.extend(r_kpc.tolist())
            vcptg_all.extend((v_cptg / 1000.0).tolist())
            vmond_all.extend((v_mond / 1000.0).tolist())
            vobs_all.extend(vobs_km_s.tolist())

            if files_override is not None or mode_filter is not None:
                selected_curve_data.append(
                    {
                        "name": galaxy_name,
                        "r_kpc": np.asarray(r_kpc, dtype=float),
                        "vobs_km_s": np.asarray(vobs_km_s, dtype=float),
                        "errv_km_s": np.asarray(errv_km_s, dtype=float),
                        "vcptg_km_s": np.asarray(v_cptg / 1000.0, dtype=float),
                        "vmond_km_s": np.asarray(v_mond / 1000.0, dtype=float),
                    }
                )

            if primary_sample_known:
                if primary_sample:
                    primary_sample_files += 1
                    primary_points += len(r_kpc)
                    chi2_cptg_total_primary += chi2_cptg
                    chi2_mond_total_primary += chi2_mond
                    chi2_cptg_total_primary_incl += chi2_cptg_incl
                    chi2_mond_total_primary_incl += chi2_mond_incl
                    vel_err_cptg_primary.extend((vobs - v_cptg).tolist())
                    vel_err_mond_primary.extend((vobs - v_mond).tolist())
                    abs_err_cptg_primary.extend(np.abs(vobs - v_cptg).tolist())
                    abs_err_mond_primary.extend(np.abs(vobs - v_mond).tolist())
                    rar_resid_cptg_primary.extend(rar_resid_cptg_gal_arr.tolist())
                    rar_resid_mond_primary.extend(rar_resid_mond_gal_arr.tolist())
                else:
                    excluded_sample_files += 1

            mode_type = detect_galaxy_type_from_mode(mode_n)
            mode_type_display = mode_type_display_from_mode(mode_n)

            vel_resid_cptg_gal = vobs - v_cptg
            vel_resid_mond_gal = vobs - v_mond
            rms_cptg_gal = float(np.sqrt(np.mean(vel_resid_cptg_gal ** 2)))
            rms_mond_gal = float(np.sqrt(np.mean(vel_resid_mond_gal ** 2)))
            mae_cptg_gal = float(np.mean(np.abs(vel_resid_cptg_gal)))
            mae_mond_gal = float(np.mean(np.abs(vel_resid_mond_gal)))
            rar_cptg_gal = float(np.std(rar_resid_cptg_gal_arr))
            rar_mond_gal = float(np.std(rar_resid_mond_gal_arr))
            gbar_cptg_gal = float(np.mean(g_bar_cptg))
            gbar_mond_gal = float(np.mean(g_bar_mond))

            galaxy_summaries.append(
                {
                    "name": os.path.basename(file),
                    "chi2_cptg": chi2_cptg,
                    "chi2_mond": chi2_mond,
                    "npts": len(r_kpc),
                    "primary_sample_known": primary_sample_known,
                    "primary_sample": primary_sample,
                    "a_star": a_star_g,
                    "gbar_cptg": gbar_cptg_gal,
                    "gbar_mond": gbar_mond_gal,
                    "rc": rc_g,
                    "N": mode_n,
                    "mode_type": mode_type,
                    "mode_type_display": mode_type_display,
                    "rms_cptg": rms_cptg_gal,
                    "rms_mond": rms_mond_gal,
                    "mae_cptg": mae_cptg_gal,
                    "mae_mond": mae_mond_gal,
                    "rar_cptg": rar_cptg_gal,
                    "rar_mond": rar_mond_gal,
                }
            )

            vmax = np.max(vobs)
            rmax = np.max(r_kpc)
            if vmax > 0 and rmax > 0:
                avg_curve_r.extend((r_kpc / rmax).tolist())
                avg_curve_obs.extend((vobs / vmax).tolist())
                avg_curve_cptg.extend((v_cptg / vmax).tolist())
                avg_curve_mond.extend((v_mond / vmax).tolist())

                r_norm = r_kpc / rmax
                r_norm_all.extend(r_norm.tolist())
                rar_resid_cptg_all.extend(rar_resid_cptg_gal_arr.tolist())
                rar_resid_mond_all.extend(rar_resid_mond_gal_arr.tolist())
                rar_resid_newtonian_all.extend(rar_resid_newtonian_gal_arr.tolist())

            used_files += 1

        except Exception as e:
            skipped_files += 1
            print(f"Skipped {file}: {e}")
            continue

    if used_files == 0:
        if mode_filter is not None:
            print(f"No galaxies matched mode filter: {mode_filter['label']}")
            print(f"Database files processed for mode filter: {mode_filter_processed_files}")
            print(f"Files skipped due to load/solve errors    : {skipped_files}")
        else:
            print("No galaxies were successfully processed.")
        input("Press Enter to exit...")
        return

    gbar_all = np.asarray(gbar_all)
    gbar_mond_all = np.asarray(gbar_mond_all)
    gobs_all = np.asarray(gobs_all)
    gcptg_all = np.asarray(gcptg_all)
    gmond_all = np.asarray(gmond_all)
    radius_all = np.asarray(radius_all)
    vcptg_all = np.asarray(vcptg_all)
    vmond_all = np.asarray(vmond_all)
    vobs_all = np.asarray(vobs_all)

    vel_err_cptg = np.asarray(vel_err_cptg)
    vel_err_mond = np.asarray(vel_err_mond)
    abs_err_cptg = np.asarray(abs_err_cptg)
    abs_err_mond = np.asarray(abs_err_mond)

    vel_rms_cptg = rms(vel_err_cptg)
    vel_rms_mond = rms(vel_err_mond)

    mae_cptg = float(np.mean(abs_err_cptg)) if abs_err_cptg.size else float("nan")
    mae_mond = float(np.mean(abs_err_mond)) if abs_err_mond.size else float("nan")

    rar_scatter_cptg = scatter(rar_resid_cptg)
    rar_scatter_mond = scatter(rar_resid_mond)

    vel_err_cptg_primary = np.asarray(vel_err_cptg_primary)
    vel_err_mond_primary = np.asarray(vel_err_mond_primary)
    abs_err_cptg_primary = np.asarray(abs_err_cptg_primary)
    abs_err_mond_primary = np.asarray(abs_err_mond_primary)

    vel_rms_cptg_primary = rms(vel_err_cptg_primary)
    vel_rms_mond_primary = rms(vel_err_mond_primary)
    mae_cptg_primary = float(np.mean(abs_err_cptg_primary)) if abs_err_cptg_primary.size else float("nan")
    mae_mond_primary = float(np.mean(abs_err_mond_primary)) if abs_err_mond_primary.size else float("nan")
    rar_scatter_cptg_primary = scatter(rar_resid_cptg_primary)
    rar_scatter_mond_primary = scatter(rar_resid_mond_primary)

    mode_type_stats: dict[str, dict[str, object]] = {}
    for row in galaxy_summaries:
        mode_type_label = row.get("mode_type", "Unknown")
        if mode_type_label not in mode_type_stats:
            mode_type_stats[mode_type_label] = {
                "chi2_cptg": 0.0,
                "chi2_mond": 0.0,
                "points": 0,
                "galaxies": 0,
                "rms_cptg": [],
                "rms_mond": [],
                "mae_cptg": [],
                "mae_mond": [],
                "rar_cptg": [],
                "rar_mond": [],
            }

        stats = mode_type_stats[mode_type_label]
        stats["chi2_cptg"] += row["chi2_cptg"]
        stats["chi2_mond"] += row["chi2_mond"]
        stats["points"] += row["npts"]
        stats["galaxies"] += 1
        stats["rms_cptg"].append(row["rms_cptg"])
        stats["rms_mond"].append(row["rms_mond"])
        stats["mae_cptg"].append(row["mae_cptg"])
        stats["mae_mond"].append(row["mae_mond"])
        stats["rar_cptg"].append(row["rar_cptg"])
        stats["rar_mond"].append(row["rar_mond"])

    # =========================
    # SUMMARY PRINTS
    # =========================
    print()

    print('=' * 160)
    if mode_filter is not None:
        print('=' * 59, "Mode-Filtered Galaxy-Level Derived Values", '=' * 58)
        rows_to_print = [
            row for row in galaxy_summaries
            if mode_number_matches_filter(row.get("N", float("nan")), mode_filter)
        ]
    else:
        print('=' * 63, "All Galaxy-Level Derived Values", '=' * 64)
        rows_to_print = galaxy_summaries
    print('=' * 160)
    print(f"{'Galaxy Name':<25} {'Mode':>5} {'Mode Type':>38} {'a\u204e CPTG':>14} {'R_c[m] CPTG':>14} {'gbar_cptg':>14} {'gbar_mond':>14} {'\u03C7\u00B2 CPTG':>14} {'\u03C7\u00B2 MOND':>14}")
    print('-' * 160)
    for row in rows_to_print:
        N_val = row.get("N", float("nan"))
        N_str = f"{N_val:.3f}" if np.isfinite(N_val) else "nan"
        display_name = row['name'] + (PRIMARY_SAMPLE_MARKER if row.get('primary_sample_known', False) and not row.get('primary_sample', True) else '') + ('*' if row['chi2_mond'] < row['chi2_cptg'] else '')
        print(
            f"{display_name:<25} "
            f"{N_str:>5} "
            f"{row.get('mode_type_display', 'Unknown'):>38} "
            f"{row['a_star']:>14.6e} "
            f"{row['rc']:>14.6e} "
            f"{row['gbar_cptg']:>14.6e} "
            f"{row['gbar_mond']:>14.6e} "
            f"{row['chi2_cptg']:>14.6e} "
            f"{row['chi2_mond']:>14.6e}"
        )
    print('-' * 160)

    print("* indicates MOND win")
    if primary_sample_known_files:
        print(f"{PRIMARY_SAMPLE_MARKER} indicates not in the metadata-defined primary science sample")
    print()
    print()

    print('=' * 160)
    print('=' * 71, "CSMI PERFORMANCE", '=' * 71)
    print('=' * 160)
    print(f"{'Mode Type':<25} {'Galaxies':>8} {'Pts':>5} {'\u03C7\u00B2/pt CPTG':>16} {'\u03C7\u00B2/pt MOND':>16} {'RMS CPTG':>14} {'RMS MOND':>14} {'MAE CPTG':>14} {'MAE MOND':>14} {'RAR CPTG':>12} {'RAR MOND':>12}")
    print('-' * 160)

    mode_type_order = [label for _, label in MODE_TYPE_THRESHOLDS]
    seen_mode_types = set()
    ordered_mode_types = []
    for label in mode_type_order:
        if label in mode_type_stats and label not in seen_mode_types:
            ordered_mode_types.append(label)
            seen_mode_types.add(label)
    for label in sorted(mode_type_stats):
        if label not in seen_mode_types:
            ordered_mode_types.append(label)
            seen_mode_types.add(label)

    for mode_type_label in ordered_mode_types:
        stats = mode_type_stats[mode_type_label]
        galaxies_in_type = int(stats["galaxies"])
        points = int(stats["points"])
        if galaxies_in_type == 0 or points == 0:
            continue

        chi2_pt_cptg = stats["chi2_cptg"] / points
        chi2_pt_mond = stats["chi2_mond"] / points
        rms_cptg_band = float(np.mean(stats["rms_cptg"])) if stats["rms_cptg"] else float("nan")
        rms_mond_band = float(np.mean(stats["rms_mond"])) if stats["rms_mond"] else float("nan")
        mae_cptg_band = float(np.mean(stats["mae_cptg"])) if stats["mae_cptg"] else float("nan")
        mae_mond_band = float(np.mean(stats["mae_mond"])) if stats["mae_mond"] else float("nan")
        rar_cptg_band = float(np.mean(stats["rar_cptg"])) if stats["rar_cptg"] else float("nan")
        rar_mond_band = float(np.mean(stats["rar_mond"])) if stats["rar_mond"] else float("nan")
        print(
            f"{mode_type_label:<25} {galaxies_in_type:>8d} {points:>5d} {chi2_pt_cptg:>16.6f} {chi2_pt_mond:>16.6f} {rms_cptg_band:>14.6f} {rms_mond_band:>14.6f} {mae_cptg_band:>14.6f} {mae_mond_band:>14.6f} {rar_cptg_band:>12.6f} {rar_mond_band:>12.6f}"
        )
    print('-' * 160)
    print("Mode: Curvature-Weighted Structural Mode Index (CSMI); Type: Mode-Only Auto Galaxy Type")
    print(f"Mode Type Transition Width = {MODE_TYPE_TRANSITION_WIDTH}; Minimum Points for Structure = {MIN_POINTS_FOR_STRUCTURE}")
    print()
    print()

    primary_rows = [row for row in galaxy_summaries if row.get("primary_sample_known", False)]
    cptg_wins_primary = sum(1 for row in primary_rows if row.get("primary_sample", False) and row['chi2_cptg'] < row['chi2_mond'])
    mond_wins_primary = sum(1 for row in primary_rows if row.get("primary_sample", False) and row['chi2_mond'] < row['chi2_cptg'])
    ties_primary = sum(1 for row in primary_rows if row.get("primary_sample", False) and row['chi2_cptg'] == row['chi2_mond'])
    total_primary_results = cptg_wins_primary + mond_wins_primary + ties_primary
    cptg_win_pct_primary = (100.0 * cptg_wins_primary / total_primary_results) if total_primary_results else float("nan")

    dof_cptg_primary = max(primary_points - primary_sample_files, 1)
    dof_mond_primary = max(primary_points, 1)
    dof_cptg_full = max(len(vobs_all) - used_files, 1)
    dof_mond_full = max(len(vobs_all), 1)

    if primary_sample_known_files:
        print('=' * 72)
        print("=========== Metadata-Defined Primary Science Sample Benchmark ==========")
        print('=' * 72)
        if metadata_loaded:
            print(f"Metadata source          : {metadata_source}")
        print(f"Primary sample galaxies  : {primary_sample_files}")
        print(f"Excluded galaxies        : {excluded_sample_files}")
        print(f"Primary sample points    : {primary_points}\n")

        print("Total Rotation-Curve \u03C7\u00B2 Benchmark:")
        print(f"CPTG : {chi2_cptg_total_primary:.6f}")
        print(f"MOND : {chi2_mond_total_primary:.6f}")
        print(f"Improvement vs MOND: {improvement(chi2_mond_total_primary, chi2_cptg_total_primary):.6f}%")
        print()

        chi2_per_point_cptg_primary = chi2_cptg_total_primary / max(primary_points, 1)
        chi2_per_point_mond_primary = chi2_mond_total_primary / max(primary_points, 1)
        chi2_per_dof_cptg_primary = chi2_cptg_total_primary / dof_cptg_primary
        chi2_per_dof_mond_primary = chi2_mond_total_primary / dof_mond_primary

        print("Datapoint-Normalized \u03C7\u00B2 Benchmark:")
        print(f"CPTG: \u03C7\u00B2 / datapoint (N={primary_points}): {chi2_per_point_cptg_primary:.6f}")
        print(f"MOND: \u03C7\u00B2 / datapoint (N={primary_points}): {chi2_per_point_mond_primary:.6f}")
        print(f"Improvement vs MOND: {improvement(chi2_per_point_mond_primary, chi2_per_point_cptg_primary):.6f}%")
        print()

        print("Conservative Effective-DOF \u03C7\u00B2 Audit:")
        print(f"CPTG: \u03C7\u00B2 / effective DOF (effective DOF={dof_cptg_primary}): {chi2_per_dof_cptg_primary:.6f}")
        print(f"MOND: \u03C7\u00B2 / DOF (DOF={dof_mond_primary}; fixed {A0_MOND}): {chi2_per_dof_mond_primary:.6f}")
        print(f"Improvement vs MOND: {improvement(chi2_per_dof_mond_primary, chi2_per_dof_cptg_primary):.6f}%")
        print()

        if inclination_sample_files:
            print("Inclination-Aware Rotation-Curve \u03C7\u00B2 Benchmark:")
            print(f"CPTG : {chi2_cptg_total_primary_incl:.6f}")
            print(f"MOND : {chi2_mond_total_primary_incl:.6f}")
            print(f"Improvement vs MOND: {improvement(chi2_mond_total_primary_incl, chi2_cptg_total_primary_incl):.6f}%")
            print()

            chi2_per_point_cptg_primary_incl = chi2_cptg_total_primary_incl / max(primary_points, 1)
            chi2_per_point_mond_primary_incl = chi2_mond_total_primary_incl / max(primary_points, 1)
            chi2_per_dof_cptg_primary_incl = chi2_cptg_total_primary_incl / dof_cptg_primary
            chi2_per_dof_mond_primary_incl = chi2_mond_total_primary_incl / dof_mond_primary

            print("Inclination-Aware \u03C7\u00B2 / Datapoint Benchmark:")
            print(f"CPTG: \u03C7\u00B2 / datapoint (N={primary_points}): {chi2_per_point_cptg_primary_incl:.6f}")
            print(f"MOND: \u03C7\u00B2 / datapoint (N={primary_points}): {chi2_per_point_mond_primary_incl:.6f}")
            print(f"Improvement vs MOND: {improvement(chi2_per_point_mond_primary_incl, chi2_per_point_cptg_primary_incl):.6f}%")
            print()

            print("Inclination-Aware Conservative Effective-DOF \u03C7\u00B2 Audit:")
            print(f"CPTG: \u03C7\u00B2 / effective DOF (effective DOF={dof_cptg_primary}): {chi2_per_dof_cptg_primary_incl:.6f}")
            print(f"MOND: \u03C7\u00B2 / DOF (DOF={dof_mond_primary}; fixed {A0_MOND}): {chi2_per_dof_mond_primary_incl:.6f}")
            print(f"Improvement vs MOND: {improvement(chi2_per_dof_mond_primary_incl, chi2_per_dof_cptg_primary_incl):.6f}%")
            print()

        print("Velocity RMS Residual Benchmark:")
        print(f"CPTG : {vel_rms_cptg_primary:.6f}")
        print(f"MOND : {vel_rms_mond_primary:.6f}")
        print(f"Improvement vs MOND: {improvement(vel_rms_mond_primary, vel_rms_cptg_primary):.6f}%")
        print()

        print("Mean Absolute Velocity Error Benchmark:")
        print(f"CPTG : {mae_cptg_primary:.6f}")
        print(f"MOND : {mae_mond_primary:.6f}")
        print(f"Improvement vs MOND: {improvement(mae_mond_primary, mae_cptg_primary):.6f}%")
        print()

        print("RAR Residual Scatter Benchmark:")
        print(f"CPTG : {rar_scatter_cptg_primary:.6f}")
        print(f"MOND : {rar_scatter_mond_primary:.6f}")
        print(f"Improvement vs MOND: {improvement(rar_scatter_mond_primary, rar_scatter_cptg_primary):.6f}%")
        print()

        print("Galaxy-Level \u03C7\u00B2 Win Benchmark:")
        print(f"CPTG wins : {cptg_wins_primary}")
        print(f"MOND wins : {mond_wins_primary}")
        print(f"CPTG win rate: {cptg_win_pct_primary:.6f}%")

        if ties_primary:
            print(f"Ties      : {ties_primary}")
        print('=' * 72)
        print()
        print()

    print('=' * 72)
    if mode_filter is not None:
        print("============ Mode-Filtered Galaxy Benchmark Results ============")
    else:
        print("============= All Galaxy Benchmark Results =============")
    print('=' * 72)

    print(f"Unique {GALAXY_FILE_EXTENSION} files found : {len(files)}")
    print(f"Files used              : {used_files}")
    print(f"Files skipped           : {skipped_files}")
    if mode_filter is not None:
        print(f"Mode-filter files processed : {mode_filter_processed_files}")
        print(f"Mode-filter files excluded  : {mode_filter_excluded_files}")
        print(f"Mode-filter files included  : {used_files}")
    print(f"Total datapoints        : {len(vobs_all)}")
    if metadata_loaded:
        print(f"Metadata source         : {metadata_source}")
    else:
        print("Metadata source         : none (file-only benchmark mode)")
    print()

    print("Total Rotation-Curve \u03C7\u00B2 Benchmark:")
    print(f"CPTG : {chi2_cptg_total:.6f}")
    print(f"MOND : {chi2_mond_total:.6f}")
    print(f"Improvement vs MOND: {improvement(chi2_mond_total, chi2_cptg_total):.6f}%")
    print()

    chi2_per_point_cptg_full = chi2_cptg_total / max(len(vobs_all), 1)
    chi2_per_point_mond_full = chi2_mond_total / max(len(vobs_all), 1)
    chi2_per_dof_cptg_full = chi2_cptg_total / dof_cptg_full
    chi2_per_dof_mond_full = chi2_mond_total / dof_mond_full

    print("Datapoint-Normalized \u03C7\u00B2 Benchmark:")
    print(f"CPTG: \u03C7\u00B2 / datapoint (N={len(vobs_all)}): {chi2_per_point_cptg_full:.6f}")
    print(f"MOND: \u03C7\u00B2 / datapoint (N={len(vobs_all)}): {chi2_per_point_mond_full:.6f}")
    print(f"Improvement vs MOND: {improvement(chi2_per_point_mond_full, chi2_per_point_cptg_full):.6f}%")
    print()

    print("Conservative Effective-DOF \u03C7\u00B2 Audit:")
    print(f"CPTG: \u03C7\u00B2 / effective DOF (effective DOF={dof_cptg_full}): {chi2_per_dof_cptg_full:.6f}")
    print(f"MOND: \u03C7\u00B2 / DOF (DOF={dof_mond_full}; fixed {A0_MOND}): {chi2_per_dof_mond_full:.6f}")
    print(f"Improvement vs MOND: {improvement(chi2_per_dof_mond_full, chi2_per_dof_cptg_full):.6f}%")
    print()

    if inclination_sample_files:
        print("Inclination-Aware Total Rotation-Curve \u03C7\u00B2 Benchmark:")
        print(f"CPTG : {chi2_cptg_total_incl:.6f}")
        print(f"MOND : {chi2_mond_total_incl:.6f}")
        print(f"Improvement vs MOND: {improvement(chi2_mond_total_incl, chi2_cptg_total_incl):.6f}%")
        print()

        chi2_per_point_cptg_full_incl = chi2_cptg_total_incl / max(len(vobs_all), 1)
        chi2_per_point_mond_full_incl = chi2_mond_total_incl / max(len(vobs_all), 1)
        chi2_per_dof_cptg_full_incl = chi2_cptg_total_incl / dof_cptg_full
        chi2_per_dof_mond_full_incl = chi2_mond_total_incl / dof_mond_full

        print("Inclination-Aware Datapoint-Normalized \u03C7\u00B2 Benchmark:")
        print(f"CPTG: \u03C7\u00B2 / datapoint (N={len(vobs_all)}): {chi2_per_point_cptg_full_incl:.6f}")
        print(f"MOND: \u03C7\u00B2 / datapoint (N={len(vobs_all)}): {chi2_per_point_mond_full_incl:.6f}")
        print(f"Improvement vs MOND: {improvement(chi2_per_point_mond_full_incl, chi2_per_point_cptg_full_incl):.6f}%")
        print()

        print("Inclination-Aware Conservative Effective-DOF \u03C7\u00B2 Audit:")
        print(f"CPTG: \u03C7\u00B2 / effective DOF (effective DOF={dof_cptg_full}): {chi2_per_dof_cptg_full_incl:.6f}")
        print(f"MOND: \u03C7\u00B2 / DOF (DOF={dof_mond_full}; fixed {A0_MOND}): {chi2_per_dof_mond_full_incl:.6f}")
        print(f"Improvement vs MOND: {improvement(chi2_per_dof_mond_full_incl, chi2_per_dof_cptg_full_incl):.6f}%")
        print()

    print("Velocity RMS Residual Benchmark:")
    print(f"CPTG : {vel_rms_cptg:.6f}")
    print(f"MOND : {vel_rms_mond:.6f}")
    print(f"Improvement vs MOND: {improvement(vel_rms_mond, vel_rms_cptg):.6f}%\n")

    print("Mean Absolute Velocity Error Benchmark:")
    print(f"CPTG : {mae_cptg:.6f}")
    print(f"MOND : {mae_mond:.6f}")
    print(f"Improvement vs MOND: {improvement(mae_mond, mae_cptg):.6f}%\n")

    print("RAR Residual Scatter Benchmark:")
    print(f"CPTG : {rar_scatter_cptg:.6f}")
    print(f"MOND : {rar_scatter_mond:.6f}")
    print(f"Improvement vs MOND: {improvement(rar_scatter_mond, rar_scatter_cptg):.6f}%\n")

    cptg_wins = sum(1 for row in galaxy_summaries if row['chi2_cptg'] < row['chi2_mond'])
    mond_wins = sum(1 for row in galaxy_summaries if row['chi2_mond'] < row['chi2_cptg'])
    ties = sum(1 for row in galaxy_summaries if row['chi2_cptg'] == row['chi2_mond'])
    total_galaxy_results = cptg_wins + mond_wins + ties
    cptg_win_pct = (100.0 * cptg_wins / total_galaxy_results) if total_galaxy_results else float("nan")

    print("Galaxy-Level \u03C7\u00B2 Win Benchmark:")
    print(f"CPTG wins : {cptg_wins}")
    print(f"MOND wins : {mond_wins}")
    print(f"CPTG win rate: {cptg_win_pct:.6f}%")
    if ties:
        print(f"Ties      : {ties}")
    print('=' * 72)
    print()

    # =========================
    # PLOTS
    # =========================

    if matplotlib.get_backend().lower() == "agg":
        print("Plot display skipped because a non-interactive backend is active.")
        return

    r_norm_all_arr = np.array(r_norm_all)
    rar_resid_cptg_all_arr = np.array(rar_resid_cptg_all)
    rar_resid_mond_all_arr = np.array(rar_resid_mond_all)
    rar_resid_newtonian_all_arr = np.array(rar_resid_newtonian_all)

    # Adaptive RAR radial diagnostic.
    # Full database runs keep the original population-scatter behavior.
    # Selected small samples lower the bin threshold; if too few statistical
    # bins remain, the panel falls back to point-by-point residual magnitude
    # so comma-separated galaxy selections still show useful RAR structure.
    small_selected_sample = (files_override is not None or mode_filter is not None) and used_files <= 5
    bins = np.linspace(0.0, 1.0, 20)
    digitized = np.digitize(r_norm_all_arr, bins)
    min_rar_bin_points = 2 if small_selected_sample else 10

    r_centers = []
    scatter_newtonian = []
    scatter_cptg = []
    scatter_mond_vals = []

    for i in range(1, len(bins)):
        idx = np.where(digitized == i)[0]
        if len(idx) < min_rar_bin_points:
            continue
        r_centers.append(np.mean(r_norm_all_arr[idx]))
        scatter_newtonian.append(np.std(rar_resid_newtonian_all_arr[idx]))
        scatter_cptg.append(np.std(rar_resid_cptg_all_arr[idx]))
        scatter_mond_vals.append(np.std(rar_resid_mond_all_arr[idx]))

    use_rar_residual_fallback = small_selected_sample and len(r_centers) < 2

    avg_curve_r_arr = np.array(avg_curve_r)
    avg_curve_obs_arr = np.array(avg_curve_obs)
    avg_curve_cptg_arr = np.array(avg_curve_cptg)
    avg_curve_mond_arr = np.array(avg_curve_mond)

    bins2 = np.linspace(0.0, 1.0, 30)
    digitized2 = np.digitize(avg_curve_r_arr, bins2)

    r_avg = []
    vobs_avg = []
    vcptg_avg = []
    vmond_avg = []

    for i in range(1, len(bins2)):
        idx = np.where(digitized2 == i)[0]
        if len(idx) == 0:
            continue
        r_avg.append(np.mean(avg_curve_r_arr[idx]))
        vobs_avg.append(np.mean(avg_curve_obs_arr[idx]))
        vcptg_avg.append(np.mean(avg_curve_cptg_arr[idx]))
        vmond_avg.append(np.mean(avg_curve_mond_arr[idx]))

    # Build smoothed median curves in log-space for population-level plots.
    def binned_median_curve(x_vals, y_vals, n_bins=30, min_points=5):
        x_vals = np.asarray(x_vals, dtype=float)
        y_vals = np.asarray(y_vals, dtype=float)
        finite = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0.0) & (y_vals > 0.0)
        if np.count_nonzero(finite) < 2:
            return np.array([]), np.array([])

        logx = np.log10(x_vals[finite])
        logy = np.log10(y_vals[finite])
        edges = np.linspace(np.min(logx), np.max(logx), n_bins + 1)
        centers = []
        medians = []
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (logx >= edges[i]) & (logx < edges[i + 1])
            else:
                mask = (logx >= edges[i]) & (logx <= edges[i + 1])
            if np.count_nonzero(mask) < min_points:
                continue
            centers.append(10.0 ** np.median(logx[mask]))
            medians.append(10.0 ** np.median(logy[mask]))
        return np.asarray(centers), np.asarray(medians)

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
    ]
    rar_grid = grid[1, 1].subgridspec(1, 2, wspace=0.300000)
    ax_rar_cptg = fig.add_subplot(rar_grid[0, 0])
    ax_rar_mond = fig.add_subplot(rar_grid[0, 1])

    # Plot 1 second: RAR scatter vs radius, with single-galaxy fallback.
    if use_rar_residual_fallback:
        axes[0].scatter(r_norm_all_arr, np.abs(rar_resid_newtonian_all_arr), s=25, color="royalblue", label="Newtonian (CPTG source)")
        axes[0].scatter(r_norm_all_arr, np.abs(rar_resid_mond_all_arr), s=25, color="limegreen", label="MOND")
        axes[0].scatter(r_norm_all_arr, np.abs(rar_resid_cptg_all_arr), s=25, color="orange", label="CPTG")
        axes[0].set_ylabel(r"RAR Residual Magnitude $|\log_{10}(g_{obs}/g_{model})|$")
        axes[0].set_title("RAR Residual Magnitude vs Radius")
    else:
        axes[0].plot(r_centers, scatter_newtonian, color="royalblue", label="Newtonian (CPTG source)")
        axes[0].plot(r_centers, scatter_mond_vals, color="limegreen", label="MOND")
        axes[0].plot(r_centers, scatter_cptg, color="orange", label="CPTG")
        axes[0].set_ylabel(r"RAR Scatter ($\sigma$)")
        axes[0].set_title("RAR Scatter vs Radius")
    axes[0].set_xlabel("Normalized Radius (r / Rmax)")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2 first: population-averaged normalized rotation curve.
    axes[1].plot(r_avg, vobs_avg, color="royalblue", label="Observed")
    axes[1].plot(r_avg, vmond_avg, color="limegreen", label="MOND")
    axes[1].plot(r_avg, vcptg_avg, color="orange", label="CPTG")
    axes[1].set_xlabel("Normalized Radius (r / Rmax)")
    axes[1].set_ylabel("Normalized Velocity (v / Vmax)")
    axes[1].set_title("Population-Averaged Normalized Rotation Curve")
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3 third: selected-galaxy rotation curves or population velocity medians.
    if small_selected_sample and selected_curve_data:
        for curve_index, curve in enumerate(selected_curve_data):
            order = np.argsort(curve["r_kpc"])
            observed_label = "Observed ± errV" if curve_index == 0 else None
            mond_label = "MOND" if curve_index == 0 else None
            cptg_label = "CPTG" if curve_index == 0 else None

            axes[2].errorbar(
                curve["r_kpc"][order],
                curve["vobs_km_s"][order],
                yerr=curve["errv_km_s"][order],
                fmt="o",
                markersize=4,
                capsize=2,
                color="royalblue",
                ecolor="royalblue",
                label=observed_label,
            )
            axes[2].plot(curve["r_kpc"][order], curve["vmond_km_s"][order], color="limegreen", label=mond_label)
            axes[2].plot(curve["r_kpc"][order], curve["vcptg_km_s"][order], color="orange", label=cptg_label)

        axes[2].set_title(
            f"Selected-Galaxy Rotation Curve: {selected_curve_data[0]['name']}"
            if len(selected_curve_data) == 1 else
            "Selected-Galaxy Rotation Curves"
        )
    else:
        axes[2].scatter(radius_all, vobs_all, s=8, color="lightgray", label="Observed points")
        r_obs_med, vobs_med = binned_median_curve(radius_all, vobs_all, n_bins=30, min_points=10)
        r_mond_med, vmond_med = binned_median_curve(radius_all, vmond_all, n_bins=30, min_points=10)
        r_cptg_med, vcptg_med = binned_median_curve(radius_all, vcptg_all, n_bins=30, min_points=10)

        if r_obs_med.size:
            axes[2].plot(r_obs_med, vobs_med, color="royalblue", label="Observed median")
        if r_mond_med.size:
            axes[2].plot(r_mond_med, vmond_med, color="limegreen", label="MOND median")
        if r_cptg_med.size:
            axes[2].plot(r_cptg_med, vcptg_med, color="orange", label="CPTG median")
        axes[2].set_title("All-Galaxy Velocity Point Cloud + Median Trends")

    axes[2].set_xlabel("Radius [kpc]")
    axes[2].set_ylabel("Velocity [km/s]")
    axes[2].legend()
    axes[2].grid(True)

    # Plot 4 fourth: model-specific RAR acceleration panels.
    # CPTG and MOND are shown on their own baryonic source fields so the
    # x-axis semantics remain explicit rather than mixing model definitions.
    if small_selected_sample:
        ax_rar_cptg.scatter(gbar_all, gobs_all, s=22, color="lightgray", edgecolors="black", linewidths=0.4, label="Observed")
        order_cptg = np.argsort(gbar_all)
        ax_rar_cptg.plot(gbar_all[order_cptg], gbar_all[order_cptg], color="royalblue", label="Newtonian (CPTG source)")
        ax_rar_cptg.plot(gbar_all[order_cptg], gcptg_all[order_cptg], marker="o", markersize=3, color="orange", label="CPTG")

        ax_rar_mond.scatter(gbar_mond_all, gobs_all, s=22, color="lightgray", edgecolors="black", linewidths=0.4, label="Observed")
        order_mond = np.argsort(gbar_mond_all)
        ax_rar_mond.plot(gbar_mond_all[order_mond], gbar_mond_all[order_mond], color="royalblue", label="Newtonian (MOND source)")
        ax_rar_mond.plot(gbar_mond_all[order_mond], gmond_all[order_mond], marker="o", markersize=3, color="limegreen", label="MOND")
    else:
        ax_rar_cptg.scatter(gbar_all, gobs_all, s=8, color="lightgray", label="Observed")
        order_cptg = np.argsort(gbar_all)
        ax_rar_cptg.plot(gbar_all[order_cptg], gbar_all[order_cptg], color="royalblue", label="Newtonian (CPTG source)")
        x_cptg_curve, y_cptg_curve = binned_median_curve(gbar_all, gcptg_all)
        if x_cptg_curve.size:
            ax_rar_cptg.plot(x_cptg_curve, y_cptg_curve, color="orange", label="CPTG")

        ax_rar_mond.scatter(gbar_mond_all, gobs_all, s=8, color="lightgray", label="Observed")
        order_mond = np.argsort(gbar_mond_all)
        ax_rar_mond.plot(gbar_mond_all[order_mond], gbar_mond_all[order_mond], color="royalblue", label="Newtonian (MOND source)")
        x_mond_curve, y_mond_curve = binned_median_curve(gbar_mond_all, gmond_all)
        if x_mond_curve.size:
            ax_rar_mond.plot(x_mond_curve, y_mond_curve, color="limegreen", label="MOND")

    ax_rar_cptg.set_title("RAR: CPTG Source Field")
    ax_rar_cptg.set_xscale("log")
    ax_rar_cptg.set_yscale("log")
    ax_rar_cptg.set_xlabel(r"$g_{\mathrm{bar,CPTG}}$")
    ax_rar_cptg.set_ylabel(r"Acceleration")
    ax_rar_cptg.legend(fontsize=8)
    ax_rar_cptg.grid(True, which="both")

    ax_rar_mond.set_title("RAR: MOND Source Field")
    ax_rar_mond.set_xscale("log")
    ax_rar_mond.set_yscale("log")
    ax_rar_mond.set_xlabel(r"$g_{\mathrm{bar,MOND}}$")
    ax_rar_mond.set_ylabel(r"Acceleration")
    ax_rar_mond.legend(fontsize=8)
    ax_rar_mond.grid(True, which="both")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.350000)
    plt.show()

if __name__ == "__main__":
    extract_dir_name = os.path.basename(os.path.normpath(EXTRACT_DIR))
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    output_filename = f"CPTG_Benchmark_{extract_dir_name}_{timestamp}.txt"
    log_path = os.path.join(SCRIPT_DIR, output_filename)

    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    try:
        current_files_override = None
        current_label = "full database"

        main()

        while True:
            user_input = input(
                "\nEnter galaxy file, partial filename, directory path, comma-separated galaxies,\n"
                "m=<mode> mode filter, or <db> to return to database. Enter <q> to exit program.\n"
                ">"
            ).strip()
            print()

            user_command = user_input.lower()

            if user_command in {"q", "exit"}:
                break

            if user_command in {"db"}:
                current_files_override = None
                current_label = "full database"

                print()
                print("=" * 72)
                print("==================== Full Database Benchmark Run ====================")
                print("=" * 72)
                main(files_override=None, run_label=current_label)
                continue

            if not user_input:
                continue

            selected_mode_filter, mode_filter_error = parse_mode_filter_input(user_input)
            if mode_filter_error:
                print(mode_filter_error)
                continue

            if selected_mode_filter is not None:
                print()
                print("=" * 72)
                print("==================== Mode-Filtered Database Run ====================")
                print("=" * 72)
                main(files_override=None, run_label=selected_mode_filter["label"], mode_filter=selected_mode_filter)
                continue

            selected_files, selected_label = resolve_user_file_selection(user_input)
            if selected_files:
                current_files_override = selected_files
                current_label = selected_label

                print()
                print("=" * 72)
                print("==================== User-Selected Benchmark Run ====================")
                print("=" * 72)
                main(files_override=current_files_override, run_label=current_label)
            else:
                print(f"No usable galaxy files found for input: {user_input}")
    finally:
        sys.stdout = original_stdout
        log_file.close()
