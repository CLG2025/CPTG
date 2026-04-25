
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter, map_coordinates

# ========================================================================
# REFERENCE
# ========================================================================
#
# CPTG BULLET-CLUSTER CURVATURE TRANSPORT MODEL
#
# Curvature Polarization Transport Gravity:
# A Variational Framework for Galaxies and Cluster Mergers
#
# - Author: Carter L Glass Jr
# - E-mail: carterglass@bellsouth.net
# - Orchid: https://orcid.org/0009-0005-7538-543X
#
# ========================================================================

# ========================================================================
# HOW THIS CPTG BULLET CLUSTER MODEL WORKS
# ========================================================================
#
# This script is a reduced merger-scale CPTG proof-of-concept for the
# Bullet Cluster. It is not a full relativistic field solver. Its purpose
# is to test whether curvature transport plus curvature-induced
# polarization can reproduce the observed separation between gas, galaxy,
# and lensing structures without introducing a dark matter component.
#
# ------------------------------------------------------------------------
# 1. THEORY IDEA
# ------------------------------------------------------------------------
#
# The underlying CPTG picture is that matter does not merely source an
# ordinary Newtonian field. Matter also induces a nonlinear curvature
# response, including:
#
#     - curvature polarization
#     - curvature transport
#     - curvature-supported persistence
#
# In this view, displaced lensing structure does not require an added dark
# matter density component inside the script. Instead, the observed lensing
# morphology is modeled as the result of:
#
#     baryonic geometry
#     + curvature-derived polarization
#     + transported curvature
#     + curvature-weighted lensing projection
#
# The reduced model is therefore testing whether a merger-scale CPTG
# realization can generate the Bullet Cluster offset pattern in a single
# framework.
#
# ------------------------------------------------------------------------
# 2. MASTER CPTG FRAMEWORK BEHIND THE REDUCED MODEL
# ------------------------------------------------------------------------
#
# The script is based on the CPTG action:
#
#     S = integral d^4x sqrt(-g) [
#             R/(16 pi G)
#           + L_m
#           + beta a_star^(2/3) C^(1/3)
#           + L_trans
#         ]
#
# where:
#
#     R       = Ricci scalar
#     L_m     = matter lagrangian
#     a_star  = structure-dependent acceleration scale
#     C       = curvature-polarization invariant
#     L_trans = curvature transport sector
#
# The key nonlinear invariant is:
#
#     C = g^2 + kappa^2 (grad g)^2
#
# and the associated polarization field is represented as:
#
#     P = beta C^(-2/3) grad C
#
# In weak-field language, the source relation is:
#
#     div g = 4 pi G rho + div P
#
# In the full theory, div P acts as an additional effective source term.
# This script keeps that physical interpretation, but implements it with
# reduced proxy fields suitable for a merger-plane proof-of-concept.
#
# ------------------------------------------------------------------------
# 3. WHAT THE SCRIPT ACTUALLY MODELS
# ------------------------------------------------------------------------
#
# The merger plane is built from four baryonic components:
#
#     - Bullet galaxy concentration
#     - Main-cluster galaxy concentration
#     - Bullet gas component
#     - Main-cluster gas component
#
# These are represented as smooth Gaussian fields.
#
# No dark matter density field is inserted anywhere in the script.
#
# The baryonic geometry therefore provides the physical input structure from
# which the reduced curvature fields are constructed.
#
# ------------------------------------------------------------------------
# 4. BACKGROUND CURVATURE PROXIES
# ------------------------------------------------------------------------
#
# The script does not evolve the full spacetime metric or the full
# gravitational field g directly. Instead it builds reduced background
# proxies.
#
# 4a. Galaxy-dominated background proxy
#
# The script first builds a galaxy-dominated potential proxy:
#
#     phi_gal
#
# from the two galaxy components.
#
# From phi_gal it derives a background curvature-structure measure:
#
#     chi_background
#
# This field is used to shape:
#
#     - the effective transport velocity field
#     - the Bullet-anchored initial transported packet T0
#     - curvature-flux weighting during evolution
#
# In other words, chi_background tells the reduced solver where transported
# curvature is naturally guided, supported, and organized by the merger
# geometry.
#
# 4b. Full baryonic background polarization proxy
#
# The script also builds a full baryonic potential proxy:
#
#     phi_baryonic
#
# from gas + galaxy components together.
#
# From this, the code constructs a reduced background polarization solve:
#
#     C_bg = g_bg^2 + kappa^2 |grad g_bg|^2
#     P_bg = beta C_bg^(-2/3) grad C_bg
#     sigma_pol ~ max(0, div P_bg)
#
# This gives the large-scale polarization background used in the lensing
# projection.
#
# This is important conceptually:
#
#     - chi_background organizes transport
#     - sigma_pol provides the static large-scale polarization baseline
#
# ------------------------------------------------------------------------
# 5. TRANSPORTED CURVATURE FIELD
# ------------------------------------------------------------------------
#
# The active evolved field in the model is the positive transported-curvature
# mode:
#
#     T_pos(x,y,t)
#
# The code also carries a bounded negative bookkeeping mode:
#
#     T_neg(x,y,t)
#
# but the observable transported structure is driven mainly by T_pos.
#
# The reduced evolution is schematically:
#
#     dT_pos/dt = diffusion
#                 - advection
#                 - decay
#                 + source
#                 + curvature support
#
# Each term has a physical role:
#
#     diffusion
#         spreads transported curvature locally
#
#     advection
#         moves transported curvature along the effective merger-plane flow
#
#     decay
#         prevents indefinite buildup of unsupported transported structure
#
#     source
#         reinforces the channel through transported-curvature polarization
#
#     curvature support
#         allows an already-formed ridge to persist coherently
#
# The transport flow itself is built from the Bullet gas/galaxy geometry and
# the background curvature structure. Some scenario families additionally use:
#
#     - divergence-free projection
#     - axis-blend reshaping
#     - flank-bias reshaping
#     - axis/flank hybrid reshaping
#
# to test which transport morphology best matches the observed Bullet
# Cluster offsets.
#
# ------------------------------------------------------------------------
# 6. LOCAL TRANSPORTED-CURVATURE INVARIANT
# ------------------------------------------------------------------------
#
# Because the full field g is not evolved directly inside the transport step,
# the code constructs a local transported-curvature proxy from T_pos:
#
#     C_T = T_pos^2 + kappa_T^2 |grad T_pos|^2
#
# This mirrors the CPTG invariant:
#
#     C = g^2 + kappa^2 (grad g)^2
#
# and serves as the local measure of:
#
#     - transported curvature strength
#     - transported curvature-gradient structure
#
# This local invariant is the basis for both source reinforcement and
# curvature-support persistence in the active channel.
#
# ------------------------------------------------------------------------
# 7. SOURCE REINFORCEMENT
# ------------------------------------------------------------------------
#
# The source term is built from the transported-curvature polarization proxy:
#
#     P_T = beta C_T^(-2/3) grad C_T
#     source ~ max(0, div P_T)
#
# In the code, this source is then:
#
#     - smoothed
#     - optionally advected forward along the flow
#     - corridor-gated
#
# so that reinforcement remains tied to the populated transport channel
# rather than appearing arbitrarily throughout the grid.
#
# Physically, this means the transported ridge is not just an imposed shape.
# It can reinforce itself where transported curvature already exists in a
# coherent way.
#
# ------------------------------------------------------------------------
# 8. CURVATURE-SUPPORT OPERATOR
# ------------------------------------------------------------------------
#
# A second reinforcement mechanism is the curvature-support operator.
#
# This is built from:
#
#     - local transported-curvature strength
#     - local transported-curvature-gradient strength
#     - positive alignment between grad C_T and the transport direction
#     - the already-present transported amplitude
#
# Schematically:
#
#     support ~ C_T_norm^(1/3)
#               * (1 + coherence)^(1/3)
#               * grad_strength^(7/20)
#               * T_pos
#
# with
#
#     coherence = max(0, gradC_T_hat . u_hat)^2
#
# This term is what lets the script produce a persistent displaced
# transported-curvature ridge rather than a short-lived numerical wake.
#
# In plain language:
#
#     the more coherent the transported-curvature structure is along the
#     flow, the more strongly the model allows it to persist.
#
# ------------------------------------------------------------------------
# 9. LENSING PROJECTION
# ------------------------------------------------------------------------
#
# The script builds the final observable lensing proxy from two pieces:
#
#     1) a broad static polarization background
#     2) an evolved transported-curvature contribution
#
# The background piece is:
#
#     Sigma_background
#
# The transported piece is built from T_pos:
#
#     Sigma_T ~ |grad T_pos|_norm + T_pos_norm * C_T_norm^(1/3)
#
# The code then applies a ridge enhancement tied to the populated transport
# channel and constructs:
#
#     Sigma_final = Sigma_background + alpha_t_eff * Sigma_T_enhanced
#
# where:
#
#     alpha_t_eff
#
# is not imposed as a fixed global constant. It is derived from the active
# transported-curvature source/ridge region.
#
# The exported convergence map is then a normalized kappa reconstruction
# built from Sigma_final.
#
# This means the displaced lensing structure comes from the combination of:
#
#     broad polarization support
#     + transported-curvature ridge structure
#
# not from a dark matter halo inserted by hand.
#
# ------------------------------------------------------------------------
# 10. OBSERVATIONAL INTERPRETATION
# ------------------------------------------------------------------------
#
# In the reduced model:
#
#     gas
#         traces collisional baryonic plasma
#
#     galaxies
#         trace collisionless cluster concentrations
#
#     T_pos
#         traces the transported-curvature ridge
#
#     Sigma_final / kappa
#         gives the observable lensing morphology proxy
#
# The model therefore asks whether CPTG transport and polarization can
# reproduce the Bullet Cluster's observed dissociation pattern:
#
#     - mass vs gas offset in the Bullet side
#     - mass vs gas offsets in the main north/south structure
#     - main subclump separation
#     - cluster-scale separation
#
# ------------------------------------------------------------------------
# 11. HOW THE MODEL IS SCORED
# ------------------------------------------------------------------------
#
# The final scenario ranking is tied to the locked external Bullet Cluster
# benchmark used by the script:
#
#     - Bullet mass-galaxy offset: 17.78 +/- 0.66 kpc
#     - Bullet mass-ICM offset:    ~150 kpc
#     - Main-north mass-ICM:       ~200 kpc
#     - Main-south mass-ICM:       ~400 kpc
#     - Main subclump separation:  ~200 kpc
#     - Cluster-scale separation:  ~720 kpc
#
# The script compares the model output to these observables and builds a
# total score from squared z residuals using the benchmark-specific scoring
# sigmas.
#
# Therefore:
#
#     lower score = better agreement with the locked Bullet Cluster benchmark
#
# This score is used for controlled scenario ranking inside the reduced CPTG
# framework.
#
# ------------------------------------------------------------------------
# 12. WHAT THIS MODEL IS AND IS NOT
# ------------------------------------------------------------------------
#
# This model IS:
#
#     - a reduced CPTG merger-plane proof-of-concept
#     - a morphology/offset matching tool
#     - a test of whether transported curvature plus polarization can
#       reproduce Bullet Cluster structure without adding dark matter
#
# This model is NOT:
#
#     - a full relativistic field solve
#     - a full cosmological merger simulation
#     - a final statistical likelihood model for the Bullet Cluster
#
# It is best understood as a researcher-readable reduced model whose purpose
# is to test whether:
#
#     curvature transport
#     + curvature-derived source reinforcement
#     + curvature-support persistence
#     + curvature-weighted lensing projection
#
# can reproduce the key Bullet Cluster merger morphology in one CPTG
# framework while remaining explicit about the reduced numerical choices used
# by the code.
#
# ========================================================================

# ========================================================================
# RESEARCH BASIS / CITATION BLOCK
# ========================================================================
#
#  PRIMARY JWST / LENSING BENCHMARK REFERENCE
#
#    Cha, S., Cho, B. Y., Joo, H., Lee, W., HyeongHan, K., Scofield, Z. P.,
#    Finner, K., & Jee, M. J. 2025,
#    "A High-Caliber View of the Bullet Cluster through JWST Strong and
#     Weak Lensing Analyses,"
#    The Astrophysical Journal Letters, 987, L15.
#
#    Used in this model for:
#    - the high-resolution JWST mass-map interpretation
#    - the Bullet subcluster mass-galaxy offset benchmark
#      (17.78 +/- 0.66 kpc)
#    - the Bullet mass-ICM offset target (~150 kpc)
#    - the main-north and main-south mass-ICM offset targets
#      (~200 kpc and ~400 kpc)
#    - the interpretation that the main cluster contains resolved
#      north/south substructure and that the merger is more complex
#      than a simple binary-merger picture
#
#  CLASSIC OBSERVATIONAL BACKGROUND
#
#    The current benchmark logic is also historically informed by the
#    classic Bullet Cluster lensing / X-ray literature cited inside the
#    JWST paper, especially:
#
#    - Markevitch et al. 2002
#    - Clowe et al. 2004
#    - Clowe et al. 2006
#
#    These are the classic background references for the Bullet Cluster's
#    dissociative merger picture, while the active benchmark values in this
#    script are tied primarily to the newer JWST analysis above.
#
# ========================================================================

# Purpose: Compute Cartesian x/y gradients for a 2D field on the current grid.
def grad2(field, dx, dy):
    dfdY, dfdX = np.gradient(field, dy, dx)
    return dfdX, dfdY

# Purpose: Compute the 2D divergence of a vector field on the current grid.
def div2(vx, vy, dx, dy):
    dvx_dx = np.gradient(vx, dy, dx)[1]
    dvy_dy = np.gradient(vy, dy, dx)[0]
    return dvx_dx + dvy_dy

# Purpose: Build the scalar Laplacian through the shared gradient/divergence helpers.
def lap2(field, dx, dy):
    fx, fy = grad2(field, dx, dy)
    return div2(fx, fy, dx, dy)

# Purpose: Apply the compact edge-preserving five-point smoother used throughout the reduced solver.
def smooth5(field, passes=1):
    """Apply a compact 5-point smoother without periodic edge wrapping.

    The earlier proof-of-concept version used np.roll, which implicitly wrapped
    the grid edges onto each other. That is convenient numerically, but for a
    finite merger field it can create unphysical boundary reinforcement and can
    leak edge structure back into the interior. Here we keep the same 5-point
    stencil weights while copying the nearest physical edge value outward
    instead of wrapping to the opposite side of the domain.
    """
    out = np.array(field, copy=True, dtype=float)
    for _ in range(max(0, passes)):
        pad = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            4.0 * pad[1:-1, 1:-1]
            + pad[:-2, 1:-1]
            + pad[2:, 1:-1]
            + pad[1:-1, :-2]
            + pad[1:-1, 2:]
        ) / 8.0
    return out

# Purpose: Clip a field to nonnegative values and normalize its total weight to unity.
def normalize_nonnegative(field, eps=1e-30):
    out = np.maximum(np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    s = np.sum(out)
    if (not np.isfinite(s)) or s <= eps:
        return np.zeros_like(field)
    return out / s

# Purpose: Return the positive-weighted centroid of a field for anchor and peak readout.
def centroid_from_positive(field, X, Y, eps=1e-30):
    w = np.maximum(np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    s = np.sum(w)
    if (not np.isfinite(s)) or s <= eps:
        return (0.0, 0.0)
    return (float(np.sum(X * w) / s), float(np.sum(Y * w) / s))

# Purpose: Find the brightest finite peak inside a supplied measurement mask.
def peak_in_mask(field, X, Y, mask):
    safe = np.nan_to_num(field, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    arr = np.where(mask, safe, -np.inf)
    vmax = np.max(arr)
    if (not np.isfinite(vmax)) or vmax <= 0.0:
        return (float("nan"), float("nan"))
    idx = np.argmax(arr)
    iy, ix = np.unravel_index(idx, arr.shape)
    return (float(X[iy, ix]), float(Y[iy, ix]))

# Purpose: Mimic an observational smoothed local peak centroid inside a measurement window.
def centroid_peak_in_mask(field, X, Y, mask, smooth_sigma_pix=1.5, eps=1e-30):
    """Return an observational-style local peak centroid inside a mask.

    Weak-lensing peaks are typically inferred from smoothed aperture-filtered
    mass maps rather than a single raw pixel maximum. This helper mirrors that
    behavior by lightly smoothing the local map and then returning the positive
    weighted centroid within the requested window.
    """
    safe = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth_sigma_pix and smooth_sigma_pix > 0.0:
        safe = gaussian_filter(safe, sigma=smooth_sigma_pix)
    local = np.where(mask, np.maximum(safe, 0.0), 0.0)
    s = np.sum(local)
    if (not np.isfinite(s)) or s <= eps:
        return (float("nan"), float("nan"))
    return (float(np.sum(X * local) / s), float(np.sum(Y * local) / s))

# Purpose: Return the Euclidean separation between two model points in kpc.
def dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))

# Purpose: Evaluate the first-order upwind advection derivative for a transported field.
def upwind_first_order(T, ux, uy, dx, dy):
    T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

    T_left = np.roll(T, 1, axis=1)
    T_right = np.roll(T, -1, axis=1)
    T_down = np.roll(T, 1, axis=0)
    T_up = np.roll(T, -1, axis=0)

    dTdx_backward = (T - T_left) / dx
    dTdx_forward = (T_right - T) / dx
    dTdy_backward = (T - T_down) / dy
    dTdy_forward = (T_up - T) / dy

    dTdx = np.where(ux >= 0.0, dTdx_backward, dTdx_forward)
    dTdy = np.where(uy >= 0.0, dTdy_backward, dTdy_forward)

    adv = ux * dTdx + uy * dTdy
    return np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)

# Purpose: Project a velocity field to its approximately divergence-free component with an FFT Helmholtz solve.
def project_velocity_fft(u_x, u_y, dx, dy):
    ny, nx = u_x.shape
    kx = fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = fftfreq(ny, d=dy) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)

    Ux = fft2(u_x)
    Uy = fft2(u_y)

    k2 = KX * KX + KY * KY
    k2[0, 0] = 1.0

    div_hat = 1j * (KX * Ux + KY * Uy)
    phi_hat = -div_hat / k2
    phi_hat[0, 0] = 0.0

    Ux_proj = Ux - 1j * KX * phi_hat
    Uy_proj = Uy - 1j * KY * phi_hat

    uxp = np.real(ifft2(Ux_proj))
    uyp = np.real(ifft2(Uy_proj))
    return uxp, uyp

# Purpose: Sample a field upstream along the transport flow for semi-Lagrangian source carry-forward.
def advect_field_along_flow(field, u_x, u_y, dt, dx, dy):
    """Sample a field one step upstream along the transport flow."""
    arr = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    ny, nx = arr.shape
    iy, ix = np.indices(arr.shape, dtype=float)
    shift_x = (u_x * dt) / max(dx, 1e-12)
    shift_y = (u_y * dt) / max(dy, 1e-12)
    sample_y = np.clip(iy - shift_y, 0.0, ny - 1.0)
    sample_x = np.clip(ix - shift_x, 0.0, nx - 1.0)
    return map_coordinates(arr, [sample_y, sample_x], order=1, mode="nearest")

# Purpose: Generate an elliptical Gaussian component for the reduced baryonic geometry.
def gaussian2d(X, Y, x0, y0, sx, sy, amp):
    return amp * np.exp(-0.5 * (((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2))

# Purpose: Return the RMS-style field norm used for timestep diagnostics.
def l2norm(field):
    arr = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean(arr * arr)))

@dataclass
class Config:
    # ====================================================================
    # GRID / DOMAIN GEOMETRY
    # ====================================================================

    # Horizontal grid resolution in x for the merger-plane computational mesh.
    NX: int = 420
    # Vertical grid resolution in y for the merger-plane computational mesh.
    NY: int = 300
    # Minimum x coordinate of the computational domain in kpc.
    XMIN: float = -900.0
    # Maximum x coordinate of the computational domain in kpc.
    XMAX: float = 900.0
    # Minimum y coordinate of the computational domain in kpc.
    YMIN: float = -550.0
    # Maximum y coordinate of the computational domain in kpc.
    YMAX: float = 550.0

    # ====================================================================
    # MERGER GEOMETRY ANCHORS
    # ====================================================================

    # Bullet-side galaxy anchor x position in kpc.
    BULLET_GAL_X: float = -260.0
    # Bullet-side galaxy anchor y position in kpc.
    BULLET_GAL_Y: float = 35.0
    # Main-cluster galaxy anchor x position in kpc.
    MAIN_GAL_X: float = 458 * (24 / 25)
    # Main-cluster galaxy anchor y position in kpc.
    MAIN_GAL_Y: float = -20.0

    # Bullet-side collisional gas anchor x position in kpc.
    BULLET_GAS_X: float = -130.0
    # Bullet-side collisional gas anchor y position in kpc.
    BULLET_GAS_Y: float = 0.0
    # Main-cluster collisional gas anchor x position in kpc.
    MAIN_GAS_X: float = 387
    # Main-cluster collisional gas anchor y position in kpc.
    MAIN_GAS_Y: float = 273

    # ====================================================================
    # MEASUREMENT WINDOWS / OBSERVATIONAL READOUT
    # ====================================================================

    # Radius of the Bullet-side lens-peak extraction window in kpc.
    BULLET_MEASURE_WINDOW_KPC: float = 120.0
    # Radius of the main-cluster primary lens-peak extraction window in kpc.
    MAIN_MEASURE_WINDOW_KPC: float = 120.0
    # Northward offset from the main galaxy anchor for the north subclump search center.
    MAIN_NORTH_MEASURE_DY: float = 100.0
    # Southward offset from the main galaxy anchor for the south subclump search center.
    MAIN_SOUTH_MEASURE_DY: float = -100.0
    # Radius of the north/south main-subclump extraction windows in kpc.
    MAIN_NS_MEASURE_WINDOW_KPC: float = 50.0

    # ====================================================================
    # BARYONIC MORPHOLOGY MODEL
    # ====================================================================

    # Bullet galaxy Gaussian x width for the baryonic morphology model.
    BULLET_GAL_SX: float = 95.0
    # Bullet galaxy Gaussian y width for the baryonic morphology model.
    BULLET_GAL_SY: float = 80.0
    # Main galaxy Gaussian x width for the baryonic morphology model.
    MAIN_GAL_SX: float = 130.0
    # Main galaxy Gaussian y width for the baryonic morphology model.
    MAIN_GAL_SY: float = 190
    # Bullet gas Gaussian x width for the baryonic morphology model.
    BULLET_GAS_SX: float = 180.0
    # Bullet gas Gaussian y width for the baryonic morphology model.
    BULLET_GAS_SY: float = 115.0
    # Main gas Gaussian x width for the baryonic morphology model.
    MAIN_GAS_SX: float = 210.0
    # Main gas Gaussian y width for the baryonic morphology model.
    MAIN_GAS_SY: float = 85.0

    # Bullet galaxy Gaussian amplitude for the baryonic morphology model.
    BULLET_GAL_AMP: float = 1.05
    # Main galaxy Gaussian amplitude for the baryonic morphology model.
    MAIN_GAL_AMP: float = 1.55
    # Bullet gas Gaussian amplitude for the baryonic morphology model.
    BULLET_GAS_AMP: float = 0.95
    # Main gas Gaussian amplitude for the baryonic morphology model.
    MAIN_GAS_AMP: float = 1.05

    # ====================================================================
    # OUTPUT / LENSING READOUT CONTROLS
    # ====================================================================

    # Gaussian smoothing applied to the exported normalized kappa map, in pixels.
    LENSING_SMOOTH_SIGMA_PIX: float = 1.0
    # Gaussian smoothing used during local lens-peak centroid extraction, in pixels.
    PEAK_SMOOTH_SIGMA_PIX: float = 1.5

    # ====================================================================
    # CURVATURE / POLARIZATION BACKGROUND
    # ====================================================================

    # Curvature-gradient length scale used in the chi background construction.
    KAPPA_C: float = 120.0
    # CPTG polarization coupling used in the reduced curvature source terms.
    BETA_POL: float = 21.0 / 100.0
    # Numerical floor preventing singular behavior in polarization-related powers.
    POL_FLOOR: float = 1e-18
    # Number of compact smoothing passes applied to curvature gradients.
    C_GRAD_SMOOTHING_PASSES: int = 1

    # ====================================================================
    # TRANSPORT INTEGRATION / TIMESTEPPING
    # ====================================================================

    # Number of transport steps in the reduced merger evolution.
    NSTEPS: int = 120
    # Base numerical timestep used in the explicit transport update.
    NUM_DT: float = 0.35
    # Whether to rescale the numerical timestep by the grid spacing.
    CFL_NORMALIZE_NUM_DT: bool = True
    # Reference grid spacing in kpc used for the numerical timestep normalization.
    NUM_DT_REF_DX_KPC: float = 1800.0 / 419.0
    # Diffusion coefficient for the transported-curvature packet.
    D_TRANS_NUM: float = 14.0
    # Decay coefficient for the transported-curvature packet.
    GAMMA_TRANS_NUM: float = 1.0 / 100.0
    # Physical time represented by each evolution step.
    PHYS_DT_PER_STEP: float = 1.0e13

    # ====================================================================
    # INITIAL TRANSPORTED-CURVATURE PACKET
    # ====================================================================

    # Overall amplitude of the Bullet-anchored initial transported-curvature packet.
    T0_STRENGTH: float = 1.0
    # Forward shift of the lead component of the initial packet, in kpc.
    T0_LEAD_SHIFT_KPC: float = 60.0
    # Wake shift of the trailing component of the initial packet, in kpc.
    T0_WAKE_SHIFT_KPC: float = 110.0
    # Parallel width of the lead component of the initial packet.
    T0_LEAD_SIGMA_PAR: float = 60.0
    # Perpendicular width of the lead component of the initial packet.
    T0_LEAD_SIGMA_PERP: float = 55.0
    # Parallel width of the wake component of the initial packet.
    T0_WAKE_SIGMA_PAR: float = 723.0 / 5.0
    # Perpendicular width of the wake component of the initial packet.
    T0_WAKE_SIGMA_PERP: float = 85.0
    # Relative weight of the wake subtraction in the initial packet.
    T0_WAKE_WEIGHT: float = 11.0 / 20.0
    # Scale factor boosting the lead component in the initial packet build.
    T0_SCALE: float = 5.0

    # Number of compact smoothing passes applied to the chi background field.
    CHI_SMOOTHING_PASSES: int = 1
    # Number of compact smoothing passes that maintain finite transverse corridor width.
    TRANSPORT_COHERENCE_PASSES: int = 1
    # Gas-repulsion scale suppressing the initial packet too near the gas peak.
    GAS_REPEL_SIGMA: float = 170.0

    # ====================================================================
    # TRANSPORT FLOW / STABILITY CONTROLS
    # ====================================================================

    # Upper numerical clip for positive transported curvature.
    CLIP_T_MAX: float = 5.0
    # Small numerical epsilon used throughout the solver to avoid singular divisions.
    EPS: float = 1e-30

    # Overall transport-speed scale for the curvature-derived velocity field.
    V_TRANSPORT: float = 5.0
    # Maximum allowed transport speed after velocity scaling.
    U_MAX: float = 40.0
    # Minimum curvature transport weighting retained in low-curvature regions.
    CURVATURE_TRANSPORT_FLOOR: float = 0.01
    # Global prefactor applied to the diffusion term in the transport update.
    DIFFUSION_SCALE: float = 1.0 / 5.0
    # Damping scale controlling how strongly gradients in chi suppress transport speed.
    GRAD_CHI_SCALE: float = 1.0 / 2000000.0
    # Overall scale of the curvature-weighted flux transport term.
    FLUX_TRANSPORT_SCALE: float = 500.0
    # Power applied to the curvature transport gate in the flux coefficient.
    FLUX_CURVATURE_POWER: float = 1.0
    # Power applied to the flow/curvature-alignment factor in the flux coefficient.
    FLUX_ALIGNMENT_POWER: float = 1.0 / 2.0
    # Number of compact smoothing passes applied to the flux coefficient field.
    FLUX_SMOOTHING_PASSES: int = 1

    # ====================================================================
    # RIDGE / SUPPORT DIAGNOSTICS
    # ====================================================================

    # Preferred off-axis ridge location relative to the Bullet-side axis, in kpc.
    RIDGE_Q_TARGET: float = 55.0
    # Width of the preferred off-axis ridge band, in kpc.
    RIDGE_Q_SIGMA: float = 22.0
    # Preferred radial distance of the active ridge from the Bullet anchor, in kpc.
    RIDGE_R_TARGET: float = 55.0
    # Width of the preferred radial ridge band, in kpc.
    RIDGE_R_SIGMA: float = 28.0
    # Width of the axis-center penalty used in ridge diagnostics, in kpc.
    RIDGE_CENTER_SIGMA: float = 18.0
    # Softening factor controlling how harshly axis-centered ridge power is penalized.
    RIDGE_AXIS_SOFTEN: float = 0
    # Number of late-time steps included in the ridge persistence summary.
    RIDGE_PERSISTENCE_WINDOW: int = 40
    # Characteristic scale for ridge-peak wander suppression.
    RIDGE_WANDER_SIGMA: float = 18.0
    # Preferred Bullet-side gas-to-ridge offset in the ridge objective, in kpc.
    RIDGE_OFFSET_TARGET: float = 235.0
    # Width of the preferred gas-to-ridge offset band, in kpc.
    RIDGE_OFFSET_SIGMA: float = 45.0
    # Preferred scale for ridge-to-transport support separation, in kpc.
    RIDGE_SUPPORT_SEP_SIGMA: float = 40.0
    # Preferred Bullet-side transport-lobe detachment from the galaxy anchor, in kpc.
    TPOS_BULLET_DETACH_TARGET: float = 40.0
    # Width of the preferred Bullet-side transport-lobe detachment band, in kpc.
    TPOS_BULLET_DETACH_SIGMA: float = 18.0
    # Preferred ridge-to-gradient separation in the ridge support diagnostics, in kpc.
    RIDGE_GRAD_TARGET: float = 40.0
    # Width of the preferred ridge-to-gradient separation band, in kpc.
    RIDGE_GRAD_SIGMA: float = 22.0
    # Forward shift of the off-axis flank preference in the velocity family, in kpc.
    VELOCITY_FLANK_FORWARD_SHIFT: float = 55.0
    # Width of the off-axis flank preference in the velocity family, in kpc.
    VELOCITY_FLANK_SIGMA: float = 65.0
    # Weight of the merger-axis blend in the axis-biased velocity family.
    VELOCITY_AXIS_BLEND: float = 0.35
    # Weight of the off-axis flank bias in the flank and hybrid velocity families.
    VELOCITY_FLANK_BLEND: float = 0.75
    # Weight of the transported-curvature amplitude term in the lensing projection.
    LENS_TPOS_BLEND: float = 1.0
    # Curvature-gradient length scale used in the local transported-curvature proxy.
    KAPPA_CURV_SUPPORT: float = 1.0

    # Forward shift of the static ridge template relative to the Bullet galaxy, in kpc.
    SOURCE_RIDGE_FORWARD_SHIFT: float = 112.0
    # Longitudinal width of the ridge template along the transport corridor.
    SOURCE_RIDGE_LENGTH_SIGMA: float = 125.0
    # Transverse width of the ridge template across the transport corridor.
    SOURCE_RIDGE_WIDTH_SIGMA: float = 26.0
    # Off-axis activation scale for ridge enhancement away from the corridor center.
    SOURCE_RIDGE_OFFAXIS_SIGMA: float = 40.0
    # Strength of the transported-ridge enhancement in the final lensing build.
    KAPPA_RIDGE_BLEND: float = 25

    # ====================================================================
    # FINAL LENSING / DISPLAY MIX
    # ====================================================================

    # Multiplicative weight of the static polarization background in the final observable mix.
    POLBG_FINAL_BLEND: float = 1.5
    # Display-only gamma used to reveal low-amplitude saturation background in the exported kappa map.
    POLBG_DISPLAY_GAMMA: float = 0.6
    # Display-only weight of the static polarization background in the rendered kappa image.
    DISPLAY_POLBG_BLEND: float = 1.250
    # Display-only gamma compression applied to the static polarization background.
    DISPLAY_POLBG_GAMMA: float = 3.75
    # Display-only gamma compression applied to the full final map.
    DISPLAY_FINAL_GAMMA: float = 1.00

SCENARIO_DESCRIPTIONS = {
    "full_baseline": "Reference: CPTG transport run: diffusion, advection, decay, and the native curvature-derived velocity field are enabled; no divergence-free projection or velocity-family reshaping is applied.",
    "divfree_projection": "Reference: CPTG transport run with the same physics as full_baseline, but with the velocity field projected to be approximately divergence-free before evolution.",
    "axis_blend": "Reference: CPTG transport run with the transport direction partially blended toward the merger axis to encourage large-scale axis-following flow.",
    "axis_blend_divfree": "Axis-blended CPTG transport with the velocity field additionally projected to be approximately divergence-free before evolution.",
    "flank_bias": "Reference: CPTG transport run with extra off-axis flank preference in the velocity family to favor detached side-ridge development.",
    "flank_bias_divfree": "Flank-biased CPTG transport with the velocity field additionally projected to be approximately divergence-free before evolution.",
    "axis_flank_hybrid": "Reference: CPTG transport run using the hybrid velocity family that mixes axis-following flow and off-axis flank preference.",
    "axis_flank_hybrid_divfree": "Hybrid axis-plus-flank CPTG transport with an additional divergence-free velocity projection; this is the main proof-of-concept scenario family for the locked Bullet Cluster solution.",
}

SCENARIO_REQUIRED_KEYS = {
    "use_diffusion",
    "use_advection",
    "use_decay",
    "use_initial_condition",
    "use_projected_velocity",
    "velocity_mode",
}

VALID_VELOCITY_MODES = {
    "curv",
    "axis_blend",
    "flank_bias",
    "axis_flank_hybrid",
}

# Locked external Bullet Cluster observational benchmark used for final scenario
# selection and reporting.
#
# Observed values tied to Cha et al. 2025 when explicitly quoted:
#   - Bullet mass-galaxy offset: 17.78 +/- 0.66 kpc (quoted uncertainty)
#   - Bullet mass-ICM offset:    ~150 kpc
#   - Main-north mass-ICM:       ~200 kpc
#   - Main-south mass-ICM:       ~400 kpc
#   - Main-cluster subclump:     ~200 kpc
# The cluster-scale separation remains locked at 720 kpc from the classic lensing
# benchmark.
#
# Updated uncertainty treatment:
#   - observed_sigma_kpc stores a real quoted uncertainty when the literature
#     provides one directly.
#   - For mass-ICM offsets, when only component-position precisions are available,
#     we use derived/working 1-sigma values:
#       * bullet mass-ICM: derived +/-11.0 kpc from JWST subcluster peak precision
#         (~4 kpc) and Chandra gas centroiding (<10 kpc).
#       * main-north mass-ICM: working +/-22.4 kpc using the low end of the JWST/WL
#         main-cluster centroid range (20 kpc) combined with <10 kpc gas centroiding.
#       * main-south mass-ICM: working +/-51.0 kpc using the broad-end JWST/WL
#         centroid range (50 kpc) combined with <10 kpc gas centroiding.
#   - score_sigma_kpc matches these observed/derived margins where used.
#
# Fractions are used for active script coefficients where practical, but the
# north/south observed offsets remain plain observational targets.
TARGET_BULLET_GAS_LENS_OFFSET_KPC = 150.0
TARGET_BULLET_GAS_LENS_OBS_SIGMA_KPC = 11.0
TARGET_BULLET_GAS_LENS_SCORE_SIGMA_KPC = 11.0
TARGET_BULLET_GAL_LENS_OFFSET_KPC = 17.78
TARGET_BULLET_GAL_LENS_OBS_SIGMA_KPC = 0.66
TARGET_BULLET_GAL_LENS_SCORE_SIGMA_KPC = 0.66
TARGET_MAIN_NORTH_GAS_LENS_OFFSET_KPC = 200.0
TARGET_MAIN_NORTH_GAS_LENS_OBS_SIGMA_KPC = 22.4
TARGET_MAIN_NORTH_GAS_LENS_SCORE_SIGMA_KPC = 22.4
TARGET_MAIN_SOUTH_GAS_LENS_OFFSET_KPC = 400.0
TARGET_MAIN_SOUTH_GAS_LENS_OBS_SIGMA_KPC = 51.0
TARGET_MAIN_SOUTH_GAS_LENS_SCORE_SIGMA_KPC = 51.0
TARGET_LENS_PEAK_SEPARATION_KPC = 720.0
TARGET_LENS_PEAK_SEPARATION_OBS_SIGMA_KPC = None
TARGET_LENS_PEAK_SEPARATION_SCORE_SIGMA_KPC = 50.0
TARGET_MAIN_SUBCLUMP_SEPARATION_KPC = 200.0
TARGET_MAIN_SUBCLUMP_SEPARATION_OBS_SIGMA_KPC = None
TARGET_MAIN_SUBCLUMP_SEPARATION_SCORE_SIGMA_KPC = 30.0

# Purpose: Define the locked observed Bullet Cluster benchmark targets used for scoring.
def benchmark_specs():
    """Return independent observed-data benchmark definitions."""
    return {
        "bullet_mass_galaxy_offset": {
            "model_key": "lens_gal_bullet_sep",
            "target": TARGET_BULLET_GAL_LENS_OFFSET_KPC,
            "observed_sigma_kpc": TARGET_BULLET_GAL_LENS_OBS_SIGMA_KPC,
            "score_sigma_kpc": TARGET_BULLET_GAL_LENS_SCORE_SIGMA_KPC,
            "observed_label": f"{TARGET_BULLET_GAL_LENS_OFFSET_KPC:.2f} ± {TARGET_BULLET_GAL_LENS_OBS_SIGMA_KPC:.2f} kpc",
        },
        "bullet_mass_icm_offset": {
            "model_key": "offset_from_gas",
            "target": TARGET_BULLET_GAS_LENS_OFFSET_KPC,
            "observed_sigma_kpc": TARGET_BULLET_GAS_LENS_OBS_SIGMA_KPC,
            "score_sigma_kpc": TARGET_BULLET_GAS_LENS_SCORE_SIGMA_KPC,
            "observed_label": f"~{TARGET_BULLET_GAS_LENS_OFFSET_KPC:.0f} kpc (derived ±{TARGET_BULLET_GAS_LENS_OBS_SIGMA_KPC:.1f} kpc)",
        },
        "main_north_mass_icm_offset": {
            "model_key": "main_north_offset_from_gas",
            "target": TARGET_MAIN_NORTH_GAS_LENS_OFFSET_KPC,
            "observed_sigma_kpc": TARGET_MAIN_NORTH_GAS_LENS_OBS_SIGMA_KPC,
            "score_sigma_kpc": TARGET_MAIN_NORTH_GAS_LENS_SCORE_SIGMA_KPC,
            "observed_label": f"~{TARGET_MAIN_NORTH_GAS_LENS_OFFSET_KPC:.0f} kpc (working ±{TARGET_MAIN_NORTH_GAS_LENS_OBS_SIGMA_KPC:.1f} kpc)",
        },
        "main_south_mass_icm_offset": {
            "model_key": "main_south_offset_from_gas",
            "target": TARGET_MAIN_SOUTH_GAS_LENS_OFFSET_KPC,
            "observed_sigma_kpc": TARGET_MAIN_SOUTH_GAS_LENS_OBS_SIGMA_KPC,
            "score_sigma_kpc": TARGET_MAIN_SOUTH_GAS_LENS_SCORE_SIGMA_KPC,
            "observed_label": f"~{TARGET_MAIN_SOUTH_GAS_LENS_OFFSET_KPC:.0f} kpc (working ±{TARGET_MAIN_SOUTH_GAS_LENS_OBS_SIGMA_KPC:.1f} kpc)",
        },
        "cluster_scale_separation": {
            "model_key": "lens_peak_separation",
            "target": TARGET_LENS_PEAK_SEPARATION_KPC,
            "observed_sigma_kpc": TARGET_LENS_PEAK_SEPARATION_OBS_SIGMA_KPC,
            "score_sigma_kpc": TARGET_LENS_PEAK_SEPARATION_SCORE_SIGMA_KPC,
            "observed_label": f"~{TARGET_LENS_PEAK_SEPARATION_KPC:.0f} kpc (no quoted ±)",
        },
        "main_cluster_subclump_separation": {
            "model_key": "main_subclump_separation",
            "target": TARGET_MAIN_SUBCLUMP_SEPARATION_KPC,
            "observed_sigma_kpc": TARGET_MAIN_SUBCLUMP_SEPARATION_OBS_SIGMA_KPC,
            "score_sigma_kpc": TARGET_MAIN_SUBCLUMP_SEPARATION_SCORE_SIGMA_KPC,
            "observed_label": f"~{TARGET_MAIN_SUBCLUMP_SEPARATION_KPC:.0f} kpc (no quoted ±)",
        },
    }

# Purpose: Score one observed benchmark against a model summary row.
def calculate_single_benchmark(summary_row, benchmark_name):
    spec = benchmark_specs()[benchmark_name]
    model_val = float(summary_row[spec["model_key"]])
    residual = model_val - float(spec["target"])
    score_sigma = float(spec["score_sigma_kpc"])
    z = residual / max(score_sigma, 1e-12)
    observed_sigma = spec["observed_sigma_kpc"]
    within_observed_margin = None
    if observed_sigma is not None:
        within_observed_margin = abs(residual) <= float(observed_sigma)
    return {
        "benchmark": benchmark_name,
        "model_key": spec["model_key"],
        "observed": float(spec["target"]),
        "observed_sigma_kpc": observed_sigma,
        "score_sigma_kpc": score_sigma,
        "observed_label": spec["observed_label"],
        "model": model_val,
        "residual": residual,
        "z": z,
        "score": z ** 2,
        "within_observed_margin": within_observed_margin,
    }

# Purpose: Score all observed benchmarks for a model summary row.
def calculate_all_benchmarks(summary_row):
    return {name: calculate_single_benchmark(summary_row, name) for name in benchmark_specs().keys()}

# Purpose: Build ordered observed-vs-model comparison rows for reporting and plotting.
def observed_comparison_rows(summary_row):
    order = [
        "bullet_mass_galaxy_offset",
        "bullet_mass_icm_offset",
        "main_north_mass_icm_offset",
        "main_south_mass_icm_offset",
        "cluster_scale_separation",
        "main_cluster_subclump_separation",
    ]
    rows = []
    for name in order:
        item = calculate_single_benchmark(summary_row, name)
        rows.append({
            "benchmark": item["benchmark"],
            "observed_label": item["observed_label"],
            "observed": item["observed"],
            "observed_sigma_kpc": item["observed_sigma_kpc"],
            "score_sigma_kpc": item["score_sigma_kpc"],
            "model": item["model"],
            "residual": item["residual"],
            "z": item["z"],
            "score": item["score"],
            "within_observed_margin": item["within_observed_margin"],
        })
    return rows

# Purpose: Return the total Bullet Cluster score as the sum of independent benchmark terms.
def bullet_cluster_score(summary_row):
    """Return the total observed-data score as the sum of independent benchmark scores."""
    return float(sum(item["score"] for item in calculate_all_benchmarks(summary_row).values()))

SCENARIOS = {
    "full_baseline": {
        "use_diffusion": True,
        "use_advection": True,
        "use_decay": True,
        "use_initial_condition": True,
        "use_projected_velocity": False,
        "velocity_mode": "curv",
    },
    "divfree_projection": {
        "use_diffusion": True,
        "use_advection": True,
        "use_decay": True,
        "use_initial_condition": True,
        "use_projected_velocity": True,
        "velocity_mode": "curv",
    },
    "axis_blend": {
        "use_diffusion": True,
        "use_advection": True,
        "use_decay": True,
        "use_initial_condition": True,
        "use_projected_velocity": False,
        "velocity_mode": "axis_blend",
    },
    "axis_blend_divfree": {
        "use_diffusion": True,
        "use_advection": True,
        "use_decay": True,
        "use_initial_condition": True,
        "use_projected_velocity": True,
        "velocity_mode": "axis_blend",
    },
    "flank_bias": {
        "use_diffusion": True,
        "use_advection": True,
        "use_decay": True,
        "use_initial_condition": True,
        "use_projected_velocity": False,
        "velocity_mode": "flank_bias",
    },
    "flank_bias_divfree": {
        "use_diffusion": True,
        "use_advection": True,
        "use_decay": True,
        "use_initial_condition": True,
        "use_projected_velocity": True,
        "velocity_mode": "flank_bias",
    },
    "axis_flank_hybrid": {
        "use_diffusion": True,
        "use_advection": True,
        "use_decay": True,
        "use_initial_condition": True,
        "use_projected_velocity": False,
        "velocity_mode": "axis_flank_hybrid",
    },
    "axis_flank_hybrid_divfree": {
        "use_diffusion": True,
        "use_advection": True,
        "use_decay": True,
        "use_initial_condition": True,
        "use_projected_velocity": True,
        "velocity_mode": "axis_flank_hybrid",
    },
}

print()
print('*' * 60)
print("* CPTG BULLET-CLUSTER CURVATURE TRANSPORT MODEL            *")
print('*' * 60)
print("* Research Paper Reference:                                *")
print("*                                                          *")
print("* Curvature Polarization Transport Gravity                 *")
print("*                                                          *")
print("* A Variational Framework for Galaxies and Cluster Mergers *")
print("*                                                          *")
print("* - Author: Carter L Glass Jr                              *")
print("* - E-mail: carterglass@bellsouth.net                      *")
print("* - Orchid: https://orcid.org/0009-0005-7538-543X          *")
print('*' * 60)
print()
print("Please wait. This may take a several minutes or more to complete...\n")
print()

# Purpose: Format a human-readable summary of the active scenario switch settings.
def scenario_switch_summary(switches):
    """Return a compact human-readable summary of a scenario switch set."""
    mode = switches.get("velocity_mode", "curv")
    parts = [
        f"velocity_mode={mode}",
        f"diffusion={'on' if switches.get('use_diffusion', False) else 'off'}",
        f"advection={'on' if switches.get('use_advection', False) else 'off'}",
        f"decay={'on' if switches.get('use_decay', False) else 'off'}",
        f"initial_condition={'on' if switches.get('use_initial_condition', False) else 'off'}",
        f"projected_velocity={'on' if switches.get('use_projected_velocity', False) else 'off'}",
    ]
    return ", ".join(parts)

# Purpose: Enforce that scenario names, descriptions, and switch keys stay synchronized.
def validate_scenario_definitions():
    """Validate that scenario labels and implementation switches stay synchronized.

    This turns SCENARIO_DESCRIPTIONS from passive text into an enforceable
    documentation layer. If a scenario is renamed, added, or given invalid
    switches, the script fails early instead of silently running mislabeled
    experiments.
    """
    scenario_names = set(SCENARIOS.keys())
    description_names = set(SCENARIO_DESCRIPTIONS.keys())

    missing_descriptions = sorted(scenario_names - description_names)
    dangling_descriptions = sorted(description_names - scenario_names)
    if missing_descriptions or dangling_descriptions:
        raise ValueError(
            "Scenario name mismatch between SCENARIOS and SCENARIO_DESCRIPTIONS: "
            f"missing_descriptions={missing_descriptions}, "
            f"dangling_descriptions={dangling_descriptions}"
        )

    for name, switches in SCENARIOS.items():
        missing_keys = sorted(SCENARIO_REQUIRED_KEYS - set(switches.keys()))
        extra_keys = sorted(set(switches.keys()) - SCENARIO_REQUIRED_KEYS)
        if missing_keys or extra_keys:
            raise ValueError(
                f"Scenario '{name}' has invalid switch keys: "
                f"missing={missing_keys}, extra={extra_keys}"
            )

        mode = switches["velocity_mode"]
        if mode not in VALID_VELOCITY_MODES:
            raise ValueError(
                f"Scenario '{name}' uses unsupported velocity_mode '{mode}'. "
                f"Valid modes: {sorted(VALID_VELOCITY_MODES)}"
            )

        for bool_key in sorted(SCENARIO_REQUIRED_KEYS - {"velocity_mode"}):
            if not isinstance(switches[bool_key], bool):
                raise TypeError(
                    f"Scenario '{name}' key '{bool_key}' must be bool, "
                    f"got {type(switches[bool_key]).__name__}"
                )

        desc = SCENARIO_DESCRIPTIONS[name].lower()
        mentions_divfree = ("divergence-free" in desc) or ("divergence free" in desc) or ("divfree" in desc)
        negates_divfree = ("no divergence-free" in desc) or ("no divergence free" in desc) or ("without divergence-free" in desc) or ("without divergence free" in desc)
        if switches["use_projected_velocity"]:
            if (not mentions_divfree) or negates_divfree:
                raise ValueError(
                    f"Scenario '{name}' projection switch/description mismatch: "
                    f"description='{SCENARIO_DESCRIPTIONS[name]}', "
                    f"use_projected_velocity={switches['use_projected_velocity']}"
                )
        else:
            contradictory_positive = mentions_divfree and (not negates_divfree)
            if contradictory_positive:
                raise ValueError(
                    f"Scenario '{name}' projection switch/description mismatch: "
                    f"description='{SCENARIO_DESCRIPTIONS[name]}', "
                    f"use_projected_velocity={switches['use_projected_velocity']}"
                )

        if "axis" in name and mode not in {"axis_blend", "axis_flank_hybrid"}:
            raise ValueError(
                f"Scenario '{name}' name suggests axis handling, but velocity_mode='{mode}'"
            )
        if "flank" in name and mode not in {"flank_bias", "axis_flank_hybrid"}:
            raise ValueError(
                f"Scenario '{name}' name suggests flank handling, but velocity_mode='{mode}'"
            )
        if "hybrid" in name and mode != "axis_flank_hybrid":
            raise ValueError(
                f"Scenario '{name}' name suggests hybrid handling, but velocity_mode='{mode}'"
            )
        if name == "full_baseline" and mode != "curv":
            raise ValueError("Scenario 'full_baseline' must use velocity_mode='curv'.")
        if name == "divfree_projection" and (mode != "curv" or not switches["use_projected_velocity"]):
            raise ValueError(
                "Scenario 'divfree_projection' must use velocity_mode='curv' and projected velocity."
            )

WORKING_BASELINE_SCENARIO = "axis_flank_hybrid_divfree"

# Purpose: Print the observed-vs-model audit block for a scenario summary.
def print_benchmark_audit(summary_row, label="scenario"):
    print(f"Observed Bullet Cluster audit for {label}:")
    for row in observed_comparison_rows(summary_row):
        if row["observed_sigma_kpc"] is None:
            margin_text = "observed_margin=not quoted"
        else:
            status = "inside" if row["within_observed_margin"] else "outside"
            margin_text = f"observed_margin=±{row['observed_sigma_kpc']:.2f} kpc ({status})"
        print(
            "  "
            f"{row['benchmark']}: observed={row['observed_label']} | "
            f"model={row['model']:.2f} kpc | residual={row['residual']:+.2f} kpc | "
            f"{margin_text} | score_sigma={row['score_sigma_kpc']:.2f} kpc | z={row['z']:+.2f}"
        )
    print(f"  total_observed_score={bullet_cluster_score(summary_row):.3f}")

class BulletClusterDiagnosticModel:
    # Purpose: Initialize the grid, configuration, and all cached static merger fields.
    def __init__(self, cfg: Config):
        validate_scenario_definitions()
        self.c = cfg
        self.x = np.linspace(cfg.XMIN, cfg.XMAX, cfg.NX)
        self.y = np.linspace(cfg.YMIN, cfg.YMAX, cfg.NY)
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        self.X, self.Y = np.meshgrid(self.x, self.y)

        if cfg.CFL_NORMALIZE_NUM_DT:
            dx_scale = self.dx / max(cfg.NUM_DT_REF_DX_KPC, 1e-12)
            self.num_dt_eff = cfg.NUM_DT * dx_scale
        else:
            self.num_dt_eff = cfg.NUM_DT

        self._build_geometry()
        self._build_baseline_fields()

    # Purpose: Build the reduced Gaussian galaxy/gas geometry and its anchor centroids.
    def _build_geometry(self):
        c = self.c
        X, Y = self.X, self.Y

        self.rho_gal_bullet = gaussian2d(
            X, Y, c.BULLET_GAL_X, c.BULLET_GAL_Y,
            c.BULLET_GAL_SX, c.BULLET_GAL_SY, c.BULLET_GAL_AMP
        )
        self.rho_gal_main = gaussian2d(
            X, Y, c.MAIN_GAL_X, c.MAIN_GAL_Y,
            c.MAIN_GAL_SX, c.MAIN_GAL_SY, c.MAIN_GAL_AMP
        )
        self.rho_gas_bullet = gaussian2d(
            X, Y, c.BULLET_GAS_X, c.BULLET_GAS_Y,
            c.BULLET_GAS_SX, c.BULLET_GAS_SY, c.BULLET_GAS_AMP
        )
        self.rho_gas_main = gaussian2d(
            X, Y, c.MAIN_GAS_X, c.MAIN_GAS_Y,
            c.MAIN_GAS_SX, c.MAIN_GAS_SY, c.MAIN_GAS_AMP
        )
        self.rho_gas = self.rho_gas_bullet + self.rho_gas_main

        self.rho_gal = self.rho_gal_bullet + self.rho_gal_main
        self.rho_b = self.rho_gal + self.rho_gas

        self.gal_centroid_bullet = centroid_from_positive(
            self.rho_gal_bullet, self.X, self.Y, c.EPS
        )
        self.gal_centroid_main = centroid_from_positive(
            self.rho_gal_main, self.X, self.Y, c.EPS
        )
        self.gas_centroid_bullet = centroid_from_positive(
            self.rho_gas_bullet, self.X, self.Y, c.EPS
        )
        self.gas_centroid_main = centroid_from_positive(
            self.rho_gas_main, self.X, self.Y, c.EPS
        )

    # Purpose: Derive the background chi curvature proxy from the galaxy potential proxy.
    def _compute_chi_from_phi(self, phi_field):
        c = self.c

        grad_phi_x, grad_phi_y = grad2(phi_field, self.dx, self.dy)
        gmag = np.sqrt(grad_phi_x ** 2 + grad_phi_y ** 2) + c.EPS

        r = np.sqrt(self.X ** 2 + self.Y ** 2)
        r = np.maximum(r, 1e-6)

        log_g = np.log(gmag)
        log_r = np.log(r)

        dlogg_dx, dlogg_dy = grad2(log_g, self.dx, self.dy)
        dlogr_dx, dlogr_dy = grad2(log_r, self.dx, self.dy)

        grad_logr_mag = np.sqrt(dlogr_dx ** 2 + dlogr_dy ** 2) + c.EPS
        grad_logg_parallel = (
            dlogg_dx * dlogr_dx + dlogg_dy * dlogr_dy
        ) / grad_logr_mag

        chi = (c.KAPPA_C / r) ** 2 * (grad_logg_parallel ** 2)
        chi = np.maximum(chi, c.EPS)
        chi = smooth5(chi, c.CHI_SMOOTHING_PASSES)

        return chi, gmag

    # Purpose: Build the reduced static polarization background as a broad saturation field from the full baryonic potential proxy.
    def build_background_polarization_map(self):
        c = self.c

        grad_phi_x, grad_phi_y = grad2(self.phi_baryonic, self.dx, self.dy)
        gmag_bg = np.sqrt(grad_phi_x ** 2 + grad_phi_y ** 2) + c.EPS

        dg_dx, dg_dy = grad2(gmag_bg, self.dx, self.dy)
        grad_g_sq = dg_dx ** 2 + dg_dy ** 2
        C_bg = gmag_bg ** 2 + (c.KAPPA_C ** 2) * grad_g_sq
        C_bg = np.maximum(C_bg, c.POL_FLOOR)

        dCdx, dCdy = grad2(C_bg, self.dx, self.dy)
        dCdx = smooth5(dCdx, c.C_GRAD_SMOOTHING_PASSES)
        dCdy = smooth5(dCdy, c.C_GRAD_SMOOTHING_PASSES)

        P_bg_x = c.BETA_POL * np.power(C_bg, -2.0 / 3.0) * dCdx
        P_bg_y = c.BETA_POL * np.power(C_bg, -2.0 / 3.0) * dCdy
        P_bg_mag = np.sqrt(P_bg_x ** 2 + P_bg_y ** 2)

        divP_bg = div2(P_bg_x, P_bg_y, self.dx, self.dy)

        # Build the observable background as a broad connected
        # polarization-support saturation field rather than a filament finder
        # or a sum of disconnected component blobs.
        #
        # Purpose: Convert one static potential into a smooth local
        # polarization-support saturation scalar.
        def _component_pol_scalar(phi_component):
            grad_x, grad_y = grad2(phi_component, self.dx, self.dy)
            gmag_i = np.sqrt(grad_x ** 2 + grad_y ** 2) + c.EPS
            dg_i_x, dg_i_y = grad2(gmag_i, self.dx, self.dy)
            C_i = gmag_i ** 2 + (c.KAPPA_C ** 2) * (dg_i_x ** 2 + dg_i_y ** 2)
            C_i = np.maximum(C_i, c.POL_FLOOR)

            dC_i_x, dC_i_y = grad2(C_i, self.dx, self.dy)
            dC_i_x = smooth5(dC_i_x, c.C_GRAD_SMOOTHING_PASSES)
            dC_i_y = smooth5(dC_i_y, c.C_GRAD_SMOOTHING_PASSES)

            P_i_x = c.BETA_POL * np.power(C_i, -2.0 / 3.0) * dC_i_x
            P_i_y = c.BETA_POL * np.power(C_i, -2.0 / 3.0) * dC_i_y
            P_i_mag = np.sqrt(P_i_x ** 2 + P_i_y ** 2)

            phi_i_norm = np.maximum(
                np.nan_to_num(phi_component, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            )
            phi_i_norm = phi_i_norm / (np.max(phi_i_norm) + c.EPS)
            P_i_norm = P_i_mag / (np.max(P_i_mag) + c.EPS)
            C_i_norm = C_i / (np.max(C_i) + c.EPS)

            # Use a softer potential exponent than the previous candidate so
            # the static background reads as a visible saturation field rather
            # than as a pair of compact isolated component humps.
            return np.power(phi_i_norm, 7.0 / 5.0) * (
                0.75
                + 0.25 * np.power(C_i_norm, 1.0 / 3.0)
            )

        bullet_cluster_support = _component_pol_scalar(self.phi_gal_bullet + self.phi_gas_bullet)
        main_cluster_support = _component_pol_scalar(self.phi_gal_main + self.phi_gas_main)
        full_cluster_support = _component_pol_scalar(self.phi_baryonic)

        component_sum = (
            _component_pol_scalar(self.phi_gal_bullet)
            + _component_pol_scalar(self.phi_gal_main)
            + _component_pol_scalar(self.phi_gas_bullet)
            + _component_pol_scalar(self.phi_gas_main)
        )

        # Blend broad cluster-scale envelopes with a lighter component sum so
        # the right side remains connected while retaining some substructure.
        sigma_pol_raw = (
            0.20 * component_sum
            + 0.45 * bullet_cluster_support
            + 0.75 * main_cluster_support
            + 0.20 * full_cluster_support
        )

        # Add a weak baryonic bridge gate so the main gas and main lens region
        # share one background support field instead of detaching into islands.
        phi_bridge = np.maximum(
            np.nan_to_num(self.phi_gal_main + self.phi_gas_main, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        )
        phi_bridge = phi_bridge / (np.max(phi_bridge) + c.EPS)
        sigma_pol_raw = sigma_pol_raw * (0.65 + 0.35 * np.power(phi_bridge, 0.8))

        sigma_pol_raw = gaussian_filter(
            sigma_pol_raw,
            sigma=max(10.5, 10.5 * c.LENSING_SMOOTH_SIGMA_PIX),
        )
        sigma_pol_raw = smooth5(sigma_pol_raw, c.C_GRAD_SMOOTHING_PASSES + 3)
        sigma_pol_only = normalize_nonnegative(
            np.maximum(
                np.nan_to_num(sigma_pol_raw, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            ),
            c.EPS,
        )

        return {
            "grad_phi_x": grad_phi_x,
            "grad_phi_y": grad_phi_y,
            "gmag_bg": gmag_bg,
            "C_bg": C_bg,
            "P_bg_x": P_bg_x,
            "P_bg_y": P_bg_y,
            "P_bg_mag": P_bg_mag,
            "divP_bg": divP_bg,
            "sigma_pol_only": sigma_pol_only,
        }

    # Purpose: Construct the native curvature-driven Bullet-side transport velocity field.
    def build_transport_velocity(self):
        c = self.c

        X, Y = self.X, self.Y
        gas_x, gas_y = self.gas_centroid_bullet
        bul_x, bul_y = self.gal_centroid_bullet

        dx_g = X - gas_x
        dy_g = Y - gas_y
        r_g = np.sqrt(dx_g**2 + dy_g**2) + 1e-12

        dx_b = X - bul_x
        dy_b = Y - bul_y
        r_b = np.sqrt(dx_b**2 + dy_b**2) + 1e-12

        ux_g = dx_g / r_g
        uy_g = dy_g / r_g

        ux_b = dx_b / r_b
        uy_b = dy_b / r_b

        ux = ux_g - ux_b
        uy = uy_g - uy_b

        mag = np.sqrt(ux**2 + uy**2) + 1e-12
        ux /= mag
        uy /= mag

        env_g = np.exp(-(r_g / 300.0)**2)
        env_b = np.exp(-(r_b / 120.0)**2)
        envelope = env_b * env_g

        chi = self.chi_background
        dchidx, dchidy = np.gradient(chi, self.dx, self.dy)
        grad_chi = np.sqrt(dchidx**2 + dchidy**2)

        chi_velocity_damp = np.exp(-grad_chi / c.GRAD_CHI_SCALE)

        u_x = c.V_TRANSPORT * envelope * chi_velocity_damp * ux
        u_y = c.V_TRANSPORT * envelope * chi_velocity_damp * uy

        u_mag = np.sqrt(u_x**2 + u_y**2)
        scale = np.minimum(1.0, c.U_MAX / (u_mag + 1e-12))

        u_x *= scale
        u_y *= scale

        return u_x, u_y

    # Purpose: Reshape the native velocity field into the scenario-specific axis/flank families.
    def build_velocity_family(self, base_u_x, base_u_y, mode):
        c = self.c
        if mode == "curv":
            return base_u_x, base_u_y

        speed = np.sqrt(base_u_x**2 + base_u_y**2) + c.EPS
        dir_x = base_u_x / speed
        dir_y = base_u_y / speed

        ax = self.axis_ux
        ay = self.axis_uy
        px = -ay
        py = ax

        dx = self.X - self.gal_centroid_bullet[0]
        dy = self.Y - self.gal_centroid_bullet[1]
        p = dx * ax + dy * ay
        q = -dx * ay + dy * ax

        flank_env = np.exp(-0.5 * (((p - c.VELOCITY_FLANK_FORWARD_SHIFT) / (1.35 * c.VELOCITY_FLANK_SIGMA)) ** 2 + (q / c.VELOCITY_FLANK_SIGMA) ** 2))
        flank_sign = np.sign(q)
        flank_x = ax + flank_sign * px
        flank_y = ay + flank_sign * py
        flank_mag = np.sqrt(flank_x**2 + flank_y**2) + c.EPS
        flank_x /= flank_mag
        flank_y /= flank_mag

        axis_x = np.full_like(base_u_x, ax)
        axis_y = np.full_like(base_u_y, ay)
        if mode == "axis_blend":
            blend = c.VELOCITY_AXIS_BLEND
            out_x = (1.0 - blend) * dir_x + blend * axis_x
            out_y = (1.0 - blend) * dir_y + blend * axis_y
        elif mode == "flank_bias":
            flank_w = np.clip(c.VELOCITY_FLANK_BLEND * flank_env, 0.0, 0.92)
            out_x = (1.0 - flank_w) * dir_x + flank_w * flank_x
            out_y = (1.0 - flank_w) * dir_y + flank_w * flank_y
        elif mode == "axis_flank_hybrid":
            flank_w = np.clip(c.VELOCITY_FLANK_BLEND * flank_env, 0.0, 0.85)
            axis_w = c.VELOCITY_AXIS_BLEND
            out_x = ((1.0 - axis_w) * dir_x + axis_w * axis_x)
            out_y = ((1.0 - axis_w) * dir_y + axis_w * axis_y)
            out_x = (1.0 - flank_w) * out_x + flank_w * flank_x
            out_y = (1.0 - flank_w) * out_y + flank_w * flank_y
        else:
            out_x, out_y = dir_x, dir_y

        out_mag = np.sqrt(out_x**2 + out_y**2) + c.EPS
        out_x = speed * (out_x / out_mag)
        out_y = speed * (out_y / out_mag)
        return out_x, out_y

    # Purpose: Fetch a cached velocity family with or without divergence-free projection.
    def get_velocity_bundle(self, use_projected_velocity, velocity_mode="curv"):
        mode = velocity_mode or "curv"
        key = f"{mode}_proj" if use_projected_velocity else mode
        bundle = self.velocity_bundles[key]
        return bundle["u_x"], bundle["u_y"]

    # Purpose: Seed the initial transported-curvature packet anchored to the Bullet-side geometry.
    def build_bullet_anchored_T0(self):
        c = self.c

        dx = self.X - self.gal_centroid_bullet[0]
        dy = self.Y - self.gal_centroid_bullet[1]

        ax = -self.axis_ux
        ay = -self.axis_uy

        u_par = dx * ax + dy * ay
        u_perp = -dx * ay + dy * ax

        lead = np.exp(
            -0.5 * (
                ((u_par - c.T0_LEAD_SHIFT_KPC) / c.T0_LEAD_SIGMA_PAR) ** 2
                + (u_perp / c.T0_LEAD_SIGMA_PERP) ** 2
            )
        )

        wake = np.exp(
            -0.5 * (
                ((u_par + c.T0_WAKE_SHIFT_KPC) / c.T0_WAKE_SIGMA_PAR) ** 2
                + (u_perp / c.T0_WAKE_SIGMA_PERP) ** 2
            )
        )

        chi_norm = self.chi_background / (np.max(self.chi_background) + c.EPS)

        forward_gate = np.clip(u_par / (np.abs(u_par) + c.EPS), 0.0, 1.0)

        dxg = self.X - self.gas_centroid_bullet[0]
        dyg = self.Y - self.gas_centroid_bullet[1]
        rg = np.sqrt(dxg**2 + dyg**2) + c.EPS
        gas_suppress = 1.0 - np.exp(-0.5 * (rg / c.GAS_REPEL_SIGMA) ** 2)

        T0_raw = c.T0_STRENGTH * chi_norm * (
            c.T0_SCALE * lead * forward_gate * gas_suppress
            - c.T0_WAKE_WEIGHT * wake
        )

        T_pos = np.maximum(T0_raw, 0.0)
        pos_scale = np.max(T_pos) + c.EPS

        T0 = T0_raw / pos_scale

        return T0

    # Purpose: Build all static background fields, cached velocity bundles, and source templates used by every scenario.
    def _build_baseline_fields(self):
        c = self.c

        self.phi_gal_bullet = gaussian2d(
            self.X, self.Y,
            c.BULLET_GAL_X, c.BULLET_GAL_Y,
            (27.0 / 20.0) * c.BULLET_GAL_SX, (27.0 / 20.0) * c.BULLET_GAL_SY,
            (23.0 / 20.0) * c.BULLET_GAL_AMP,
        )
        self.phi_gal_main = gaussian2d(
            self.X, self.Y,
            c.MAIN_GAL_X, c.MAIN_GAL_Y,
            (27.0 / 20.0) * c.MAIN_GAL_SX, (27.0 / 20.0) * c.MAIN_GAL_SY,
            (23.0 / 20.0) * c.MAIN_GAL_AMP,
        )
        self.phi_gas_bullet = gaussian2d(
            self.X, self.Y,
            c.BULLET_GAS_X, c.BULLET_GAS_Y,
            (23.0 / 20.0) * c.BULLET_GAS_SX, (23.0 / 20.0) * c.BULLET_GAS_SY,
            (3.0 / 4.0) * c.BULLET_GAS_AMP,
        )
        self.phi_gas_main = gaussian2d(
            self.X, self.Y,
            c.MAIN_GAS_X, c.MAIN_GAS_Y,
            (23.0 / 20.0) * c.MAIN_GAS_SX, (23.0 / 20.0) * c.MAIN_GAS_SY,
            (3.0 / 4.0) * c.MAIN_GAS_AMP,
        )

        self.phi_gal = self.phi_gal_bullet + self.phi_gal_main
        self.phi_gas = self.phi_gas_bullet + self.phi_gas_main
        self.phi_baryonic = self.phi_gal + self.phi_gas

        background_pol = self.build_background_polarization_map()
        self.gmag_background = background_pol["gmag_bg"]
        self.C_background = background_pol["C_bg"]
        self.P_bg_x = background_pol["P_bg_x"]
        self.P_bg_y = background_pol["P_bg_y"]
        self.divP_background = background_pol["divP_bg"]
        self.sigma_pol_background = background_pol["sigma_pol_only"]

        self.sigma_b = normalize_nonnegative(self.rho_b, c.EPS)

        self.chi_background, _ = self._compute_chi_from_phi(self.phi_gal)

        dx_dir = self.gal_centroid_bullet[0] - self.gas_centroid_bullet[0]
        dy_dir = self.gal_centroid_bullet[1] - self.gas_centroid_bullet[1]
        norm = np.hypot(dx_dir, dy_dir) + c.EPS
        self.axis_ux = dx_dir / norm
        self.axis_uy = dy_dir / norm

        self.u_x_curv, self.u_y_curv = self.build_transport_velocity()

        family_modes = ["curv", "axis_blend", "flank_bias", "axis_flank_hybrid"]
        self.velocity_bundles = {}

        for mode in family_modes:
            if mode == "curv":
                uxf, uyf = self.u_x_curv, self.u_y_curv
            else:
                uxf, uyf = self.build_velocity_family(self.u_x_curv, self.u_y_curv, mode)

            self.velocity_bundles[mode] = {"u_x": uxf, "u_y": uyf}

            uxp, uyp = project_velocity_fft(uxf, uyf, self.dx, self.dy)
            self.velocity_bundles[f"{mode}_proj"] = {"u_x": uxp, "u_y": uyp}

        self.T0_background = self.build_bullet_anchored_T0()

        self.curv_norm_cached = self.chi_background / (np.max(self.chi_background) + c.EPS)
        self.curv_grad_x_cached, self.curv_grad_y_cached = grad2(self.curv_norm_cached, self.dx, self.dy)

        # Positive-source template: tighter core + flank-biased arc-pair
        ax_src = self.axis_ux
        ay_src = self.axis_uy

        dx = self.X - self.gal_centroid_bullet[0]
        dy = self.Y - self.gal_centroid_bullet[1]
        rsrc = np.hypot(dx, dy) + c.EPS

        source_core = np.exp(-0.5 * ((dx / 28.0) ** 2 + (dy / 28.0) ** 2))

        dx_tail = self.X - self.gal_centroid_bullet[0]
        dy_tail = self.Y - self.gal_centroid_bullet[1]
        p_tail = dx_tail * ax_src + dy_tail * ay_src
        q_tail = -dx_tail * ay_src + dy_tail * ax_src

        tail = np.exp(-0.5 * ((p_tail / 46.0) ** 2 + (q_tail / 30.0) ** 2))

        # The legacy source-ring scaffold is no longer part of the active source
        # template or lensing projection path, so the dead ring computation is omitted.

        dx_ridge = self.X - (self.gal_centroid_bullet[0] + c.SOURCE_RIDGE_FORWARD_SHIFT * ax_src)
        dy_ridge = self.Y - (self.gal_centroid_bullet[1] + c.SOURCE_RIDGE_FORWARD_SHIFT * ay_src)
        p_ridge = dx_ridge * ax_src + dy_ridge * ay_src
        q_ridge = -dx_ridge * ay_src + dy_ridge * ax_src
        ridge_axis = np.exp(-0.5 * (p_ridge / c.SOURCE_RIDGE_LENGTH_SIGMA) ** 2)
        ridge_width = np.exp(-0.5 * (q_ridge / c.SOURCE_RIDGE_WIDTH_SIGMA) ** 2)
        ridge_offaxis = 1.0 - np.exp(-0.5 * (q_ridge / c.SOURCE_RIDGE_OFFAXIS_SIGMA) ** 2)
        ridge = ridge_axis * ridge_width * ((7.0 / 20.0) + (13.0 / 20.0) * ridge_offaxis)

        curv_gate = np.clip(self.curv_norm_cached, 0.0, 1.0) ** (4.0 / 5.0)
        forward_gate = np.clip((dx * ax_src + dy * ay_src) / rsrc, 0.0, 1.0)
        self.source_ridge = ridge
        self.source_template = (
            source_core
            + 7.0 / 100.0 * tail * curv_gate * forward_gate * source_core.max()
        )
        source_scale = float(np.max(self.source_template) + c.EPS)
        # Cache invariant normalizations used by the lensing build so the
        # scenario loop does not repeatedly renormalize geometry-only fields.
        self.source_norm_cached = self.source_template / source_scale
        source_ridge_scale = float(np.max(self.source_ridge) + c.EPS)
        self.source_ridge_norm_cached = self.source_ridge / source_ridge_scale

    # Purpose: Wrap the shared first-order upwind advection operator for the model grid.
    def advective_term(self, T, u_x, u_y):
        return upwind_first_order(T, u_x, u_y, self.dx, self.dy)

    # Purpose: Return the numerical diffusion term for a transported field.
    def diffuse_term(self, T):
        return self.c.D_TRANS_NUM * lap2(T, self.dx, self.dy)

    # Purpose: Build the aligned curvature-weighted transport flux divergence used in the positive channel.
    def curvature_flux_term(self, T, u_x, u_y, curv_norm, chi_velocity_damp):
        c = self.c

        T_safe = np.clip(
            np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            c.CLIP_T_MAX,
        )

        speed = np.sqrt(u_x**2 + u_y**2) + c.EPS
        dir_x = u_x / speed
        dir_y = u_y / speed

        dchidx = self.curv_grad_x_cached
        dchidy = self.curv_grad_y_cached
        grad_mag = np.sqrt(dchidx**2 + dchidy**2) + c.EPS
        chi_hat_x = dchidx / grad_mag
        chi_hat_y = dchidy / grad_mag

        align = np.maximum(dir_x * chi_hat_x + dir_y * chi_hat_y, 0.0)
        align = np.power(align, c.FLUX_ALIGNMENT_POWER)

        curvature_gate = np.power(
            c.CURVATURE_TRANSPORT_FLOOR
            + (1.0 - c.CURVATURE_TRANSPORT_FLOOR) * curv_norm,
            c.FLUX_CURVATURE_POWER,
        )

        flux_coeff = (
            c.FLUX_TRANSPORT_SCALE
            * speed
            * curvature_gate
            * chi_velocity_damp
            * align
        )

        flux_coeff = smooth5(flux_coeff, c.FLUX_SMOOTHING_PASSES)

        dTdx, dTdy = grad2(T_safe, self.dx, self.dy)
        Fx = flux_coeff * dTdx
        Fy = flux_coeff * dTdy

        divF = np.zeros_like(T_safe)
        divF[1:-1, 1:-1] = (
            (Fx[1:-1, 1:-1] - Fx[:-2, 1:-1]) / self.dx
            + (Fy[1:-1, 1:-1] - Fy[1:-1, :-2]) / self.dy
        )
        divF = np.nan_to_num(divF, nan=0.0, posinf=0.0, neginf=0.0)

        return divF, flux_coeff, Fx, Fy

    # Purpose: Compute the local transported-curvature proxy C_T and its gradient diagnostics.
    def compute_transport_curvature(self, T_pos):
        c = self.c
        T_safe = np.maximum(np.nan_to_num(T_pos, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        dTdx, dTdy = grad2(T_safe, self.dx, self.dy)
        grad_sq = dTdx**2 + dTdy**2
        C_local = T_safe**2 + (c.KAPPA_CURV_SUPPORT**2) * grad_sq
        C_local = np.maximum(C_local, c.EPS)
        grad_mag = np.sqrt(grad_sq)
        return C_local, dTdx, dTdy, grad_mag

    # Purpose: Measure flow-aligned coherence of transported curvature along the effective transport direction.
    def transport_coherence_fields(self, T_pos, flow_x, flow_y):
        """Return flow-aligned coherence diagnostics for transported curvature.

        The key quantity is the directional derivative of transported
        curvature along the effective transport flow. This emphasizes coherent
        curvature transport along streamlines rather than static concentration
        at a single high-curvature point.
        """
        c = self.c
        C_local, _, _, _ = self.compute_transport_curvature(T_pos)
        Cx, Cy = grad2(C_local, self.dx, self.dy)
        C_grad_mag = np.sqrt(Cx**2 + Cy**2) + c.EPS

        flow_mag = np.sqrt(flow_x**2 + flow_y**2) + c.EPS
        flow_dir_x = flow_x / flow_mag
        flow_dir_y = flow_y / flow_mag

        flow_alignment = (Cx * flow_dir_x + Cy * flow_dir_y) / C_grad_mag
        coherence = np.maximum(flow_alignment, 0.0) ** 2

        return C_local, C_grad_mag, coherence, flow_dir_x, flow_dir_y

    # Purpose: Apply the CPTG support operator that reinforces coherent transported-curvature structure.
    def curvature_support_operator(self, T_pos, flow_x, flow_y):
        c = self.c
        C_local, C_grad_mag, coherence, _, _ = self.transport_coherence_fields(T_pos, flow_x, flow_y)
        C_norm = C_local / (np.max(C_local) + c.EPS)
        grad_strength = C_grad_mag / (np.max(C_grad_mag) + c.EPS)
        support_strength = np.power(C_norm, 1.0 / 3.0)
        coherence_boost = np.power(1.0 + coherence, 1.0 / 3.0)

        support = (
            support_strength
            * coherence_boost
            * np.power(grad_strength, 7.0 / 20.0)
            * np.maximum(T_pos, 0.0)
        )

        return np.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0)

    # Purpose: Combine the reduced polarization background and transported-curvature contribution into the lensing maps.
    def build_lensing_maps(self, T):
        c = self.c

        # Use the reduced polarization background solve as the large-scale
        # lensing baseline for the CPTG convergence reconstruction.
        sigma_pol_only = np.array(self.sigma_pol_background, copy=True)

        T_pos = np.maximum(T, 0.0)
        C_local, _, _, grad_mag = self.compute_transport_curvature(T_pos)
        sigma_grad = grad_mag / (np.max(grad_mag) + c.EPS)
        sigma_tpos = T_pos / (np.max(T_pos) + c.EPS)
        C_norm = C_local / (np.max(C_local) + c.EPS)

        # Keep the observable kappa reconstruction tied to transported
        # curvature strength itself rather than reusing the transport-coherence
        # boost. This preserves Bullet-side detachment while preventing the
        # lensing peak from being over-pulled away from the galaxy anchor.
        sigma_T_only = sigma_grad + c.LENS_TPOS_BLEND * sigma_tpos * np.power(C_norm, 1.0 / 3.0)

        # Use the transported-curvature field directly without an additional
        # hand-built transport gate. This keeps the observable contribution tied
        # to the evolved CPTG transport structure rather than to an imposed
        # corridor-localizing mask.
        source_norm = self.source_norm_cached
        ridge_norm = self.source_ridge_norm_cached

        # Tie the forward ridge enhancement to the transported-curvature field
        # itself, so the ridge boost only appears where transport actually
        # populates the Bullet-side corridor rather than acting as a static
        # morphology overlay.
        ridge_transport_weight = ridge_norm * sigma_tpos
        ridge_enhance = 1.0 + c.KAPPA_RIDGE_BLEND * ridge_transport_weight * (
            (2.0 / 5.0) + (3.0 / 5.0) * np.power(C_norm, 1.0 / 3.0)
        )
        sigma_T_only = sigma_T_only * ridge_enhance
        alpha_t_eff = float(
            np.sum(np.power(C_norm, 1.0 / 3.0) * (source_norm + 9.0 / 25.0 * ridge_transport_weight))
            / (np.sum(source_norm + 9.0 / 25.0 * ridge_transport_weight) + c.EPS)
        )
        alpha_t_eff = float(np.clip(alpha_t_eff, 0.0, 1.0))
        sigma_final = c.POLBG_FINAL_BLEND * sigma_pol_only + alpha_t_eff * sigma_T_only

        return sigma_pol_only, sigma_T_only, sigma_final

    # Purpose: Measure left-side peak locations across the baryonic, transport-only, and final lens maps.
    def measure_three_maps(self, sigma_pol_only, sigma_T_only, sigma_final):
        pol_left = peak_in_mask(sigma_pol_only, self.X, self.Y, self.X < 0.0)
        t_left = peak_in_mask(sigma_T_only, self.X, self.Y, self.X < 0.0)
        final_left = peak_in_mask(sigma_final, self.X, self.Y, self.X < 0.0)

        return {
            "pol_left_x": pol_left[0],
            "pol_left_y": pol_left[1],
            "t_left_x": t_left[0],
            "t_left_y": t_left[1],
            "final_left_x": final_left[0],
            "final_left_y": final_left[1],
            "pol_offset_from_gas": dist(pol_left, self.gas_centroid_bullet),
            "t_offset_from_gas": dist(t_left, self.gas_centroid_bullet),
            "final_offset_from_gas": dist(final_left, self.gas_centroid_bullet),
            "pol_sep_from_bullet_gal": dist(pol_left, self.gal_centroid_bullet),
            "t_sep_from_bullet_gal": dist(t_left, self.gal_centroid_bullet),
            "final_sep_from_bullet_gal": dist(final_left, self.gal_centroid_bullet),
        }

    # Purpose: Measure the positive, negative, and gradient transport lobes on the Bullet side.
    def measure_transport_lobes(self, T):
        T_pos = np.maximum(T, 0.0)
        T_neg = np.maximum(-T, 0.0)

        dTdx, dTdy = grad2(T, self.dx, self.dy)
        grad_mag = np.sqrt(dTdx**2 + dTdy**2)

        pos_left = peak_in_mask(T_pos, self.X, self.Y, self.X < 0.0)
        neg_left = peak_in_mask(T_neg, self.X, self.Y, self.X < 0.0)
        grad_left = peak_in_mask(grad_mag, self.X, self.Y, self.X < 0.0)

        return {
            "Tpos_left_x": pos_left[0],
            "Tpos_left_y": pos_left[1],
            "Tneg_left_x": neg_left[0],
            "Tneg_left_y": neg_left[1],
            "gradT_left_x": grad_left[0],
            "gradT_left_y": grad_left[1],
            "Tpos_offset_from_gas": dist(pos_left, self.gas_centroid_bullet),
            "Tneg_offset_from_gas": dist(neg_left, self.gas_centroid_bullet),
            "gradT_offset_from_gas": dist(grad_left, self.gas_centroid_bullet),
            "Tpos_sep_from_bullet": dist(pos_left, self.gal_centroid_bullet),
            "Tneg_sep_from_bullet": dist(neg_left, self.gal_centroid_bullet),
            "gradT_sep_from_bullet": dist(grad_left, self.gal_centroid_bullet),
        }

    # Purpose: Quantify the active Bullet-side ridge morphology used for persistence diagnostics.
    def measure_active_region_diagnostics(self, sigma_T_only):
        c = self.c

        dxb = self.X - self.gal_centroid_bullet[0]
        dyb = self.Y - self.gal_centroid_bullet[1]

        q = -dxb * self.axis_uy + dyb * self.axis_ux
        r = np.hypot(dxb, dyb)

        left_mask = self.X < 0.0

        q_abs = np.abs(q)
        flank_pref = np.exp(-0.5 * ((q_abs - c.RIDGE_Q_TARGET) / c.RIDGE_Q_SIGMA) ** 2)
        radial_pref = np.exp(-0.5 * ((r - c.RIDGE_R_TARGET) / c.RIDGE_R_SIGMA) ** 2)
        center_penalty = np.exp(-0.5 * (q / c.RIDGE_CENTER_SIGMA) ** 2)
        axis_soft = c.RIDGE_AXIS_SOFTEN + (1.0 - c.RIDGE_AXIS_SOFTEN) * flank_pref

        ridge_field = np.nan_to_num(sigma_T_only, nan=0.0, posinf=0.0, neginf=0.0)
        ridge_weight = ridge_field * radial_pref * axis_soft * (1.0 - center_penalty)
        ridge_weight = np.where(left_mask, ridge_weight, 0.0)

        ridge_peak = peak_in_mask(ridge_weight, self.X, self.Y, left_mask)

        ridge_mass = np.sum(ridge_weight) + c.EPS
        ridge_centroid = (
            float(np.sum(self.X * ridge_weight) / ridge_mass),
            float(np.sum(self.Y * ridge_weight) / ridge_mass),
        )

        axis_suppression = float(np.sum(ridge_field[left_mask] * center_penalty[left_mask]) / (np.sum(ridge_field[left_mask]) + c.EPS))
        flank_fraction = float(np.sum(ridge_field[left_mask] * flank_pref[left_mask]) / (np.sum(ridge_field[left_mask]) + c.EPS))
        ridge_score = float(np.max(ridge_weight))

        return {
            "ridge_peak_x": ridge_peak[0],
            "ridge_peak_y": ridge_peak[1],
            "ridge_offset_from_gas": dist(ridge_peak, self.gas_centroid_bullet),
            "ridge_sep_from_bullet": dist(ridge_peak, self.gal_centroid_bullet),
            "ridge_centroid_x": ridge_centroid[0],
            "ridge_centroid_y": ridge_centroid[1],
            "ridge_score": ridge_score,
            "ridge_flank_fraction": flank_fraction,
            "ridge_axis_suppression": axis_suppression,
        }

    # Purpose: Aggregate late-time ridge diagnostics into the summary persistence/support metrics.
    def compute_ridge_objective_summary(self, df):
        c = self.c

        tail_n = int(max(8, min(c.RIDGE_PERSISTENCE_WINDOW, len(df))))
        tail = df.tail(tail_n).copy()

        ridge_scores = tail["ridge_score"].to_numpy(dtype=float)
        ridge_offsets = tail["ridge_offset_from_gas"].to_numpy(dtype=float)
        ridge_flank = tail["ridge_flank_fraction"].to_numpy(dtype=float)
        ridge_axis = tail["ridge_axis_suppression"].to_numpy(dtype=float)

        peak_x = tail["ridge_peak_x"].to_numpy(dtype=float)
        peak_y = tail["ridge_peak_y"].to_numpy(dtype=float)
        peak_motion = np.sqrt(np.diff(peak_x) ** 2 + np.diff(peak_y) ** 2)
        peak_motion = np.nan_to_num(peak_motion, nan=0.0, posinf=0.0, neginf=0.0)

        ridge_tpos_sep = tail["ridge_tpos_sep"].to_numpy(dtype=float)
        ridge_grad_sep = tail["ridge_grad_sep"].to_numpy(dtype=float)
        carrier_detach = tail["tpos_bullet_detach"].to_numpy(dtype=float)

        score_mean = float(np.mean(ridge_scores))
        score_std = float(np.std(ridge_scores))
        offset_mean = float(np.mean(ridge_offsets))
        offset_std = float(np.std(ridge_offsets))
        flank_mean = float(np.mean(ridge_flank))
        axis_mean = float(np.mean(ridge_axis))
        wander_mean = float(np.mean(peak_motion)) if peak_motion.size > 0 else 0.0

        tpos_sep_mean = float(np.mean(ridge_tpos_sep))
        grad_sep_mean = float(np.mean(ridge_grad_sep))
        carrier_detach_mean = float(np.mean(carrier_detach))

        persistence = score_mean / (1.0 + score_std)
        stability = float(np.exp(-wander_mean / c.RIDGE_WANDER_SIGMA))
        offset_pref = float(np.exp(-0.5 * ((offset_mean - c.RIDGE_OFFSET_TARGET) / c.RIDGE_OFFSET_SIGMA) ** 2))
        flank_term = 0.5 + 0.5 * flank_mean
        axis_term = max(0.0, 1.0 - axis_mean)

        support_coherence = float(np.exp(-0.5 * (tpos_sep_mean / c.RIDGE_SUPPORT_SEP_SIGMA) ** 2))
        grad_coherence = float(np.exp(-0.5 * ((grad_sep_mean - c.RIDGE_GRAD_TARGET) / c.RIDGE_GRAD_SIGMA) ** 2))
        carrier_term = float(np.exp(-0.5 * ((carrier_detach_mean - c.TPOS_BULLET_DETACH_TARGET) / c.TPOS_BULLET_DETACH_SIGMA) ** 2))

        ridge_objective = persistence * stability * offset_pref * flank_term * axis_term
        ridge_supported_objective = ridge_objective * support_coherence * grad_coherence * carrier_term

        return {
            "ridge_tail_window": tail_n,
            "ridge_score_tail_mean": score_mean,
            "ridge_score_tail_std": score_std,
            "ridge_offset_tail_mean": offset_mean,
            "ridge_offset_tail_std": offset_std,
            "ridge_flank_fraction_tail_mean": flank_mean,
            "ridge_axis_suppression_tail_mean": axis_mean,
            "ridge_peak_wander_tail_mean": wander_mean,
            "ridge_tpos_sep_tail_mean": tpos_sep_mean,
            "ridge_grad_sep_tail_mean": grad_sep_mean,
            "tpos_bullet_detach_tail_mean": carrier_detach_mean,
            "ridge_persistence": float(persistence),
            "ridge_stability": float(stability),
            "ridge_offset_preference": float(offset_pref),
            "ridge_support_coherence": float(support_coherence),
            "ridge_grad_coherence": float(grad_coherence),
            "tpos_bullet_detach_term": float(carrier_term),
            "ridge_objective": float(ridge_objective),
            "ridge_supported_objective": float(ridge_supported_objective),
        }

    # Purpose: Measure how the active ridge aligns with the positive and gradient transport carriers.
    def measure_ridge_support_diagnostics(self, ridge, lobe):
        # ridge_tpos_sep: ridge peak to positive transport-lobe peak
        # ridge_grad_sep: ridge peak to gradient-of-transport peak
        # tpos_bullet_detach: positive transport-lobe peak to bullet-galaxy centroid
        ridge_peak = (ridge["ridge_peak_x"], ridge["ridge_peak_y"])
        tpos_peak = (lobe["Tpos_left_x"], lobe["Tpos_left_y"])
        grad_peak = (lobe["gradT_left_x"], lobe["gradT_left_y"])

        ridge_tpos_sep = dist(ridge_peak, tpos_peak)
        ridge_grad_sep = dist(ridge_peak, grad_peak)
        carrier_detach = lobe["Tpos_sep_from_bullet"]

        return {
            "ridge_tpos_sep": ridge_tpos_sep,
            "ridge_grad_sep": ridge_grad_sep,
            "tpos_bullet_detach": carrier_detach,
        }

    # Purpose: Export the named observational elements and measurement windows used in the reduced model.
    def build_element_map(self, sigma):
        """Map the modeled Bullet Cluster elements onto explicit named points.

        This readout layer keeps the observational elements separate from the
        solver field itself. The returned structure records the anchor centers,
        the local measurement windows, and the extracted lensing peaks used in
        the benchmark.
        """
        lens_centroid = centroid_from_positive(sigma, self.X, self.Y, self.c.EPS)

        bullet_window_kpc = self.c.BULLET_MEASURE_WINDOW_KPC
        main_window_kpc = self.c.MAIN_MEASURE_WINDOW_KPC
        north_ref_y = self.gal_centroid_main[1] + self.c.MAIN_NORTH_MEASURE_DY
        south_ref_y = self.gal_centroid_main[1] + self.c.MAIN_SOUTH_MEASURE_DY
        ns_window_kpc = self.c.MAIN_NS_MEASURE_WINDOW_KPC

        bullet_mask = (
            (self.X - self.gal_centroid_bullet[0]) ** 2
            + (self.Y - self.gal_centroid_bullet[1]) ** 2
            <= bullet_window_kpc ** 2
        )
        main_mask = (
            (self.X - self.gal_centroid_main[0]) ** 2
            + (self.Y - self.gal_centroid_main[1]) ** 2
            <= main_window_kpc ** 2
        )
        north_mask = (
            ((self.X - self.gal_centroid_main[0]) ** 2 + (self.Y - north_ref_y) ** 2 <= ns_window_kpc ** 2)
            & (self.Y >= self.gal_centroid_main[1])
        )
        south_mask = (
            ((self.X - self.gal_centroid_main[0]) ** 2 + (self.Y - south_ref_y) ** 2 <= ns_window_kpc ** 2)
            & (self.Y <= self.gal_centroid_main[1])
        )

        bullet_lens = centroid_peak_in_mask(
            sigma, self.X, self.Y, bullet_mask,
            smooth_sigma_pix=self.c.PEAK_SMOOTH_SIGMA_PIX,
            eps=self.c.EPS,
        )
        main_lens = centroid_peak_in_mask(
            sigma, self.X, self.Y, main_mask,
            smooth_sigma_pix=max(0.75, (461 / 40) * self.c.PEAK_SMOOTH_SIGMA_PIX),
            eps=self.c.EPS,
        )
        main_north_lens = centroid_peak_in_mask(
            sigma, self.X, self.Y, north_mask,
            smooth_sigma_pix=max(0.75, (461 / 40) * self.c.PEAK_SMOOTH_SIGMA_PIX),
            eps=self.c.EPS,
        )
        main_south_lens = centroid_peak_in_mask(
            sigma, self.X, self.Y, south_mask,
            smooth_sigma_pix=max(0.75, (461 / 40) * self.c.PEAK_SMOOTH_SIGMA_PIX),
            eps=self.c.EPS,
        )

        # Purpose: Package one named element point for the exported element map structure.
        def _point(xy, role):
            return {"x": float(xy[0]), "y": float(xy[1]), "role": role}

        return {
            "elements": {
                "lens_global": _point(lens_centroid, "Positive-field centroid of the full CPTG lens map."),
                "bullet_galaxy": _point(self.gal_centroid_bullet, "Collisionless Bullet galaxy anchor."),
                "bullet_gas": _point(self.gas_centroid_bullet, "Collisional Bullet gas anchor."),
                "bullet_lens": _point(bullet_lens, "Local Bullet lensing peak extracted inside the Bullet window."),
                "main_galaxy": _point(self.gal_centroid_main, "Collisionless main-cluster galaxy anchor."),
                "main_gas": _point(self.gas_centroid_main, "Collisional main-cluster gas anchor."),
                "main_lens": _point(main_lens, "Primary main-cluster lensing peak extracted inside the main window."),
                "main_north_lens": _point(main_north_lens, "Northern main-cluster subclump lensing peak."),
                "main_south_lens": _point(main_south_lens, "Southern main-cluster subclump lensing peak."),
                "main_north_reference": _point((self.gal_centroid_main[0], north_ref_y), "Reference center for north subclump peak search."),
                "main_south_reference": _point((self.gal_centroid_main[0], south_ref_y), "Reference center for south subclump peak search."),
            },
            "measurement_windows": {
                "bullet": {
                    "center_x": float(self.gal_centroid_bullet[0]),
                    "center_y": float(self.gal_centroid_bullet[1]),
                    "radius_kpc": float(bullet_window_kpc),
                },
                "main": {
                    "center_x": float(self.gal_centroid_main[0]),
                    "center_y": float(self.gal_centroid_main[1]),
                    "radius_kpc": float(main_window_kpc),
                },
                "main_north": {
                    "center_x": float(self.gal_centroid_main[0]),
                    "center_y": float(north_ref_y),
                    "radius_kpc": float(ns_window_kpc),
                },
                "main_south": {
                    "center_x": float(self.gal_centroid_main[0]),
                    "center_y": float(south_ref_y),
                    "radius_kpc": float(ns_window_kpc),
                },
            },
        }

    # Purpose: Convert the extracted lens peaks into the benchmark distance measurements used for scoring.
    def measure_map(self, sigma):
        element_map = self.build_element_map(sigma)
        elems = element_map["elements"]

        lens_centroid = (elems["lens_global"]["x"], elems["lens_global"]["y"])
        bullet_peak = (elems["bullet_lens"]["x"], elems["bullet_lens"]["y"])
        main_peak = (elems["main_lens"]["x"], elems["main_lens"]["y"])
        north_peak = (elems["main_north_lens"]["x"], elems["main_north_lens"]["y"])
        south_peak = (elems["main_south_lens"]["x"], elems["main_south_lens"]["y"])

        return {
            "lens_centroid_x": lens_centroid[0],
            "lens_centroid_y": lens_centroid[1],
            "bullet_peak_x": bullet_peak[0],
            "bullet_peak_y": bullet_peak[1],
            "main_peak_x": main_peak[0],
            "main_peak_y": main_peak[1],
            "main_north_peak_x": north_peak[0],
            "main_north_peak_y": north_peak[1],
            "main_south_peak_x": south_peak[0],
            "main_south_peak_y": south_peak[1],
            "offset_from_gas": dist(bullet_peak, self.gas_centroid_bullet),
            "main_offset_from_gas": dist(main_peak, self.gas_centroid_main),
            "main_north_offset_from_gas": dist(north_peak, self.gas_centroid_main),
            "main_south_offset_from_gas": dist(south_peak, self.gas_centroid_main),
            "lens_gal_bullet_sep": dist(bullet_peak, self.gal_centroid_bullet),
            "main_gal_sep": dist(main_peak, self.gal_centroid_main),
            "main_subclump_separation": dist(north_peak, south_peak),
            "lens_peak_separation": dist(bullet_peak, main_peak),
        }

    # Purpose: Estimate the characteristic Bullet offset timescale from the tracked lens-peak motion.
    def estimate_tau_from_track(self, track, dt_phys):
        if len(track) < 8:
            return 0.0
        xs = np.array([p[0] for p in track], dtype=float)
        ys = np.array([p[1] for p in track], dtype=float)
        motion = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        motion = np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)
        thresh = 0.02 * (np.max(motion) + 1e-30)
        small = motion < thresh
        if np.any(small):
            idx = int(np.argmax(small)) + 1
            return idx * dt_phys
        return len(track) * dt_phys

    # Purpose: Evolve one full transport scenario in time and record all per-step and final diagnostics.
    def run_scenario(self, scenario_name, switches):
        c = self.c

        if switches["use_initial_condition"]:
            T_pos = np.maximum(self.T0_background, 0.0)
            T_neg = np.maximum(-self.T0_background, 0.0)
        else:
            T_pos = np.zeros_like(self.sigma_b)
            T_neg = np.zeros_like(self.sigma_b)

        bullet_track = []
        rows = []

        use_diffusion = switches["use_diffusion"]
        use_advection = switches["use_advection"]
        use_decay = switches["use_decay"]
        use_projected_velocity = switches.get("use_projected_velocity", False)

        velocity_mode = switches.get("velocity_mode", "curv")
        u_x, u_y = self.get_velocity_bundle(
            use_projected_velocity=use_projected_velocity,
            velocity_mode=velocity_mode,
        )

        u_mag = np.sqrt(u_x**2 + u_y**2) + c.EPS
        u_scale = np.minimum(1.0, c.U_MAX / u_mag)
        u_x = u_x * u_scale
        u_y = u_y * u_scale

        curv = self.chi_background
        curv_norm = curv / (np.max(curv) + c.EPS)

        dchidx, dchidy = grad2(curv_norm, self.dx, self.dy)
        grad_chi_mag = np.sqrt(dchidx**2 + dchidy**2)
        chi_velocity_damp = np.exp(-grad_chi_mag / c.GRAD_CHI_SCALE)
        chi_velocity_damp = np.clip(chi_velocity_damp, 1.0 / 25.0, 1.0)

        print(f"\n=== RUNNING {scenario_name} ===")
        print(SCENARIO_DESCRIPTIONS[scenario_name])
        print(scenario_switch_summary(switches))

        for n in range(c.NSTEPS):
            if use_diffusion:
                diff_pos = c.DIFFUSION_SCALE * self.diffuse_term(T_pos)
                diff_neg = c.DIFFUSION_SCALE * self.diffuse_term(T_neg)
            else:
                diff_pos = np.zeros_like(T_pos)
                diff_neg = np.zeros_like(T_neg)

            if use_advection:
                velocity_gate = chi_velocity_damp

                u_x_eff = u_x * velocity_gate
                u_y_eff = u_y * velocity_gate

                flow_dir = u_x_eff * self.axis_ux + u_y_eff * self.axis_uy
                forward_mask = (flow_dir > 0.0).astype(float)

                u_x_eff = u_x_eff * forward_mask
                u_y_eff = u_y_eff * forward_mask

                divF_pos, flux_coeff_pos, _, _ = self.curvature_flux_term(
                    T_pos, u_x_eff, u_y_eff, curv_norm, chi_velocity_damp
                )
                advective_pos = self.advective_term(T_pos, u_x_eff, u_y_eff)
                adv_pos = divF_pos + advective_pos
                adv_neg = np.zeros_like(T_neg)

                u_eff_max = float(np.max(np.sqrt(u_x_eff**2 + u_y_eff**2)))
                flux_coeff_max = float(np.max(flux_coeff_pos))
            else:
                adv_pos = np.zeros_like(T_pos)
                adv_neg = np.zeros_like(T_neg)
                u_x_eff = np.zeros_like(T_pos)
                u_y_eff = np.zeros_like(T_pos)
                u_eff_max = 0.0
                flux_coeff_max = 0.0

            if use_decay:
                C_local_pos, _, _, _ = self.compute_transport_curvature(T_pos)
                curvature_decay_gate = 1.0 / (1.0 + C_local_pos)
                decay_pos = c.GAMMA_TRANS_NUM * curvature_decay_gate * T_pos
                decay_neg = c.GAMMA_TRANS_NUM * T_neg
            else:
                decay_pos = np.zeros_like(T_pos)
                decay_neg = np.zeros_like(T_neg)

            C_local_pos, _, _, _ = self.compute_transport_curvature(T_pos)
            Ctx, Cty = grad2(C_local_pos, self.dx, self.dy)
            Ptx = c.BETA_POL * np.power(C_local_pos, -2.0 / 3.0) * Ctx
            Pty = c.BETA_POL * np.power(C_local_pos, -2.0 / 3.0) * Cty
            divP_transport = div2(Ptx, Pty, self.dx, self.dy)
            source_base = np.maximum(np.nan_to_num(divP_transport, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            source_base = smooth5(source_base, 1)

            if use_advection:
                source_adv_1 = advect_field_along_flow(source_base, u_x_eff, u_y_eff, self.num_dt_eff, self.dx, self.dy)
                source_adv_2 = advect_field_along_flow(source_adv_1, u_x_eff, u_y_eff, self.num_dt_eff, self.dx, self.dy)
                source_pos = source_base + 0.5 * source_adv_1 + (1.0 / 3.0) * source_adv_2
            else:
                source_pos = source_base

            corridor_gate = np.sqrt(np.clip(T_pos / (np.max(T_pos) + c.EPS), 0.0, 1.0))
            source_pos = source_pos * corridor_gate
            # Theory-consistent source amplitude: keep the physical polarization-divergence
            # source strength after corridor selection instead of imposing a fixed empirical scale.
            source_pos = np.maximum(source_pos, 0.0)

            rhs_pos = diff_pos - adv_pos - decay_pos + source_pos
            rhs_neg = diff_neg - adv_neg - decay_neg

            rhs_pos = np.nan_to_num(rhs_pos, nan=0.0, posinf=0.0, neginf=0.0)
            rhs_neg = np.nan_to_num(rhs_neg, nan=0.0, posinf=0.0, neginf=0.0)

            T_pos = T_pos + self.num_dt_eff * rhs_pos
            T_neg = T_neg + self.num_dt_eff * rhs_neg

            if use_advection:
                T_pos = T_pos + self.num_dt_eff * self.curvature_support_operator(T_pos, u_x_eff, u_y_eff)

            # Maintain a finite transverse transport-channel width so the
            # transported-curvature packet behaves like a coherent corridor
            # rather than a numerically razor-thin filament.
            T_pos = smooth5(T_pos, c.TRANSPORT_COHERENCE_PASSES)
            T_neg = smooth5(T_neg, c.TRANSPORT_COHERENCE_PASSES)

            T_pos = np.clip(
                np.nan_to_num(T_pos, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
                c.CLIP_T_MAX,
            )
            T_neg = np.clip(
                np.nan_to_num(T_neg, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
                1.0,
            )

            T_signed = T_pos - 0.2 * T_neg
            T_eff = T_pos

            sigma_pol_only, sigma_T_only, sigma_transport = self.build_lensing_maps(T_eff)

            meas = self.measure_map(sigma_transport)
            diag = self.measure_three_maps(sigma_pol_only, sigma_T_only, sigma_transport)
            lobe = self.measure_transport_lobes(T_signed)
            ridge = self.measure_active_region_diagnostics(sigma_T_only)
            support = self.measure_ridge_support_diagnostics(ridge, lobe)

            bullet_peak = (meas["bullet_peak_x"], meas["bullet_peak_y"])
            bullet_track.append(bullet_peak)

            adv_total = adv_pos + adv_neg
            diff_total = diff_pos + diff_neg
            decay_total = decay_pos + decay_neg
            rhs_total = rhs_pos + rhs_neg

            adv_norm = l2norm(adv_total)
            adv_pos_norm = l2norm(adv_pos)
            adv_neg_norm = l2norm(adv_neg)
            diff_norm = l2norm(diff_total)
            decay_norm = l2norm(decay_total)

            rows.append({
                "step": n + 1,
                "time_phys": (n + 1) * c.PHYS_DT_PER_STEP,
                **meas,
                **diag,
                **lobe,
                **ridge,
                **support,
                "adv_norm": adv_norm,
                "adv_pos_norm": adv_pos_norm,
                "adv_neg_norm": adv_neg_norm,
                "diff_norm": diff_norm,
                "decay_norm": decay_norm,
                "rhs_norm": l2norm(rhs_total),
                "source_norm": l2norm(source_pos),
                "u_eff_max": u_eff_max,
                "flux_coeff_max": flux_coeff_max,
                "chi_velocity_damp_mean": float(np.mean(chi_velocity_damp)),
                "transport_coherence_mean": float(np.mean(self.transport_coherence_fields(T_pos, u_x_eff, u_y_eff)[2])) if use_advection else 0.0,
                "T_norm": l2norm(T_eff),
                "T_max": float(np.max(T_eff)),
                "T_min": float(np.min(T_eff)),
                "T_signed_norm": l2norm(T_signed),
                "T_signed_max": float(np.max(T_signed)),
                "T_signed_min": float(np.min(T_signed)),
                "T_pos_norm": l2norm(T_pos),
                "T_neg_norm": l2norm(T_neg),
                "T_pos_max": float(np.max(T_pos)),
                "T_neg_max": float(np.max(T_neg)),
            })

        tau_offset = self.estimate_tau_from_track(bullet_track, c.PHYS_DT_PER_STEP)

        T_signed = T_pos - 0.2 * T_neg
        T_eff = T_pos
        sigma_pol_only, sigma_T_only, sigma_final = self.build_lensing_maps(T_eff)
        final_meas = self.measure_map(sigma_final)

        df = pd.DataFrame(rows)

        ridge_summary = self.compute_ridge_objective_summary(df)

        summary = {
            "scenario": scenario_name,
            "scenario_description": SCENARIO_DESCRIPTIONS[scenario_name],
            "scenario_switch_summary": scenario_switch_summary(switches),
            "use_diffusion": use_diffusion,
            "use_advection": use_advection,
            "use_decay": use_decay,
            "use_initial_condition": switches["use_initial_condition"],
            "velocity_mode": velocity_mode,
            "num_dt_eff": float(self.num_dt_eff),
            "tau_offset_bullet": float(tau_offset),
            "offset_from_gas": float(final_meas["offset_from_gas"]),
            "main_offset_from_gas": float(final_meas["main_offset_from_gas"]),
            "lens_gal_bullet_sep": float(final_meas["lens_gal_bullet_sep"]),
            "main_gal_sep": float(final_meas["main_gal_sep"]),
            "main_north_offset_from_gas": float(final_meas["main_north_offset_from_gas"]),
            "main_south_offset_from_gas": float(final_meas["main_south_offset_from_gas"]),
            "main_subclump_separation": float(final_meas["main_subclump_separation"]),
            "lens_peak_separation": float(final_meas["lens_peak_separation"]),
            "bullet_gas_residual_kpc": float(final_meas["offset_from_gas"] - TARGET_BULLET_GAS_LENS_OFFSET_KPC),
            "bullet_gal_residual_kpc": float(final_meas["lens_gal_bullet_sep"] - TARGET_BULLET_GAL_LENS_OFFSET_KPC),
            "main_north_residual_kpc": float(final_meas["main_north_offset_from_gas"] - TARGET_MAIN_NORTH_GAS_LENS_OFFSET_KPC),
            "main_south_residual_kpc": float(final_meas["main_south_offset_from_gas"] - TARGET_MAIN_SOUTH_GAS_LENS_OFFSET_KPC),
            "main_subclump_residual_kpc": float(final_meas["main_subclump_separation"] - TARGET_MAIN_SUBCLUMP_SEPARATION_KPC),
            "lens_peak_separation_residual_kpc": float(final_meas["lens_peak_separation"] - TARGET_LENS_PEAK_SEPARATION_KPC),
            "bullet_gas_z": float((final_meas["offset_from_gas"] - TARGET_BULLET_GAS_LENS_OFFSET_KPC) / max(TARGET_BULLET_GAS_LENS_SCORE_SIGMA_KPC, 1e-12)),
            "bullet_gal_z": float((final_meas["lens_gal_bullet_sep"] - TARGET_BULLET_GAL_LENS_OFFSET_KPC) / max(TARGET_BULLET_GAL_LENS_SCORE_SIGMA_KPC, 1e-12)),
            "main_north_z": float((final_meas["main_north_offset_from_gas"] - TARGET_MAIN_NORTH_GAS_LENS_OFFSET_KPC) / max(TARGET_MAIN_NORTH_GAS_LENS_SCORE_SIGMA_KPC, 1e-12)),
            "main_south_z": float((final_meas["main_south_offset_from_gas"] - TARGET_MAIN_SOUTH_GAS_LENS_OFFSET_KPC) / max(TARGET_MAIN_SOUTH_GAS_LENS_SCORE_SIGMA_KPC, 1e-12)),
            "main_subclump_z": float((final_meas["main_subclump_separation"] - TARGET_MAIN_SUBCLUMP_SEPARATION_KPC) / max(TARGET_MAIN_SUBCLUMP_SEPARATION_SCORE_SIGMA_KPC, 1e-12)),
            "lens_peak_separation_z": float((final_meas["lens_peak_separation"] - TARGET_LENS_PEAK_SEPARATION_KPC) / max(TARGET_LENS_PEAK_SEPARATION_SCORE_SIGMA_KPC, 1e-12)),
            "bullet_cluster_score": float(bullet_cluster_score(final_meas)),
            "adv_norm_final": float(df["adv_norm"].iloc[-1]),
            "adv_pos_norm_final": float(df["adv_pos_norm"].iloc[-1]),
            "adv_neg_norm_final": float(df["adv_neg_norm"].iloc[-1]),
            "diff_norm_final": float(df["diff_norm"].iloc[-1]),
            "decay_norm_final": float(df["decay_norm"].iloc[-1]),
            "rhs_norm_final": float(df["rhs_norm"].iloc[-1]),
            "source_norm_final": float(df["source_norm"].iloc[-1]),
            "T_pos_max_final": float(df["T_pos_max"].iloc[-1]),
            "T_neg_max_final": float(df["T_neg_max"].iloc[-1]),
            "u_eff_max_final": float(df["u_eff_max"].iloc[-1]),
            "flux_coeff_max_final": float(df["flux_coeff_max"].iloc[-1]),
            "chi_velocity_damp_mean_final": float(df["chi_velocity_damp_mean"].iloc[-1]),
            "transport_coherence_mean_final": float(df["transport_coherence_mean"].iloc[-1]),
            "ridge_score_final": float(df["ridge_score"].iloc[-1]),
            "ridge_score_best": float(df["ridge_score"].max()),
            "ridge_offset_final": float(df["ridge_offset_from_gas"].iloc[-1]),
            "ridge_flank_fraction_final": float(df["ridge_flank_fraction"].iloc[-1]),
            "ridge_axis_suppression_final": float(df["ridge_axis_suppression"].iloc[-1]),
            "ridge_tpos_sep_final": float(df["ridge_tpos_sep"].iloc[-1]),
            "ridge_grad_sep_final": float(df["ridge_grad_sep"].iloc[-1]),
            "tpos_bullet_detach_final": float(df["tpos_bullet_detach"].iloc[-1]),
            **ridge_summary,
        }

        return {
            "summary": summary,
            "step_df": df,
            "sigma_pol_only": sigma_pol_only,
            "sigma_T_only": sigma_T_only,
            "sigma_final": sigma_final,
            "T_final": T_eff,
            "T_signed_final": T_signed,
            "T_pos_final": T_pos,
            "T_neg_final": T_neg,
            "final_meas": final_meas,
        }

    # Purpose: Smooth and normalize the effective lensing field into the exported kappa proxy.
    def build_normalized_kappa_map(self, sigma_eff):
        """Return a normalized convergence reconstruction.

        This reduced script does not solve for an absolute Sigma_crit, so the
        reconstruction is reported as a dimensionless normalized convergence
        field, kappa_norm, proportional to Sigma_eff / Sigma_crit.
        """
        c = self.c
        arr = np.maximum(np.nan_to_num(sigma_eff, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        arr = gaussian_filter(arr, sigma=c.LENSING_SMOOTH_SIGMA_PIX)
        arr = arr / (np.max(arr) + c.EPS)
        arr = np.power(arr, c.POLBG_DISPLAY_GAMMA)
        return normalize_nonnegative(arr, c.EPS)

    # Purpose: Build a display-only kappa map that makes the broad polarization background legible.
    def build_display_kappa_map(self, sigma_pol_only, sigma_final):
        c = self.c
        bg = np.maximum(np.nan_to_num(sigma_pol_only, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        bg = gaussian_filter(bg, sigma=max(c.LENSING_SMOOTH_SIGMA_PIX, 1.8))
        bg = bg / (np.max(bg) + c.EPS)
        bg = np.power(bg, c.DISPLAY_POLBG_GAMMA)

        final = np.maximum(np.nan_to_num(sigma_final, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        final = gaussian_filter(final, sigma=c.LENSING_SMOOTH_SIGMA_PIX)
        final = final / (np.max(final) + c.EPS)
        final = np.power(final, c.DISPLAY_FINAL_GAMMA)

        display = c.DISPLAY_POLBG_BLEND * bg + final
        return normalize_nonnegative(display, c.EPS)

    # Purpose: Open the researcher-facing kappa figure in a plot window for the best scenario.
    def show_synthetic_observables(self, scenario_result):
        """Display the normalized CPTG kappa reconstruction in a plot window.

        The display layer is restricted to the single benchmark-facing CPTG
        convergence plot. No image files or JSON sidecar files are written.
        """
        c = self.c

        cptg_display_map = self.build_display_kappa_map(scenario_result["sigma_pol_only"], scenario_result["sigma_final"])

        extent = [c.XMIN, c.XMAX, c.YMIN, c.YMAX]
        cptg_meas = scenario_result["final_meas"]

        bullet_gas = (self.gas_centroid_bullet[0], self.gas_centroid_bullet[1])
        main_gas = (self.gas_centroid_main[0], self.gas_centroid_main[1])
        bullet_gal = self.gal_centroid_bullet
        main_gal = self.gal_centroid_main

        # Purpose: Draw the fixed physical scale bar on the displayed kappa figure.
        def _draw_scale_bar(ax, length_kpc=100.0):
            x0 = c.XMIN + 70.0
            y0 = c.YMIN + 60.0
            ax.plot([x0, x0 + length_kpc], [y0, y0], color="white", linewidth=3)
            ax.text(
                x0 + 0.5 * length_kpc,
                y0 + 18.0,
                f"{int(length_kpc)} kpc",
                color="white",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Purpose: Overlay gas, galaxy, and lens-peak markers plus callouts on the displayed figure.
        def _add_common_markers(ax, meas):
            bullet_lens = (meas["bullet_peak_x"], meas["bullet_peak_y"])
            main_lens = (meas["main_peak_x"], meas["main_peak_y"])
            main_north_lens = (meas["main_north_peak_x"], meas["main_north_peak_y"])
            main_south_lens = (meas["main_south_peak_x"], meas["main_south_peak_y"])

            ax.scatter(*bullet_gas, marker="x", s=90, color="red", linewidths=2.0, label="Bullet gas peak", zorder=11)
            ax.scatter(*main_gas, marker="x", s=90, color="darkorange", linewidths=2.0, label="Main gas peak", zorder=11)
            ax.scatter(*bullet_lens, marker="s", s=14, color="cyan", label="Bullet lens peak", zorder=12)
            ax.scatter(*main_lens, marker="D", s=12, color="deepskyblue", label="Main lens peak", zorder=12)
            ax.scatter(*main_north_lens, marker="^", s=50, color="lawngreen", label="Main-north lens peak", zorder=11)
            ax.scatter(*main_south_lens, marker="v", s=50, color="magenta", label="Main-south lens peak", zorder=11)
            ax.scatter(*bullet_gal, marker="o", s=50, facecolors="none", color="violet", linewidths=1.8, label="Bullet galaxy peak", zorder=11)
            ax.scatter(*main_gal, marker="o", s=50, facecolors="none", color="blueviolet", linewidths=1.8, label="Main galaxy peak", zorder=11)

            ax.annotate(
                "Bullet gas peak",
                xy=bullet_gas,
                xytext=(10.0, -140.0),
                color="white",
                fontsize=10,
                ha="right",
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color="white",
                    zorder=12,
                    lw=2.2,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=18,
                ),
            )
            ax.annotate(
                "Main gas peak",
                xy=main_gas,
                xytext=(210.0, 260.0),
                color="white",
                fontsize=10,
                zorder=12,
                ha="right",
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color="white",
                    lw=2.2,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=18,
                ),
            )
            _draw_scale_bar(ax)

        # Purpose: Add the observed-vs-model benchmark summary box to the displayed figure.
        def _add_metric_box(ax, meas):
            benchmark_rows = observed_comparison_rows(meas)
            label_map = {
                "bullet_mass_galaxy_offset": "Bullet mass-galaxy",
                "bullet_mass_icm_offset": "Bullet mass-ICM",
                "main_north_mass_icm_offset": "Main-north mass-ICM",
                "main_south_mass_icm_offset": "Main-south mass-ICM",
                "cluster_scale_separation": "Cluster separation",
                "main_cluster_subclump_separation": "Main subclump separation",
            }
            lines = ["CPTG kappa reconstruction (display-enhanced)"]
            for row in benchmark_rows:
                lines.append(
                    f"{label_map[row['benchmark']]}: {row['model']:.2f} kpc "
                    f"(obs {row['observed_label']}, d={row['residual']:+.2f})"
                )
            lines.append(f"Score: {bullet_cluster_score(meas):.6f}")
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5, edgecolor="white"),
            )

        # Purpose: Collapse duplicate legend entries and finalize the styled figure legend.
        def _finalize_legend(ax):
            handles, labels = ax.get_legend_handles_labels()
            seen = set()
            uniq_h, uniq_l = [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    uniq_h.append(h)
                    uniq_l.append(l)
            ax.legend(
                uniq_h,
                uniq_l,
                loc="lower right",
                fontsize=8,
                framealpha=0.75,
                facecolor="black",
                edgecolor="white",
                labelcolor="white",
            )

        fig, ax = plt.subplots(figsize=(10.2, 6.4), constrained_layout=True)
        im = ax.imshow(cptg_display_map, origin="lower", extent=extent, aspect="auto", cmap="viridis")

        positive_kappa = cptg_display_map[cptg_display_map > 0.0]
        contour_levels = np.array([], dtype=float)
        if positive_kappa.size > 0:
            contour_quantiles = [94.3, 95.0, 96.5, 98.0, 99.0, 99.6]
            contour_levels = np.unique(
                np.array([np.percentile(positive_kappa, q) for q in contour_quantiles], dtype=float)
            )
            contour_levels = contour_levels[np.isfinite(contour_levels) & (contour_levels > 0.0)]
        if contour_levels.size > 0:
            contour_widths = np.linspace(0.75, 1.55, contour_levels.size)
            ax.contour(
                self.X,
                self.Y,
                cptg_display_map,
                levels=contour_levels,
                colors="white",
                linewidths=contour_widths,
                alpha=0.92,
            )

        _add_common_markers(ax, cptg_meas)
        _add_metric_box(ax, cptg_meas)
        ax.set_title("CPTG Curvature Transport Model — Normalized Kappa Reconstruction")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        _finalize_legend(ax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized convergence kappa")
        plt.show()

    # Purpose: Run every named scenario and return the score-ranked summary table.
    def run_all(self):
        scenario_results = {}
        summaries = []
        for name, switches in SCENARIOS.items():
            res = self.run_scenario(name, switches)
            scenario_results[name] = res
            summaries.append(res["summary"])

        summary_df = pd.DataFrame(summaries)
        summary_df = summary_df.sort_values(
            [
                "bullet_cluster_score",
                "bullet_gas_z",
                "bullet_gal_z",
                "main_north_z",
                "main_south_z",
                "main_subclump_z",
                "lens_peak_separation_z",
            ],
            ascending=[True, True, True, True, True, True, True],
        ).reset_index(drop=True)

        return scenario_results, summary_df

# Purpose: Execute the full scenario sweep, save the best synthetic outputs, and print the final audit.
def main():

    cfg = Config()
    model = BulletClusterDiagnosticModel(cfg)
    scenario_results, summary_df = model.run_all()

    best = summary_df.iloc[0]
    best_name = best["scenario"]
    model.show_synthetic_observables(scenario_results[best_name])
    print("CPTG Bullet Cluster proof-of-concept run complete.")
    print(f"Restored working baseline scenario for controlled refinement: {WORKING_BASELINE_SCENARIO}")
    print("Final scenario ranking now uses the locked external Bullet Cluster observational benchmark.")
    print("Targets: bullet mass-galaxy=17.78+/-0.66 kpc, bullet mass-ICM~150 kpc, main-north mass-ICM~200 kpc, main-south mass-ICM~400 kpc, main subclump separation~200 kpc, cluster-scale separation~720 kpc.")
    print("A single normalized CPTG kappa reconstruction was opened for the best observationally matched scenario.")
    print(
        f"Best scenario by Bullet Cluster match: {best_name} | "
        f"bullet_gas_offset={best['offset_from_gas']:.2f} kpc | "
        f"bullet_gal_lens_sep={best['lens_gal_bullet_sep']:.2f} kpc | "
        f"main_gas_offset={best['main_offset_from_gas']:.2f} kpc | "
        f"main_gal_lens_sep={best['main_gal_sep']:.2f} kpc | "
        f"lens_peak_sep={best['lens_peak_separation']:.2f} kpc | "
        f"score={best['bullet_cluster_score']:.3f}"
    )
    print(
        f"Residuals: d_bullet_mass_icm={best['bullet_gas_residual_kpc']:.2f} kpc | "
        f"d_bullet_mass_galaxy={best['bullet_gal_residual_kpc']:.2f} kpc | "
        f"d_main_north_mass_icm={best['main_north_residual_kpc']:.2f} kpc | "
        f"d_main_south_mass_icm={best['main_south_residual_kpc']:.2f} kpc | "
        f"d_main_subclump_sep={best['main_subclump_residual_kpc']:.2f} kpc | "
        f"d_cluster_sep={best['lens_peak_separation_residual_kpc']:.2f} kpc"
    )
    print_benchmark_audit(best, label=f"best scenario ({best_name})")
    if WORKING_BASELINE_SCENARIO in scenario_results:
        print_benchmark_audit(scenario_results[WORKING_BASELINE_SCENARIO]["summary"], label=f"locked baseline ({WORKING_BASELINE_SCENARIO})")

if __name__ == "__main__":
    try:
        main()
    finally:
        input("\nPress Enter to exit...")

