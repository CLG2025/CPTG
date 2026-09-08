#!/usr/bin/env python3
"""
Visual CMB vs CPTG comparison from the same data source.

What it does
------------
Given a Planck-style component-separation FITS map, this script:

1. Reads the observed CMB temperature map.
2. Applies the temperature mask from the same file (or a fallback mask source).
3. Downgrades to a chosen Nside.
4. Removes monopole and dipole on the valid sky.
5. Builds a CPTG comparison map from the *same observed phase scaffold* using
   the locked CPTG geometric-pi CMB envelope.
6. Saves side-by-side visual comparisons using the same projection, same mask,
   and comparable color scaling.

Main outputs
------------
- cmb_vs_cptg_side_by_side.png
- cmb_vs_cptg_side_by_side_fitted.png
- observed_cmb_map_uK.png
- cptg_comparison_map_uK.png
- residual_observed_minus_cptg_uK.png
- cmb_vs_cptg_scatter.png
- cmb_vs_cptg_power_compare.png
- cptg_comparison_map_uK.fits
- cptg_comparison_map_fitted_uK.fits
- comparison_summary.txt

Example
-------
python cptg_cmb_visual_compare_v2.py ^
  --cmb-map COM_CMB_IQU-smica_2048_R3.00_full.fits ^
  --nside 256 ^
  --out cptg_cmb_visual_compare_v2

Optional inpainted run
----------------------
python cptg_cmb_visual_compare_v2.py ^
  --cmb-map COM_CMB_IQU-smica_2048_R3.00_full.fits ^
  --use-inpainted ^
  --nside 256 ^
  --out cptg_cmb_visual_compare_v2_inpainted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits


def require_camb():
    try:
        import camb
        return camb
    except Exception as exc:
        raise SystemExit(
            "This script requires CAMB. Install it in your active environment with:\n"
            "    conda install -c conda-forge camb -y\n"
            f"Original import error: {exc}"
        )


def get_field_names(path: str) -> List[str]:
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "columns") and hdu.columns is not None:
                names = list(hdu.columns.names)
                if names:
                    return [str(n).strip() for n in names]
    return []


def get_field_index(path: str, field_name: str) -> int:
    names = get_field_names(path)
    lookup = {name.upper(): i for i, name in enumerate(names)}
    key = field_name.upper()
    if key not in lookup:
        raise KeyError(f"Field {field_name} not found in {path}. Available fields: {names}")
    return int(lookup[key])


def has_field(path: str, field_name: str) -> bool:
    try:
        get_field_index(path, field_name)
        return True
    except Exception:
        return False


def read_healpix_field(path: str, field_name: str) -> np.ndarray:
    idx = get_field_index(path, field_name)
    return hp.read_map(path, field=idx, dtype=np.float64)


def finite_map(m: np.ndarray) -> np.ndarray:
    out = np.asarray(m, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    bad |= np.isclose(out, hp.UNSEEN)
    out[bad] = np.nan
    return out


def robust_std(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    lo, hi = np.nanpercentile(vals, [0.1, 99.9])
    clipped = vals[(vals >= lo) & (vals <= hi)]
    if clipped.size == 0:
        clipped = vals
    return float(np.nanstd(clipped))


def convert_temperature_field_to_uK(raw_temp: np.ndarray) -> tuple[np.ndarray, float, str, float]:
    raw = finite_map(raw_temp)
    sigma = robust_std(raw)
    if not np.isfinite(sigma):
        raise ValueError("Cannot infer temperature units: non-finite raw robust RMS.")
    if sigma < 1.0e-2:
        return raw * 1.0e6, 1.0e6, "K_CMB_to_uK", sigma
    if sigma < 10.0:
        return raw * 1.0e3, 1.0e3, "mK_CMB_to_uK", sigma
    return raw.copy(), 1.0, "uK_native_or_already_scaled", sigma


def select_temperature_field(path: str, use_inpainted: bool) -> str:
    if use_inpainted and has_field(path, "I_STOKES_INP"):
        return "I_STOKES_INP"
    if has_field(path, "I_STOKES"):
        return "I_STOKES"
    for candidate in ["TEMPERATURE", "TEMP", "MAP", "SIGNAL"]:
        if has_field(path, candidate):
            return candidate
    raise KeyError(f"No usable temperature field in {path}. Available fields: {get_field_names(path)}")


def select_mask(path: str, use_inpainted: bool, mask_source_map: str | None) -> tuple[np.ndarray, str, str]:
    if use_inpainted and has_field(path, "TMASKINP"):
        return finite_map(read_healpix_field(path, "TMASKINP")), "TMASKINP", path
    if has_field(path, "TMASK"):
        return finite_map(read_healpix_field(path, "TMASK")), "TMASK", path

    if mask_source_map and Path(mask_source_map).exists():
        if use_inpainted and has_field(mask_source_map, "TMASKINP"):
            return finite_map(read_healpix_field(mask_source_map, "TMASKINP")), "TMASKINP", mask_source_map
        if has_field(mask_source_map, "TMASK"):
            return finite_map(read_healpix_field(mask_source_map, "TMASK")), "TMASK", mask_source_map

    temp_field = select_temperature_field(path, use_inpainted)
    temp = finite_map(read_healpix_field(path, temp_field))
    nside = hp.get_nside(temp)
    return np.ones(hp.nside2npix(nside), dtype=np.float64), "ALL_SKY_NO_MASK_AVAILABLE", path


def downgrade_map_and_mask(temp_uk: np.ndarray, mask: np.ndarray, nside_out: int, mask_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if hp.get_nside(temp_uk) == nside_out:
        temp_low = temp_uk.copy()
    else:
        temp_low = hp.ud_grade(temp_uk, nside_out=nside_out, power=0)

    if hp.get_nside(mask) == nside_out:
        mask_frac = mask.astype(float).copy()
    else:
        mask_frac = hp.ud_grade(mask.astype(float), nside_out=nside_out, power=0)

    valid = np.isfinite(temp_low) & (mask_frac >= mask_threshold)
    return temp_low, valid


def remove_monopole_dipole_masked(m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    work = np.full_like(m, hp.UNSEEN, dtype=np.float64)
    work[valid] = m[valid]
    cleaned = hp.remove_dipole(work, fitval=False, copy=True)
    return finite_map(cleaned)


def camb_tt_cl(
    lmax: int,
    *,
    H0: float,
    ombh2: float,
    omch2: float,
    ns: float,
    Neff: float,
    As: float,
    tau: float,
) -> np.ndarray:
    camb = require_camb()
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0, mnu=0.06, nnu=Neff, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    if hasattr(pars, "Alens"):
        pars.Alens = 1.0
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, lmax=lmax, CMB_unit="muK", raw_cl=True)
    cl_tt = powers["total"][: lmax + 1, 0].astype(np.float64)
    cl_tt[0:2] = 0.0
    return cl_tt


def build_phase_locked_map(
    alm_obs: np.ndarray,
    cl_obs: np.ndarray,
    cl_target: np.ndarray,
    nside: int,
    lmax: int,
) -> np.ndarray:
    eps = 1.0e-30
    scale = np.sqrt(np.maximum(cl_target[: lmax + 1], 0.0) / np.maximum(cl_obs[: lmax + 1], eps))
    scale[0:2] = 0.0

    alm_new = alm_obs.copy()
    for ell in range(lmax + 1):
        factor = scale[ell]
        for m in range(ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            alm_new[idx] *= factor

    return hp.alm2map(alm_new, nside=nside, lmax=lmax)


def write_masked_map_fits(path: Path, m: np.ndarray) -> None:
    out = np.asarray(m, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    out[bad] = hp.UNSEEN
    hp.write_map(str(path), out, overwrite=True, dtype=np.float64)


def common_map_scale(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    vals = np.concatenate([a[valid], b[valid]])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 300.0
    vmax = float(np.nanpercentile(np.abs(vals), 99.0))
    return max(vmax, 1.0)


def residual_scale(r: np.ndarray, valid: np.ndarray) -> float:
    vals = r[valid]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 100.0
    vmax = float(np.nanpercentile(np.abs(vals), 99.0))
    return max(vmax, 1.0)


def masked_for_display(m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full_like(m, hp.UNSEEN, dtype=np.float64)
    good = valid & np.isfinite(m)
    out[good] = m[good]
    return out


def save_single_map(
    path: Path,
    m: np.ndarray,
    title: str,
    unit: str,
    cmap: str,
    minv: float,
    maxv: float,
) -> None:
    plt.figure(figsize=(10, 5.8))
    hp.mollview(
        m,
        fig=plt.gcf().number,
        title=title,
        unit=unit,
        cmap=cmap,
        min=minv,
        max=maxv,
        cbar=True,
        hold=True,
    )
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_side_by_side(
    path: Path,
    obs_disp: np.ndarray,
    cptg_disp: np.ndarray,
    resid_disp: np.ndarray,
    map_vmax: float,
    resid_vmax: float,
) -> None:
    fig = plt.figure(figsize=(16, 10))

    hp.mollview(
        obs_disp,
        fig=fig.number,
        sub=(3, 1, 1),
        title="Observed CMB map (cleaned, masked)",
        unit="uK",
        cmap="coolwarm",
        min=-map_vmax,
        max=map_vmax,
        cbar=True,
        hold=True,
    )

    hp.mollview(
        cptg_disp,
        fig=fig.number,
        sub=(3, 1, 2),
        title="CPTG comparison map from same phase scaffold",
        unit="uK",
        cmap="coolwarm",
        min=-map_vmax,
        max=map_vmax,
        cbar=True,
        hold=True,
    )

    hp.mollview(
        resid_disp,
        fig=fig.number,
        sub=(3, 1, 3),
        title="Residual: observed minus CPTG",
        unit="uK",
        cmap="PRGn",
        min=-resid_vmax,
        max=resid_vmax,
        cbar=True,
        hold=True,
    )

    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def masked_corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    x = a[valid]
    y = b[valid]
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(x * x) * np.nansum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.nansum(x * y) / denom)


def rms(x: np.ndarray, valid: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(x[valid] ** 2)))


def normalize_template_shape(t: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(t, dtype=np.float64).copy()
    mean = np.nanmean(out[valid])
    std = np.nanstd(out[valid])
    if not np.isfinite(std) or std == 0:
        raise ValueError("Template has zero or invalid standard deviation on valid sky.")
    return (out - mean) / std


def fit_amplitude_and_offset(obs: np.ndarray, template: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Produce the amplitude+offset-fitted comparison map used by the envelope tests.
    This is for visual comparison only; the raw CPTG comparison map is also saved.
    """
    use = valid & np.isfinite(obs) & np.isfinite(template)
    shape = normalize_template_shape(template, use)
    x = shape[use]
    y = obs[use]
    A = np.vstack([x, np.ones_like(x)]).T
    amp, offset = np.linalg.lstsq(A, y, rcond=None)[0]
    fitted = amp * shape + offset
    return fitted, float(amp), float(offset)


def compose_pngs(path: Path, image_paths: list[Path], titles: list[str], vertical: bool = True) -> None:
    """
    Robust side-by-side/stacked composite that avoids healpy subplot/colorbar issues.
    """
    imgs = [plt.imread(str(p)) for p in image_paths]
    n = len(imgs)

    if vertical:
        fig, axes = plt.subplots(n, 1, figsize=(13, 5.2 * n))
    else:
        fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.5))

    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scatter_plot(path: Path, obs: np.ndarray, cptg: np.ndarray, valid: np.ndarray, label: str = "CPTG") -> tuple[float, float]:
    x = obs[valid]
    y = cptg[valid]
    corr = masked_corr(obs, cptg, valid)
    rms_resid = rms(obs - cptg, valid)

    if x.size > 60000:
        rng = np.random.default_rng(12345)
        idx = rng.choice(x.size, size=60000, replace=False)
        x = x[idx]
        y = y[idx]

    plt.figure(figsize=(7.5, 7))
    plt.scatter(x, y, s=1, alpha=0.15)
    lo = float(np.nanpercentile(np.concatenate([x, y]), 1))
    hi = float(np.nanpercentile(np.concatenate([x, y]), 99))
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("Observed CMB map [uK]")
    plt.ylabel(f"{label} comparison map [uK]")
    plt.title(f"Observed vs {label} valid-sky pixels\ncorr={corr:.6f}, residual RMS={rms_resid:.6f} uK")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

    return corr, rms_resid


def save_power_compare(path: Path, cl_obs: np.ndarray, cl_cptg_target: np.ndarray, cl_planck: np.ndarray, ell_max_plot: int) -> None:
    ell = np.arange(len(cl_obs))
    good = (ell >= 2) & (ell <= ell_max_plot)

    def d_ell(cl):
        return ell * (ell + 1.0) * cl / (2.0 * np.pi)

    plt.figure(figsize=(8.5, 6))
    plt.plot(ell[good], d_ell(cl_obs)[good], label="Observed phase-scaffold Cl")
    plt.plot(ell[good], d_ell(cl_cptg_target)[good], label="CPTG target Cl")
    plt.plot(ell[good], d_ell(cl_planck)[good], label="Planck baseline Cl")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\ell(\ell+1)C_\ell / 2\pi\ [\mu K^2]$")
    plt.title("Angular power comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual CMB vs CPTG comparison from the same data source, v2.")
    parser.add_argument("--cmb-map", required=True, help="Observed CMB FITS map, e.g. COM_CMB_IQU-smica_2048_R3.00_full.fits")
    parser.add_argument("--mask-source-map", default=None, help="Optional fallback map to supply TMASK/TMASKINP.")
    parser.add_argument("--use-inpainted", action="store_true", help="Use I_STOKES_INP / TMASKINP when available.")
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.99)
    parser.add_argument("--ell-max-plot", type=int, default=200)
    parser.add_argument("--out", default="cptg_cmb_visual_compare")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not Path(args.cmb_map).exists():
        raise SystemExit(f"CMB map not found: {args.cmb_map}")

    mask_source_map = args.mask_source_map
    if mask_source_map is None:
        mask_source_map = args.cmb_map

    available_fields = get_field_names(args.cmb_map)
    temperature_field = select_temperature_field(args.cmb_map, args.use_inpainted)
    raw_temp = finite_map(read_healpix_field(args.cmb_map, temperature_field))
    temp_uk, temp_scale, temp_unit, raw_std = convert_temperature_field_to_uK(raw_temp)

    mask_raw, mask_field, mask_source_used = select_mask(args.cmb_map, args.use_inpainted, mask_source_map)
    obs_low, valid = downgrade_map_and_mask(temp_uk, mask_raw, args.nside, args.mask_threshold)
    obs_clean = remove_monopole_dipole_masked(obs_low, valid)

    obs_filled = np.zeros_like(obs_clean)
    good = valid & np.isfinite(obs_clean)
    obs_filled[good] = obs_clean[good]

    lmax = 3 * args.nside - 1
    alm_obs = hp.map2alm(obs_filled, lmax=lmax, iter=0)
    cl_obs = hp.alm2cl(alm_obs)
    cl_obs[0:2] = 0.0

    cl_cptg = camb_tt_cl(
        lmax,
        H0=67.4777967351,
        ombh2=0.022527857494,
        omch2=0.117685841620526,
        ns=0.968584073464,
        Neff=2.968584073464,
        As=2.136283004441e-9,
        tau=0.058930875934,
    )

    cl_planck = camb_tt_cl(
        lmax,
        H0=67.36,
        ombh2=0.02237,
        omch2=0.1200,
        ns=0.9649,
        Neff=3.046,
        As=2.100549e-9,
        tau=0.0544,
    )

    cptg_map = build_phase_locked_map(alm_obs, cl_obs, cl_cptg, args.nside, lmax)
    residual = obs_clean - cptg_map

    cptg_fitted_map, fitted_amp, fitted_offset = fit_amplitude_and_offset(obs_clean, cptg_map, valid)
    fitted_residual = obs_clean - cptg_fitted_map

    obs_disp = masked_for_display(obs_clean, valid)
    cptg_disp = masked_for_display(cptg_map, valid)
    resid_disp = masked_for_display(residual, valid)
    cptg_fitted_disp = masked_for_display(cptg_fitted_map, valid)
    fitted_resid_disp = masked_for_display(fitted_residual, valid)

    map_vmax = common_map_scale(obs_clean, cptg_map, valid)
    fitted_map_vmax = common_map_scale(obs_clean, cptg_fitted_map, valid)
    resid_vmax = residual_scale(residual, valid)
    fitted_resid_vmax = residual_scale(fitted_residual, valid)

    save_single_map(
        out / "observed_cmb_map_uK.png",
        obs_disp,
        "Observed CMB map (cleaned, masked)",
        "uK",
        "coolwarm",
        -map_vmax,
        map_vmax,
    )

    save_single_map(
        out / "cptg_comparison_map_uK.png",
        cptg_disp,
        "CPTG comparison map from same phase scaffold",
        "uK",
        "coolwarm",
        -map_vmax,
        map_vmax,
    )

    save_single_map(
        out / "residual_observed_minus_cptg_uK.png",
        resid_disp,
        "Residual: observed minus CPTG",
        "uK",
        "PRGn",
        -resid_vmax,
        resid_vmax,
    )

    save_single_map(
        out / "cptg_comparison_map_fitted_uK.png",
        cptg_fitted_disp,
        "CPTG comparison map, amplitude+offset fitted",
        "uK",
        "coolwarm",
        -fitted_map_vmax,
        fitted_map_vmax,
    )

    save_single_map(
        out / "residual_observed_minus_cptg_fitted_uK.png",
        fitted_resid_disp,
        "Residual: observed minus fitted CPTG",
        "uK",
        "PRGn",
        -fitted_resid_vmax,
        fitted_resid_vmax,
    )

    # Keep the original healpy side-by-side output, then overwrite it with a robust PNG composite.
    save_side_by_side(
        out / "cmb_vs_cptg_side_by_side_healpy_raw.png",
        obs_disp,
        cptg_disp,
        resid_disp,
        map_vmax,
        resid_vmax,
    )

    compose_pngs(
        out / "cmb_vs_cptg_side_by_side.png",
        [
            out / "observed_cmb_map_uK.png",
            out / "cptg_comparison_map_uK.png",
            out / "residual_observed_minus_cptg_uK.png",
        ],
        [
            "Observed CMB",
            "Raw CPTG comparison map",
            "Observed minus raw CPTG",
        ],
        vertical=True,
    )

    compose_pngs(
        out / "cmb_vs_cptg_side_by_side_fitted.png",
        [
            out / "observed_cmb_map_uK.png",
            out / "cptg_comparison_map_fitted_uK.png",
            out / "residual_observed_minus_cptg_fitted_uK.png",
        ],
        [
            "Observed CMB",
            "Amplitude+offset-fitted CPTG comparison map",
            "Observed minus fitted CPTG",
        ],
        vertical=True,
    )

    corr, resid_rms = save_scatter_plot(
        out / "cmb_vs_cptg_scatter.png",
        obs_clean,
        cptg_map,
        valid,
        label="raw CPTG",
    )

    fitted_corr, fitted_resid_rms = save_scatter_plot(
        out / "cmb_vs_cptg_fitted_scatter.png",
        obs_clean,
        cptg_fitted_map,
        valid,
        label="fitted CPTG",
    )

    save_power_compare(
        out / "cmb_vs_cptg_power_compare.png",
        cl_obs,
        cl_cptg,
        cl_planck,
        min(args.ell_max_plot, lmax),
    )

    write_masked_map_fits(out / "cptg_comparison_map_uK.fits", cptg_map)
    write_masked_map_fits(out / "cptg_comparison_map_fitted_uK.fits", cptg_fitted_map)

    summary = {
        "cmb_map": args.cmb_map,
        "available_fields": available_fields,
        "temperature_field": temperature_field,
        "temperature_unit_inference": temp_unit,
        "temperature_scale_applied": temp_scale,
        "raw_temperature_robust_std": raw_std,
        "mask_field": mask_field,
        "mask_source_used": mask_source_used,
        "nside": args.nside,
        "mask_threshold": args.mask_threshold,
        "valid_pixels": int(np.sum(valid)),
        "valid_sky_fraction": float(np.sum(valid) / len(valid)),
        "map_common_vmax_uK": map_vmax,
        "residual_vmax_uK": resid_vmax,
        "observed_rms_uK": rms(obs_clean, valid),
        "cptg_raw_rms_uK": rms(cptg_map, valid),
        "cptg_raw_residual_rms_uK": resid_rms,
        "observed_vs_cptg_raw_corr": corr,
        "cptg_fitted_rms_uK": rms(cptg_fitted_map, valid),
        "cptg_fitted_residual_rms_uK": fitted_resid_rms,
        "observed_vs_cptg_fitted_corr": fitted_corr,
        "fitted_amp_uK_per_template_sigma": fitted_amp,
        "fitted_offset_uK": fitted_offset,
    }

    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("Visual CMB vs CPTG comparison summary")
    lines.append("=" * 38)
    lines.append("")
    lines.append(f"CMB map: {args.cmb_map}")
    lines.append(f"Temperature field: {temperature_field}")
    lines.append(f"Temperature unit inference: {temp_unit}")
    lines.append(f"Mask field: {mask_field}")
    lines.append(f"Mask source: {mask_source_used}")
    lines.append(f"Nside: {args.nside}")
    lines.append(f"Mask threshold: {args.mask_threshold}")
    lines.append(f"Valid sky fraction: {summary['valid_sky_fraction']:.6f}")
    lines.append("")
    lines.append("Comparison metrics")
    lines.append("------------------")
    lines.append(f"Observed RMS: {summary['observed_rms_uK']:.6f} uK")
    lines.append(f"Raw CPTG RMS: {summary['cptg_raw_rms_uK']:.6f} uK")
    lines.append(f"Raw residual RMS: {summary['cptg_raw_residual_rms_uK']:.6f} uK")
    lines.append(f"Observed vs raw CPTG correlation: {summary['observed_vs_cptg_raw_corr']:.6f}")
    lines.append(f"Fitted CPTG RMS: {summary['cptg_fitted_rms_uK']:.6f} uK")
    lines.append(f"Fitted residual RMS: {summary['cptg_fitted_residual_rms_uK']:.6f} uK")
    lines.append(f"Observed vs fitted CPTG correlation: {summary['observed_vs_cptg_fitted_corr']:.6f}")
    lines.append(f"Fitted amplitude: {summary['fitted_amp_uK_per_template_sigma']:.6f} uK/template-sigma")
    lines.append(f"Fitted offset: {summary['fitted_offset_uK']:.6f} uK")
    lines.append("")
    lines.append("Outputs")
    lines.append("-------")
    lines.append("observed_cmb_map_uK.png")
    lines.append("cptg_comparison_map_uK.png")
    lines.append("residual_observed_minus_cptg_uK.png")
    lines.append("cmb_vs_cptg_side_by_side.png")
    lines.append("cmb_vs_cptg_side_by_side_fitted.png")
    lines.append("cmb_vs_cptg_scatter.png")
    lines.append("cmb_vs_cptg_fitted_scatter.png")
    lines.append("cmb_vs_cptg_power_compare.png")
    lines.append("cptg_comparison_map_uK.fits")
    lines.append("cptg_comparison_map_fitted_uK.fits")

    (out / "comparison_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("")
    print(f"Outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
