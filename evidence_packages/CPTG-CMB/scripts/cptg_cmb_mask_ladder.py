#!/usr/bin/env python3
"""
CPTG CMB mask / sky-fraction ladder.

Purpose:
    Test whether the tiny CPTG-minus-Planck envelope gap is sensitive to
    sky selection: Planck temperature mask threshold, Galactic latitude cuts,
    and their combination.

Main question:
    Is the sub-microkelvin Planck-over-CPTG edge stable under mask/sky-fraction
    changes, or is it partly a mask-boundary / foreground-cleaning /
    sky-selection residual?

Default auto-detected Planck component files in the current directory:
    COM_CMB_IQU-smica_2048_R3.00_full.fits
    COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits
    COM_CMB_IQU-nilc_2048_R3.00_full.fits
    COM_CMB_IQU-sevem_2048_R3.01_full.fits
    COM_CMB_IQU-sevem_2048_R3.00_full.fits
    COM_CMB_IQU-commander_2048_R3.00_full.fits

Default mask threshold ladder:
    0.50,0.70,0.80,0.90,0.95,0.99

Default Galactic cuts:
    none,10,20,30,40 degrees

Examples:
    python cptg_cmb_mask_ladder.py ^
      --nside 256 ^
      --out cptg_cmb_mask_ladder

    python cptg_cmb_mask_ladder.py ^
      --nside 256 ^
      --use-inpainted ^
      --out cptg_cmb_mask_ladder_inpainted

    python cptg_cmb_mask_ladder.py ^
      --nside 256 ^
      --bands 2-30,2-40,2-64,2-95 ^
      --out cptg_cmb_mask_ladder_bands

Outputs:
    mask_ladder_summary.csv
    mask_ladder_delta.csv
    mask_ladder_stability.csv
    mask_ladder_report.txt
    gap_by_mask_threshold.png
    gap_by_galactic_cut.png
    residual_fraction_by_sky_fraction.png

If --bands is supplied:
    mask_ladder_band_summary.csv
    mask_ladder_band_delta.csv
    gap_by_mask_band.png

Interpretation boundary:
    This is a phase-locked angular-power / transport-envelope comparison.
    It does not establish native CPTG prediction of exact CMB pixel phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits


DEFAULT_MAP_CANDIDATES = {
    "SMICA": ["COM_CMB_IQU-smica_2048_R3.00_full.fits"],
    "SMICA_noSZ": ["COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"],
    "NILC": ["COM_CMB_IQU-nilc_2048_R3.00_full.fits"],
    "SEVEM": [
        "COM_CMB_IQU-sevem_2048_R3.01_full.fits",
        "COM_CMB_IQU-sevem_2048_R3.00_full.fits",
    ],
    "COMMANDER": ["COM_CMB_IQU-commander_2048_R3.00_full.fits"],
}


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


def parse_map_args(map_args: List[str] | None) -> Dict[str, str]:
    if not map_args:
        found = {}
        for label, candidates in DEFAULT_MAP_CANDIDATES.items():
            for fname in candidates:
                if Path(fname).exists():
                    found[label] = fname
                    break
        return found

    maps = {}
    for item in map_args:
        if "=" not in item:
            raise SystemExit(f"Invalid --map entry: {item}. Use LABEL=path.fits")
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip().strip('"')
        if not label or not path:
            raise SystemExit(f"Invalid --map entry: {item}. Use LABEL=path.fits")
        maps[label] = path
    return maps


def parse_float_list(s: str) -> list[float]:
    vals = []
    for item in s.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise SystemExit("No float values supplied.")
    return vals


def parse_galactic_cuts(s: str) -> list[Optional[float]]:
    vals: list[Optional[float]] = []
    for item in s.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in ["none", "no", "all", "0", "0.0"]:
            vals.append(None)
        else:
            vals.append(float(item))
    if not vals:
        raise SystemExit("No Galactic cut values supplied.")
    return vals


def cut_label(cut: Optional[float]) -> str:
    if cut is None:
        return "no_galactic_cut"
    if abs(cut - round(cut)) < 1.0e-9:
        return f"abs_b_gt_{int(round(cut))}deg"
    return f"abs_b_gt_{cut:g}deg"


def parse_bands(s: Optional[str]) -> list[tuple[int, int]]:
    if not s:
        return []
    bands = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise SystemExit(f"Invalid band '{item}'. Use format 2-30,2-64")
        a, b = item.split("-", 1)
        lo = int(a)
        hi = int(b)
        if lo < 2 or hi < lo:
            raise SystemExit(f"Invalid band '{item}'. Require 2 <= lo <= hi.")
        bands.append((lo, hi))
    return bands


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


def select_mask(path: str, use_inpainted: bool, mask_source_map: str | None) -> Tuple[np.ndarray, str, str]:
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


def downgrade_map(m: np.ndarray, nside_out: int) -> np.ndarray:
    if hp.get_nside(m) == nside_out:
        return m.copy()
    return hp.ud_grade(m, nside_out=nside_out, power=0)


def remove_monopole_dipole_masked(m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    work = np.full_like(m, hp.UNSEEN, dtype=np.float64)
    work[valid] = m[valid]
    cleaned = hp.remove_dipole(work, fitval=False, copy=True)
    return finite_map(cleaned)


def normalize_template_shape(t: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(t, dtype=np.float64).copy()
    mean = np.nanmean(out[valid])
    std = np.nanstd(out[valid])
    if not np.isfinite(std) or std == 0:
        raise ValueError("Template has zero or invalid standard deviation on valid sky.")
    return (out - mean) / std


def fit_amplitude_and_offset(obs: np.ndarray, tmpl: np.ndarray, valid: np.ndarray) -> Tuple[float, float]:
    x = tmpl[valid]
    y = obs[valid]
    A = np.vstack([x, np.ones_like(x)]).T
    amp, offset = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(amp), float(offset)


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


def evaluate_template(obs_map: np.ndarray, template_map: np.ndarray, valid: np.ndarray) -> dict:
    use = valid & np.isfinite(obs_map) & np.isfinite(template_map)
    template_shape = normalize_template_shape(template_map, use)
    amp, offset = fit_amplitude_and_offset(obs_map, template_shape, use)
    fit_map = amp * template_shape + offset
    residual = obs_map - fit_map
    obs_rms = rms(obs_map, use)
    fit_rms = rms(fit_map, use)
    res_rms = rms(residual, use)
    return {
        "masked_corr": masked_corr(obs_map, fit_map, use),
        "observed_rms_uK": obs_rms,
        "fit_rms_uK": fit_rms,
        "residual_rms_uK": res_rms,
        "residual_fraction": res_rms / obs_rms if obs_rms != 0 else np.nan,
        "amp_uK_per_template_sigma": amp,
        "offset_uK": offset,
        "valid_pixels": int(np.sum(use)),
        "valid_sky_fraction": float(np.sum(use) / len(use)),
    }


def camb_tt_cl(lmax: int, *, H0: float, ombh2: float, omch2: float, ns: float, Neff: float, As: float, tau: float) -> np.ndarray:
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


def build_phase_locked_map(alm_obs: np.ndarray, cl_obs: np.ndarray, cl_target: np.ndarray, nside: int, lmax: int) -> np.ndarray:
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


def bandlimit_alm(alm: np.ndarray, lmax: int, ell_min: int, ell_max: int, scale_by_ell: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.zeros_like(alm)
    for ell in range(ell_min, ell_max + 1):
        factor = 1.0 if scale_by_ell is None else scale_by_ell[ell]
        for m in range(ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            out[idx] = alm[idx] * factor
    return out


def build_observed_band_map(alm_obs: np.ndarray, nside: int, lmax: int, ell_min: int, ell_max: int) -> np.ndarray:
    alm_band = bandlimit_alm(alm_obs, lmax, ell_min, ell_max)
    return hp.alm2map(alm_band, nside=nside, lmax=lmax)


def build_band_template_map(alm_obs: np.ndarray, cl_obs: np.ndarray, cl_target: np.ndarray, nside: int, lmax: int, ell_min: int, ell_max: int) -> np.ndarray:
    eps = 1.0e-30
    scale = np.sqrt(np.maximum(cl_target[: lmax + 1], 0.0) / np.maximum(cl_obs[: lmax + 1], eps))
    scale[0:2] = 0.0
    alm_band = bandlimit_alm(alm_obs, lmax, ell_min, ell_max, scale_by_ell=scale)
    return hp.alm2map(alm_band, nside=nside, lmax=lmax)


def galactic_cut_mask(nside: int, cut_deg: Optional[float]) -> np.ndarray:
    if cut_deg is None:
        return np.ones(hp.nside2npix(nside), dtype=bool)
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    lat_deg = 90.0 - np.degrees(theta)
    return np.abs(lat_deg) > cut_deg


def make_gap_by_threshold(path: Path, delta: pd.DataFrame) -> None:
    if delta.empty:
        return
    grouped = delta.groupby("mask_threshold")["cptg_minus_planck_residual_fraction"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(grouped["mask_threshold"], grouped["cptg_minus_planck_residual_fraction"], marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Mask threshold")
    plt.ylabel("Mean CPTG minus Planck residual fraction")
    plt.title("Gap by mask threshold")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_gap_by_cut(path: Path, delta: pd.DataFrame) -> None:
    if delta.empty:
        return
    grouped = delta.groupby("galactic_cut_label")["cptg_minus_planck_residual_fraction"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(grouped["galactic_cut_label"], grouped["cptg_minus_planck_residual_fraction"])
    plt.axhline(0.0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean CPTG minus Planck residual fraction")
    plt.title("Gap by Galactic latitude cut")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_gap_by_sky_fraction(path: Path, delta: pd.DataFrame) -> None:
    if delta.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(delta["valid_sky_fraction"], delta["cptg_minus_planck_residual_fraction"], alpha=0.8)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Valid sky fraction")
    plt.ylabel("CPTG minus Planck residual fraction")
    plt.title("Gap versus usable sky fraction")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_band_plot(path: Path, band_delta: pd.DataFrame) -> None:
    if band_delta.empty:
        return
    grouped = band_delta.groupby(["ell_band", "mask_threshold"])["cptg_minus_planck_residual_fraction"].mean().reset_index()
    plt.figure(figsize=(11, 7))
    for band in sorted(grouped["ell_band"].unique(), key=lambda x: int(str(x).split("-")[1])):
        sub = grouped[grouped["ell_band"] == band]
        plt.plot(sub["mask_threshold"], sub["cptg_minus_planck_residual_fraction"], marker="o", label=f"ell={band}")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Mask threshold")
    plt.ylabel("Mean CPTG minus Planck residual fraction")
    plt.title("Band-limited gap by mask threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPTG CMB mask / sky-fraction ladder.")
    parser.add_argument("--map", action="append", default=None, help="Map as LABEL=path.fits. May be repeated.")
    parser.add_argument("--mask-source-map", default=None)
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--thresholds", default="0.50,0.70,0.80,0.90,0.95,0.99")
    parser.add_argument("--galactic-cuts", default="none,10,20,30,40")
    parser.add_argument("--min-sky-fraction", type=float, default=0.10)
    parser.add_argument("--use-inpainted", action="store_true")
    parser.add_argument("--bands", default=None, help="Optional bands, e.g. 2-30,2-40,2-64,2-95")
    parser.add_argument("--out", default="cptg_cmb_mask_ladder")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    maps = parse_map_args(args.map)
    if not maps:
        raise SystemExit("No map files found. Put Planck FITS files in this directory or pass --map LABEL=path.fits.")

    mask_source_map = args.mask_source_map
    if mask_source_map is None:
        if "SMICA" in maps and Path(maps["SMICA"]).exists():
            mask_source_map = maps["SMICA"]
        elif Path("COM_CMB_IQU-smica_2048_R3.00_full.fits").exists():
            mask_source_map = "COM_CMB_IQU-smica_2048_R3.00_full.fits"

    thresholds = parse_float_list(args.thresholds)
    cuts = parse_galactic_cuts(args.galactic_cuts)
    bands = parse_bands(args.bands)
    max_band_ell = max([hi for _, hi in bands], default=0)
    lmax = max(3 * args.nside - 1, max_band_ell)

    print(f"Computing locked envelopes to lmax={lmax}")
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

    summary_rows = []
    delta_rows = []
    band_summary_rows = []
    band_delta_rows = []
    skipped_rows = []

    for component, path in maps.items():
        print("")
        print(f"Component: {component}")
        print(f"File: {path}")

        if not Path(path).exists():
            skipped_rows.append({"component": component, "path": path, "reason": "file_not_found"})
            print("  skipped: file not found")
            continue

        try:
            available_fields = get_field_names(path)
            temperature_field = select_temperature_field(path, args.use_inpainted)
            raw_temp = finite_map(read_healpix_field(path, temperature_field))
            temp_uk, temp_scale, temp_unit, raw_std = convert_temperature_field_to_uK(raw_temp)
            temp_low = downgrade_map(temp_uk, args.nside)

            mask_raw, mask_field, mask_source_used = select_mask(path, args.use_inpainted, mask_source_map)
            mask_low = downgrade_map(mask_raw, args.nside)

            for threshold in thresholds:
                threshold_valid = np.isfinite(temp_low) & np.isfinite(mask_low) & (mask_low >= threshold)

                for cut in cuts:
                    gcut = galactic_cut_mask(args.nside, cut)
                    valid = threshold_valid & gcut
                    sky_fraction = float(np.sum(valid) / len(valid))
                    c_label = cut_label(cut)
                    m_label = f"mask_ge_{threshold:.2f}_{c_label}"

                    if sky_fraction < args.min_sky_fraction:
                        skipped_rows.append(
                            {
                                "component": component,
                                "path": path,
                                "reason": f"sky_fraction_below_min:{m_label}:{sky_fraction:.6f}",
                            }
                        )
                        continue

                    print(f"  Mask {m_label}, sky_fraction={sky_fraction:.4f}")

                    obs_clean = remove_monopole_dipole_masked(temp_low, valid)
                    obs_filled = np.zeros_like(obs_clean)
                    good = valid & np.isfinite(obs_clean)
                    obs_filled[good] = obs_clean[good]

                    alm_obs = hp.map2alm(obs_filled, lmax=lmax, iter=0)
                    cl_obs = hp.alm2cl(alm_obs)
                    cl_obs[0:2] = 0.0

                    envelopes = {
                        "cptg_locked_pi_cmb": build_phase_locked_map(alm_obs, cl_obs, cl_cptg, args.nside, lmax),
                        "planck2018_baseline_lcdm": build_phase_locked_map(alm_obs, cl_obs, cl_planck, args.nside, lmax),
                        "component_self_spectrum_control": build_phase_locked_map(alm_obs, cl_obs, cl_obs.copy(), args.nside, lmax),
                    }

                    metrics_by_model = {}
                    for model, template in envelopes.items():
                        metrics = evaluate_template(obs_clean, template, valid)
                        metrics.update(
                            {
                                "component": component,
                                "model": model,
                                "mask_label": m_label,
                                "mask_threshold": threshold,
                                "galactic_cut_deg": -1.0 if cut is None else cut,
                                "galactic_cut_label": c_label,
                                "path": path,
                                "available_fields": "|".join(available_fields),
                                "temperature_field": temperature_field,
                                "temperature_unit_inference": temp_unit,
                                "temperature_scale_applied": temp_scale,
                                "raw_temperature_robust_std": raw_std,
                                "mask_field": mask_field,
                                "mask_source_used": mask_source_used,
                                "nside": args.nside,
                                "valid_sky_fraction": sky_fraction,
                            }
                        )
                        summary_rows.append(metrics)
                        metrics_by_model[model] = metrics

                    c = metrics_by_model["cptg_locked_pi_cmb"]
                    p = metrics_by_model["planck2018_baseline_lcdm"]
                    delta_rows.append(
                        {
                            "component": component,
                            "mask_label": m_label,
                            "mask_threshold": threshold,
                            "galactic_cut_deg": -1.0 if cut is None else cut,
                            "galactic_cut_label": c_label,
                            "cptg_residual_fraction": c["residual_fraction"],
                            "planck_residual_fraction": p["residual_fraction"],
                            "cptg_minus_planck_residual_fraction": c["residual_fraction"] - p["residual_fraction"],
                            "cptg_residual_rms_uK": c["residual_rms_uK"],
                            "planck_residual_rms_uK": p["residual_rms_uK"],
                            "cptg_minus_planck_residual_rms_uK": c["residual_rms_uK"] - p["residual_rms_uK"],
                            "cptg_corr": c["masked_corr"],
                            "planck_corr": p["masked_corr"],
                            "observed_rms_uK": c["observed_rms_uK"],
                            "valid_sky_fraction": sky_fraction,
                            "temperature_field": temperature_field,
                            "temperature_unit_inference": temp_unit,
                            "mask_field": mask_field,
                            "mask_source_used": mask_source_used,
                        }
                    )

                    if bands:
                        for ell_min, ell_max in bands:
                            band_label = f"{ell_min}-{ell_max}"
                            obs_band = build_observed_band_map(alm_obs, args.nside, lmax, ell_min, ell_max)

                            band_metrics = {}
                            for model, cl in [
                                ("cptg_locked_pi_cmb", cl_cptg),
                                ("planck2018_baseline_lcdm", cl_planck),
                                ("component_self_spectrum_control", cl_obs.copy()),
                            ]:
                                template_band = build_band_template_map(alm_obs, cl_obs, cl, args.nside, lmax, ell_min, ell_max)
                                metrics = evaluate_template(obs_band, template_band, valid)
                                metrics.update(
                                    {
                                        "component": component,
                                        "model": model,
                                        "ell_min": ell_min,
                                        "ell_max": ell_max,
                                        "ell_band": band_label,
                                        "mask_label": m_label,
                                        "mask_threshold": threshold,
                                        "galactic_cut_deg": -1.0 if cut is None else cut,
                                        "galactic_cut_label": c_label,
                                        "temperature_field": temperature_field,
                                        "temperature_unit_inference": temp_unit,
                                        "mask_field": mask_field,
                                        "mask_source_used": mask_source_used,
                                        "nside": args.nside,
                                        "valid_sky_fraction": sky_fraction,
                                    }
                                )
                                band_summary_rows.append(metrics)
                                band_metrics[model] = metrics

                            bc = band_metrics["cptg_locked_pi_cmb"]
                            bp = band_metrics["planck2018_baseline_lcdm"]
                            band_delta_rows.append(
                                {
                                    "component": component,
                                    "ell_min": ell_min,
                                    "ell_max": ell_max,
                                    "ell_band": band_label,
                                    "mask_label": m_label,
                                    "mask_threshold": threshold,
                                    "galactic_cut_deg": -1.0 if cut is None else cut,
                                    "galactic_cut_label": c_label,
                                    "cptg_residual_fraction": bc["residual_fraction"],
                                    "planck_residual_fraction": bp["residual_fraction"],
                                    "cptg_minus_planck_residual_fraction": bc["residual_fraction"] - bp["residual_fraction"],
                                    "cptg_residual_rms_uK": bc["residual_rms_uK"],
                                    "planck_residual_rms_uK": bp["residual_rms_uK"],
                                    "cptg_minus_planck_residual_rms_uK": bc["residual_rms_uK"] - bp["residual_rms_uK"],
                                    "cptg_corr": bc["masked_corr"],
                                    "planck_corr": bp["masked_corr"],
                                    "observed_band_rms_uK": bc["observed_rms_uK"],
                                    "valid_sky_fraction": sky_fraction,
                                    "temperature_field": temperature_field,
                                    "temperature_unit_inference": temp_unit,
                                    "mask_field": mask_field,
                                    "mask_source_used": mask_source_used,
                                }
                            )

            print(f"  done: temp={temperature_field}, unit={temp_unit}, mask={mask_field}, mask_source={mask_source_used}")

        except Exception as exc:
            print(f"  skipped after analysis error: {exc}")
            skipped_rows.append({"component": component, "path": path, "reason": f"analysis_failed: {exc}"})

    summary = pd.DataFrame(summary_rows)
    delta = pd.DataFrame(delta_rows)
    band_summary = pd.DataFrame(band_summary_rows)
    band_delta = pd.DataFrame(band_delta_rows)
    skipped = pd.DataFrame(skipped_rows)

    summary.to_csv(out / "mask_ladder_summary.csv", index=False)
    delta.to_csv(out / "mask_ladder_delta.csv", index=False)
    band_summary.to_csv(out / "mask_ladder_band_summary.csv", index=False)
    band_delta.to_csv(out / "mask_ladder_band_delta.csv", index=False)
    skipped.to_csv(out / "skipped_masks.csv", index=False)

    stability_rows = []
    if not delta.empty:
        for threshold in sorted(delta["mask_threshold"].unique()):
            sub = delta[delta["mask_threshold"] == threshold]
            stability_rows.append(
                {
                    "scope": f"threshold_{threshold:.2f}_all_cuts_components",
                    "n_rows": len(sub),
                    "mean_sky_fraction": sub["valid_sky_fraction"].mean(),
                    "mean_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].mean(),
                    "std_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].std(ddof=1),
                    "min_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].min(),
                    "max_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].max(),
                    "span_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].max() - sub["cptg_minus_planck_residual_fraction"].min(),
                    "mean_gap_rms_uK": sub["cptg_minus_planck_residual_rms_uK"].mean(),
                    "std_gap_rms_uK": sub["cptg_minus_planck_residual_rms_uK"].std(ddof=1),
                    "span_gap_rms_uK": sub["cptg_minus_planck_residual_rms_uK"].max() - sub["cptg_minus_planck_residual_rms_uK"].min(),
                    "planck_ahead_all_rows": bool((sub["cptg_minus_planck_residual_fraction"] > 0).all()),
                }
            )

        for c_label in delta["galactic_cut_label"].unique():
            sub = delta[delta["galactic_cut_label"] == c_label]
            stability_rows.append(
                {
                    "scope": f"galactic_cut_{c_label}_all_thresholds_components",
                    "n_rows": len(sub),
                    "mean_sky_fraction": sub["valid_sky_fraction"].mean(),
                    "mean_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].mean(),
                    "std_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].std(ddof=1),
                    "min_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].min(),
                    "max_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].max(),
                    "span_gap_residual_fraction": sub["cptg_minus_planck_residual_fraction"].max() - sub["cptg_minus_planck_residual_fraction"].min(),
                    "mean_gap_rms_uK": sub["cptg_minus_planck_residual_rms_uK"].mean(),
                    "std_gap_rms_uK": sub["cptg_minus_planck_residual_rms_uK"].std(ddof=1),
                    "span_gap_rms_uK": sub["cptg_minus_planck_residual_rms_uK"].max() - sub["cptg_minus_planck_residual_rms_uK"].min(),
                    "planck_ahead_all_rows": bool((sub["cptg_minus_planck_residual_fraction"] > 0).all()),
                }
            )

    stability = pd.DataFrame(stability_rows)
    stability.to_csv(out / "mask_ladder_stability.csv", index=False)

    make_gap_by_threshold(out / "gap_by_mask_threshold.png", delta)
    make_gap_by_cut(out / "gap_by_galactic_cut.png", delta)
    make_gap_by_sky_fraction(out / "residual_fraction_by_sky_fraction.png", delta)
    make_band_plot(out / "gap_by_mask_band.png", band_delta)

    report = []
    report.append("CPTG CMB mask / sky-fraction ladder report")
    report.append("=" * 47)
    report.append("")
    report.append(f"Nside: {args.nside}")
    report.append(f"Requested inpainted fields: {args.use_inpainted}")
    report.append(f"Mask source map: {mask_source_map}")
    report.append(f"Mask thresholds: {', '.join([f'{x:.2f}' for x in thresholds])}")
    report.append(f"Galactic cuts: {', '.join([cut_label(x) for x in cuts])}")
    report.append(f"Minimum sky fraction: {args.min_sky_fraction}")
    report.append(f"Bands: {args.bands if args.bands else 'not requested'}")
    report.append("")

    if delta.empty:
        report.append("No mask configurations completed.")
    else:
        report.append("Mean full-band gap by mask threshold:")
        for threshold in sorted(delta["mask_threshold"].unique()):
            sub = delta[delta["mask_threshold"] == threshold]
            report.append(
                f"  threshold={threshold:.2f}: mean_sky={sub['valid_sky_fraction'].mean():.6f}, "
                f"mean_delta={sub['cptg_minus_planck_residual_fraction'].mean():+.8f}, "
                f"mean_delta_RMS={sub['cptg_minus_planck_residual_rms_uK'].mean():+.8f} uK, "
                f"Planck_ahead_all={bool((sub['cptg_minus_planck_residual_fraction'] > 0).all())}"
            )

        report.append("")
        report.append("Mean full-band gap by Galactic cut:")
        for c_label in delta["galactic_cut_label"].unique():
            sub = delta[delta["galactic_cut_label"] == c_label]
            report.append(
                f"  {c_label}: mean_sky={sub['valid_sky_fraction'].mean():.6f}, "
                f"mean_delta={sub['cptg_minus_planck_residual_fraction'].mean():+.8f}, "
                f"mean_delta_RMS={sub['cptg_minus_planck_residual_rms_uK'].mean():+.8f} uK, "
                f"Planck_ahead_all={bool((sub['cptg_minus_planck_residual_fraction'] > 0).all())}"
            )

    if not band_delta.empty:
        report.append("")
        report.append("Band-limited mean gap by mask threshold:")
        for threshold in sorted(band_delta["mask_threshold"].unique()):
            for band in sorted(band_delta["ell_band"].unique(), key=lambda x: int(str(x).split("-")[1])):
                sub = band_delta[(band_delta["mask_threshold"] == threshold) & (band_delta["ell_band"] == band)]
                if sub.empty:
                    continue
                report.append(
                    f"  threshold={threshold:.2f} ell={band}: mean_delta={sub['cptg_minus_planck_residual_fraction'].mean():+.8f}, "
                    f"mean_delta_RMS={sub['cptg_minus_planck_residual_rms_uK'].mean():+.8f} uK"
                )

    if not stability.empty:
        report.append("")
        report.append("Stability rows:")
        for _, row in stability.iterrows():
            report.append(
                f"  {row['scope']}: mean_sky={row['mean_sky_fraction']:.6f}, "
                f"mean_gap={row['mean_gap_residual_fraction']:+.8f}, "
                f"span_gap={row['span_gap_residual_fraction']:.8f}, "
                f"mean_gap_RMS={row['mean_gap_rms_uK']:+.8f} uK, "
                f"span_gap_RMS={row['span_gap_rms_uK']:.8f} uK, "
                f"Planck_ahead_all={row['planck_ahead_all_rows']}"
            )

    if not skipped.empty:
        report.append("")
        report.append("Skipped masks/maps:")
        for _, row in skipped.iterrows():
            report.append(f"  {row['component']}: {row['reason']}")

    report.append("")
    report.append("Interpretation boundary:")
    report.append("  If the CPTG-minus-Planck gap changes strongly with sky fraction,")
    report.append("  the residual should be treated as mask/foreground/sky-selection")
    report.append("  sensitive rather than assigned uniquely to CPTG.")
    report.append("  If the gap is stable under mask variation, it is a stable feature")
    report.append("  of the selected phase-locked RMS envelope metric.")
    report.append("  This remains a phase-locked envelope comparison, not a native")
    report.append("  CPTG pixel-phase prediction.")

    report_text = "\n".join(report)
    (out / "mask_ladder_report.txt").write_text(report_text, encoding="utf-8")

    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "maps": maps,
                "mask_source_map": mask_source_map,
                "nside": args.nside,
                "thresholds": thresholds,
                "galactic_cuts": ["none" if x is None else x for x in cuts],
                "min_sky_fraction": args.min_sky_fraction,
                "use_inpainted": args.use_inpainted,
                "bands": bands,
                "lmax": lmax,
            },
            f,
            indent=2,
        )

    print("")
    print(report_text)
    print("")
    print(f"Outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
