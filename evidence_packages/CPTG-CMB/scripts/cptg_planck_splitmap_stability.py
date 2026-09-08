#!/usr/bin/env python3
"""
CPTG Planck SMICA split-map stability test.

Purpose:
    Test whether the tiny CPTG-minus-Planck envelope gap is stable under
    Planck internal split maps.

Default auto-detected files in the current directory:
    COM_CMB_IQU-smica_2048_R3.00_full.fits
    COM_CMB_IQU-smica_2048_R3.00_hm1.fits
    COM_CMB_IQU-smica_2048_R3.00_hm2.fits
    COM_CMB_IQU-smica_2048_R3.00_oe1.fits
    COM_CMB_IQU-smica_2048_R3.00_oe2.fits

Main question:
    Is the CPTG-minus-Planck residual gap stable across full, half-mission,
    and odd/even ring splits? If split-to-split variations are comparable to
    the reported sub-microkelvin gap, then the residual cannot be assigned
    uniquely to CPTG.

What this tests:
    - Same phase-locked envelope competition as prior runs.
    - Uses the split map itself as the phase scaffold.
    - Uses the full SMICA TMASK/TMASKINP as a common mask when split maps
      do not contain mask fields.
    - Compares CPTG locked pi-CMB envelope against Planck 2018 baseline.

What this does not claim:
    This is still a phase-locked angular-power / transport-envelope comparison.
    It is not a native CPTG prediction of exact random CMB pixel phases.

Example:
    python cptg_planck_splitmap_stability.py ^
      --nside 256 ^
      --out cptg_planck_splitmap_stability

Optional inpainted/fallback run:
    python cptg_planck_splitmap_stability.py ^
      --nside 256 ^
      --use-inpainted ^
      --out cptg_planck_splitmap_stability_inpainted

Optional low-ell bands:
    python cptg_planck_splitmap_stability.py ^
      --nside 256 ^
      --bands 2-30,2-40,2-64,2-95 ^
      --out cptg_planck_splitmap_stability_bands

Outputs:
    splitmap_envelope_summary.csv
    splitmap_cptg_vs_planck_delta.csv
    splitmap_stability_summary.csv
    splitmap_report.txt
    cptg_minus_planck_gap_by_split.png
    residual_fraction_by_split.png

If --bands is provided:
    splitmap_band_envelope_summary.csv
    splitmap_band_cptg_vs_planck_delta.csv
    cptg_minus_planck_gap_by_split_band.png
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


DEFAULT_SPLIT_CANDIDATES = {
    "full": ["COM_CMB_IQU-smica_2048_R3.00_full.fits"],
    "hm1": ["COM_CMB_IQU-smica_2048_R3.00_hm1.fits"],
    "hm2": ["COM_CMB_IQU-smica_2048_R3.00_hm2.fits"],
    "oe1": ["COM_CMB_IQU-smica_2048_R3.00_oe1.fits"],
    "oe2": ["COM_CMB_IQU-smica_2048_R3.00_oe2.fits"],
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


def parse_split_args(split_args: List[str] | None) -> Dict[str, str]:
    if not split_args:
        found = {}
        for label, candidates in DEFAULT_SPLIT_CANDIDATES.items():
            for fname in candidates:
                if Path(fname).exists():
                    found[label] = fname
                    break
        return found

    splits = {}
    for item in split_args:
        if "=" not in item:
            raise SystemExit(f"Invalid --split entry: {item}. Use LABEL=path.fits")
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip().strip('"')
        if not label or not path:
            raise SystemExit(f"Invalid --split entry: {item}. Use LABEL=path.fits")
        splits[label] = path
    return splits


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
    # Split maps sometimes have only TEMPERATURE/TEMP-like names.
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


def downgrade_map_and_mask(temp_uk: np.ndarray, mask: np.ndarray, nside_out: int, mask_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
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


def make_gap_plot(path: Path, delta_df: pd.DataFrame) -> None:
    if delta_df.empty:
        return
    plot = delta_df.copy()
    plt.figure(figsize=(10, 6))
    plt.bar(plot["split"], plot["cptg_minus_planck_residual_fraction"])
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("CPTG residual fraction minus Planck residual fraction")
    plt.title("CPTG vs Planck envelope gap by Planck split")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_residual_plot(path: Path, delta_df: pd.DataFrame) -> None:
    if delta_df.empty:
        return
    x = np.arange(len(delta_df))
    h = 0.38
    plt.figure(figsize=(10, 6))
    plt.bar(x - h/2, delta_df["cptg_residual_fraction"], width=h, label="CPTG")
    plt.bar(x + h/2, delta_df["planck_residual_fraction"], width=h, label="Planck baseline")
    plt.xticks(x, delta_df["split"])
    plt.ylabel("Residual RMS / observed RMS")
    plt.title("Residual fraction by Planck split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_band_gap_plot(path: Path, delta_df: pd.DataFrame) -> None:
    if delta_df.empty:
        return
    plot = delta_df.copy()
    plot["label"] = plot["split"] + " " + plot["ell_band"]
    plt.figure(figsize=(12, max(7, 0.3 * len(plot))))
    plt.barh(plot["label"], plot["cptg_minus_planck_residual_fraction"])
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("CPTG residual fraction minus Planck residual fraction")
    plt.title("Band-limited CPTG vs Planck gap by Planck split")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPTG Planck SMICA split-map stability test.")
    parser.add_argument("--split", action="append", default=None, help="Split map as LABEL=path.fits. May be repeated.")
    parser.add_argument("--mask-source-map", default=None, help="Optional map to supply TMASK/TMASKINP. Defaults to full SMICA if present.")
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.99)
    parser.add_argument("--use-inpainted", action="store_true")
    parser.add_argument("--bands", default=None, help="Optional low-ell bands, e.g. 2-30,2-40,2-64,2-95")
    parser.add_argument("--out", default="cptg_planck_splitmap_stability")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = parse_split_args(args.split)
    if not splits:
        raise SystemExit("No split maps found. Put SMICA full/hm/oe FITS files in this directory or pass --split LABEL=path.fits.")

    mask_source_map = args.mask_source_map
    if mask_source_map is None:
        if "full" in splits and Path(splits["full"]).exists():
            mask_source_map = splits["full"]
        elif Path("COM_CMB_IQU-smica_2048_R3.00_full.fits").exists():
            mask_source_map = "COM_CMB_IQU-smica_2048_R3.00_full.fits"

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

    for split, path in splits.items():
        print("")
        print(f"Split: {split}")
        print(f"File: {path}")

        if not Path(path).exists():
            skipped_rows.append({"split": split, "path": path, "reason": "file_not_found"})
            print("  skipped: file not found")
            continue

        try:
            available_fields = get_field_names(path)
            temperature_field = select_temperature_field(path, args.use_inpainted)
            raw_temp = finite_map(read_healpix_field(path, temperature_field))
            temp_uk, temp_scale, temp_unit, raw_std = convert_temperature_field_to_uK(raw_temp)

            mask_raw, mask_field, mask_source_used = select_mask(path, args.use_inpainted, mask_source_map)
            mask = np.isfinite(mask_raw) & (mask_raw > 0.5)

            obs_low, valid = downgrade_map_and_mask(temp_uk, mask, args.nside, args.mask_threshold)
            obs_clean = remove_monopole_dipole_masked(obs_low, valid)

            obs_filled = np.zeros_like(obs_clean)
            good = valid & np.isfinite(obs_clean)
            obs_filled[good] = obs_clean[good]

            print(f"  Computing phase scaffold alms to lmax={lmax}")
            alm_obs = hp.map2alm(obs_filled, lmax=lmax, iter=0)
            cl_obs = hp.alm2cl(alm_obs)
            cl_obs[0:2] = 0.0

            envelopes = {
                "cptg_locked_pi_cmb": cl_cptg,
                "planck2018_baseline_lcdm": cl_planck,
                "split_self_spectrum_control": cl_obs.copy(),
            }

            split_metrics = {}
            for model, cl in envelopes.items():
                template = build_phase_locked_map(alm_obs, cl_obs, cl, args.nside, lmax)
                metrics = evaluate_template(obs_clean, template, valid)
                metrics.update(
                    {
                        "split": split,
                        "model": model,
                        "path": path,
                        "available_fields": "|".join(available_fields),
                        "temperature_field": temperature_field,
                        "temperature_unit_inference": temp_unit,
                        "temperature_scale_applied": temp_scale,
                        "raw_temperature_robust_std": raw_std,
                        "mask_field": mask_field,
                        "mask_source_used": mask_source_used,
                        "nside": args.nside,
                        "mask_threshold": args.mask_threshold,
                    }
                )
                summary_rows.append(metrics)
                split_metrics[model] = metrics

            c = split_metrics["cptg_locked_pi_cmb"]
            p = split_metrics["planck2018_baseline_lcdm"]
            delta_rows.append(
                {
                    "split": split,
                    "cptg_residual_fraction": c["residual_fraction"],
                    "planck_residual_fraction": p["residual_fraction"],
                    "cptg_minus_planck_residual_fraction": c["residual_fraction"] - p["residual_fraction"],
                    "cptg_residual_rms_uK": c["residual_rms_uK"],
                    "planck_residual_rms_uK": p["residual_rms_uK"],
                    "cptg_minus_planck_residual_rms_uK": c["residual_rms_uK"] - p["residual_rms_uK"],
                    "cptg_corr": c["masked_corr"],
                    "planck_corr": p["masked_corr"],
                    "observed_rms_uK": c["observed_rms_uK"],
                    "valid_sky_fraction": c["valid_sky_fraction"],
                    "temperature_field": temperature_field,
                    "temperature_unit_inference": temp_unit,
                    "mask_field": mask_field,
                    "mask_source_used": mask_source_used,
                }
            )

            for ell_min, ell_max in bands:
                band_label = f"{ell_min}-{ell_max}"
                obs_band = build_observed_band_map(alm_obs, args.nside, lmax, ell_min, ell_max)
                band_metrics = {}
                for model, cl in envelopes.items():
                    template_band = build_band_template_map(alm_obs, cl_obs, cl, args.nside, lmax, ell_min, ell_max)
                    metrics = evaluate_template(obs_band, template_band, valid)
                    metrics.update(
                        {
                            "split": split,
                            "model": model,
                            "ell_min": ell_min,
                            "ell_max": ell_max,
                            "ell_band": band_label,
                            "path": path,
                            "temperature_field": temperature_field,
                            "temperature_unit_inference": temp_unit,
                            "mask_field": mask_field,
                            "mask_source_used": mask_source_used,
                            "nside": args.nside,
                            "mask_threshold": args.mask_threshold,
                        }
                    )
                    band_summary_rows.append(metrics)
                    band_metrics[model] = metrics

                bc = band_metrics["cptg_locked_pi_cmb"]
                bp = band_metrics["planck2018_baseline_lcdm"]
                band_delta_rows.append(
                    {
                        "split": split,
                        "ell_min": ell_min,
                        "ell_max": ell_max,
                        "ell_band": band_label,
                        "cptg_residual_fraction": bc["residual_fraction"],
                        "planck_residual_fraction": bp["residual_fraction"],
                        "cptg_minus_planck_residual_fraction": bc["residual_fraction"] - bp["residual_fraction"],
                        "cptg_residual_rms_uK": bc["residual_rms_uK"],
                        "planck_residual_rms_uK": bp["residual_rms_uK"],
                        "cptg_minus_planck_residual_rms_uK": bc["residual_rms_uK"] - bp["residual_rms_uK"],
                        "cptg_corr": bc["masked_corr"],
                        "planck_corr": bp["masked_corr"],
                        "observed_band_rms_uK": bc["observed_rms_uK"],
                        "valid_sky_fraction": bc["valid_sky_fraction"],
                        "temperature_field": temperature_field,
                        "temperature_unit_inference": temp_unit,
                        "mask_field": mask_field,
                        "mask_source_used": mask_source_used,
                    }
                )

            print(f"  done: temp={temperature_field}, unit={temp_unit}, mask={mask_field}, mask_source={mask_source_used}")

        except Exception as exc:
            print(f"  skipped after analysis error: {exc}")
            skipped_rows.append({"split": split, "path": path, "reason": f"analysis_failed: {exc}"})

    summary = pd.DataFrame(summary_rows)
    delta = pd.DataFrame(delta_rows)
    band_summary = pd.DataFrame(band_summary_rows)
    band_delta = pd.DataFrame(band_delta_rows)
    skipped = pd.DataFrame(skipped_rows)

    summary.to_csv(out / "splitmap_envelope_summary.csv", index=False)
    delta.to_csv(out / "splitmap_cptg_vs_planck_delta.csv", index=False)
    band_summary.to_csv(out / "splitmap_band_envelope_summary.csv", index=False)
    band_delta.to_csv(out / "splitmap_band_cptg_vs_planck_delta.csv", index=False)
    skipped.to_csv(out / "skipped_splitmaps.csv", index=False)

    stability_rows = []
    if not delta.empty:
        gaps = delta["cptg_minus_planck_residual_fraction"].to_numpy(dtype=float)
        gaps_rms = delta["cptg_minus_planck_residual_rms_uK"].to_numpy(dtype=float)
        full_gap = np.nan
        full_gap_rms = np.nan
        if "full" in set(delta["split"]):
            full_row = delta[delta["split"] == "full"].iloc[0]
            full_gap = full_row["cptg_minus_planck_residual_fraction"]
            full_gap_rms = full_row["cptg_minus_planck_residual_rms_uK"]

        stability_rows.append(
            {
                "scope": "full_band_splits",
                "n_splits": len(delta),
                "mean_gap_residual_fraction": np.nanmean(gaps),
                "std_gap_residual_fraction": np.nanstd(gaps, ddof=1) if len(gaps) > 1 else np.nan,
                "min_gap_residual_fraction": np.nanmin(gaps),
                "max_gap_residual_fraction": np.nanmax(gaps),
                "span_gap_residual_fraction": np.nanmax(gaps) - np.nanmin(gaps),
                "full_gap_residual_fraction": full_gap,
                "mean_gap_rms_uK": np.nanmean(gaps_rms),
                "std_gap_rms_uK": np.nanstd(gaps_rms, ddof=1) if len(gaps_rms) > 1 else np.nan,
                "min_gap_rms_uK": np.nanmin(gaps_rms),
                "max_gap_rms_uK": np.nanmax(gaps_rms),
                "span_gap_rms_uK": np.nanmax(gaps_rms) - np.nanmin(gaps_rms),
                "full_gap_rms_uK": full_gap_rms,
                "planck_ahead_all_splits": bool((delta["cptg_minus_planck_residual_fraction"] > 0).all()),
            }
        )

    if not band_delta.empty:
        for band in sorted(band_delta["ell_band"].unique(), key=lambda x: int(str(x).split("-")[1])):
            sub = band_delta[band_delta["ell_band"] == band]
            gaps = sub["cptg_minus_planck_residual_fraction"].to_numpy(dtype=float)
            gaps_rms = sub["cptg_minus_planck_residual_rms_uK"].to_numpy(dtype=float)
            stability_rows.append(
                {
                    "scope": f"band_{band}_splits",
                    "n_splits": len(sub),
                    "mean_gap_residual_fraction": np.nanmean(gaps),
                    "std_gap_residual_fraction": np.nanstd(gaps, ddof=1) if len(gaps) > 1 else np.nan,
                    "min_gap_residual_fraction": np.nanmin(gaps),
                    "max_gap_residual_fraction": np.nanmax(gaps),
                    "span_gap_residual_fraction": np.nanmax(gaps) - np.nanmin(gaps),
                    "full_gap_residual_fraction": sub[sub["split"] == "full"]["cptg_minus_planck_residual_fraction"].iloc[0] if "full" in set(sub["split"]) else np.nan,
                    "mean_gap_rms_uK": np.nanmean(gaps_rms),
                    "std_gap_rms_uK": np.nanstd(gaps_rms, ddof=1) if len(gaps_rms) > 1 else np.nan,
                    "min_gap_rms_uK": np.nanmin(gaps_rms),
                    "max_gap_rms_uK": np.nanmax(gaps_rms),
                    "span_gap_rms_uK": np.nanmax(gaps_rms) - np.nanmin(gaps_rms),
                    "full_gap_rms_uK": sub[sub["split"] == "full"]["cptg_minus_planck_residual_rms_uK"].iloc[0] if "full" in set(sub["split"]) else np.nan,
                    "planck_ahead_all_splits": bool((sub["cptg_minus_planck_residual_fraction"] > 0).all()),
                }
            )

    stability = pd.DataFrame(stability_rows)
    stability.to_csv(out / "splitmap_stability_summary.csv", index=False)

    make_gap_plot(out / "cptg_minus_planck_gap_by_split.png", delta)
    make_residual_plot(out / "residual_fraction_by_split.png", delta)
    make_band_gap_plot(out / "cptg_minus_planck_gap_by_split_band.png", band_delta)

    report = []
    report.append("CPTG Planck SMICA split-map stability report")
    report.append("=" * 49)
    report.append("")
    report.append(f"Nside: {args.nside}")
    report.append(f"Mask threshold: {args.mask_threshold}")
    report.append(f"Requested inpainted fields: {args.use_inpainted}")
    report.append(f"Mask source map: {mask_source_map}")
    report.append(f"Bands: {args.bands if args.bands else 'not requested'}")
    report.append("")

    if delta.empty:
        report.append("No split maps completed.")
    else:
        report.append("Full-band CPTG vs Planck by split:")
        for _, row in delta.iterrows():
            report.append(
                f"  {row['split']}: CPTG={row['cptg_residual_fraction']:.8f}, "
                f"Planck={row['planck_residual_fraction']:.8f}, "
                f"delta={row['cptg_minus_planck_residual_fraction']:+.8f}, "
                f"delta_RMS={row['cptg_minus_planck_residual_rms_uK']:+.8f} uK"
            )

    if not stability.empty:
        report.append("")
        report.append("Stability summary:")
        for _, row in stability.iterrows():
            report.append(
                f"  {row['scope']}: mean_gap={row['mean_gap_residual_fraction']:+.8f}, "
                f"std_gap={row['std_gap_residual_fraction']:.8f}, "
                f"span_gap={row['span_gap_residual_fraction']:.8f}, "
                f"mean_gap_RMS={row['mean_gap_rms_uK']:+.8f} uK, "
                f"span_gap_RMS={row['span_gap_rms_uK']:.8f} uK, "
                f"Planck_ahead_all={row['planck_ahead_all_splits']}"
            )

    if not band_delta.empty:
        report.append("")
        report.append("Band-limited CPTG vs Planck by split:")
        for _, row in band_delta.iterrows():
            report.append(
                f"  {row['split']} ell={row['ell_band']}: "
                f"CPTG={row['cptg_residual_fraction']:.8f}, "
                f"Planck={row['planck_residual_fraction']:.8f}, "
                f"delta={row['cptg_minus_planck_residual_fraction']:+.8f}, "
                f"delta_RMS={row['cptg_minus_planck_residual_rms_uK']:+.8f} uK"
            )

    if not skipped.empty:
        report.append("")
        report.append("Skipped split maps:")
        for _, row in skipped.iterrows():
            report.append(f"  {row['split']}: {row['reason']}")

    report.append("")
    report.append("Interpretation boundary:")
    report.append("  This tests stability of the sub-microkelvin CPTG-minus-Planck gap")
    report.append("  under Planck internal split maps. If the split-to-split span is")
    report.append("  comparable to the mean gap, the residual should not be assigned")
    report.append("  uniquely to CPTG.")
    report.append("  This remains a phase-locked envelope comparison, not a native")
    report.append("  CPTG pixel-phase prediction.")

    report_text = "\n".join(report)
    (out / "splitmap_report.txt").write_text(report_text, encoding="utf-8")

    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "splits": splits,
                "mask_source_map": mask_source_map,
                "nside": args.nside,
                "mask_threshold": args.mask_threshold,
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
