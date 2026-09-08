#!/usr/bin/env python3
"""
CPTG WMAP ILC band-limited envelope competition.

Why this exists:
    The first WMAP all-sky run compared the full WMAP ILC map to unsmoothed
    theory envelopes through lmax=3*Nside-1. That is a useful cross-mission
    sanity check, but it can be dominated by WMAP ILC smoothing / transfer /
    high-ell differences rather than the large-angle CMB geometry we are
    trying to test.

This script repeats the competition in declared multipole bands, e.g.
    ell=2-30
    ell=2-40
    ell=2-64
    ell=2-95

For each band:
    - Observed WMAP map is band-limited to that ell range.
    - CPTG and Planck envelopes use the same WMAP phase scaffold in that band.
    - Same amplitude+offset comparison rule.
    - Same mask.
    - Same unit inference.

This is still a phase-locked angular-power / transport-envelope comparison,
not a native CPTG prediction of exact CMB phases.

Example:
    python cptg_wmap_lowell_bandtest.py ^
      --wmap wmap_ilc_9yr_v5.fits ^
      --nside 128 ^
      --bands 2-30,2-40,2-64,2-95 ^
      --out cptg_wmap_lowell_bandtest

Masked example:
    python cptg_wmap_lowell_bandtest.py ^
      --wmap wmap_ilc_9yr_v5.fits ^
      --mask wmap_temperature_analysis_mask_r9_9yr_v5.fits ^
      --nside 128 ^
      --bands 2-30,2-40,2-64,2-95 ^
      --out cptg_wmap_lowell_bandtest_masked

Outputs:
    wmap_band_envelope_summary.csv
    wmap_band_cptg_vs_planck_delta.csv
    wmap_band_report.txt
    residual_fraction_by_band.png
    cptg_minus_planck_by_band.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
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


def parse_bands(s: str) -> list[tuple[int, int]]:
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
    if not bands:
        raise SystemExit("No valid bands supplied.")
    return bands


def get_field_names(path: str) -> List[str]:
    try:
        with fits.open(path, memmap=True) as hdul:
            for hdu in hdul:
                if hasattr(hdu, "columns") and hdu.columns is not None:
                    names = list(hdu.columns.names)
                    if names:
                        return [str(n).strip() for n in names]
    except Exception:
        return []
    return []


def get_field_index(path: str, field_name: str) -> int:
    names = get_field_names(path)
    lookup = {name.upper(): i for i, name in enumerate(names)}
    key = field_name.upper()
    if key not in lookup:
        raise KeyError(f"Field {field_name} not found in {path}. Available fields: {names}")
    return int(lookup[key])


def read_healpix_map(path: str, field: Optional[str | int] = None) -> np.ndarray:
    if field is None:
        names = get_field_names(path)
        preferred = ["TEMPERATURE", "TEMP", "I_STOKES", "I_STOKES_INP", "ILC", "MAP", "SIGNAL"]
        upper_names = {name.upper(): name for name in names}
        for p in preferred:
            if p in upper_names:
                idx = get_field_index(path, upper_names[p])
                return hp.read_map(path, field=idx, dtype=np.float64)
        return hp.read_map(path, field=0, dtype=np.float64)

    if isinstance(field, int):
        return hp.read_map(path, field=field, dtype=np.float64)

    idx = get_field_index(path, field)
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


def convert_temperature_to_uK(raw_temp: np.ndarray) -> tuple[np.ndarray, float, str, float]:
    raw = finite_map(raw_temp)
    sigma = robust_std(raw)

    if not np.isfinite(sigma):
        raise ValueError("Cannot infer temperature units: non-finite robust RMS.")

    if sigma < 1.0e-2:
        return raw * 1.0e6, 1.0e6, "K_CMB_to_uK", sigma

    if sigma < 10.0:
        return raw * 1.0e3, 1.0e3, "mK_CMB_to_uK", sigma

    return raw.copy(), 1.0, "uK_native_or_already_scaled", sigma


def read_mask(mask_path: Optional[str], mask_field: Optional[str | int], nside_ref: int) -> tuple[np.ndarray, str]:
    if not mask_path:
        return np.ones(hp.nside2npix(nside_ref), dtype=np.float64), "ALL_SKY_NO_MASK"
    raw = finite_map(read_healpix_map(mask_path, field=mask_field))
    return raw, str(mask_field if mask_field is not None else "field0_or_auto")


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


def evaluate_template(obs_band: np.ndarray, template_band: np.ndarray, valid: np.ndarray) -> dict:
    use = valid & np.isfinite(obs_band) & np.isfinite(template_band)
    template_shape = normalize_template_shape(template_band, use)
    amp, offset = fit_amplitude_and_offset(obs_band, template_shape, use)
    fit_map = amp * template_shape + offset
    residual = obs_band - fit_map

    obs_rms = rms(obs_band, use)
    fit_rms = rms(fit_map, use)
    res_rms = rms(residual, use)

    return {
        "masked_corr": masked_corr(obs_band, fit_map, use),
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


def bandlimit_alm(alm: np.ndarray, lmax: int, ell_min: int, ell_max: int, scale_by_ell: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.zeros_like(alm)
    for ell in range(ell_min, ell_max + 1):
        factor = 1.0 if scale_by_ell is None else scale_by_ell[ell]
        for m in range(ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            out[idx] = alm[idx] * factor
    return out


def build_band_template_map(alm_obs: np.ndarray, cl_obs: np.ndarray, cl_target: np.ndarray, nside: int, lmax: int, ell_min: int, ell_max: int) -> np.ndarray:
    eps = 1.0e-30
    scale = np.sqrt(np.maximum(cl_target[: lmax + 1], 0.0) / np.maximum(cl_obs[: lmax + 1], eps))
    scale[0:2] = 0.0
    alm_band = bandlimit_alm(alm_obs, lmax, ell_min, ell_max, scale_by_ell=scale)
    return hp.alm2map(alm_band, nside=nside, lmax=lmax)


def build_observed_band_map(alm_obs: np.ndarray, nside: int, lmax: int, ell_min: int, ell_max: int) -> np.ndarray:
    alm_band = bandlimit_alm(alm_obs, lmax, ell_min, ell_max, scale_by_ell=None)
    return hp.alm2map(alm_band, nside=nside, lmax=lmax)


def make_residual_plot(path: Path, summary: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 7))
    for model in summary["model"].unique():
        sub = summary[summary["model"] == model]
        plt.plot(sub["ell_band"], sub["residual_fraction"], marker="o", label=model)
    plt.ylabel("Residual RMS / observed-band RMS")
    plt.xlabel("Multipole band")
    plt.title("WMAP ILC band-limited envelope competition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_delta_plot(path: Path, delta_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(delta_df["ell_band"], delta_df["cptg_minus_planck_residual_fraction"])
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("CPTG residual fraction minus Planck residual fraction")
    plt.xlabel("Multipole band")
    plt.title("WMAP ILC CPTG vs Planck band-limited gap")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPTG WMAP ILC band-limited envelope competition.")
    parser.add_argument("--wmap", required=True)
    parser.add_argument("--field", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--mask-field", default=None)
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--mask-threshold", type=float, default=0.99)
    parser.add_argument("--bands", default="2-30,2-40,2-64,2-95")
    parser.add_argument("--out", default="cptg_wmap_lowell_bandtest")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bands = parse_bands(args.bands)
    max_band_ell = max(hi for _, hi in bands)
    lmax = max(max_band_ell, 3 * args.nside - 1)

    if args.field is not None:
        try:
            field: Optional[str | int] = int(args.field)
        except ValueError:
            field = args.field
    else:
        field = None

    if args.mask_field is not None:
        try:
            mask_field: Optional[str | int] = int(args.mask_field)
        except ValueError:
            mask_field = args.mask_field
    else:
        mask_field = None

    print(f"Reading WMAP ILC map: {args.wmap}")
    available_fields = get_field_names(args.wmap)
    raw_temp = finite_map(read_healpix_map(args.wmap, field=field))
    temp_uk, temp_scale, temp_unit, raw_std = convert_temperature_to_uK(raw_temp)
    print(f"Temperature unit inference: {temp_unit}; raw robust std={raw_std:.8g}; scale={temp_scale:g}")

    nside_in = hp.get_nside(temp_uk)
    mask_raw, mask_label = read_mask(args.mask, mask_field, nside_ref=nside_in)

    print(f"Downgrading to Nside={args.nside}")
    obs_low, valid = downgrade_map_and_mask(temp_uk, mask_raw, args.nside, args.mask_threshold)

    print("Removing monopole/dipole.")
    obs_clean = remove_monopole_dipole_masked(obs_low, valid)

    obs_filled = np.zeros_like(obs_clean)
    good = valid & np.isfinite(obs_clean)
    obs_filled[good] = obs_clean[good]

    print(f"Computing WMAP phase scaffold alms to lmax={lmax}")
    alm_obs = hp.map2alm(obs_filled, lmax=lmax, iter=0)
    cl_obs = hp.alm2cl(alm_obs)
    cl_obs[0:2] = 0.0

    print("Computing locked CPTG and Planck TT envelopes.")
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

    for ell_min, ell_max in bands:
        band_label = f"{ell_min}-{ell_max}"
        print(f"Evaluating band ell={band_label}")

        obs_band = build_observed_band_map(alm_obs, args.nside, lmax, ell_min, ell_max)

        envelopes = {
            "cptg_locked_pi_cmb": cl_cptg,
            "planck2018_baseline_lcdm": cl_planck,
            "wmap_band_self_spectrum_control": cl_obs.copy(),
        }

        band_metrics = {}

        for label, cl in envelopes.items():
            template_band = build_band_template_map(alm_obs, cl_obs, cl, args.nside, lmax, ell_min, ell_max)
            metrics = evaluate_template(obs_band, template_band, valid)
            metrics.update(
                {
                    "model": label,
                    "ell_min": ell_min,
                    "ell_max": ell_max,
                    "ell_band": band_label,
                    "wmap_file": args.wmap,
                    "wmap_available_fields": "|".join(available_fields),
                    "wmap_requested_field": str(field if field is not None else "auto"),
                    "temperature_unit_inference": temp_unit,
                    "temperature_scale_applied": temp_scale,
                    "raw_temperature_robust_std": raw_std,
                    "mask_file": args.mask if args.mask else "",
                    "mask_field": mask_label,
                    "nside": args.nside,
                    "mask_threshold": args.mask_threshold,
                }
            )
            summary_rows.append(metrics)
            band_metrics[label] = metrics

        c = band_metrics["cptg_locked_pi_cmb"]
        p = band_metrics["planck2018_baseline_lcdm"]
        delta_rows.append(
            {
                "ell_min": ell_min,
                "ell_max": ell_max,
                "ell_band": band_label,
                "cptg_residual_fraction": c["residual_fraction"],
                "planck_residual_fraction": p["residual_fraction"],
                "cptg_minus_planck_residual_fraction": c["residual_fraction"] - p["residual_fraction"],
                "cptg_residual_rms_uK": c["residual_rms_uK"],
                "planck_residual_rms_uK": p["residual_rms_uK"],
                "cptg_minus_planck_residual_rms_uK": c["residual_rms_uK"] - p["residual_rms_uK"],
                "cptg_corr": c["masked_corr"],
                "planck_corr": p["masked_corr"],
                "observed_band_rms_uK": c["observed_rms_uK"],
                "valid_sky_fraction": c["valid_sky_fraction"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    delta_df = pd.DataFrame(delta_rows)

    summary.to_csv(out / "wmap_band_envelope_summary.csv", index=False)
    delta_df.to_csv(out / "wmap_band_cptg_vs_planck_delta.csv", index=False)

    make_residual_plot(out / "residual_fraction_by_band.png", summary)
    make_delta_plot(out / "cptg_minus_planck_by_band.png", delta_df)

    report = []
    report.append("CPTG WMAP ILC band-limited envelope report")
    report.append("=" * 47)
    report.append("")
    report.append(f"WMAP file: {args.wmap}")
    report.append(f"WMAP requested field: {field if field is not None else 'auto'}")
    report.append(f"WMAP available fields: {available_fields if available_fields else 'none listed by astropy'}")
    report.append(f"Temperature unit inference: {temp_unit}")
    report.append(f"Raw temperature robust std: {raw_std:.10g}")
    report.append(f"Temperature scale applied: {temp_scale:g}")
    report.append(f"Mask file: {args.mask if args.mask else 'ALL_SKY_NO_MASK'}")
    report.append(f"Mask field: {mask_label}")
    report.append(f"Nside: {args.nside}")
    report.append(f"Mask threshold: {args.mask_threshold}")
    report.append(f"Bands: {', '.join([f'{a}-{b}' for a, b in bands])}")
    report.append("")
    report.append("CPTG vs Planck by band:")
    for _, row in delta_df.iterrows():
        report.append(
            f"  ell={row['ell_band']}: CPTG={row['cptg_residual_fraction']:.8f}, "
            f"Planck={row['planck_residual_fraction']:.8f}, "
            f"delta={row['cptg_minus_planck_residual_fraction']:+.8f}, "
            f"delta_RMS={row['cptg_minus_planck_residual_rms_uK']:+.8f} uK"
        )
    report.append("")
    report.append("Interpretation boundary:")
    report.append("  This is a band-limited cross-mission phase-locked envelope comparison.")
    report.append("  It reduces sensitivity to WMAP ILC smoothing / transfer / high-ell details.")
    report.append("  It does not by itself establish native CPTG prediction of exact CMB phases.")

    report_text = "\n".join(report)
    (out / "wmap_band_report.txt").write_text(report_text, encoding="utf-8")

    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "wmap": args.wmap,
                "field": str(field if field is not None else "auto"),
                "available_fields": available_fields,
                "temperature_unit_inference": temp_unit,
                "temperature_scale_applied": temp_scale,
                "raw_temperature_robust_std": raw_std,
                "mask": args.mask,
                "mask_field": str(mask_field if mask_field is not None else "auto_or_none"),
                "nside": args.nside,
                "mask_threshold": args.mask_threshold,
                "bands": bands,
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
