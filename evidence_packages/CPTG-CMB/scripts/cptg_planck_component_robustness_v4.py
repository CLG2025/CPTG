#!/usr/bin/env python3
"""
CPTG Planck component-map robustness runner, v4.

Fixes through v4:
    - Never passes string field names into healpy.read_map().
      Some healpy/astropy combinations interpret a string like "I_STOKES"
      as an iterable and fail with: Key 'I' does not exist.
      This script looks up the FITS column index with astropy, then calls
      healpy.read_map(..., field=<integer index>).
    - Auto-detects temperature units per field. Most Planck component maps
      are stored in K_CMB, but some derived fields can already be in uK.
      The script now converts only when the raw robust RMS is K-scale.

Purpose:
    Run the same phase-locked envelope competition across multiple real
    Planck component-separation CMB maps.

Default auto-detected filenames in the current directory:
    COM_CMB_IQU-smica_2048_R3.00_full.fits
    COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits
    COM_CMB_IQU-nilc_2048_R3.00_full.fits
    COM_CMB_IQU-sevem_2048_R3.01_full.fits
    COM_CMB_IQU-sevem_2048_R3.00_full.fits
    COM_CMB_IQU-commander_2048_R3.00_full.fits

Examples:
    python cptg_planck_component_robustness_v4.py ^
      --nside 256 ^
      --out cptg_planck_component_robustness_v4

    python cptg_planck_component_robustness_v4.py ^
      --nside 256 ^
      --use-inpainted ^
      --out cptg_planck_component_robustness_v4_inpainted

Outputs:
    component_envelope_summary.csv
    component_cptg_vs_planck_delta.csv
    component_lowL_single_scan.csv
    component_lowL_cumulative_scan.csv
    component_robustness_report.txt
    residual_fraction_by_component.png
    cptg_minus_planck_by_component.png

Interpretation boundary:
    This is a component-separation robustness test for the phase-locked
    angular-power envelope comparison. It does not by itself establish
    native CPTG prediction of exact CMB pixel phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def convert_temperature_field_to_uK(raw_temp: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Convert a Planck component temperature field to uK, with unit inference.

    Most Planck CMB component fields are in K_CMB, so their raw RMS is roughly
    1e-4. Some derived/inpainted fields in certain products can already be
    stored in uK-scale values, with raw RMS around 100.

    Rule:
        raw robust RMS < 0.01  -> treat as K_CMB and multiply by 1e6
        raw robust RMS >= 0.01 -> treat as already uK-scale

    The threshold is deliberately far from both expected regimes.
    """
    raw = finite_map(raw_temp)
    sigma = robust_std(raw)

    if not np.isfinite(sigma):
        raise ValueError("Cannot infer temperature units: non-finite raw robust RMS.")

    if sigma < 1.0e-2:
        return raw * 1.0e6, 1.0e6, "K_CMB_to_uK"

    return raw.copy(), 1.0, "uK_native_or_already_scaled"


def select_temperature_field(path: str, use_inpainted: bool) -> str:
    if use_inpainted and has_field(path, "I_STOKES_INP"):
        return "I_STOKES_INP"
    if has_field(path, "I_STOKES"):
        return "I_STOKES"
    raise KeyError(f"No usable temperature field in {path}. Available fields: {get_field_names(path)}")


def select_mask(path: str, use_inpainted: bool, mask_source_map: str | None) -> Tuple[np.ndarray, str, str]:
    """
    Returns mask array, mask field name, mask source path.

    Preference:
        1. Same map TMASKINP for inpainted, if available.
        2. Same map TMASK.
        3. mask_source_map TMASKINP for inpainted, if available.
        4. mask_source_map TMASK.
        5. All-sky mask.
    """
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


def downgrade_map_and_mask(
    temp_uk: np.ndarray,
    mask: np.ndarray,
    nside_out: int,
    mask_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nside_in = hp.get_nside(temp_uk)
    nside_mask = hp.get_nside(mask)

    if nside_in == nside_out:
        temp_low = temp_uk.copy()
    else:
        temp_low = hp.ud_grade(temp_uk, nside_out=nside_out, power=0)

    if nside_mask == nside_out:
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


def evaluate_template(obs_clean: np.ndarray, template_raw: np.ndarray, valid: np.ndarray) -> dict:
    template_clean = remove_monopole_dipole_masked(template_raw, valid)
    use = valid & np.isfinite(obs_clean) & np.isfinite(template_clean)

    template_shape = normalize_template_shape(template_clean, use)
    amp, offset = fit_amplitude_and_offset(obs_clean, template_shape, use)
    fit_map = amp * template_shape + offset
    residual = obs_clean - fit_map

    obs_rms = rms(obs_clean, use)
    fit_rms = rms(fit_map, use)
    res_rms = rms(residual, use)

    return {
        "masked_corr": masked_corr(obs_clean, fit_map, use),
        "observed_rms_uK": obs_rms,
        "fit_rms_uK": fit_rms,
        "residual_rms_uK": res_rms,
        "residual_fraction": res_rms / obs_rms if obs_rms != 0 else np.nan,
        "amp_uK_per_template_sigma": amp,
        "offset_uK": offset,
        "valid_pixels": int(np.sum(use)),
        "valid_sky_fraction": float(np.sum(use) / len(use)),
    }


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
    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=0.0,
        mnu=0.06,
        nnu=Neff,
        tau=tau,
    )
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)

    if hasattr(pars, "Alens"):
        pars.Alens = 1.0

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(
        pars,
        lmax=lmax,
        CMB_unit="muK",
        raw_cl=True,
    )

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


def evaluate_envelope(
    label: str,
    cl: np.ndarray,
    alm_obs: np.ndarray,
    cl_obs: np.ndarray,
    obs_clean: np.ndarray,
    valid: np.ndarray,
    nside: int,
    lmax: int,
) -> dict:
    template = build_phase_locked_map(alm_obs, cl_obs, cl, nside, lmax)
    metrics = evaluate_template(obs_clean, template, valid)
    metrics["model"] = label
    return metrics


def make_residual_bar(path: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    pivot = summary.pivot_table(index="component", columns="model", values="residual_fraction", aggfunc="first")
    cols = [c for c in ["cptg_locked_pi_cmb", "planck2018_baseline_lcdm", "smica_self_spectrum_control"] if c in pivot.columns]
    pivot = pivot[cols]

    ax = pivot.plot(kind="bar", figsize=(12, 7))
    ax.set_ylabel("Residual RMS / observed RMS")
    ax.set_title("Phase-locked envelope competition by component map")
    ax.legend(title="Envelope")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_delta_plot(path: Path, delta_df: pd.DataFrame) -> None:
    if delta_df.empty:
        return

    plot_df = delta_df.sort_values("cptg_minus_planck_residual_fraction")
    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["component"], plot_df["cptg_minus_planck_residual_fraction"])
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("CPTG residual fraction minus Planck residual fraction")
    plt.title("CPTG vs Planck envelope gap by component")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPTG Planck component-map robustness runner, v4.")
    parser.add_argument("--map", action="append", default=None, help="Component map as LABEL=path.fits. May be repeated.")
    parser.add_argument("--mask-source-map", default=None, help="Optional map to supply TMASK/TMASKINP for maps without masks.")
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.99)
    parser.add_argument("--use-inpainted", action="store_true")
    parser.add_argument("--ell-max-scan", type=int, default=30)
    parser.add_argument("--out", default="cptg_planck_component_robustness_v4")
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

    lmax = 3 * args.nside - 1
    ell_max_scan = min(args.ell_max_scan, lmax)

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
    low_single_rows = []
    low_cumulative_rows = []
    skipped = []

    for component, path in maps.items():
        print("")
        print(f"Component: {component}")
        print(f"File: {path}")

        if not Path(path).exists():
            print("  skipped: file not found")
            skipped.append({"component": component, "path": path, "reason": "file_not_found"})
            continue

        try:
            available_fields = get_field_names(path)
            temperature_field = select_temperature_field(path, args.use_inpainted)
            temp = finite_map(read_healpix_field(path, temperature_field))
            mask_raw, mask_field, mask_source_used = select_mask(path, args.use_inpainted, mask_source_map)

            temp_uk, temperature_scale_applied, temperature_unit_inference = convert_temperature_field_to_uK(temp)
            raw_temperature_robust_std = robust_std(temp)
            mask = np.isfinite(mask_raw) & (mask_raw > 0.5)

            obs_low, valid = downgrade_map_and_mask(temp_uk, mask, args.nside, args.mask_threshold)
            obs_clean = remove_monopole_dipole_masked(obs_low, valid)

            obs_filled = np.zeros_like(obs_clean)
            good = valid & np.isfinite(obs_clean)
            obs_filled[good] = obs_clean[good]

            alm_obs = hp.map2alm(obs_filled, lmax=lmax, iter=0)
            cl_obs = hp.alm2cl(alm_obs)
            cl_obs[0:2] = 0.0

            envelopes = {
                "cptg_locked_pi_cmb": cl_cptg,
                "planck2018_baseline_lcdm": cl_planck,
                "smica_self_spectrum_control": cl_obs.copy(),
            }

            base_metrics = {}

            for model, cl in envelopes.items():
                metrics = evaluate_envelope(model, cl, alm_obs, cl_obs, obs_clean, valid, args.nside, lmax)
                metrics.update(
                    {
                        "component": component,
                        "path": path,
                        "available_fields": "|".join(available_fields),
                        "temperature_field": temperature_field,
                        "temperature_unit_inference": temperature_unit_inference,
                        "temperature_scale_applied": temperature_scale_applied,
                        "raw_temperature_robust_std": raw_temperature_robust_std,
                        "mask_field": mask_field,
                        "mask_source_used": mask_source_used,
                        "nside": args.nside,
                        "mask_threshold": args.mask_threshold,
                    }
                )
                summary_rows.append(metrics)
                if model in ["cptg_locked_pi_cmb", "planck2018_baseline_lcdm"]:
                    base_metrics[model] = metrics["residual_fraction"]

            for model, cl_base in {
                "cptg_locked_pi_cmb": cl_cptg,
                "planck2018_baseline_lcdm": cl_planck,
            }.items():
                baseline = base_metrics[model]

                for ell in range(2, ell_max_scan + 1):
                    cl_mod = cl_base.copy()
                    cl_mod[ell] = cl_obs[ell]
                    metrics = evaluate_envelope(model, cl_mod, alm_obs, cl_obs, obs_clean, valid, args.nside, lmax)
                    low_single_rows.append(
                        {
                            "component": component,
                            "model": model,
                            "ell": ell,
                            "baseline_residual_fraction": baseline,
                            "residual_fraction": metrics["residual_fraction"],
                            "improvement_abs": baseline - metrics["residual_fraction"],
                            "improvement_percent": 100.0 * (baseline - metrics["residual_fraction"]) / baseline,
                            "masked_corr": metrics["masked_corr"],
                            "residual_rms_uK": metrics["residual_rms_uK"],
                        }
                    )

                for ell_max in range(2, ell_max_scan + 1):
                    cl_mod = cl_base.copy()
                    cl_mod[2 : ell_max + 1] = cl_obs[2 : ell_max + 1]
                    metrics = evaluate_envelope(model, cl_mod, alm_obs, cl_obs, obs_clean, valid, args.nside, lmax)
                    low_cumulative_rows.append(
                        {
                            "component": component,
                            "model": model,
                            "ell_range": f"2-{ell_max}",
                            "ell_max": ell_max,
                            "baseline_residual_fraction": baseline,
                            "residual_fraction": metrics["residual_fraction"],
                            "improvement_abs": baseline - metrics["residual_fraction"],
                            "improvement_percent": 100.0 * (baseline - metrics["residual_fraction"]) / baseline,
                            "masked_corr": metrics["masked_corr"],
                            "residual_rms_uK": metrics["residual_rms_uK"],
                        }
                    )

            print(f"  done: temp={temperature_field}, unit={temperature_unit_inference}, mask={mask_field}, mask_source={mask_source_used}")

        except Exception as exc:
            skipped.append({"component": component, "path": path, "reason": f"analysis_failed: {exc}"})
            print(f"  skipped after analysis error: {exc}")

    summary = pd.DataFrame(summary_rows)
    single = pd.DataFrame(low_single_rows)
    cumulative = pd.DataFrame(low_cumulative_rows)
    skipped_df = pd.DataFrame(skipped)

    summary.to_csv(out / "component_envelope_summary.csv", index=False)
    single.to_csv(out / "component_lowL_single_scan.csv", index=False)
    cumulative.to_csv(out / "component_lowL_cumulative_scan.csv", index=False)
    skipped_df.to_csv(out / "skipped_maps.csv", index=False)

    delta_rows = []
    if not summary.empty:
        for component in summary["component"].unique():
            sub = summary[summary["component"] == component]
            cptg = sub[sub["model"] == "cptg_locked_pi_cmb"]
            planck = sub[sub["model"] == "planck2018_baseline_lcdm"]
            control = sub[sub["model"] == "smica_self_spectrum_control"]
            if len(cptg) and len(planck):
                c = cptg.iloc[0]
                p = planck.iloc[0]
                row = {
                    "component": component,
                    "cptg_residual_fraction": c["residual_fraction"],
                    "planck_residual_fraction": p["residual_fraction"],
                    "cptg_minus_planck_residual_fraction": c["residual_fraction"] - p["residual_fraction"],
                    "cptg_residual_rms_uK": c["residual_rms_uK"],
                    "planck_residual_rms_uK": p["residual_rms_uK"],
                    "cptg_minus_planck_residual_rms_uK": c["residual_rms_uK"] - p["residual_rms_uK"],
                    "cptg_corr": c["masked_corr"],
                    "planck_corr": p["masked_corr"],
                    "valid_sky_fraction": c["valid_sky_fraction"],
                    "temperature_field": c["temperature_field"],
                    "temperature_unit_inference": c["temperature_unit_inference"],
                    "temperature_scale_applied": c["temperature_scale_applied"],
                    "raw_temperature_robust_std": c["raw_temperature_robust_std"],
                    "mask_field": c["mask_field"],
                    "mask_source_used": c["mask_source_used"],
                }
                if len(control):
                    s = control.iloc[0]
                    row["self_spectrum_control_residual_fraction"] = s["residual_fraction"]
                delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out / "component_cptg_vs_planck_delta.csv", index=False)

    make_residual_bar(out / "residual_fraction_by_component.png", summary)
    make_delta_plot(out / "cptg_minus_planck_by_component.png", delta_df)

    report = []
    report.append("CPTG Planck component-map robustness report, v4")
    report.append("=" * 53)
    report.append("")
    report.append(f"Nside: {args.nside}")
    report.append(f"Mask threshold: {args.mask_threshold}")
    report.append(f"Requested inpainted fields: {args.use_inpainted}")
    report.append(f"Mask source map: {mask_source_map}")
    report.append(f"Ell scan range: 2-{ell_max_scan}")
    report.append("")

    if delta_df.empty:
        report.append("No component maps completed.")
    else:
        report.append("CPTG vs Planck envelope gap by component:")
        for _, row in delta_df.iterrows():
            report.append(
                f"  {row['component']}: CPTG={row['cptg_residual_fraction']:.8f}, "
                f"Planck={row['planck_residual_fraction']:.8f}, "
                f"delta={row['cptg_minus_planck_residual_fraction']:+.8f}, "
                f"delta_RMS={row['cptg_minus_planck_residual_rms_uK']:+.6f} uK, "
                f"temp={row['temperature_field']}, unit={row['temperature_unit_inference']}, mask={row['mask_field']}"
            )

        report.append("")
        report.append("Best single-ell diagnostic replacements, CPTG rows:")
        cptg_single = single[single["model"] == "cptg_locked_pi_cmb"] if not single.empty else pd.DataFrame()
        if not cptg_single.empty:
            for component in cptg_single["component"].unique():
                sub = cptg_single[cptg_single["component"] == component].sort_values("improvement_abs", ascending=False).head(5)
                report.append(f"  {component}:")
                for _, row in sub.iterrows():
                    report.append(
                        f"    ell={int(row['ell'])}: improvement={row['improvement_abs']:.8f} "
                        f"({row['improvement_percent']:.2f}%), residual_fraction={row['residual_fraction']:.8f}"
                    )

        report.append("")
        report.append("Best cumulative diagnostic replacements, CPTG rows:")
        cptg_cum = cumulative[cumulative["model"] == "cptg_locked_pi_cmb"] if not cumulative.empty else pd.DataFrame()
        if not cptg_cum.empty:
            for component in cptg_cum["component"].unique():
                sub = cptg_cum[cptg_cum["component"] == component].sort_values("improvement_abs", ascending=False).head(5)
                report.append(f"  {component}:")
                for _, row in sub.iterrows():
                    report.append(
                        f"    ell={row['ell_range']}: improvement={row['improvement_abs']:.8f} "
                        f"({row['improvement_percent']:.2f}%), residual_fraction={row['residual_fraction']:.8f}"
                    )

    if skipped:
        report.append("")
        report.append("Skipped maps:")
        for row in skipped:
            report.append(f"  {row['component']}: {row['reason']}")

    report.append("")
    report.append("Interpretation boundary:")
    report.append("  This is a component-separation robustness test for the phase-locked")
    report.append("  angular-power envelope comparison. It does not by itself establish")
    report.append("  native CPTG prediction of exact CMB pixel phases.")

    report_text = "\n".join(report)
    (out / "component_robustness_report.txt").write_text(report_text, encoding="utf-8")

    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "maps": maps,
                "mask_source_map": mask_source_map,
                "nside": args.nside,
                "mask_threshold": args.mask_threshold,
                "use_inpainted": args.use_inpainted,
                "ell_max_scan": ell_max_scan,
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
