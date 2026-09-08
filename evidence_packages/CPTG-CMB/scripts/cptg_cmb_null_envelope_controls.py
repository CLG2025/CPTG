#!/usr/bin/env python3
"""
CPTG CMB null-envelope / uniqueness controls.

Purpose
-------
Test whether the near-degeneracy between the locked CPTG geometric-pi CMB
comparison envelope and the Planck baseline is nontrivial.

Main question
-------------
Is CPTG close only because the map comparison uses the observed phase scaffold,
or does the locked CPTG angular-power envelope carry real information?

Method
------
For each Planck-style CMB map:

1. Read the observed CMB temperature map.
2. Apply the temperature mask from the same map or a fallback source.
3. Downgrade to Nside.
4. Remove monopole/dipole on the valid sky.
5. Extract the observed phase scaffold.
6. Build phase-locked comparison maps for:
      - CPTG locked pi-CMB
      - Planck 2018 baseline
      - observed self-spectrum control
      - smoothed observed spectrum control
      - flat C_l null
      - flat D_l null
      - fitted power-law D_l null
      - shuffled-spectrum nulls
      - CPTG tilt/ripple perturbations
      - Planck tilt/ripple perturbations
7. Fit amplitude+offset on the valid sky for every envelope.
8. Rank envelopes by residual RMS / observed RMS.

Default auto-detected Planck component files in the current directory:
    COM_CMB_IQU-smica_2048_R3.00_full.fits
    COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits
    COM_CMB_IQU-nilc_2048_R3.00_full.fits
    COM_CMB_IQU-sevem_2048_R3.01_full.fits
    COM_CMB_IQU-sevem_2048_R3.00_full.fits
    COM_CMB_IQU-commander_2048_R3.00_full.fits

Examples
--------
python cptg_cmb_null_envelope_controls.py ^
  --nside 256 ^
  --out cptg_cmb_null_envelope_controls

Inpainted:
python cptg_cmb_null_envelope_controls.py ^
  --nside 256 ^
  --use-inpainted ^
  --out cptg_cmb_null_envelope_controls_inpainted

SMICA only:
python cptg_cmb_null_envelope_controls.py ^
  --map SMICA=COM_CMB_IQU-smica_2048_R3.00_full.fits ^
  --nside 256 ^
  --out cptg_cmb_null_envelope_controls_smica

Outputs
-------
null_envelope_results.csv
null_envelope_rankings.csv
null_envelope_summary_by_model.csv
null_envelope_family_summary.csv
null_envelope_report.txt
ranking_by_component.png
residual_fraction_by_model.png
cptg_planck_vs_nulls.png

Interpretation boundary
-----------------------
This is still a phase-locked angular-power / transport-envelope comparison.
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
        raise SystemExit("No perturbation values supplied.")
    return vals


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


def moving_average_spectrum(cl: np.ndarray, width: int) -> np.ndarray:
    width = max(int(width), 3)
    if width % 2 == 0:
        width += 1
    pad = width // 2
    src = np.asarray(cl, dtype=np.float64).copy()
    src[0:2] = 0.0

    out = src.copy()
    for i in range(2, len(src)):
        lo = max(2, i - pad)
        hi = min(len(src), i + pad + 1)
        vals = src[lo:hi]
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        out[i] = np.nanmean(vals) if vals.size else src[i]
    out[0:2] = 0.0
    return np.maximum(out, 0.0)


def flat_cl_null(cl_obs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(cl_obs)
    vals = cl_obs[2:]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    mean_cl = float(np.nanmean(vals)) if vals.size else 1.0
    out[2:] = mean_cl
    return out


def flat_dl_null(cl_obs: np.ndarray) -> np.ndarray:
    ell = np.arange(len(cl_obs), dtype=np.float64)
    dl = ell * (ell + 1.0) * cl_obs / (2.0 * np.pi)
    vals = dl[2:]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    mean_dl = float(np.nanmean(vals)) if vals.size else 1.0

    out = np.zeros_like(cl_obs)
    good = ell >= 2
    out[good] = mean_dl * (2.0 * np.pi) / (ell[good] * (ell[good] + 1.0))
    return np.maximum(out, 0.0)


def fit_powerlaw_dl_null(cl_obs: np.ndarray, ell_min: int = 20, ell_max: Optional[int] = None) -> np.ndarray:
    ell = np.arange(len(cl_obs), dtype=np.float64)
    if ell_max is None:
        ell_max = len(cl_obs) - 1

    dl = ell * (ell + 1.0) * cl_obs / (2.0 * np.pi)
    use = (ell >= ell_min) & (ell <= ell_max) & np.isfinite(dl) & (dl > 0)
    if np.sum(use) < 5:
        return flat_dl_null(cl_obs)

    x = np.log(ell[use])
    y = np.log(dl[use])
    slope, intercept = np.polyfit(x, y, deg=1)

    out_dl = np.zeros_like(cl_obs)
    good = ell >= 2
    out_dl[good] = np.exp(intercept) * np.power(ell[good], slope)

    out_cl = np.zeros_like(cl_obs)
    out_cl[good] = out_dl[good] * (2.0 * np.pi) / (ell[good] * (ell[good] + 1.0))
    return np.maximum(out_cl, 0.0)


def shuffled_spectrum_null(cl_base: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(cl_base)
    vals = cl_base[2:].copy()
    vals = np.maximum(vals, 0.0)
    rng.shuffle(vals)
    out[2:] = vals
    return out


def tilt_perturbation(cl_base: np.ndarray, pct: float, sign: int, pivot_ell: float = 80.0) -> np.ndarray:
    """
    Apply a gentle endpoint-normalized tilt perturbation.

    pct=2 means approximately +/-2 percent from one end of the ell range to the other.
    This survives amplitude fitting because it changes the relative shape.
    """
    ell = np.arange(len(cl_base), dtype=np.float64)
    out = cl_base.copy()
    good = ell >= 2
    x = np.log(np.maximum(ell[good], 2.0) / pivot_ell)
    x = x / max(np.nanmax(np.abs(x)), 1.0)
    factor = 1.0 + sign * (pct / 100.0) * x
    factor = np.maximum(factor, 1.0e-6)
    out[good] = np.maximum(out[good] * factor, 0.0)
    out[0:2] = 0.0
    return out


def ripple_perturbation(cl_base: np.ndarray, pct: float, sign: int, period: float = 55.0) -> np.ndarray:
    """
    Apply a small acoustic-scale sinusoidal perturbation.
    """
    ell = np.arange(len(cl_base), dtype=np.float64)
    out = cl_base.copy()
    good = ell >= 2
    factor = 1.0 + sign * (pct / 100.0) * np.sin(2.0 * np.pi * ell[good] / period)
    factor = np.maximum(factor, 1.0e-6)
    out[good] = np.maximum(out[good] * factor, 0.0)
    out[0:2] = 0.0
    return out


def build_envelope_library(
    cl_obs: np.ndarray,
    cl_cptg: np.ndarray,
    cl_planck: np.ndarray,
    *,
    smooth_width: int,
    null_shuffles: int,
    perturb_pcts: list[float],
    seed: int,
) -> list[dict]:
    envelopes = []

    def add(model: str, family: str, cl: np.ndarray, description: str) -> None:
        clean = np.asarray(cl, dtype=np.float64).copy()
        clean[~np.isfinite(clean)] = 0.0
        clean = np.maximum(clean, 0.0)
        clean[0:2] = 0.0
        envelopes.append(
            {
                "model": model,
                "family": family,
                "cl": clean,
                "description": description,
            }
        )

    add("planck2018_baseline_lcdm", "benchmark", cl_planck, "Planck 2018 baseline LCDM TT envelope")
    add("cptg_locked_pi_cmb", "cptg", cl_cptg, "Locked CPTG geometric-pi CMB comparison envelope")
    add("observed_self_spectrum_control", "data_control", cl_obs, "Observed self-spectrum upper control")
    add("observed_smoothed_spectrum_control", "data_control", moving_average_spectrum(cl_obs, smooth_width), f"Observed spectrum smoothed with width={smooth_width}")

    add("flat_cl_null", "bad_null", flat_cl_null(cl_obs), "Flat C_l null matched to mean observed C_l")
    add("flat_dl_null", "bad_null", flat_dl_null(cl_obs), "Flat D_l null matched to mean observed D_l")
    add("powerlaw_dl_null", "smooth_null", fit_powerlaw_dl_null(cl_obs), "Fitted smooth power-law D_l null")

    for i in range(null_shuffles):
        add(
            f"shuffled_spectrum_null_seed_{seed + i}",
            "shuffled_null",
            shuffled_spectrum_null(cl_obs, seed + i),
            "Observed spectrum values shuffled across ell",
        )

    for pct in perturb_pcts:
        label_pct = str(pct).replace(".", "p")
        for sign in [-1, 1]:
            s = "plus" if sign > 0 else "minus"
            add(
                f"cptg_tilt_{s}_{label_pct}pct",
                "cptg_perturbation",
                tilt_perturbation(cl_cptg, pct, sign),
                f"CPTG endpoint-normalized tilt perturbation {s} {pct} percent",
            )
            add(
                f"cptg_ripple_{s}_{label_pct}pct",
                "cptg_perturbation",
                ripple_perturbation(cl_cptg, pct, sign),
                f"CPTG sinusoidal ripple perturbation {s} {pct} percent",
            )
            add(
                f"planck_tilt_{s}_{label_pct}pct",
                "planck_perturbation",
                tilt_perturbation(cl_planck, pct, sign),
                f"Planck endpoint-normalized tilt perturbation {s} {pct} percent",
            )
            add(
                f"planck_ripple_{s}_{label_pct}pct",
                "planck_perturbation",
                ripple_perturbation(cl_planck, pct, sign),
                f"Planck sinusoidal ripple perturbation {s} {pct} percent",
            )

    return envelopes


def make_ranking_plot(path: Path, rankings: pd.DataFrame) -> None:
    if rankings.empty:
        return

    focus = rankings[rankings["model"].isin([
        "planck2018_baseline_lcdm",
        "cptg_locked_pi_cmb",
        "observed_self_spectrum_control",
        "observed_smoothed_spectrum_control",
        "flat_cl_null",
        "flat_dl_null",
        "powerlaw_dl_null",
    ])].copy()

    if focus.empty:
        return

    pivot = focus.pivot_table(index="component", columns="model", values="rank", aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(12, 7))
    ax.set_ylabel("Rank, lower is better")
    ax.set_title("Envelope rank by component")
    ax.legend(title="Envelope", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_residual_plot(path: Path, results: pd.DataFrame) -> None:
    if results.empty:
        return

    keep = [
        "planck2018_baseline_lcdm",
        "cptg_locked_pi_cmb",
        "observed_self_spectrum_control",
        "observed_smoothed_spectrum_control",
        "flat_cl_null",
        "flat_dl_null",
        "powerlaw_dl_null",
    ]
    focus = results[results["model"].isin(keep)].copy()
    if focus.empty:
        return

    pivot = focus.pivot_table(index="component", columns="model", values="residual_fraction", aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(12, 7))
    ax.set_ylabel("Residual RMS / observed RMS")
    ax.set_title("Residual fractions for target envelopes and null controls")
    ax.legend(title="Envelope", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_target_vs_null_plot(path: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    plot = summary.sort_values("mean_residual_fraction")
    plt.figure(figsize=(12, max(7, 0.24 * len(plot))))
    plt.barh(plot["model"], plot["mean_residual_fraction"])
    plt.xlabel("Mean residual fraction across components")
    plt.title("CPTG and Planck versus null-envelope controls")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPTG CMB null-envelope / uniqueness controls.")
    parser.add_argument("--map", action="append", default=None, help="Map as LABEL=path.fits. May be repeated.")
    parser.add_argument("--mask-source-map", default=None)
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.99)
    parser.add_argument("--use-inpainted", action="store_true")
    parser.add_argument("--smooth-width", type=int, default=31, help="Moving-average width for observed smoothed-spectrum control.")
    parser.add_argument("--null-shuffles", type=int, default=20)
    parser.add_argument("--perturb-pcts", default="0.5,1,2")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--out", default="cptg_cmb_null_envelope_controls")
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

    perturb_pcts = parse_float_list(args.perturb_pcts)
    lmax = 3 * args.nside - 1

    print(f"Computing locked CPTG and Planck envelopes to lmax={lmax}")
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

    result_rows = []
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

            mask_raw, mask_field, mask_source_used = select_mask(path, args.use_inpainted, mask_source_map)
            obs_low, valid = downgrade_map_and_mask(temp_uk, mask_raw, args.nside, args.mask_threshold)
            obs_clean = remove_monopole_dipole_masked(obs_low, valid)

            obs_filled = np.zeros_like(obs_clean)
            good = valid & np.isfinite(obs_clean)
            obs_filled[good] = obs_clean[good]

            print("  Computing observed phase scaffold")
            alm_obs = hp.map2alm(obs_filled, lmax=lmax, iter=0)
            cl_obs = hp.alm2cl(alm_obs)
            cl_obs[0:2] = 0.0

            library = build_envelope_library(
                cl_obs,
                cl_cptg,
                cl_planck,
                smooth_width=args.smooth_width,
                null_shuffles=args.null_shuffles,
                perturb_pcts=perturb_pcts,
                seed=args.seed,
            )

            print(f"  Evaluating {len(library)} envelopes")
            for item in library:
                template = build_phase_locked_map(alm_obs, cl_obs, item["cl"], args.nside, lmax)
                metrics = evaluate_template(obs_clean, template, valid)
                metrics.update(
                    {
                        "component": component,
                        "model": item["model"],
                        "family": item["family"],
                        "description": item["description"],
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
                result_rows.append(metrics)

            print(f"  done: temp={temperature_field}, unit={temp_unit}, mask={mask_field}, mask_source={mask_source_used}")

        except Exception as exc:
            print(f"  skipped after analysis error: {exc}")
            skipped_rows.append({"component": component, "path": path, "reason": f"analysis_failed: {exc}"})

    results = pd.DataFrame(result_rows)
    skipped = pd.DataFrame(skipped_rows)

    if not results.empty:
        results = results.sort_values(["component", "residual_fraction"]).reset_index(drop=True)
        results["rank"] = results.groupby("component")["residual_fraction"].rank(method="min", ascending=True).astype(int)

    results.to_csv(out / "null_envelope_results.csv", index=False)
    skipped.to_csv(out / "skipped_maps.csv", index=False)

    rankings = results[
        [
            "component",
            "model",
            "family",
            "rank",
            "residual_fraction",
            "residual_rms_uK",
            "masked_corr",
            "description",
        ]
    ].copy() if not results.empty else pd.DataFrame()
    rankings.to_csv(out / "null_envelope_rankings.csv", index=False)

    if not results.empty:
        summary = (
            results.groupby(["model", "family", "description"], as_index=False)
            .agg(
                mean_rank=("rank", "mean"),
                median_rank=("rank", "median"),
                best_rank=("rank", "min"),
                worst_rank=("rank", "max"),
                mean_residual_fraction=("residual_fraction", "mean"),
                std_residual_fraction=("residual_fraction", "std"),
                mean_residual_rms_uK=("residual_rms_uK", "mean"),
                std_residual_rms_uK=("residual_rms_uK", "std"),
                mean_corr=("masked_corr", "mean"),
                n_components=("component", "nunique"),
                n_rows=("model", "size"),
            )
            .sort_values(["mean_residual_fraction", "mean_rank"])
        )
    else:
        summary = pd.DataFrame()

    summary.to_csv(out / "null_envelope_summary_by_model.csv", index=False)

    if not results.empty:
        family_summary = (
            results.groupby("family", as_index=False)
            .agg(
                mean_rank=("rank", "mean"),
                median_rank=("rank", "median"),
                best_rank=("rank", "min"),
                worst_rank=("rank", "max"),
                mean_residual_fraction=("residual_fraction", "mean"),
                std_residual_fraction=("residual_fraction", "std"),
                mean_residual_rms_uK=("residual_rms_uK", "mean"),
                std_residual_rms_uK=("residual_rms_uK", "std"),
                n_models=("model", "nunique"),
                n_rows=("model", "size"),
            )
            .sort_values("mean_residual_fraction")
        )
    else:
        family_summary = pd.DataFrame()

    family_summary.to_csv(out / "null_envelope_family_summary.csv", index=False)

    make_ranking_plot(out / "ranking_by_component.png", rankings)
    make_residual_plot(out / "residual_fraction_by_model.png", results)
    make_target_vs_null_plot(out / "cptg_planck_vs_nulls.png", summary)

    report = []
    report.append("CPTG CMB null-envelope / uniqueness controls")
    report.append("=" * 50)
    report.append("")
    report.append(f"Nside: {args.nside}")
    report.append(f"Mask threshold: {args.mask_threshold}")
    report.append(f"Requested inpainted fields: {args.use_inpainted}")
    report.append(f"Mask source map: {mask_source_map}")
    report.append(f"Null shuffles: {args.null_shuffles}")
    report.append(f"Perturbation percentages: {', '.join([str(x) for x in perturb_pcts])}")
    report.append("")

    if results.empty:
        report.append("No map rows completed.")
    else:
        report.append("Top envelopes by mean residual fraction:")
        for _, row in summary.head(20).iterrows():
            report.append(
                f"  rank_mean={row['mean_rank']:.2f}, model={row['model']}, "
                f"family={row['family']}, mean_residual_fraction={row['mean_residual_fraction']:.8f}, "
                f"mean_RMS={row['mean_residual_rms_uK']:.6f} uK"
            )

        report.append("")
        report.append("Component-level top five:")
        for component in sorted(results["component"].unique()):
            report.append(f"  {component}:")
            sub = results[results["component"] == component].sort_values("residual_fraction").head(5)
            for _, row in sub.iterrows():
                report.append(
                    f"    rank={int(row['rank'])}, model={row['model']}, "
                    f"family={row['family']}, residual_fraction={row['residual_fraction']:.8f}, "
                    f"RMS={row['residual_rms_uK']:.6f} uK"
                )

        report.append("")
        report.append("Family summary:")
        for _, row in family_summary.iterrows():
            report.append(
                f"  {row['family']}: mean_rank={row['mean_rank']:.2f}, "
                f"mean_residual_fraction={row['mean_residual_fraction']:.8f}, "
                f"mean_RMS={row['mean_residual_rms_uK']:.6f} uK, "
                f"n_models={int(row['n_models'])}"
            )

        # Explicit CPTG-vs-Planck-vs-null interpretation.
        cptg = summary[summary["model"] == "cptg_locked_pi_cmb"]
        planck = summary[summary["model"] == "planck2018_baseline_lcdm"]
        if len(cptg) and len(planck):
            c = cptg.iloc[0]
            p = planck.iloc[0]
            report.append("")
            report.append("CPTG versus Planck benchmark:")
            report.append(
                f"  CPTG mean residual fraction={c['mean_residual_fraction']:.8f}, "
                f"mean RMS={c['mean_residual_rms_uK']:.6f} uK, mean rank={c['mean_rank']:.2f}"
            )
            report.append(
                f"  Planck mean residual fraction={p['mean_residual_fraction']:.8f}, "
                f"mean RMS={p['mean_residual_rms_uK']:.6f} uK, mean rank={p['mean_rank']:.2f}"
            )
            report.append(
                f"  CPTG-minus-Planck mean RMS difference="
                f"{c['mean_residual_rms_uK'] - p['mean_residual_rms_uK']:+.6f} uK"
            )

    if not skipped.empty:
        report.append("")
        report.append("Skipped maps:")
        for _, row in skipped.iterrows():
            report.append(f"  {row['component']}: {row['reason']}")

    report.append("")
    report.append("Interpretation boundary:")
    report.append("  The test asks whether CPTG's near-Planck map-space performance is")
    report.append("  automatic under phase locking. Null envelopes using the same phase")
    report.append("  scaffold provide the control. If generic nulls and perturbed envelopes")
    report.append("  perform substantially worse while CPTG remains adjacent to Planck,")
    report.append("  the near-degeneracy is not a trivial phase-locking artifact.")
    report.append("  This remains a phase-locked envelope comparison, not a native")
    report.append("  CPTG pixel-phase prediction.")

    report_text = "\n".join(report)
    (out / "null_envelope_report.txt").write_text(report_text, encoding="utf-8")

    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "maps": maps,
                "mask_source_map": mask_source_map,
                "nside": args.nside,
                "mask_threshold": args.mask_threshold,
                "use_inpainted": args.use_inpainted,
                "smooth_width": args.smooth_width,
                "null_shuffles": args.null_shuffles,
                "perturb_pcts": perturb_pcts,
                "seed": args.seed,
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
