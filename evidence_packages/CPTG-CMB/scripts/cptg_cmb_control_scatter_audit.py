#!/usr/bin/env python3
"""
CPTG CMB control-scatter / attribution audit.

Purpose
-------
Ingest the completed CPTG CMB comparison outputs and quantify whether the
CPTG-minus-Planck residual gap is smaller than, comparable to, or larger than
the scatter introduced by observational and comparison-coordinate controls.

This is the residual-attribution audit table for the CMB note/paper.

Default input folders, if present in the current directory
----------------------------------------------------------
Planck component robustness:
    cptg_planck_component_robustness_v4
    cptg_planck_component_robustness_v4_inpainted

Planck split-map stability:
    cptg_planck_splitmap_stability
    cptg_planck_splitmap_stability_bands

Planck low-ell band tests:
    cptg_planck_lowell_bandtest
    cptg_planck_lowell_bandtest_inpainted

WMAP low-ell band test:
    cptg_wmap_lowell_bandtest

Smoothing / beam-scale ladder:
    cptg_cmb_smoothing_ladder_v2
    cptg_cmb_smoothing_ladder_v2_inpainted
    cptg_cmb_smoothing_ladder_v2_bands

Mask / sky-fraction ladder:
    cptg_cmb_mask_ladder
    cptg_cmb_mask_ladder_inpainted
    cptg_cmb_mask_ladder_bands

The script also accepts .zip files with the same names or explicit --input
entries.

Examples
--------
python cptg_cmb_control_scatter_audit.py ^
  --out cptg_cmb_control_scatter_audit

With explicit inputs:
python cptg_cmb_control_scatter_audit.py ^
  --input PlanckComponents=cptg_planck_component_robustness_v4 ^
  --input MaskLadder=cptg_cmb_mask_ladder ^
  --out cptg_cmb_control_scatter_audit

Outputs
-------
all_gap_rows.csv
control_scatter_by_run.csv
control_scatter_by_family.csv
attribution_table.csv
control_scatter_report.txt
gap_vs_control_scatter.png
gap_distribution_by_family.png

Interpretation boundary
-----------------------
This audit compares phase-locked angular-power / transport-envelope metrics.
It does not establish native CPTG prediction of exact random CMB pixel phases.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUTS = [
    # label, path, family, expected csv filename
    ("Planck components non-inpainted", "cptg_planck_component_robustness_v4", "component_separation", "component_cptg_vs_planck_delta.csv"),
    ("Planck components inpainted", "cptg_planck_component_robustness_v4_inpainted", "component_separation_inpainted", "component_cptg_vs_planck_delta.csv"),

    ("Planck SMICA split full-band", "cptg_planck_splitmap_stability", "split_maps", "splitmap_cptg_vs_planck_delta.csv"),
    ("Planck SMICA split bands", "cptg_planck_splitmap_stability_bands", "split_maps_bandlimited", "splitmap_band_cptg_vs_planck_delta.csv"),

    ("Planck low-ell bands non-inpainted", "cptg_planck_lowell_bandtest", "planck_lowell_bands", "planck_band_cptg_vs_planck_delta.csv"),
    ("Planck low-ell bands inpainted", "cptg_planck_lowell_bandtest_inpainted", "planck_lowell_bands_inpainted", "planck_band_cptg_vs_planck_delta.csv"),

    ("WMAP low-ell bands", "cptg_wmap_lowell_bandtest", "wmap_lowell_bands", "wmap_band_cptg_vs_planck_delta.csv"),

    ("Smoothing ladder non-inpainted", "cptg_cmb_smoothing_ladder_v2", "smoothing_beam_transfer", "smoothing_ladder_delta.csv"),
    ("Smoothing ladder inpainted", "cptg_cmb_smoothing_ladder_v2_inpainted", "smoothing_beam_transfer_inpainted", "smoothing_ladder_delta.csv"),
    ("Smoothing ladder bands", "cptg_cmb_smoothing_ladder_v2_bands", "smoothing_beam_transfer_bandlimited", "smoothing_ladder_band_delta.csv"),

    ("Mask ladder non-inpainted", "cptg_cmb_mask_ladder", "mask_sky_fraction", "mask_ladder_delta.csv"),
    ("Mask ladder inpainted", "cptg_cmb_mask_ladder_inpainted", "mask_sky_fraction_inpainted", "mask_ladder_delta.csv"),
    ("Mask ladder bands", "cptg_cmb_mask_ladder_bands", "mask_sky_fraction_bandlimited", "mask_ladder_band_delta.csv"),
]


REQUIRED_GAP_COLUMNS = [
    "cptg_residual_fraction",
    "planck_residual_fraction",
    "cptg_minus_planck_residual_fraction",
    "cptg_residual_rms_uK",
    "planck_residual_rms_uK",
    "cptg_minus_planck_residual_rms_uK",
]


def parse_input_args(items: Optional[list[str]]) -> list[tuple[str, str, str, Optional[str]]]:
    """
    Parse explicit --input entries.

    Accepted formats:
        label=path
        label=path:csv_name
        label|family=path
        label|family=path:csv_name
    """
    if not items:
        return []

    parsed = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --input entry: {item}. Use label=path or label|family=path.")
        left, right = item.split("=", 1)
        left = left.strip()
        right = right.strip().strip('"')

        if "|" in left:
            label, family = left.split("|", 1)
            label = label.strip()
            family = family.strip()
        else:
            label = left
            family = "custom"

        csv_name = None
        # Allow Windows paths with drive letters. Only treat suffix after last colon
        # as csv override when it ends in .csv.
        if ":" in right:
            maybe_path, maybe_csv = right.rsplit(":", 1)
            if maybe_csv.lower().endswith(".csv"):
                right = maybe_path
                csv_name = maybe_csv

        parsed.append((label, right, family, csv_name))

    return parsed


def candidate_paths(base: str) -> list[Path]:
    p = Path(base)
    candidates = [p]

    if p.suffix.lower() != ".zip":
        candidates.append(Path(base + ".zip"))

    return candidates


def read_csv_from_folder_or_zip(base: str, csv_name: str) -> Optional[pd.DataFrame]:
    """
    Read a CSV from either:
        - a folder containing csv_name
        - a zip archive containing csv_name anywhere inside
        - a direct CSV file path
    """
    for p in candidate_paths(base):
        if not p.exists():
            continue

        if p.is_file() and p.suffix.lower() == ".csv":
            try:
                return pd.read_csv(p)
            except Exception as exc:
                print(f"Could not read CSV {p}: {exc}")
                return None

        if p.is_dir():
            direct = p / csv_name
            if direct.exists():
                try:
                    return pd.read_csv(direct)
                except Exception as exc:
                    print(f"Could not read {direct}: {exc}")
                    return None

            matches = list(p.rglob(csv_name))
            if matches:
                try:
                    return pd.read_csv(matches[0])
                except Exception as exc:
                    print(f"Could not read {matches[0]}: {exc}")
                    return None

        if p.is_file() and p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p, "r") as z:
                    names = z.namelist()
                    exact = [n for n in names if Path(n).name == csv_name]
                    if not exact and csv_name == "":
                        exact = [n for n in names if n.lower().endswith(".csv")]
                    if exact:
                        with z.open(exact[0]) as f:
                            return pd.read_csv(f)
            except Exception as exc:
                print(f"Could not read {csv_name} from zip {p}: {exc}")
                return None

    return None


def infer_csv_name_for_path(path: str) -> str:
    """
    Try to infer the expected CSV name from a folder/zip name.
    """
    lower = Path(path).name.lower()

    if "component_robustness" in lower:
        return "component_cptg_vs_planck_delta.csv"
    if "splitmap" in lower and "band" in lower:
        return "splitmap_band_cptg_vs_planck_delta.csv"
    if "splitmap" in lower:
        return "splitmap_cptg_vs_planck_delta.csv"
    if "planck_lowell" in lower:
        return "planck_band_cptg_vs_planck_delta.csv"
    if "wmap_lowell" in lower:
        return "wmap_band_cptg_vs_planck_delta.csv"
    if "smoothing" in lower and "band" in lower:
        return "smoothing_ladder_band_delta.csv"
    if "smoothing" in lower:
        return "smoothing_ladder_delta.csv"
    if "mask" in lower and "band" in lower:
        return "mask_ladder_band_delta.csv"
    if "mask" in lower:
        return "mask_ladder_delta.csv"

    return ""


def standardize_delta_frame(
    df: pd.DataFrame,
    run_label: str,
    family: str,
    source_path: str,
) -> pd.DataFrame:
    out = df.copy()

    missing = [c for c in REQUIRED_GAP_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"{run_label}: missing required columns: {missing}")

    out["run"] = run_label
    out["family"] = family
    out["source_path"] = source_path

    # Create a generic control_id for plotting/reporting.
    control_parts = []
    for col in ["component", "split", "ell_band", "smoothing_label", "mask_label", "galactic_cut_label"]:
        if col in out.columns:
            control_parts.append(out[col].astype(str))
    if control_parts:
        cid = control_parts[0]
        for part in control_parts[1:]:
            cid = cid + " | " + part
        out["control_id"] = cid
    else:
        out["control_id"] = out.index.astype(str)

    # Ensure numeric columns.
    for col in REQUIRED_GAP_COLUMNS + ["valid_sky_fraction"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["planck_ahead"] = out["cptg_minus_planck_residual_fraction"] > 0
    out["cptg_ahead"] = out["cptg_minus_planck_residual_fraction"] < 0
    out["abs_gap_rms_uK"] = out["cptg_minus_planck_residual_rms_uK"].abs()
    out["abs_gap_residual_fraction"] = out["cptg_minus_planck_residual_fraction"].abs()

    return out


def summarize_group(df: pd.DataFrame, group_label: str, group_type: str) -> dict:
    if df is None or df.empty:
        return {
            "group": group_label,
            "group_type": group_type,
            "status": "empty",
            "n_rows": 0,
        }

    gap = df["cptg_minus_planck_residual_fraction"].dropna()
    gap_rms = df["cptg_minus_planck_residual_rms_uK"].dropna()
    abs_gap_rms = df["abs_gap_rms_uK"].dropna()

    mean_gap = gap.mean()
    mean_gap_rms = gap_rms.mean()
    mean_abs_gap_rms = abs_gap_rms.mean()

    span_gap = gap.max() - gap.min() if len(gap) else np.nan
    span_gap_rms = gap_rms.max() - gap_rms.min() if len(gap_rms) else np.nan
    std_gap = gap.std(ddof=1) if len(gap) > 1 else np.nan
    std_gap_rms = gap_rms.std(ddof=1) if len(gap_rms) > 1 else np.nan

    sign_flip = bool((gap < 0).any() and (gap > 0).any()) if len(gap) else False
    planck_ahead_fraction = float((gap > 0).mean()) if len(gap) else np.nan
    cptg_ahead_fraction = float((gap < 0).mean()) if len(gap) else np.nan

    if sign_flip:
        attribution = "sign-changing; attribution-limited"
    elif np.isfinite(span_gap_rms) and np.isfinite(mean_abs_gap_rms) and span_gap_rms >= mean_abs_gap_rms:
        attribution = "scatter comparable/larger than mean gap; attribution-limited"
    elif np.isfinite(std_gap_rms) and np.isfinite(mean_abs_gap_rms) and std_gap_rms >= 0.5 * mean_abs_gap_rms:
        attribution = "scatter significant; attribution-limited"
    elif np.isfinite(mean_abs_gap_rms):
        attribution = "metric-stable sign; physical attribution still comparison-limited"
    else:
        attribution = "insufficient data"

    return {
        "group": group_label,
        "group_type": group_type,
        "status": "ok",
        "n_rows": int(len(df)),
        "mean_gap_residual_fraction": mean_gap,
        "median_gap_residual_fraction": gap.median() if len(gap) else np.nan,
        "std_gap_residual_fraction": std_gap,
        "min_gap_residual_fraction": gap.min() if len(gap) else np.nan,
        "max_gap_residual_fraction": gap.max() if len(gap) else np.nan,
        "span_gap_residual_fraction": span_gap,
        "mean_gap_rms_uK": mean_gap_rms,
        "median_gap_rms_uK": gap_rms.median() if len(gap_rms) else np.nan,
        "std_gap_rms_uK": std_gap_rms,
        "min_gap_rms_uK": gap_rms.min() if len(gap_rms) else np.nan,
        "max_gap_rms_uK": gap_rms.max() if len(gap_rms) else np.nan,
        "span_gap_rms_uK": span_gap_rms,
        "mean_abs_gap_rms_uK": mean_abs_gap_rms,
        "gap_to_span_ratio_rms": mean_abs_gap_rms / span_gap_rms if np.isfinite(span_gap_rms) and span_gap_rms != 0 else np.nan,
        "gap_to_std_ratio_rms": mean_abs_gap_rms / std_gap_rms if np.isfinite(std_gap_rms) and std_gap_rms != 0 else np.nan,
        "planck_ahead_fraction": planck_ahead_fraction,
        "cptg_ahead_fraction": cptg_ahead_fraction,
        "sign_flip": sign_flip,
        "attribution_read": attribution,
    }


def make_gap_vs_scatter_plot(path: Path, table: pd.DataFrame, title: str) -> None:
    if table.empty:
        return

    plot = table[table["status"] == "ok"].copy()
    if plot.empty:
        return

    plot = plot.sort_values("mean_abs_gap_rms_uK", ascending=True)

    y = np.arange(len(plot))
    plt.figure(figsize=(12, max(6, 0.45 * len(plot))))
    plt.barh(y, plot["mean_abs_gap_rms_uK"], label="mean absolute RMS gap")
    plt.errorbar(
        plot["mean_abs_gap_rms_uK"],
        y,
        xerr=plot["std_gap_rms_uK"].fillna(0.0),
        fmt="none",
        capsize=3,
        label="1-sigma control scatter",
    )
    plt.yticks(y, plot["group"])
    plt.xlabel("RMS gap / scatter [uK]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_distribution_plot(path: Path, all_rows: pd.DataFrame) -> None:
    if all_rows.empty:
        return

    families = list(all_rows["family"].dropna().unique())
    data = [all_rows.loc[all_rows["family"] == f, "cptg_minus_planck_residual_rms_uK"].dropna().to_numpy() for f in families]
    data = [d for d in data if len(d)]
    labels = [f for f, d in zip(families, data) if len(d)]

    if not data:
        return

    plt.figure(figsize=(13, max(6, 0.35 * len(labels))))
    plt.boxplot(data, vert=False, labels=labels)
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("CPTG minus Planck residual RMS [uK]")
    plt.title("Control-family distribution of CPTG-minus-Planck RMS gaps")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_hist_plot(path: Path, all_rows: pd.DataFrame) -> None:
    if all_rows.empty:
        return
    x = all_rows["cptg_minus_planck_residual_rms_uK"].dropna().to_numpy()
    if len(x) == 0:
        return
    plt.figure(figsize=(9, 6))
    plt.hist(x, bins=40)
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("CPTG minus Planck residual RMS [uK]")
    plt.ylabel("Rows")
    plt.title("All control rows: gap histogram")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def simple_table(df: pd.DataFrame, cols: list[str], max_rows: Optional[int] = None) -> str:
    if df is None or df.empty:
        return "_No rows._"

    use = df[cols].copy()
    if max_rows is not None:
        use = use.head(max_rows)

    def cell(v) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.8g}"
        return str(v)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in use.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPTG CMB control-scatter / attribution audit.")
    parser.add_argument("--input", action="append", default=None, help="Optional input as label=path or label|family=path[:csv_name]. May repeat.")
    parser.add_argument("--out", default="cptg_cmb_control_scatter_audit")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    explicit = parse_input_args(args.input)

    inputs = []
    if explicit:
        for label, path, family, csv_name in explicit:
            inputs.append((label, path, family, csv_name or infer_csv_name_for_path(path)))
    else:
        inputs = [(label, path, family, csv_name) for label, path, family, csv_name in DEFAULT_INPUTS]

    loaded_frames = []
    load_rows = []

    for label, path, family, csv_name in inputs:
        df = read_csv_from_folder_or_zip(path, csv_name)
        if df is None or df.empty:
            load_rows.append(
                {
                    "run": label,
                    "family": family,
                    "path": path,
                    "csv_name": csv_name,
                    "status": "missing_or_empty",
                    "rows": 0,
                }
            )
            continue

        try:
            std = standardize_delta_frame(df, label, family, path)
            loaded_frames.append(std)
            load_rows.append(
                {
                    "run": label,
                    "family": family,
                    "path": path,
                    "csv_name": csv_name,
                    "status": "ok",
                    "rows": len(std),
                }
            )
        except Exception as exc:
            load_rows.append(
                {
                    "run": label,
                    "family": family,
                    "path": path,
                    "csv_name": csv_name,
                    "status": f"failed: {exc}",
                    "rows": 0,
                }
            )

    load_table = pd.DataFrame(load_rows)
    load_table.to_csv(out / "input_load_status.csv", index=False)

    if loaded_frames:
        all_rows = pd.concat(loaded_frames, ignore_index=True)
    else:
        all_rows = pd.DataFrame()

    all_rows.to_csv(out / "all_gap_rows.csv", index=False)

    run_summary = []
    family_summary = []

    if not all_rows.empty:
        for run in all_rows["run"].unique():
            run_summary.append(summarize_group(all_rows[all_rows["run"] == run], run, "run"))

        for family in all_rows["family"].unique():
            family_summary.append(summarize_group(all_rows[all_rows["family"] == family], family, "family"))

        run_summary.append(summarize_group(all_rows, "ALL_CONTROL_ROWS", "overall"))

    run_table = pd.DataFrame(run_summary)
    family_table = pd.DataFrame(family_summary)
    attribution_table = run_table.copy()

    run_table.to_csv(out / "control_scatter_by_run.csv", index=False)
    family_table.to_csv(out / "control_scatter_by_family.csv", index=False)
    attribution_table.to_csv(out / "attribution_table.csv", index=False)

    make_gap_vs_scatter_plot(out / "gap_vs_control_scatter.png", run_table, "Mean RMS gap versus control scatter by run")
    make_distribution_plot(out / "gap_distribution_by_family.png", all_rows)
    make_hist_plot(out / "all_gap_histogram.png", all_rows)

    overall = summarize_group(all_rows, "ALL_CONTROL_ROWS", "overall") if not all_rows.empty else {"status": "empty"}

    report = []
    report.append("CPTG CMB control-scatter / attribution audit")
    report.append("=" * 48)
    report.append("")
    report.append("Input load status")
    report.append("-----------------")
    report.append(simple_table(load_table, ["run", "family", "status", "rows"], max_rows=None))
    report.append("")

    if all_rows.empty:
        report.append("No valid input rows loaded.")
    else:
        report.append("Overall result")
        report.append("--------------")
        report.append(f"Total loaded rows: {len(all_rows)}")
        report.append(f"Mean CPTG-minus-Planck RMS gap: {overall['mean_gap_rms_uK']:+.8f} uK")
        report.append(f"Mean absolute RMS gap: {overall['mean_abs_gap_rms_uK']:.8f} uK")
        report.append(f"RMS-gap span across all rows: {overall['span_gap_rms_uK']:.8f} uK")
        report.append(f"RMS-gap standard deviation across all rows: {overall['std_gap_rms_uK']:.8f} uK")
        report.append(f"Planck-ahead fraction: {overall['planck_ahead_fraction']:.6f}")
        report.append(f"CPTG-ahead fraction: {overall['cptg_ahead_fraction']:.6f}")
        report.append(f"Sign flip present: {overall['sign_flip']}")
        report.append(f"Attribution read: {overall['attribution_read']}")
        report.append("")

        report.append("Run-level attribution table")
        report.append("---------------------------")
        cols = [
            "group",
            "n_rows",
            "mean_gap_rms_uK",
            "std_gap_rms_uK",
            "span_gap_rms_uK",
            "planck_ahead_fraction",
            "sign_flip",
            "attribution_read",
        ]
        report.append(simple_table(run_table[run_table["group"] != "ALL_CONTROL_ROWS"], cols, max_rows=None))
        report.append("")

        report.append("Family-level attribution table")
        report.append("------------------------------")
        report.append(simple_table(family_table, cols, max_rows=None))
        report.append("")

        report.append("Interpretation")
        report.append("--------------")
        report.append("The audit compares the measured CPTG-minus-Planck gap with the scatter and span")
        report.append("generated by component choice, split maps, low-ell band choice, smoothing/beam")
        report.append("scale, mask threshold, Galactic sky selection, and WMAP cross-mission bands.")
        report.append("")
        report.append("A stable positive mean gap means the selected phase-locked RMS metric usually")
        report.append("places the Planck baseline slightly ahead. However, if the scatter/span is")
        report.append("comparable to or larger than the mean absolute gap, or if sign changes occur,")
        report.append("the residual is attribution-limited and should not be assigned uniquely to CPTG.")
        report.append("")
        report.append("Claim boundary")
        report.append("--------------")
        report.append("This audit supports the CPTG angular-power / transport-envelope comparison layer.")
        report.append("It does not claim native CPTG prediction of the exact random CMB pixel phases.")

    report_text = "\n".join(report)
    (out / "control_scatter_report.txt").write_text(report_text, encoding="utf-8")
    (out / "control_scatter_report.md").write_text(report_text, encoding="utf-8")

    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": [
                    {"label": label, "path": path, "family": family, "csv_name": csv_name}
                    for label, path, family, csv_name in inputs
                ],
                "loaded_rows": 0 if all_rows.empty else int(len(all_rows)),
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
