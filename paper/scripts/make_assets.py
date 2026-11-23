"""Build the manuscript's numbers, tables and figures from the saved result files.

    python paper/scripts/make_assets.py            # numbers.tex, tables, charts, review page (light: CSV/JSON only)
    python paper/scripts/make_assets.py --images   # + image figures (data audit, qualitative): reads fundus
                                                   #   images and probability maps -> run as a SLURM job

Rules (PAPER_PROMPT.md): every measured value comes from results/ files through this script; nothing is
typed by hand. Each macro in numbers.tex is preceded by a machine-readable provenance line
("% PROV {...}") that paper/scripts/check_numbers.py re-derives independently.

Outputs
  paper/numbers.tex             \\newcommand macros (text mode; negatives use \\textminus)
  paper/tables/*.tex            LaTeX tables (PLOS template rules: cell-based, \\hline, no nesting)
  paper/figures/*.pdf           vector charts (DejaVu Sans; white background); previews in figures/preview/*.png
  paper/figures/*_raster.png    image figures at 300 dpi, everything outside the FOV pixelated (with --images)
  paper/captions.tex            caption macros (\\Cap...) for Step 3
  paper/assets_review.md        every figure and table with its draft caption, for review
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
TABLES_IN, EVAL, TRAIN, CHECKS = RES / "tables", RES / "eval", RES / "train", RES / "checks"
PAPER = ROOT / "paper"
FIG, TAB, PREV = PAPER / "figures", PAPER / "tables", PAPER / "figures" / "preview"

# ----------------------------------------------------------------------------------------- models
MODELS = [  # run, macro code, full name, short name, family
    ("unet", "UNet", "U-Net", "U-Net", "cnn"),
    ("attention_unet", "Att", "Attention U-Net", "Att U-Net", "cnn"),
    ("multiscale_unet", "MS", "Multi-Scale U-Net", "MS U-Net", "cnn"),
    ("ms_attention_unet", "MSAtt", "Multi-Scale Attention U-Net", "MS-Att U-Net", "cnn"),
    ("sam_lora_tiled", "SAMTiled", "SAM, tiled (main SAM)", "SAM tiled", "sam"),
    ("sam_lora_tiled_overlap", "SAMOverlap", "SAM, overlapping tiles", "SAM overlap", "sam"),
    ("sam_lora_whole", "SAMWhole", "SAM, whole image", "SAM whole", "sam"),
]
RUNS = [m[0] for m in MODELS]
CODE = {m[0]: m[1] for m in MODELS}
NAME = {m[0]: m[2] for m in MODELS}
SHORT = {m[0]: m[3] for m in MODELS}
FAMILY = {m[0]: m[4] for m in MODELS}
CNN = RUNS[:4]
SEEDS = (0, 1, 2)
SEEDCODE = {0: "A", 1: "B", 2: "C"}
METRICS = [("dice", "Dice", "Dice"), ("iou", "IoU", "IoU"), ("precision", "Prec", "Precision"),
           ("recall", "Rec", "Recall"), ("cl_recall_thin", "Thin", "Thin"), ("cl_recall_medium", "Med", "Medium"),
           ("cl_recall_thick", "Thick", "Thick"), ("cldice", "ClDice", "clDice"), ("nsd", "NSD", "NSD"),
           ("cl_recall_thin_r4_8", "ThinAlt", "Thin (4/8)"), ("cl_recall_medium_r4_8", "MedAlt", "Medium (4/8)"),
           ("cl_recall_thick_r4_8", "ThickAlt", "Thick (4/8)")]
MCODE = {m[0]: m[1] for m in METRICS}
MLABEL = {m[0]: m[2] for m in METRICS}
MAIN9 = [m[0] for m in METRICS[:9]]
CATS = [("A", "AMD"), ("D", "DR"), ("G", "Glaucoma"), ("N", "Normal")]

# ----------------------------------------------------------------------------------------- style
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
# Categorical slots (dataviz reference palette), validated on white with --pairs all per family:
#   CNN family + U-Net {#2a78d6,#e34948,#1baf7a,#eda100}: pass (CVD floor 6.9 -> markers required);
#   SAM family + U-Net {#2a78d6,#e87ba4,#008300}: pass. Families are never colour-coded in one panel;
#   panels with all models identify them by direct labels and marker shape instead.
STYLE = {
    "unet": dict(color="#2a78d6", marker="o", ls="-"),
    "attention_unet": dict(color="#e34948", marker="s", ls="-"),
    "multiscale_unet": dict(color="#1baf7a", marker="^", ls="-"),
    "ms_attention_unet": dict(color="#eda100", marker="D", ls="-"),
    "sam_lora_tiled": dict(color="#e87ba4", marker="v", ls="-"),
    "sam_lora_tiled_overlap": dict(color="#e87ba4", marker="v", ls=(0, (4, 2)), hollow=True),
    "sam_lora_whole": dict(color="#008300", marker="P", ls="-"),
}
ERR = {"bg": (255, 255, 255), "tp": (150, 149, 143), "fp": (235, 104, 52), "fn": (42, 120, 214)}
FULL_W = 6.5  # inches: text width of the working template (1 in margins); within PLOS's 2.63-7.5 in


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False,
        "lines.linewidth": 1.3, "lines.markersize": 4.5, "savefig.dpi": 300})
    return plt


def seedlab(run: str, n: int = 3) -> str:
    return f"{NAME[run].replace(' (main SAM)', '')} ({n} seed{'s' if n != 1 else ''})"


# ----------------------------------------------------------------------------------------- formatting
def fmt(raw: float, kind: str) -> str:
    """Text-mode LaTeX number. kinds: fN, sfN (explicit sign), pctN (x100), int (thousands commas),
    sci (p-value: 3 decimals, or m x 10^e below 0.001)."""
    if kind == "int":
        s = f"{int(round(raw)):,}"
    elif kind.startswith("sf"):
        d = int(kind[2:])
        s = f"{abs(raw):.{d}f}"
        if float(s) != 0:
            s = ("+" if raw > 0 else "-") + s
    elif kind.startswith("pct"):
        s = f"{100 * raw:.{int(kind[3:])}f}"
    elif kind.startswith("f"):
        s = f"{raw:.{int(kind[1:])}f}"
    elif kind == "sci":
        if raw >= 0.001:
            s = f"{raw:.3f}"
        else:
            m, e = f"{raw:.1e}".split("e")
            return f"${m}\\times10^{{{int(e)}}}$"
    else:
        raise ValueError(kind)
    return s.replace("-", "\\textminus{}") if s.startswith("-") else s


TEX_SAFE = {"–": "--", "−": "$-$", "≥": "$\\geq$", "≤": "$\\leq$", "±": "$\\pm$", "×": "$\\times$", "°": "\\textdegree{}",
            "<": "$<$", ">": "$>$", "%": "\\%"}


def texsafe(s: str) -> str:
    """Unicode symbols and < > % -> commands pdfLaTeX typesets with the template's packages."""
    return "".join(TEX_SAFE.get(ch, ch) for ch in s)


def plain(tex: str) -> str:
    """LaTeX number text -> plain text (review page)."""
    out = tex.replace("\\textminus{}", "−").replace("$", "").replace("\\times10^{", "×10^").replace("}", "")
    return out


# ----------------------------------------------------------------------------------------- provenance
_CSV: dict[str, pd.DataFrame] = {}
_JSON: dict[str, object] = {}


def csv(rel: str) -> pd.DataFrame:
    if rel not in _CSV:
        _CSV[rel] = pd.read_csv(ROOT / rel)
    return _CSV[rel]


def js(rel: str):
    if rel not in _JSON:
        _JSON[rel] = json.loads((ROOT / rel).read_text())
    return _JSON[rel]


def _filter(df: pd.DataFrame, where: dict) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    for k, v in where.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            m &= np.isclose(pd.to_numeric(df[k], errors="coerce").to_numpy(float), float(v))
        else:
            m &= (df[k].astype(str) == str(v)).to_numpy()
    return df[m]


def evaluate(p: dict, nums: "Numbers") -> float:
    op = p["op"]
    if op in ("cell", "mean", "agg"):
        d = _filter(csv(p["file"]), p.get("where", {}))
        if op == "cell":
            if len(d) != 1:
                raise ValueError(f"cell lookup matched {len(d)} rows: {p}")
            return float(d[p["col"]].iloc[0])
        if op == "mean":
            return float(pd.to_numeric(d[p["col"]], errors="coerce").dropna().mean())
        if p["fn"] == "count":
            return float(len(d))
        return float(getattr(pd.to_numeric(d[p["col"]], errors="coerce").dropna(), p["fn"])())
    if op == "json":
        x = js(p["file"])
        for k in p["path"]:
            x = x[k]
        return float(x)
    if op == "json_list":
        x = js(p["file"])
        for k in p["path"]:
            x = x[k]
        x = [e for e in x if all(e.get(k) == v for k, v in p.get("match", {}).items())]
        if p["fn"] == "count":
            return float(len(x))
        vals = [float(e[p["field"]]) for e in x]
        if p["fn"] == "one":
            if len(vals) != 1:
                raise ValueError(f"json_list one matched {len(vals)}: {p}")
            return vals[0]
        return float({"min": min, "max": max}[p["fn"]](vals))
    if op == "curve":
        d = csv(p["file"])
        d = d[d.split == p["split"]].pivot(index="threshold", columns="run", values=["cl_recall_thin", "seed_min", "seed_max"])
        subj = d["cl_recall_thin"][p["subject"]]
        if p["kind"] == "gap_best":
            vals = subj - d["cl_recall_thin"][p["refs"]].max(axis=1)
        elif p["kind"] == "diff":
            vals = subj - d["cl_recall_thin"][p["refs"][0]]
        elif p["kind"] == "overlap_count":
            r = p["refs"][0]
            ov = (d["seed_min"][p["subject"]] <= d["seed_max"][r]) & (d["seed_min"][r] <= d["seed_max"][p["subject"]])
            return float(ov.sum())
        elif p["kind"] == "above_count":
            return float((subj - d["cl_recall_thin"][p["refs"]].max(axis=1) > 0).sum())
        else:
            raise ValueError(p)
        return float({"min": vals.min, "max": vals.max}[p["fn"]]())
    if op == "expr":
        env = {k: nums.raw[v] for k, v in p["args"].items()}
        return float(eval(p["expr"], {"min": min, "max": max, "abs": abs, "__builtins__": {}}, env))
    raise ValueError(op)


class Numbers:
    def __init__(self):
        self.items: list[tuple[str, str, dict]] = []
        self.raw: dict[str, float] = {}
        self.text: dict[str, str] = {}

    def add(self, name: str, kind: str, prov: dict) -> float:
        if not name.isalpha() or name in self.raw:
            raise ValueError(f"bad or duplicate macro name {name!r}")
        raw = evaluate(prov, self)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            raise ValueError(f"{name}: NaN from {prov}")
        prov = dict(prov, fmt=kind)
        self.raw[name], self.text[name] = raw, fmt(raw, kind)
        self.items.append((name, self.text[name], prov))
        return raw

    def write(self, path: Path):
        lines = ["% numbers.tex -- generated by paper/scripts/make_assets.py. NEVER edit by hand.",
                 "% Each macro is preceded by its provenance; paper/scripts/check_numbers.py re-derives it.", ""]
        for name, text, prov in self.items:
            lines.append("% PROV " + json.dumps(dict(prov, name=name), sort_keys=True))
            lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
        path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------- numbers
def build_numbers() -> Numbers:
    N = Numbers()
    tm, tw = "results/tables/test_main.csv", "results/tables/test_wilcoxon.csv"
    tf, tfw = "results/tables/test_main_fixed_t0.5.csv", "results/tables/test_wilcoxon_fixed_t0.5.csv"

    # --- dataset, split, audit
    p1, sp, ig = "results/checks/phase1_checks.json", "results/checks/split_summary.json", "results/checks/integrity.json"
    N.add("NTrainOfficial", "int", {"op": "json", "file": p1, "path": ["train_images"]})
    N.add("NTestOfficial", "int", {"op": "json", "file": p1, "path": ["test_images"]})
    N.add("NImages", "int", {"op": "expr", "expr": "a+b", "args": {"a": "NTrainOfficial", "b": "NTestOfficial"}})
    N.add("NPerCategory", "int", {"op": "json", "file": p1, "path": ["category_counts", "All", "A"]})
    N.add("NTestPerCategory", "int", {"op": "json", "file": p1, "path": ["category_counts", "test", "A"]})
    for s, nm in (("train", "NTrain"), ("val", "NVal"), ("test", "NTest"), ("excluded", "NExcluded")):
        N.add(nm, "int", {"op": "json", "file": sp, "path": ["counts", s, "All"]})
    N.add("NValPerCategory", "int", {"op": "json", "file": sp, "path": ["counts", "val", "A"]})
    N.add("NTrainVal", "int", {"op": "expr", "expr": "a+b", "args": {"a": "NTrain", "b": "NVal"}})
    N.add("SplitSeed", "int", {"op": "json", "file": sp, "path": ["seed"]})
    N.add("NGroups", "int", {"op": "json", "file": sp, "path": ["groups"]})
    N.add("NGroupedTrain", "int", {"op": "json", "file": sp, "path": ["grouped_images_by_split", "train"]})
    N.add("NGroupedVal", "int", {"op": "json", "file": sp, "path": ["grouped_images_by_split", "val"]})
    N.add("NGroupedImages", "int", {"op": "expr", "expr": "a+b", "args": {"a": "NGroupedTrain", "b": "NGroupedVal"}})
    N.add("NGroupPairsAdjacent", "int", {"op": "json", "file": sp, "path": ["group_pairs", "identical marking (IoU >= 0.8) + adjacent numbers"]})
    N.add("NGroupPairsIsolated", "int", {"op": "json", "file": sp, "path": ["group_pairs", "identical marking (IoU >= 0.8), isolated pair"]})
    N.add("NGroupPairsIdentical", "int", {"op": "json", "file": sp, "path": ["group_pairs", "identical image file"]})
    N.add("NGroupsCrossing", "int", {"op": "json", "file": sp, "path": ["groups_crossing_train_val"]})
    grp = ["identical_images", "groups"]
    N.add("NDupPairsTrain", "int", {"op": "json_list", "file": ig, "path": grp, "fn": "count", "match": {"splits": ["train", "train"]}})
    N.add("NDupPairsTest", "int", {"op": "json_list", "file": ig, "path": grp, "fn": "count", "match": {"splits": ["test", "test"]}})
    N.add("DupTrainMaskDiceMin", "f3", {"op": "json_list", "file": ig, "path": grp, "fn": "min", "field": "mask_dice", "match": {"splits": ["train", "train"]}})
    N.add("DupTrainMaskDiceMax", "f3", {"op": "json_list", "file": ig, "path": grp, "fn": "max", "field": "mask_dice", "match": {"splits": ["train", "train"]}})
    N.add("DupTestMaskDice", "f3", {"op": "json_list", "file": ig, "path": grp, "fn": "one", "field": "mask_dice", "match": {"splits": ["test", "test"]}})
    N.add("NSharedMaskTestPairs", "int", {"op": "json_list", "file": ig, "path": ["identical_masks_on_different_images"],
                                          "fn": "count", "match": {"splits": ["test", "test"]}})
    N.add("NEmptyMasks", "int", {"op": "json_list", "file": ig, "path": ["empty_masks"], "fn": "count"})
    N.add("NBorderMasks", "int", {"op": "json", "file": ig, "path": ["gt_outside_fov", "by_kind", "border_line"]})
    N.add("NBorderPx", "int", {"op": "json", "file": ig, "path": ["gt_outside_fov", "px_by_kind", "border_line"]})
    N.add("NMarkings", "int", {"op": "json", "file": p1, "path": ["markings", "n_flagged"]})
    spp = "results/checks/same_patient_pairs.json"
    iso = ["by_threshold", "0.8", "isolated_pairs"]
    N.add("NIsoPairs", "int", {"op": "json", "file": spp, "path": iso + ["n_pairs"]})
    for k, nm in (("train-test", "NIsoTrainTest"), ("val-test", "NIsoValTest"), ("train-val", "NIsoTrainVal")):
        N.add(nm, "int", {"op": "json", "file": spp, "path": iso + ["by_split", k]})
    N.add("NNoMarking", "int", {"op": "json", "file": spp, "path": ["n_images_without_top_left_marking"]})
    N.add("NSameEyePairs", "int", {"op": "agg", "file": "results/checks/mask_similar_pairs.csv",
                                   "where": {"splits": "test-excluded"}, "col": "mask_layout_iou", "fn": "count"})
    N.add("SameEyeIoU", "f3", {"op": "cell", "file": "results/checks/mask_similar_pairs.csv",
                               "where": {"image_a": "train_174_D", "image_b": "test_54_D"}, "col": "mask_layout_iou"})
    fov = "results/checks/fov_summary.json"
    N.add("FOVRadiusMin", "int", {"op": "json", "file": fov, "path": ["r_px", "0"]})
    N.add("FOVRadiusMax", "int", {"op": "json", "file": fov, "path": ["r_px", "1"]})
    N.add("FOVShareMin", "pct1", {"op": "json", "file": fov, "path": ["fov_fraction", "0"]})
    N.add("FOVShareMax", "pct1", {"op": "json", "file": fov, "path": ["fov_fraction", "1"]})
    N.add("NTextInsideFOV", "int", {"op": "json", "file": fov, "path": ["images_with_text_inside_fov"]})
    N.add("NFOVTouchFrame", "int", {"op": "json", "file": fov, "path": ["touches_frame"]})
    wr = "results/checks/width_radius_summary.json"
    for k, nm in (("thin", "ShareThin"), ("medium", "ShareMed"), ("thick", "ShareThick")):
        N.add(nm, "pct1", {"op": "json", "file": wr, "path": ["shares_proposed", k]})
    N.add("NSkelPx", "int", {"op": "json", "file": wr, "path": ["skeleton_px"]})

    # --- main test results (seed mean +- sd across seeds; image-level sd and median)
    for r in RUNS:
        for m, c, _ in METRICS:
            w = {"run": r, "metric": m}
            N.add(f"{CODE[r]}{c}Mean", "f4", {"op": "cell", "file": tm, "where": w, "col": "mean"})
            N.add(f"{CODE[r]}{c}SdSeed", "f4", {"op": "cell", "file": tm, "where": w, "col": "sd_seeds"})
            N.add(f"{CODE[r]}{c}SdImg", "f3", {"op": "cell", "file": tm, "where": w, "col": "sd_images"})
            N.add(f"{CODE[r]}{c}Median", "f4", {"op": "cell", "file": tm, "where": w, "col": "median"})
    N.add("NTestImages", "int", {"op": "cell", "file": tm, "where": {"run": "unet", "metric": "dice"}, "col": "n_images"})
    N.add("NSeeds", "int", {"op": "cell", "file": tm, "where": {"run": "unet", "metric": "dice"}, "col": "n_seeds"})

    # --- per-seed test means (from per-image files) and seed ranges
    for r in RUNS:
        for m in ("dice", "cl_recall_thin", "cl_recall_medium", "nsd", "cldice", "precision", "recall"):
            names = []
            for s in SEEDS:
                nm = f"{CODE[r]}{MCODE[m]}Seed{SEEDCODE[s]}"
                N.add(nm, "f4", {"op": "mean", "file": f"results/eval/{r}/seed{s}/test_metrics.csv", "col": m})
                names.append(nm)
            args = dict(zip("abc", names))
            N.add(f"{CODE[r]}{MCODE[m]}SeedMin", "f4", {"op": "expr", "expr": "min(a,b,c)", "args": args})
            N.add(f"{CODE[r]}{MCODE[m]}SeedMax", "f4", {"op": "expr", "expr": "max(a,b,c)", "args": args})
        for s in SEEDS:
            N.add(f"{CODE[r]}ThrSeed{SEEDCODE[s]}", "f2", {"op": "agg", "file": f"results/eval/{r}/seed{s}/test_metrics.csv",
                                                          "col": "threshold", "fn": "max"})
    # per-seed thin recall at the fixed threshold 0.5 (from the saved sweeps)
    for r in RUNS:
        names = []
        for sd in SEEDS:
            nm = f"{CODE[r]}ThinFixSeed{SEEDCODE[sd]}"
            N.add(nm, "f3", {"op": "mean", "file": f"results/eval/{r}/seed{sd}/test_sweep.csv", "where": {"threshold": 0.5},
                             "col": "cl_recall_thin"})
            names.append(nm)
        a = dict(zip("abc", names))
        N.add(f"{CODE[r]}ThinFixSeedMin", "f3", {"op": "expr", "expr": "min(a,b,c)", "args": a})
        N.add(f"{CODE[r]}ThinFixSeedMax", "f3", {"op": "expr", "expr": "max(a,b,c)", "args": a})
    # spreads across the seven configurations (means)
    for m in ("dice", "cl_recall_thin"):
        args = {chr(97 + i): f"{CODE[r]}{MCODE[m]}Mean" for i, r in enumerate(RUNS)}
        N.add(f"Spread{MCODE[m]}", "f3", {"op": "expr", "expr": f"max({','.join(args)})-min({','.join(args)})", "args": args})
    cnn_d = {chr(97 + i): f"{CODE[r]}DiceMean" for i, r in enumerate(CNN)}
    N.add("CNNDiceMin", "f4", {"op": "expr", "expr": f"min({','.join(cnn_d)})", "args": cnn_d})
    N.add("CNNDiceMax", "f4", {"op": "expr", "expr": f"max({','.join(cnn_d)})", "args": cnn_d})
    cnn_ds = {chr(97 + i): f"{CODE[r]}DiceSdSeed" for i, r in enumerate(CNN)}
    N.add("CNNDiceSdSeedMin", "f4", {"op": "expr", "expr": f"min({','.join(cnn_ds)})", "args": cnn_ds})
    N.add("CNNDiceSdSeedMax", "f4", {"op": "expr", "expr": f"max({','.join(cnn_ds)})", "args": cnn_ds})
    allthin_sd = {chr(97 + i): f"{CODE[r]}ThinSdSeed" for i, r in enumerate(RUNS)}
    N.add("ThinSdSeedMin", "f3", {"op": "expr", "expr": f"min({','.join(allthin_sd)})", "args": allthin_sd})
    N.add("ThinSdSeedMax", "f3", {"op": "expr", "expr": f"max({','.join(allthin_sd)})", "args": allthin_sd})
    cnnthin_sd = {chr(97 + i): f"{CODE[r]}ThinSdSeed" for i, r in enumerate(CNN)}
    N.add("CNNThinSdSeedMin", "f3", {"op": "expr", "expr": f"min({','.join(cnnthin_sd)})", "args": cnnthin_sd})
    N.add("CNNThinSdSeedMax", "f3", {"op": "expr", "expr": f"max({','.join(cnnthin_sd)})", "args": cnnthin_sd})
    allm = lambda m: {chr(97 + i): f"{CODE[r]}{m}Mean" for i, r in enumerate(RUNS)}
    for m in ("Thin", "Med", "Thick"):
        a = allm(m)
        N.add(f"All{m}Min", "f2", {"op": "expr", "expr": f"min({','.join(a)})", "args": a})
        N.add(f"All{m}Max", "f2", {"op": "expr", "expr": f"max({','.join(a)})", "args": a})
    sam_d = {chr(97 + i): f"{CODE[r]}DiceMean" for i, r in enumerate(RUNS[4:])}
    N.add("SAMDiceMin", "f4", {"op": "expr", "expr": "min(a,b,c)", "args": sam_d})
    N.add("SAMDiceMax", "f4", {"op": "expr", "expr": "max(a,b,c)", "args": sam_d})
    for m in ("NSD",):
        a = {chr(97 + i): f"{CODE[r]}{m}Mean" for i, r in enumerate(CNN)}
        N.add(f"CNN{m}Min", "f4", {"op": "expr", "expr": "min(a,b,c,d)", "args": a})
        N.add(f"CNN{m}Max", "f4", {"op": "expr", "expr": "max(a,b,c,d)", "args": a})
        a = {chr(97 + i): f"{CODE[r]}{m}Mean" for i, r in enumerate(RUNS[4:])}
        N.add(f"SAM{m}Min", "f4", {"op": "expr", "expr": "min(a,b,c)", "args": a})
        N.add(f"SAM{m}Max", "f4", {"op": "expr", "expr": "max(a,b,c)", "args": a})

    # --- paired image-level comparison with U-Net (seed-averaged per image; Holm within metric)
    for r in RUNS[1:]:
        for m in MAIN9 + ["cl_recall_thin_r4_8"]:
            w = {"run": r, "metric": m}
            N.add(f"{CODE[r]}{MCODE[m]}Diff", "sf4", {"op": "cell", "file": tw, "where": w, "col": "mean_diff"})
            N.add(f"{CODE[r]}{MCODE[m]}P", "sci", {"op": "cell", "file": tw, "where": w, "col": "p_holm"})
            N.add(f"{CODE[r]}{MCODE[m]}Share", "pct0", {"op": "cell", "file": tw, "where": w, "col": "share_better"})
    # --- fixed threshold 0.5 (secondary)
    for r in RUNS:
        for m in ("dice", "precision", "recall", "cl_recall_thin", "cl_recall_medium"):
            w = {"run": r, "metric": m}
            N.add(f"{CODE[r]}{MCODE[m]}FixMean", "f4", {"op": "cell", "file": tf, "where": w, "col": "mean"})
            N.add(f"{CODE[r]}{MCODE[m]}FixSdSeed", "f4", {"op": "cell", "file": tf, "where": w, "col": "sd_seeds"})
            if r != "unet":
                N.add(f"{CODE[r]}{MCODE[m]}FixDiff", "sf4", {"op": "cell", "file": tfw, "where": w, "col": "mean_diff"})
                N.add(f"{CODE[r]}{MCODE[m]}FixP", "sci", {"op": "cell", "file": tfw, "where": w, "col": "p_holm"})

    # --- 2x2 ablation (image-level bootstrap CIs)
    for f, rule in (("results/tables/test_ablation_effects.csv", "Sel"), ("results/tables/test_ablation_effects_fixed_t0.5.csv", "Fix")):
        d = csv(f)
        for eff, ec in (("attention", "Att"), ("multi-scale", "MS"), ("interaction", "Int")):
            for m in sorted(set(d.metric)):
                w = {"metric": m, "effect": eff}
                base = f"Abl{ec}{MCODE[m]}{rule}"
                N.add(base, "sf4", {"op": "cell", "file": f, "where": w, "col": "mean"})
                N.add(base + "Lo", "sf4", {"op": "cell", "file": f, "where": w, "col": "ci_low"})
                N.add(base + "Hi", "sf4", {"op": "cell", "file": f, "where": w, "col": "ci_high"})
    ab = {"a": "AblAttDiceSel", "b": "AblMSDiceSel", "c": "AblAttDiceFix", "d": "AblMSDiceFix"}
    N.add("AblDiceMainMaxAbs", "f4", {"op": "expr", "expr": "max(abs(a),abs(b),abs(c),abs(d))", "args": ab})

    # --- threshold curves (test and val), from the saved sweeps
    tc = "results/tables/thin_recall_vs_threshold.csv"
    for split, sc in (("test", "Test"), ("val", "Val")):
        for subj in ("sam_lora_tiled", "sam_lora_tiled_overlap"):
            for fn in ("min", "max"):
                N.add(f"Curve{CODE[subj]}GapBest{fn.title()}{sc}", "sf3",
                      {"op": "curve", "file": tc, "split": split, "subject": subj, "refs": CNN, "kind": "gap_best", "fn": fn})
            N.add(f"Curve{CODE[subj]}AboveBest{sc}", "int",
                  {"op": "curve", "file": tc, "split": split, "subject": subj, "refs": CNN, "kind": "above_count"})
        N.add(f"CurveSAMTiledOverlapUNet{sc}", "int",
              {"op": "curve", "file": tc, "split": split, "subject": "sam_lora_tiled", "refs": ["unet"], "kind": "overlap_count"})
        for fn in ("min", "max"):
            N.add(f"CurveOverlapMinusTiled{fn.title()}{sc}", "sf3",
                  {"op": "curve", "file": tc, "split": split, "subject": "sam_lora_tiled_overlap", "refs": ["sam_lora_tiled"],
                   "kind": "diff", "fn": fn})
    N.add("CurveSAMWholeAboveBestTest", "int", {"op": "curve", "file": tc, "split": "test", "subject": "sam_lora_whole",
                                                "refs": CNN, "kind": "above_count"})
    for thr, nm in ((0.05, "Low"), (0.5, "Mid"), (0.95, "High")):
        for r in ("unet", "sam_lora_tiled", "sam_lora_whole"):
            N.add(f"Curve{CODE[r]}Thin{nm}", "f2", {"op": "cell", "file": tc, "where": {"split": "test", "run": r, "threshold": thr},
                                                    "col": "cl_recall_thin"})
    N.add("NThresholds", "int", {"op": "agg", "file": tc, "where": {"split": "test", "run": "unet"}, "col": "threshold", "fn": "count"})

    # --- cost (seed-0 runs; one L40S)
    cost = "results/tables/cost.csv"
    for r in RUNS + ["sam_oracle_k5"]:
        cd = "Oracle" if r == "sam_oracle_k5" else CODE[r]
        w = {"run": r}
        N.add(f"{cd}InfMs", "f1", {"op": "cell", "file": cost, "where": w, "col": "inference_ms_mean"})
        N.add(f"{cd}Gflops", "int", {"op": "cell", "file": cost, "where": w, "col": "gflops_per_image"})
        N.add(f"{cd}InfMem", "f2", {"op": "cell", "file": cost, "where": w, "col": "inference_peak_gpu_gb"})
        N.add(f"{cd}Params", "int", {"op": "cell", "file": cost, "where": w, "col": "params_total"})
        N.add(f"{cd}ParamsM", "f1", {"op": "expr", "expr": "a/1e6", "args": {"a": f"{cd}Params"}})
        N.add(f"{cd}ParamsTrainable", "int", {"op": "cell", "file": cost, "where": w, "col": "params_trainable"})
        N.add(f"{cd}ParamsTrainableM", "f1", {"op": "expr", "expr": "a/1e6", "args": {"a": f"{cd}ParamsTrainable"}})
        if r not in ("sam_oracle_k5", "sam_lora_tiled_overlap"):
            N.add(f"{cd}TrainH", "f2", {"op": "cell", "file": cost, "where": w, "col": "training_time_h"})
            N.add(f"{cd}TrainMem", "f2", {"op": "cell", "file": cost, "where": w, "col": "training_peak_gpu_gb"})
    for r in ("sam_lora_tiled", "sam_lora_tiled_overlap", "sam_lora_whole"):
        N.add(f"{CODE[r]}InfRatio", "f0", {"op": "expr", "expr": "a/b", "args": {"a": f"{CODE[r]}InfMs", "b": "UNetInfMs"}})
    for r in ("sam_lora_tiled", "sam_lora_whole"):
        N.add(f"{CODE[r]}TrainRatio", "f0", {"op": "expr", "expr": "a/b", "args": {"a": f"{CODE[r]}TrainH", "b": "UNetTrainH"}})
    N.add("SAMTiledGflopsRatio", "f0", {"op": "expr", "expr": "a/b", "args": {"a": "SAMTiledGflops", "b": "UNetGflops"}})
    mp = "results/tables/model_params.csv"
    for r, mname in (("attention_unet", "Attention U-Net"), ("multiscale_unet", "Multi-Scale U-Net"), ("ms_attention_unet", "Multi-Scale Attention U-Net")):
        N.add(f"{CODE[r]}ParamsPct", "f1", {"op": "cell", "file": mp, "where": {"model": mname}, "col": "params_vs_unet_pct"})
        N.add(f"{CODE[r]}GmacsPct", "f1", {"op": "cell", "file": mp, "where": {"model": mname}, "col": "gmacs_vs_unet_pct"})
    N.add("UNetGmacs", "f1", {"op": "cell", "file": mp, "where": {"model": "U-Net"}, "col": "gmacs_conv_at_cnn_resolution"})

    # --- mask-resampling references (context, not upper bounds)
    rf = "results/tables/test_references.csv"
    for run, cd in (("reference_soft_512", "RefFive"), ("reference_soft_256", "RefTwo")):
        for m in ("dice", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick", "cldice", "nsd", "precision", "recall"):
            N.add(f"{cd}{MCODE[m]}", "f3", {"op": "cell", "file": rf, "where": {"run": run, "metric": m}, "col": "mean"})
        N.add(f"{cd}ThinVal", "f3", {"op": "mean", "file": f"results/eval/{run}/val_metrics.csv", "col": "cl_recall_thin"})
    N.add("RefTenThinVal", "f3", {"op": "mean", "file": "results/eval/reference_soft_1024/val_metrics.csv", "col": "cl_recall_thin"})

    # --- oracle-prompted SAM (separate)
    oc = "results/tables/test_sam_oracle.csv"
    for m, cd in (("dice", "Dice"), ("precision", "Prec"), ("recall", "Rec"), ("cl_recall_thin", "Thin"),
                  ("cldice", "ClDice"), ("nsd", "NSD"), ("density_pred", "DensityPred"), ("density_gt", "DensityGT")):
        N.add(f"Oracle{cd}", "f3", {"op": "cell", "file": oc, "where": {"metric": m}, "col": "mean"})
    oj = "results/eval/sam_oracle_choice.json"
    N.add("OracleK", "int", {"op": "json", "file": oj, "path": ["chosen_k"]})
    N.add("OracleThr", "f2", {"op": "json", "file": oj, "path": ["per_k", "5", "threshold"]})
    for k, kn in (("5", "Five"), ("10", "Ten"), ("20", "Twenty")):
        N.add(f"OracleValDiceK{kn}", "f3", {"op": "json", "file": oj, "path": ["per_k", k, "mean_dice"]})

    # --- validation: checkpoint selection (best val Dice at 0.5) and val thin recall at the selected threshold
    for r in ("unet", "attention_unet", "multiscale_unet", "ms_attention_unet", "sam_lora_tiled", "sam_lora_whole"):
        for s in SEEDS:
            N.add(f"{CODE[r]}ValBestDiceSeed{SEEDCODE[s]}", "f4",
                  {"op": "json", "file": f"results/train/{r}/seed{s}/summary.json", "path": ["best_val_dice_at_0.5"]})
            N.add(f"{CODE[r]}ValThinSeed{SEEDCODE[s]}", "f3",
                  {"op": "mean", "file": f"results/eval/{r}/seed{s}/val_metrics.csv", "col": "cl_recall_thin"})
            N.add(f"{CODE[r]}EpochsSeed{SEEDCODE[s]}", "int",
                  {"op": "json", "file": f"results/train/{r}/seed{s}/summary.json", "path": ["epochs_run"]})
            N.add(f"{CODE[r]}BestEpochSeed{SEEDCODE[s]}", "int",
                  {"op": "json", "file": f"results/train/{r}/seed{s}/summary.json", "path": ["best_epoch"]})
        a = {c: f"{CODE[r]}ValBestDiceSeed{c.upper()}" for c in "abc"}
        N.add(f"{CODE[r]}ValBestDiceMin", "f4", {"op": "expr", "expr": "min(a,b,c)", "args": a})
        N.add(f"{CODE[r]}ValBestDiceMax", "f4", {"op": "expr", "expr": "max(a,b,c)", "args": a})
        a = {c: f"{CODE[r]}ValThinSeed{c.upper()}" for c in "abc"}
        N.add(f"{CODE[r]}ValThinMin", "f3", {"op": "expr", "expr": "min(a,b,c)", "args": a})
        N.add(f"{CODE[r]}ValThinMax", "f3", {"op": "expr", "expr": "max(a,b,c)", "args": a})
    N.add("SAMTiledValThinRange", "f3", {"op": "expr", "expr": "b-a", "args": {"a": "SAMTiledValThinMin", "b": "SAMTiledValThinMax"}})
    vr, vs = "results/tables/val_runs.csv", "results/tables/val_sam_runs.csv"
    N.add("NCNNRunsAtCap", "int", {"op": "agg", "file": vr, "where": {"epochs_run": 150}, "col": "epochs_run", "fn": "count"})
    N.add("NCNNRuns", "int", {"op": "agg", "file": vr, "where": {}, "col": "epochs_run", "fn": "count"})
    for tr, nm in (("sam_lora_tiled", "NSAMTiledAtCap"), ("sam_lora_whole", "NSAMWholeAtCap")):
        N.add(nm, "int", {"op": "agg", "file": vs, "where": {"trained_run": tr, "epochs_run": 150,
                                                            "model": "SAM LoRA, tiled 2x2 (main)" if tr == "sam_lora_tiled" else "SAM LoRA, whole image"},
                          "col": "epochs_run", "fn": "count"})

    # --- 1024 resolution pilot (validation, U-Net seed 0)
    u5, u10 = "results/eval/unet/seed0/val_metrics.csv", "results/eval/unet_r1024/seed0/val_metrics.csv"
    for f, cd in ((u5, "PilotFive"), (u10, "PilotTen")):
        for m in ("dice", "cl_recall_thin"):
            N.add(f"{cd}{MCODE[m]}", "f4", {"op": "mean", "file": f, "col": m})
    N.add("PilotThinGain", "sf4", {"op": "expr", "expr": "b-a", "args": {"a": "PilotFiveThin", "b": "PilotTenThin"}})
    N.add("PilotDiceGain", "sf4", {"op": "expr", "expr": "b-a", "args": {"a": "PilotFiveDice", "b": "PilotTenDice"}})
    for r, cd in (("unet", "PilotFive"), ("unet_r1024", "PilotTen")):
        N.add(f"{cd}EpochS", "f1", {"op": "json", "file": f"results/train/{r}/seed0/summary.json", "path": ["epoch_time_s", "median"]})
        N.add(f"{cd}Mem", "f2", {"op": "json", "file": f"results/train/{r}/seed0/summary.json", "path": ["gpu_peak_alloc_gb"]})

    # --- tile-seam check (validation, seed 0)
    sj = "results/checks/seam_check_summary.json"
    for k, cd in (("sam_lora_tiled", "SAMTiled"), ("sam_lora_tiled_overlap", "SAMOverlap"), ("sam_lora_whole", "SAMWhole"), ("unet_mean_s012", "UNet")):
        N.add(f"Seam{cd}Jump", "f2", {"op": "json", "file": sj, "path": ["jump_ratio", k, "mean"]})
    for subj, cd in (("sam_lora_tiled", "SAMTiled"), ("sam_lora_tiled_overlap", "SAMOverlap")):
        for m, mc in (("dice", "Dice"), ("cl_recall_thin", "Thin")):
            match = {"subject": subj, "control": "unet_mean_s012", "half_width_px": 16, "metric": m}
            for fld, suf in (("did_mean", ""), ("ci_low", "Lo"), ("ci_high", "Hi")):
                N.add(f"Seam{cd}DiD{mc}{suf}", "sf3", {"op": "json_list", "file": sj, "path": ["did"], "fn": "one", "field": fld, "match": match})

    # --- disease categories
    dz = "results/tables/test_disease.csv"
    for r in RUNS:
        for c, _ in CATS:
            for m in ("dice", "cl_recall_thin"):
                N.add(f"{CODE[r]}{MCODE[m]}Cat{c}", "f4", {"op": "cell", "file": dz, "where": {"run": r, "category": c, "metric": m}, "col": "mean"})
    for m in ("Dice", "Thin"):
        for c, _ in CATS:
            a = {chr(97 + i): f"{CODE[r]}{m}Cat{c}" for i, r in enumerate(RUNS)}
            N.add(f"Cat{c}{m}Min", "f3", {"op": "expr", "expr": f"min({','.join(a)})", "args": a})
            N.add(f"Cat{c}{m}Max", "f3", {"op": "expr", "expr": f"max({','.join(a)})", "args": a})

    # --- image-derived vessel measures (optional extra)
    vm = "results/tables/test_vessel_measures.csv"
    for r in RUNS:
        for meas, cd in (("vessel density", "Dens"), ("fractal dimension", "FD")):
            w = {"run": r, "measure": meas}
            N.add(f"{CODE[r]}{cd}Bias", "sf4", {"op": "cell", "file": vm, "where": w, "col": "bias_mean_diff"})
            N.add(f"{CODE[r]}{cd}R", "f3", {"op": "cell", "file": vm, "where": w, "col": "pearson_r"})
            N.add(f"{CODE[r]}{cd}MAD", "f4", {"op": "cell", "file": vm, "where": w, "col": "mean_abs_diff"})
    for cd, meas in (("Dens", "vessel density"), ("FD", "fractal dimension")):
        for stat, col, k in (("Bias", "bias_mean_diff", "sf3"), ("R", "pearson_r", "f3")):
            for fn in ("min", "max"):
                N.add(f"All{cd}{stat}{fn.title()}", k, {"op": "agg", "file": vm, "where": {"measure": meas}, "col": col, "fn": fn})
        N.add(f"{cd}GTMean", "f3", {"op": "cell", "file": vm, "where": {"run": "unet", "measure": meas}, "col": "mean_gt"})

    # --- context checks from Phases 2-3 (loss optimum, mask resizing)
    lo = "results/checks/loss_optimum_summary.json"
    N.add("LossOptQuarter", "f2", {"op": "json", "file": lo, "path": ["mean_p_at_optimum_by_target_value", "0.25", "bce_dice"]})
    N.add("LossOptHalf", "f2", {"op": "json", "file": lo, "path": ["mean_p_at_optimum_by_target_value", "0.5", "bce_dice"]})
    mr = "results/checks/mask_resize_means.csv"
    for res, rc in ((512, "Five"), (1024, "Ten")):
        for meth, mc in (("soft", "Soft"), ("area_t50", "AreaHalf"), ("area_t25", "AreaQuarter"), ("maxpool", "Maxpool"), ("nearest", "Nearest")):
            for m, c in (("dice", "Dice"), ("cl_recall_thin", "Thin"), ("precision", "Prec")):
                w = {"resolution": res, "method": meth, "threshold_after_upsampling": 0.5}
                if len(_filter(csv(mr), w)) == 1:
                    N.add(f"Resize{rc}{mc}{c}", "f3", {"op": "cell", "file": mr, "where": w, "col": m})

    for res, rc in ((512, "Five"), (1024, "Ten")):
        w = {"resolution": res, "method": "soft", "threshold_after_upsampling": 0.25}
        if len(_filter(csv(mr), w)) == 1:
            N.add(f"Resize{rc}SoftQuarterThin", "f3", {"op": "cell", "file": mr, "where": w, "col": "cl_recall_thin"})
            N.add(f"Resize{rc}SoftQuarterDice", "f3", {"op": "cell", "file": mr, "where": w, "col": "dice"})
    # --- agreement between two available annotations of the same image (post hoc; labels only)
    ag, agd = "results/tables/annotation_agreement.csv", "results/tables/annotation_agreement_directions.csv"
    N.add("AgreeNPairs", "int", {"op": "agg", "file": ag, "where": {}, "col": "dice", "fn": "count"})
    N.add("AgreeNTrain", "int", {"op": "agg", "file": ag, "where": {"split": "train"}, "col": "dice", "fn": "count"})
    N.add("AgreeNTest", "int", {"op": "agg", "file": ag, "where": {"split": "test"}, "col": "dice", "fn": "count"})
    N.add("AgreeNCatA", "int", {"op": "agg", "file": ag, "where": {"category": "A"}, "col": "dice", "fn": "count"})
    for m in ("dice", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick", "cldice", "nsd", "cl_recall_thin_r4_8"):
        cd = MCODE[m]
        N.add(f"Agree{cd}Mean", "f3", {"op": "mean", "file": ag, "col": m})
        N.add(f"Agree{cd}Min", "f3", {"op": "agg", "file": ag, "where": {}, "col": m, "fn": "min"})
        N.add(f"Agree{cd}Max", "f3", {"op": "agg", "file": ag, "where": {}, "col": m, "fn": "max"})
    # --- revision (Step 4): within-test likely same-patient pairs; corrected cross-split count
    for k, nm in (("test-test", "NIsoTestTest"),):
        N.add(nm, "int", {"op": "json", "file": spp, "path": iso + ["by_split", k]})
    N.add("NAdjTestTest", "int", {"op": "json", "file": spp, "path": ["by_threshold", "0.8", "adjacent_numbers", "by_split", "test-test"]})
    N.add("NIsoWithTest", "int", {"op": "expr", "expr": "a+b", "args": {"a": "NIsoTrainTest", "b": "NIsoValTest"}})
    # simple effects of the 2x2 design (point estimates: differences of model means)
    for rule, suf in (("", "Sel"), ("Fix", "Fix")):
        for m in ("Thin", "Prec", "Dice", "Rec"):
            g = lambda run: f"{run}{m}{rule}Mean" if rule else f"{run}{m}Mean"
            N.add(f"SimpleAttNoMS{m}{suf}", "sf4", {"op": "expr", "expr": "a-b", "args": {"a": g("Att"), "b": g("UNet")}})
            N.add(f"SimpleAttWithMS{m}{suf}", "sf4", {"op": "expr", "expr": "a-b", "args": {"a": g("MSAtt"), "b": g("MS")}})
            N.add(f"SimpleMSNoAtt{m}{suf}", "sf4", {"op": "expr", "expr": "a-b", "args": {"a": g("MS"), "b": g("UNet")}})
            N.add(f"SimpleMSWithAtt{m}{suf}", "sf4", {"op": "expr", "expr": "a-b", "args": {"a": g("MSAtt"), "b": g("Att")}})
    # validation thin recall at the fixed threshold 0.5 (checkpoint-selection check)
    for r in ("unet", "sam_lora_tiled"):
        nm = []
        for sd in SEEDS:
            n_ = f"{CODE[r]}ValThinFixSeed{SEEDCODE[sd]}"
            N.add(n_, "f3", {"op": "mean", "file": f"results/eval/{r}/seed{sd}/val_sweep.csv", "where": {"threshold": 0.5}, "col": "cl_recall_thin"})
            nm.append(n_)
        a = dict(zip("abc", nm))
        N.add(f"{CODE[r]}ValThinFixMin", "f3", {"op": "expr", "expr": "min(a,b,c)", "args": a})
        N.add(f"{CODE[r]}ValThinFixMax", "f3", {"op": "expr", "expr": "max(a,b,c)", "args": a})
    # first-observed (seed 0) tiled-SAM thin recall against the 3-seed U-Net mean
    N.add("SAMTiledSeedAThinMinusUNet", "sf4", {"op": "expr", "expr": "a-b", "args": {"a": "SAMTiledThinSeedA", "b": "UNetThinMean"}})
    # fractal-dimension outliers: two glaucoma test images with very sparse annotation
    for iid, nm in (("test_122_G", "FDOutOneDens"), ("test_123_G", "FDOutTwoDens")):
        N.add(nm, "f3", {"op": "cell", "file": "results/eval/unet/seed0/test_metrics.csv", "where": {"image_id": iid}, "col": "density_gt"})
    for m in ("Thick",):
        a = {chr(97 + i): f"{CODE[r]}{m}Mean" for i, r in enumerate(RUNS)}
        N.add(f"All{m}MinP", "f3", {"op": "expr", "expr": f"min({','.join(a)})", "args": a})
        N.add(f"All{m}MaxP", "f3", {"op": "expr", "expr": f"max({','.join(a)})", "args": a})
    N.add("AgreeThinDirMin", "f2", {"op": "agg", "file": agd, "where": {}, "col": "cl_recall_thin", "fn": "min"})
    N.add("AgreeThinDirMax", "f2", {"op": "agg", "file": agd, "where": {}, "col": "cl_recall_thin", "fn": "max"})
    posthoc_numbers(N)
    return N


# post hoc analyses after the referee report (D2, D4, D7), all added after the main test evaluation
PML, PMM = "results/tables/posthoc_model_level.csv", "results/tables/posthoc_matched_model_level.csv"
PST, PINFO = "results/tables/posthoc_stats.csv", "results/tables/posthoc_info.json"
PMT, OFF = "results/tables/posthoc_matched_thresholds.csv", "results/checks/official_release_check.json"
TARGETS = (("matched_unet_val_precision", "U", "unet_val_precision"), ("matched_0.90", "Ninety", "0.90"),
           ("matched_0.95", "NinetyFive", "0.95"))
# (rule, metric, comparison) -> macro stem for the D4 statistics
STATS = {("selected", "cl_recall_thin", "sam_lora_tiled - unet"): "TiledThinSel",
         ("fixed_0.5", "cl_recall_thin", "sam_lora_tiled - unet"): "TiledThinFix",
         ("matched_unet_val_precision", "cl_recall_thin", "sam_lora_tiled - unet"): "TiledThinMatch",
         ("selected", "cl_recall_thin", "sam_lora_whole - unet"): "WholeThinSel",
         ("matched_unet_val_precision", "cl_recall_thin", "sam_lora_whole - unet"): "WholeThinMatch",
         ("selected", "dice", "sam_lora_tiled - unet"): "TiledDiceSel",
         ("selected", "dice", "sam_lora_whole - unet"): "WholeDiceSel",
         ("selected", "nsd", "sam_lora_tiled - unet"): "TiledNSDSel",
         ("selected", "cl_recall_thin", "multi-scale effect"): "MSThinSel",
         ("fixed_0.5", "cl_recall_thin", "multi-scale effect"): "MSThinFix",
         ("matched_unet_val_precision", "cl_recall_thin", "multi-scale effect"): "MSThinMatch",
         ("selected", "cl_recall_thin", "attention effect"): "AttThinSel",
         ("fixed_0.5", "cl_recall_thin", "attention effect"): "AttThinFix",
         ("matched_unet_val_precision", "cl_recall_thin", "attention effect"): "AttThinMatch",
         ("fixed_0.5", "cl_recall_thin", "interaction"): "IntThinFix",
         ("selected", "dice", "multi-scale effect"): "MSDiceSel",
         ("selected", "dice", "attention effect"): "AttDiceSel",
         ("selected", "dice", "interaction"): "IntDiceSel",
         ("selected", "cl_recall_thin_pruned", "sam_lora_tiled - unet"): "TiledThinPrSel",
         ("selected", "cl_recall_thin_pruned", "multi-scale effect"): "MSThinPrSel",
         ("selected", "w_prec_thin", "sam_lora_tiled - unet"): "TiledWPrecSel",
         ("fixed_0.5", "cldice", "sam_lora_tiled - unet"): "TiledClDiceFix",
         ("fixed_0.5", "nsd", "sam_lora_tiled - unet"): "TiledNSDFix",
         ("selected", "w_prec_thin", "multi-scale effect"): "MSWPrecSel",
         ("fixed_0.5", "w_prec_thin", "multi-scale effect"): "MSWPrecFix",
         ("selected", "cldice", "sam_lora_tiled - unet"): "TiledClDiceSel"}
CI_KINDS = (("Img", "ci_image"), ("Cl", "ci_cluster"), ("Sc", "ci_seed_cluster"))


def posthoc_numbers(N: Numbers) -> None:
    for r in RUNS:
        for rule, rc in (("selected", "Sel"), ("fixed_0.5", "Fix")):
            w = {"rule": rule, "model": r}
            for c, cc in (("thin", "Thin"), ("medium", "Med"), ("thick", "Thick")):
                N.add(f"PH{CODE[r]}WPrec{cc}{rc}", "f3", {"op": "cell", "file": PML, "where": w, "col": f"w_prec_{c}_mean"})
            N.add(f"PH{CODE[r]}WPrecThinPr{rc}", "f3", {"op": "cell", "file": PML, "where": w, "col": "w_prec_thin_pruned_mean"})
            N.add(f"PH{CODE[r]}ThinPr{rc}", "f3", {"op": "cell", "file": PML, "where": w, "col": "cl_recall_thin_pruned_mean"})
            N.add(f"PH{CODE[r]}Thin{rc}", "f3", {"op": "cell", "file": PML, "where": w, "col": "cl_recall_thin_mean"})
            N.add(f"PH{CODE[r]}ClDice{rc}", "f3", {"op": "cell", "file": PML, "where": w, "col": "cldice_mean"})
            N.add(f"PH{CODE[r]}NSD{rc}", "f3", {"op": "cell", "file": PML, "where": w, "col": "nsd_mean"})
        for rule, tc, tname in TARGETS:
            w = {"rule": rule, "model": r}
            for col, mc in (("cl_recall_thin_mean", "Thin"), ("cl_recall_thin_std", "ThinSd"), ("precision_mean", "Prec"),
                            ("dice_mean", "Dice"), ("recall_mean", "Rec")):
                N.add(f"PH{CODE[r]}Match{tc}{mc}", "f3", {"op": "cell", "file": PMM, "where": w, "col": col})
            for fn, fc in (("min", "TMin"), ("max", "TMax")):
                N.add(f"PH{CODE[r]}Match{tc}{fc}", "f2", {"op": "agg", "file": PMT, "where": {"model": r, "target": tname},
                                                          "col": "threshold", "fn": fn})
    N.add("PHMatchTarget", "f3", {"op": "json", "file": PINFO, "path": ["unet_mean_val_precision_target"]})
    for c, cc in (("thin", "Thin"), ("medium", "Med"), ("thick", "Thick")):
        N.add(f"PHPruned{cc}Share", "pct1", {"op": "json", "file": PINFO, "path": ["gt_pruning", f"share_{c}_skeleton_removed"]})
    N.add("PHNClusters", "int", {"op": "json", "file": PINFO, "path": ["n_test_clusters"]})
    N.add("PHNMultiClusters", "int", {"op": "json", "file": PINFO, "path": ["n_multi_image_clusters"]})
    N.add("PHNClusteredImages", "int", {"op": "json", "file": PINFO, "path": ["n_test_images_in_multi_image_clusters"]})
    N.add("PHLargestCluster", "int", {"op": "json", "file": PINFO, "path": ["largest_cluster"]})
    N.add("PHNBoot", "int", {"op": "json", "file": PINFO, "path": ["n_boot"]})
    for (rule, metric, comp), stem in STATS.items():
        w = {"rule": rule, "metric": metric, "comparison": comp}
        N.add(f"PHSt{stem}", "sf4", {"op": "cell", "file": PST, "where": w, "col": "estimate"})
        for kc, col in CI_KINDS:
            N.add(f"PHSt{stem}{kc}Lo", "sf4", {"op": "cell", "file": PST, "where": w, "col": f"{col}_low"})
            N.add(f"PHSt{stem}{kc}Hi", "sf4", {"op": "cell", "file": PST, "where": w, "col": f"{col}_high"})
        N.add(f"PHSt{stem}Signs", "int", {"op": "cell", "file": PST, "where": w, "col": "seeds_same_sign"})
        for s in SEEDS:
            N.add(f"PHSt{stem}Seed{SEEDCODE[s]}", "sf4", {"op": "cell", "file": PST, "where": w, "col": f"seed{s}"})
    # D7: the copy used against the official figshare release
    N.add("OffNPng", "int", {"op": "json", "file": OFF, "path": ["n_png"]})
    N.add("OffBytesEqualImg", "int", {"op": "json", "file": OFF, "path": ["bytes_equal_hf", "image"]})
    N.add("OffBytesEqualMask", "int", {"op": "json", "file": OFF, "path": ["bytes_equal_hf", "mask"]})
    N.add("OffPixEqualImg", "int", {"op": "json", "file": OFF, "path": ["pixels_equal_hf", "image"]})
    N.add("OffPixEqualMask", "int", {"op": "json", "file": OFF, "path": ["pixels_equal_hf", "mask"]})
    N.add("OffNDupGroups", "int", {"op": "json", "file": OFF, "path": ["findings_official", "n_byte_identical_image_groups"]})
    N.add("OffSameEyeDice", "f3", {"op": "json", "file": OFF, "path": ["findings_official", "same_eye_pair", "mask_dice_full_image"]})
    # ranges over the seven configurations (selected thresholds) and the 1024 pilot's precision (validation, seed 0)
    for c in ("Thin", "Med", "Thick"):
        a = {chr(97 + i): f"PH{CODE[r]}WPrec{c}Sel" for i, r in enumerate(RUNS)}
        N.add(f"PHWPrec{c}Min", "f3", {"op": "expr", "expr": f"min({','.join(a)})", "args": a})
        N.add(f"PHWPrec{c}Max", "f3", {"op": "expr", "expr": f"max({','.join(a)})", "args": a})
    for nm, run in (("PilotFivePrec", "unet"), ("PilotTenPrec", "unet_r1024"), ("PilotSAMTiledPrec", "sam_lora_tiled")):
        N.add(nm, "f3", {"op": "mean", "file": f"results/eval/{run}/seed0/val_metrics.csv", "col": "precision"})
    thr0 = {r: js(f"results/eval/{r}/seed0/val_threshold.json")["threshold"] for r in ("unet", "unet_r1024", "sam_lora_tiled")}
    if len(set(thr0.values())) != 1:                       # the text says "at the same threshold"
        raise ValueError(f"seed-0 validation thresholds differ: {thr0}")
    N.add("PilotThreshold", "f2", {"op": "json", "file": "results/eval/unet_r1024/seed0/val_threshold.json", "path": ["threshold"]})
    N.add("OffPixEqualMaskRGBA", "int", {"op": "json", "file": OFF, "path": ["official_mask_channels", "rgba_opaque"]})
    N.add("OffNEmptyMasks", "int", {"op": "json_list", "file": OFF, "path": ["findings_official", "empty_masks"], "fn": "count"})


def posthoc_tables(N: Numbers) -> list[Table]:
    T = []
    rng = lambda lo, hi: (f"[\\{lo}{{}}, \\{hi}{{}}]", f"[{plain(N.text[lo])}, {plain(N.text[hi])}]")
    pm = lambda a, b: (f"\\{a}{{}} (\\{b}{{}})", f"{plain(N.text[a])} ({plain(N.text[b])})")
    rows = []
    for r in RUNS:
        p = f"PH{CODE[r]}MatchU"
        rows.append([NAME[r].replace(" (main SAM)", ""), rng(f"{p}TMin", f"{p}TMax"), t(N, f"{p}Prec"), t(N, f"{p}Dice"),
                     pm(f"{p}Thin", f"{p}ThinSd"), t(N, f"PH{CODE[r]}MatchNinetyThin"), t(N, f"PH{CODE[r]}MatchNinetyFiveThin")])
    T.append(Table("si_matched", "Thin-vessel recall at matched precision (test).",
                   "Post hoc. For every run, the threshold (grid 0.01–0.99) whose mean validation precision was closest to "
                   "a target was chosen on the validation set and applied to the test set. Columns 2–5: target = the U-Net's "
                   "mean validation precision at its selected thresholds (<<PHMatchTarget>>); thresholds over the "
                   "<<NSeeds>> seeds; test precision, Dice and thin-vessel recall, mean (SD over seeds). Columns 6–7: "
                   "thin-vessel recall with targets 0.90 and 0.95.",
                   ["Model", "Thresholds", "Precision", "Dice", "Thin recall", "Thin, P = 0.90", "Thin, P = 0.95"],
                   rows, wide=True, si=True))
    rows = []
    for r in RUNS:
        p = f"PH{CODE[r]}"
        rows.append([NAME[r].replace(" (main SAM)", "")] +
                    [t(N, f"{p}WPrec{c}Sel") for c in ("Thin", "Med", "Thick")] +
                    [t(N, f"{p}ThinSel"), t(N, f"{p}ThinPrSel"), t(N, f"{p}WPrecThinPrSel"), t(N, f"{p}ClDiceFix"),
                     t(N, f"{p}NSDFix")])
    T.append(Table("si_precision", "Width-stratified precision, spur-pruned thin class, and clDice and NSD at 0.5 (test).",
                   "Post hoc, added after the main test evaluation; saved probability maps re-scored, nothing predicted again. "
                   "Means over <<NSeeds>> seeds of the means over <<NTestImages>> test images. Width-stratified precision: "
                   "share of the predicted centreline, classed by the prediction's own radius (thin < 3, medium 3–7, thick > 7 "
                   "px), that lies within 1 px of a ground-truth vessel; columns 2–7 at each run's validation-selected "
                   "threshold. Pruned: terminal skeleton branches no longer than 1.5 × the vessel radius + 2 px removed before "
                   "classing (<<PHPrunedThinShare>>% of the ground-truth thin centreline). The last two columns are at the "
                   "fixed threshold 0.5 (Table 3 gives them at the selected thresholds).",
                   ["Model", "Thin prec.", "Medium prec.", "Thick prec.", "Thin recall", "Thin recall, pruned",
                    "Thin prec., pruned", "clDice at 0.5", "NSD at 0.5"], rows, wide=True, si=True))
    lab = {"TiledThinSel": "Thin recall, tiled SAM − U-Net, selected", "TiledThinFix": "Thin recall, tiled SAM − U-Net, 0.5",
           "TiledThinMatch": "Thin recall, tiled SAM − U-Net, matched", "WholeThinSel": "Thin recall, whole SAM − U-Net, selected",
           "WholeThinMatch": "Thin recall, whole SAM − U-Net, matched", "TiledDiceSel": "Dice, tiled SAM − U-Net, selected",
           "WholeDiceSel": "Dice, whole SAM − U-Net, selected", "TiledNSDSel": "NSD, tiled SAM − U-Net, selected",
           "MSThinSel": "Thin recall, multi-scale effect, selected", "MSThinFix": "Thin recall, multi-scale effect, 0.5",
           "MSThinMatch": "Thin recall, multi-scale effect, matched", "AttThinSel": "Thin recall, attention effect, selected",
           "AttThinFix": "Thin recall, attention effect, 0.5", "AttThinMatch": "Thin recall, attention effect, matched",
           "IntThinFix": "Thin recall, interaction, 0.5", "MSDiceSel": "Dice, multi-scale effect, selected",
           "AttDiceSel": "Dice, attention effect, selected", "IntDiceSel": "Dice, interaction, selected"}
    rows = []
    for stem, l in lab.items():
        b = f"PHSt{stem}"
        rows.append([l, t(N, b)] + [rng(f"{b}{k}Lo", f"{b}{k}Hi") for k, _ in CI_KINDS] + [t(N, f"{b}Signs")])
    T.append(Table("si_stats", "Uncertainty of the main comparisons under three resampling schemes (test).",
                   "Post hoc. Estimate: difference of model means (each over <<NSeeds>> seeds and <<NTestImages>> images); "
                   "for the ablation, main effects and interaction as in Table 5. 95% percentile intervals from <<PHNBoot>> "
                   "replicates resampling images (as in the main analysis), clusters of possibly same-patient test images "
                   "(<<PHNClusters>> clusters, <<PHNMultiClusters>> with more than one image), or clusters together with "
                   "the seeds of each model (jointly for the coupled CNN seeds, independently for SAM and the U-Net). Last "
                   "column: number of the <<NSeeds>> seed-paired differences with the sign of the majority. Threshold rules: "
                   "validation-selected, fixed 0.5, or matched validation precision (S14 Table).",
                   ["Comparison", "Estimate", "Images", "Clusters", "Seeds × clusters", "Same sign"], rows, wide=True,
                   si=True))
    return T


# ----------------------------------------------------------------------------------------- tables
def t(N: Numbers, name: str) -> tuple[str, str]:
    """(LaTeX macro call, plain text) for a number."""
    return f"\\{name}{{}}", plain(N.text[name])


class Table:
    def __init__(self, key, title, caption, header, rows, colspec=None, wide=False, notes=None, si=False):
        self.key, self.title, self.caption, self.header, self.rows = key, title, caption, header, rows
        self.colspec = colspec or "l" + "c" * (len(header) - 1)
        self.wide, self.notes, self.si = wide, notes, si

    def tex(self, caption_tex: str) -> str:
        cell = lambda c: c[0] if isinstance(c, tuple) else c
        lines = [f"% tables/{self.key}.tex -- generated by paper/scripts/make_assets.py. Do not edit by hand.",
                 "\\begin{table}[!ht]"]
        if self.wide:
            lines.append("\\begin{adjustwidth}{-0.5in}{-0.5in}")   # 1 in margins after the abstract -> 7.5 in wide
        lines += ["\\centering", f"\\caption{{{{\\bf {texsafe(self.title)}}} {caption_tex}}}",
                  f"\\begin{{tabular}}{{{self.colspec}}}", "\\hline",
                  " & ".join(f"{{\\bf {texsafe(h)}}}" for h in self.header) + " \\\\ \\thickhline"]
        for r in self.rows:
            if isinstance(r, str):
                lines.append(f"\\multicolumn{{{len(self.header)}}}{{l}}{{\\it {texsafe(r)}}} \\\\ \\hline")
            else:
                lines.append(" & ".join(cell(c) if isinstance(c, tuple) else texsafe(c) for c in r) + " \\\\ \\hline")
        lines.append("\\end{tabular}")
        if self.notes:
            lines.append(f"\\begin{{flushleft}} {self.notes[0]} \\end{{flushleft}}")
        lines.append(f"\\label{{tab:{self.key}}}")
        if self.wide:
            lines.append("\\end{adjustwidth}")
        lines.append("\\end{table}")
        return "\n".join(lines) + "\n"

    def md(self) -> str:
        cell = lambda c: c[1] if isinstance(c, tuple) else c.replace("\\textminus{}", "−").replace("$", "")
        out = ["| " + " | ".join(self.header) + " |", "|" + "|".join("---" for _ in self.header) + "|"]
        for r in self.rows:
            out.append(f"| *{r}* |" + " |" * (len(self.header) - 1) if isinstance(r, str) else "| " + " | ".join(cell(c) for c in r) + " |")
        if self.notes:
            out.append("\n" + self.notes[1])
        return "\n".join(out)


class DataTable(Table):
    """Table 1: split counts (7 columns) and audit findings (spanning cells) in one cell-based table."""

    def __init__(self, N, split_rows, audit):
        super().__init__("data", "Data, split and audit.",
                         "Top: images per disease category in the official FIVES split and in the split used here "
                         "(training and validation drawn from the official training images with seed <<SplitSeed>>, "
                         "<<NValPerCategory>> validation images per category; <<NGroups>> groups of duplicate or possibly "
                         "same-patient images, <<NGroupedImages>> images, never split). Bottom: findings of the data audit "
                         "of all <<NImages>> images and masks in the copy used (Hugging Face mirror of FIVES, identical to the official "
                         "release), made before any "
                         "model was trained. Markings were detected and compared as binary patterns only and were never read.",
                         ["Category", "Official train", "Official test", "Train", "Validation", "Excluded", "Test"], split_rows,
                         colspec="p{1.45in}cccccc", wide=True)
        self.N, self.audit = N, audit

    def tex(self, caption_tex):
        body = super().tex(caption_tex).splitlines()
        k = next(i for i, l in enumerate(body) if l.startswith("\\end{tabular}"))
        add = ["\\multicolumn{7}{l}{{\\bf Audit finding}} \\\\ \\thickhline"]
        for f, detail, handling in self.audit:
            add.append(f"{texsafe(f)} & \\multicolumn{{3}}{{p{{2.0in}}}}{{{detail}}} & \\multicolumn{{3}}{{p{{1.9in}}}}{{{texsafe(handling)}}} \\\\ \\hline")
        return "\n".join(body[:k] + add + body[k:]) + "\n"

    def md(self):
        out = super().md() + "\n\n| Audit finding | Detail | Handling |\n|---|---|---|\n"
        for f, detail, handling in self.audit:
            d = detail
            for macro in sorted(self.N.text, key=len, reverse=True):
                d = d.replace(f"\\{macro}{{}}", plain(self.N.text[macro]))
            out += f"| {f} | {d.replace('--', '–')} | {handling} |\n"
        return out


def pm(N, a, b):
    """'mean ± sd' cell from two macros."""
    return (f"\\{a}{{}} $\\pm$ \\{b}{{}}", f"{plain(N.text[a])} ± {plain(N.text[b])}")


def ci(N, base):
    return (f"\\{base}{{}} [\\{base}Lo{{}}, \\{base}Hi{{}}]",
            f"{plain(N.text[base])} [{plain(N.text[base + 'Lo'])}, {plain(N.text[base + 'Hi'])}]")


def build_tables(N: Numbers) -> list[Table]:
    T = []
    sp = js("results/checks/split_summary.json")["counts"]
    p1 = js("results/checks/phase1_checks.json")["category_counts"]
    rows = []
    for c, nm in CATS + [("All", "Total")]:
        rows.append([f"{nm}" + (f" ({c})" if c != "All" else ""), f"{p1['train'][c]:,}", f"{p1['test'][c]:,}",
                     f"{sp['train'][c]:,}", f"{sp['val'][c]:,}", f"{sp['excluded'][c]:,}", f"{sp['test'][c]:,}"])
    split_rows = []
    for c, nm in CATS + [("All", "Total")]:
        split_rows.append([f"{nm}" + (f" ({c})" if c != "All" else ""), f"{p1['train'][c]:,}", f"{p1['test'][c]:,}",
                           f"{sp['train'][c]:,}", f"{sp['val'][c]:,}", f"{sp['excluded'][c]:,}", f"{sp['test'][c]:,}"])
    audit = [
        ("Same image file, two different masks", "\\NDupPairsTrain{} pairs in train, \\NDupPairsTest{} in test; mask Dice over the whole image \\DupTrainMaskDiceMin{}--\\DupTrainMaskDiceMax{} (train), \\DupTestMaskDice{} (test)",
         "Train pairs kept on one side of the split; test pair counted as two images"),
        ("Empty ground-truth mask", "\\NEmptyMasks{} very dark images (a near-identical pair)", "Excluded"),
        ("Same eye in official train and test", "\\NSameEyePairs{} pair, vessel-layout IoU \\SameEyeIoU{}", "Training image excluded; test set unchanged"),
        ("Annotation lines outside the eye", "\\NBorderMasks{} masks, \\NBorderPx{} pixels", "All scoring inside a fitted FOV mask; masks not edited"),
        ("Burned-in text markings outside the eye", "\\NMarkings{} of \\NImages{} images; possibly identifying", "Never read; pixelated in all figures; compared only as binary patterns"),
        ("Possibly same patient (identical marking pattern, isolated pair)", "\\NIsoPairs{} pairs: \\NIsoTrainTest{} train--test, \\NIsoValTest{} validation--test, \\NIsoTrainVal{} train--validation, \\NIsoTestTest{} within test",
         "Grouped within the official training set; pairs involving test remain"),
    ]
    T.append(DataTable(N, split_rows, audit))

    rows = []
    spec = {
        "unet": ("--", "CLAHE image, area-resized to 512", "soft mask at 512", "512"),
        "attention_unet": ("attention gates on 4 skips", "as U-Net", "as U-Net", "512"),
        "multiscale_unet": ("multi-scale block in 9 blocks", "as U-Net", "as U-Net", "512"),
        "ms_attention_unet": ("both", "as U-Net", "as U-Net", "512"),
        "sam_lora_tiled": ("SAM ViT-B, LoRA r=4 + decoder", "4 native 1024 crops", "binary native crop", "512 (4×256 stitched)"),
        "sam_lora_tiled_overlap": ("same models as tiled", "9 native 1024 crops, stride 512", "--", "512 (tent-blended)"),
        "sam_lora_whole": ("SAM ViT-B, LoRA r=4 + decoder", "CLAHE image, area-resized to 1024", "soft mask at 1024", "1024 (from 256)"),
    }
    for r in RUNS:
        ch, inp, tgt, out = spec[r]
        cd = CODE[r]
        params = (f"\\{cd}ParamsTrainableM{{}} / \\{cd}ParamsM{{}}", f"{plain(N.text[cd + 'ParamsTrainableM'])} / {plain(N.text[cd + 'ParamsM'])}")
        rows.append([NAME[r].replace(" (main SAM)", ""), ch, inp, tgt, out, params])
    T.append(Table("models", "Models compared.",
                   "Everything not listed is shared by all trained models: data split, CLAHE preprocessing, augmentation "
                   "(flips and 90° rotations), loss (binary cross-entropy + soft Dice), Adam with learning rate 10\\textsuperscript{-4}, "
                   "effective batch 4, early stopping on validation Dice at 2048 (patience 20, at most 150 epochs), mixed "
                   "precision, three seeds, threshold chosen on validation, and scoring at 2048 inside the FOV. Parameters in "
                   "millions (trainable / total).",
                   ["Model", "Change vs U-Net", "Network input", "Training target", "Output grid", "Params (M)"], rows,
                   colspec="p{1.2in}p{1.25in}p{1.2in}p{0.9in}p{0.9in}p{0.7in}", wide=True))

    cols = ["dice", "cldice", "nsd", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick"]
    rows = []
    for r in RUNS:
        if r == "sam_lora_tiled":
            rows.append("Fine-tuned SAM (LoRA)")
        if r == "unet":
            rows.append("CNNs (2 × 2 ablation)")
        rows.append([NAME[r].replace(" (main SAM)", "")] + [pm(N, f"{CODE[r]}{MCODE[m]}Mean", f"{CODE[r]}{MCODE[m]}SdSeed") for m in cols])
    T.append(Table("main", "Test results.",
                   "Mean $\\pm$ standard deviation across <<NSeeds>> training seeds of each seed's mean over the <<NTestImages>> test "
                   "images, each run at its own threshold chosen on validation. Thin, medium and thick: centreline recall by "
                   "ground-truth vessel radius (< 3, 3--7, > 7 px at 2048, 1-px tolerance). NSD: normalized surface Dice "
                   "(boundary-pixel approximation, 2 px). Overlapping tiles: inference-only variant of the tiled SAM models.",
                   ["Model", "Dice", "clDice", "NSD", "Thin", "Medium", "Thick"], rows, wide=True))

    rows = []
    for r in RUNS[1:]:
        rows.append([NAME[r].replace(" (main SAM)", "")] + [
            (f"\\{CODE[r]}{MCODE[m]}Diff{{}} (\\{CODE[r]}{MCODE[m]}P{{}})", f"{plain(N.text[CODE[r] + MCODE[m] + 'Diff'])} ({plain(N.text[CODE[r] + MCODE[m] + 'P'])})")
            for m in ("dice", "cl_recall_thin", "cldice", "nsd")])
    T.append(Table("paired", "Image-level comparison with U-Net.",
                   "Mean over the <<NTestImages>> test images of the per-image difference from U-Net (each image's metric "
                   "averaged over the model's seeds first), with the Holm-adjusted p-value of a two-sided paired Wilcoxon "
                   "signed-rank test (correction within each metric across the six comparisons). These tests treat images as "
                   "the sample and do not include training-seed variability.",
                   ["Model − U-Net", "Dice", "Thin", "clDice", "NSD"], rows, wide=True))

    rows = []
    for rule, rl in (("Sel", "Val-selected"), ("Fix", "Fixed 0.5")):
        rows.append(f"Threshold: {rl.lower()}")
        for m in ("dice", "cl_recall_thin", "cldice", "nsd", "precision", "recall"):
            if f"AblAtt{MCODE[m]}{rule}" not in N.raw:
                continue
            rows.append([MLABEL[m] if m != "cl_recall_thin" else "Thin recall"] +
                        [ci(N, f"Abl{e}{MCODE[m]}{rule}") for e in ("Att", "MS", "Int")])
    T.append(Table("ablation", "2 × 2 ablation of attention gates and multi-scale blocks.",
                   "Effects on test metrics with 95% bootstrap confidence intervals (resampling the <<NTestImages>> images; "
                   "each image averaged over seeds). Attention main effect: mean of (Attention U-Net − U-Net) and "
                   "(Multi-Scale Attention U-Net − Multi-Scale U-Net); multi-scale main effect likewise; interaction: "
                   "(Multi-Scale Attention U-Net − Multi-Scale U-Net) − (Attention U-Net − U-Net). clDice and NSD exist "
                   "only at the val-selected thresholds.",
                   ["Metric", "Attention", "Multi-scale", "Interaction"], rows, wide=True))

    rows = []
    for r in RUNS:
        cd = CODE[r]
        nm = NAME[r].replace(" (main SAM)", "")
        trh = t(N, f"{cd}TrainH") if f"{cd}TrainH" in N.raw else ("(tiled models)" if r == "sam_lora_tiled_overlap" else "--")
        trm = t(N, f"{cd}TrainMem") if f"{cd}TrainMem" in N.raw else "--"
        rows.append([nm, t(N, f"{cd}ParamsTrainableM"), t(N, f"{cd}InfMs"), t(N, f"{cd}Gflops"), t(N, f"{cd}InfMem"), trh, trm])
    T.append(Table("cost", "Computational cost.",
                   "One NVIDIA L40S GPU. Inference: one image per call over the <<NTestImages>> test images after 10 warm-up "
                   "calls, from the input on the GPU to the probability map (tiling, stitching or blending included); FLOPs "
                   "per image from PyTorch's FlopCounterMode; peak memory allocated. Training time and memory: seed-0 runs.",
                   ["Model", "Trainable params (M)", "Inference (ms/image)", "GFLOPs/image", "Inference memory (GB)",
                    "Training time (h)", "Training memory (GB)"], rows, wide=True))

    rows = [["SAM zero-shot, oracle prompts (K = \\OracleK{})"] + [t(N, f"Oracle{c}") for c in ("Dice", "Prec", "Rec", "Thin", "ClDice", "NSD")]]
    rows[0][0] = (f"SAM zero-shot, oracle prompts (K = \\OracleK{{}})", f"SAM zero-shot, oracle prompts (K = {plain(N.text['OracleK'])})")
    for cd, nm in (("RefFive", "Reference: ground truth through the 512 grid"), ("RefTwo", "Reference: ground truth through the 256 grid")):
        rows.append([nm] + [t(N, f"{cd}{c}") for c in ("Dice", "Prec", "Rec", "Thin", "ClDice", "NSD")])
    T.append(Table("oracle", "Oracle-prompted SAM and mask-resampling references.",
                   "Test set, means over <<NTestImages>> images. The oracle rows use five positive and five negative points "
                   "sampled from each image's own ground truth, so they are not comparable with the automatic models and are "
                   "not ranked against them. The references push each test ground truth through an output grid and back to "
                   "2048 (soft downsampling, bilinear upsampling, same threshold rule); they show what the grid itself loses "
                   "and are not upper bounds. Oracle prompts: K = <<OracleK>> chosen on validation (validation Dice <<OracleValDiceKFive>>, "
                   "<<OracleValDiceKTen>> and <<OracleValDiceKTwenty>> for K = 5, 10 and 20); inference <<OracleInfMs>> ms per image "
                   "(SAM's default single-precision path, against mixed precision for the fine-tuned models).",
                   ["Row", "Dice", "Precision", "Recall", "Thin", "clDice", "NSD"], rows, wide=True, si=True))
    rows = []
    for _, r in csv("results/tables/annotation_agreement.csv").iterrows():
        a, b = r["pair"].split("|")
        rows.append([f"{a} / {b}", r["split"]] + [f"{r[m]:.3f}" for m in ("dice", "cl_recall_thin", "cl_recall_medium",
                                                                           "cl_recall_thick", "cldice", "nsd")])
    rows.append([("Mean", "Mean"), ""] + [t(N, f"Agree{c}Mean") for c in ("Dice", "Thin", "Med", "Thick", "ClDice", "NSD")])
    T.append(Table("si_agreement", "Agreement between two available annotations of the same image.",
                   "Post hoc descriptive analysis added during writing, using labels only (no model predictions). FIVES "
                   "contains <<AgreeNPairs>> byte-identical image files with two different masks (all AMD images); the dataset "
                   "does not say who drew them, so this is not inter-rater agreement. Each mask was scored against the other "
                   "with the evaluation metrics (2048, inside the FOV, 1-px tolerance for centreline recall); values are "
                   "the mean of the two directions (lowest single direction for thin vessels: <<AgreeThinDirMin>>). Dice here is "
                   "computed inside the FOV; Table 1 gives the same pairs' mask Dice over the whole image. n = <<AgreeNPairs>>.",
                   ["Images", "Split", "Dice", "Thin", "Medium", "Thick", "clDice", "NSD"], rows, wide=True, si=True))

    # ---- supporting tables
    rows = []
    for r in RUNS:
        rows.append([NAME[r].replace(" (main SAM)", "")] + [
            (pm(N, f"{CODE[r]}{MCODE[m]}Mean", f"{CODE[r]}{MCODE[m]}SdSeed")[0] + ("" if r == "unet" else f" (\\{CODE[r]}{MCODE[m]}P{{}})"),
             pm(N, f"{CODE[r]}{MCODE[m]}Mean", f"{CODE[r]}{MCODE[m]}SdSeed")[1] + ("" if r == "unet" else f" ({plain(N.text[CODE[r] + MCODE[m] + 'P'])})"))
            for m in ("iou", "precision", "recall")])
    T.append(Table("si_overlap", "IoU, precision and recall on the test set.",
                   "Mean $\\pm$ standard deviation across <<NSeeds>> seeds (Holm-adjusted Wilcoxon p-value vs U-Net in brackets).",
                   ["Model", "IoU", "Precision", "Recall"], rows, si=True))
    rows = []
    for r in RUNS:
        rows.append([NAME[r].replace(" (main SAM)", "")] + [pm(N, f"{CODE[r]}{MCODE[m]}Mean", f"{CODE[r]}{MCODE[m]}SdSeed")
                                                            for m in ("cl_recall_thin_r4_8", "cl_recall_medium_r4_8", "cl_recall_thick_r4_8")])
    T.append(Table("si_width", "Width classes with alternative cut-offs.",
                   "Centreline recall with radius cut-offs 4 / 8 px instead of 3 / 7 (same predictions; sensitivity check). "
                   "Mean $\\pm$ standard deviation across <<NSeeds>> seeds.",
                   ["Model", "Thin (r < 4)", "Medium (4–8)", "Thick (r > 8)"], rows, si=True))
    rows = []
    for r in RUNS:
        rows.append([NAME[r].replace(" (main SAM)", "")] + [
            (pm(N, f"{CODE[r]}{MCODE[m]}FixMean", f"{CODE[r]}{MCODE[m]}FixSdSeed")[0] + ("" if r == "unet" else f" (\\{CODE[r]}{MCODE[m]}FixP{{}})"),
             pm(N, f"{CODE[r]}{MCODE[m]}FixMean", f"{CODE[r]}{MCODE[m]}FixSdSeed")[1] + ("" if r == "unet" else f" ({plain(N.text[CODE[r] + MCODE[m] + 'FixP'])})"))
            for m in ("dice", "precision", "recall", "cl_recall_thin")])
    T.append(Table("si_fixed", "Test results with every run at a fixed threshold of 0.5.",
                   "Secondary analysis decided before test evaluation. Mean $\\pm$ standard deviation across <<NSeeds>> seeds "
                   "(Holm-adjusted Wilcoxon p-value vs U-Net). clDice and NSD were computed only at the val-selected thresholds.",
                   ["Model", "Dice", "Precision", "Recall", "Thin"], rows, wide=True, si=True))
    rows = []
    for r in RUNS:
        rows.append([NAME[r].replace(" (main SAM)", "")] + [t(N, f"{CODE[r]}{m}Cat{c}") for m in ("Dice", "Thin") for c, _ in CATS])
    T.append(Table("si_disease", "Test results by disease category.",
                   "Seed means; each category has <<NTestPerCategory>> test images. Descriptive only: no tests between categories "
                   "were planned. All metrics: test\\_disease.csv in the repository.",
                   ["Model"] + [f"Dice {c}" for c, _ in CATS] + [f"Thin {c}" for c, _ in CATS], rows, wide=True, si=True))
    rows = []
    for r in RUNS:
        cd = CODE[r]
        rows.append([NAME[r].replace(" (main SAM)", ""), t(N, f"{cd}DensBias"), t(N, f"{cd}DensMAD"), t(N, f"{cd}DensR"),
                     t(N, f"{cd}FDBias"), t(N, f"{cd}FDMAD"), t(N, f"{cd}FDR")])
    T.append(Table("si_vessel", "Image-derived vessel measures.",
                   "Vessel density (vessel pixels per FOV pixel; ground-truth mean \\DensGTMean{}) and box-counting fractal "
                   "dimension (ground-truth mean \\FDGTMean{}), predicted vs ground truth on the <<NTestImages>> test images "
                   "(per-image values averaged over seeds). Bias: mean of predicted − ground truth; MAD: mean absolute "
                   "difference; r: Pearson correlation. Not clinically validated.",
                   ["Model", "Density bias", "Density MAD", "Density r", "FD bias", "FD MAD", "FD r"], rows, wide=True, si=True))
    rows = []
    for r in ("unet", "attention_unet", "multiscale_unet", "ms_attention_unet", "sam_lora_tiled", "sam_lora_whole"):
        for s in SEEDS:
            sc = SEEDCODE[s]
            rows.append([NAME[r].replace(" (main SAM)", ""), str(s), t(N, f"{CODE[r]}EpochsSeed{sc}"), t(N, f"{CODE[r]}BestEpochSeed{sc}"),
                         t(N, f"{CODE[r]}ThrSeed{sc}"), t(N, f"{CODE[r]}ValBestDiceSeed{sc}"), t(N, f"{CODE[r]}ValThinSeed{sc}")])
    for sd in SEEDS:
        rows.append(["SAM, overlapping tiles", str(sd), "(tiled)", "(tiled)", t(N, f"SAMOverlapThrSeed{SEEDCODE[sd]}"), "--", "--"])
    T.append(Table("si_runs", "Training runs.",
                   "Epochs run (maximum 150), best epoch, threshold chosen on validation, best validation Dice at threshold 0.5 "
                   "(the checkpoint-selection criterion) and validation thin-vessel recall at the chosen threshold. Overlapping "
                   "tiles reuse the tiled models; only their thresholds were chosen separately on validation.",
                   ["Model", "Seed", "Epochs", "Best epoch", "Threshold", "Val Dice (0.5)", "Val thin"], rows, wide=True, si=True))
    rows = []
    for res, rc in ((512, "Five"), (1024, "Ten")):
        for meth, mc, lab in (("nearest", "Nearest", "nearest neighbour"), ("area_t50", "AreaHalf", "area average ≥ 0.5"),
                              ("area_t25", "AreaQuarter", "area average ≥ 0.25"), ("maxpool", "Maxpool", "max-pooling"),
                              ("soft", "Soft", "area average, no threshold (used)")):
            if f"Resize{rc}{mc}Dice" in N.raw:
                rows.append([str(res), lab, t(N, f"Resize{rc}{mc}Dice"), t(N, f"Resize{rc}{mc}Prec"), t(N, f"Resize{rc}{mc}Thin")])
        if f"Resize{rc}SoftQuarterThin" in N.raw:
            rows.append([str(res), "soft, scored at threshold 0.25 (illustration)", t(N, f"Resize{rc}SoftQuarterDice"), "--",
                         t(N, f"Resize{rc}SoftQuarterThin")])
    T.append(Table("si_resize", "Training-mask resizing.",
                   "Thirty training masks (stratified by category, seed 0) downsampled by each method, upsampled back to 2048 "
                   "as predictions are (bilinear, threshold 0.5) and scored against the original masks inside the FOV. Chosen "
                   "before any model was trained. At 1024, max-pooling and area average ≥ 0.25 coincide for 2 × 2 blocks.",
                   ["Size", "Method", "Dice", "Precision", "Thin recall"], rows, si=True))
    rows = [["U-Net, 512 (used)", t(N, "PilotFiveDice"), t(N, "PilotFiveThin"), t(N, "PilotFiveEpochS"), t(N, "PilotFiveMem")],
            ["U-Net, 1024 (pilot)", t(N, "PilotTenDice"), t(N, "PilotTenThin"), t(N, "PilotTenEpochS"), t(N, "PilotTenMem")],
            ["Difference", t(N, "PilotDiceGain"), t(N, "PilotThinGain"), "", ""]]
    T.append(Table("si_pilot", "Resolution pilot (validation).",
                   "U-Net seed 0 at 512 and at 1024, each at its own threshold chosen on validation (means over <<NVal>> images). "
                   "The rule fixed before the pilot required a thin-recall gain of at least 0.05 with a Dice loss of at most "
                   "0.005 to move the CNNs to 1024; the thin-recall gain was below 0.05, so all CNNs stayed at 512.",
                   ["Run", "Dice", "Thin recall", "Epoch (s)", "Peak memory (GB)"], rows, si=True))
    rows = [["Tiled SAM (non-overlapping)", t(N, "SeamSAMTiledJump"), ci(N, "SeamSAMTiledDiDDice"), ci(N, "SeamSAMTiledDiDThin")],
            ["Overlapping tiles", t(N, "SeamSAMOverlapJump"), ci(N, "SeamSAMOverlapDiDDice"), ci(N, "SeamSAMOverlapDiDThin")],
            ["Whole-image SAM (no seams)", t(N, "SeamSAMWholeJump"), "--", "--"],
            ["U-Net (no seams)", t(N, "SeamUNetJump"), "--", "--"]]
    T.append(Table("si_seam", "Tile-seam check (validation, seed 0).",
                   "Jump ratio: mean probability jump across the tile seam divided by the median jump between other adjacent "
                   "rows and columns (about 1: no seam). Difference-in-differences against U-Net in a ±16 px band around the "
                   "seam, relative to bands around the tile midlines, with 95% bootstrap CIs over <<NVal>> images.",
                   ["Run", "Jump ratio", "Dice DiD", "Thin recall DiD"], rows, wide=True, si=True))
    rows = []
    for rule, rl in (("Sel", "validation-selected"), ("Fix", "fixed 0.5")):
        for m, ml in (("Thin", "Thin recall"), ("Prec", "Precision"), ("Rec", "Recall"), ("Dice", "Dice")):
            rows.append([f"{ml}, {rl}"] + [t(N, f"Simple{e}{m}{rule}") for e in ("AttNoMS", "AttWithMS", "MSNoAtt", "MSWithAtt")])
    T.append(Table("si_simple", "Simple effects in the 2 × 2 ablation.",
                   "Differences of model means on the <<NTestImages>> test images (each a mean over <<NSeeds>> seeds): attention "
                   "without multi-scale blocks (Attention U-Net − U-Net), attention with them (Multi-Scale Attention U-Net − "
                   "Multi-Scale U-Net), and the multi-scale block without and with attention. Point estimates only; the main "
                   "effects and interaction with confidence intervals are in Table 5.",
                   ["Metric, threshold rule", "Attention, no MS", "Attention, with MS", "MS, no attention", "MS, with attention"],
                   rows, wide=True, si=True))
    rows = []
    for r in RUNS:
        rows.append([NAME[r].replace(" (main SAM)", "")] + [
            (f"\\{CODE[r]}{MCODE[m]}SdImg{{}} / \\{CODE[r]}{MCODE[m]}Median{{}}",
             f"{plain(N.text[CODE[r] + MCODE[m] + 'SdImg'])} / {plain(N.text[CODE[r] + MCODE[m] + 'Median'])}")
            for m in ("dice", "cl_recall_thin", "cldice", "nsd")])
    T.append(Table("si_imagelevel", "Image-level spread of the test results.",
                   "Standard deviation over the <<NTestImages>> test images / median, of the per-image values averaged over "
                   "<<NSeeds>> seeds.", ["Model", "Dice", "Thin", "clDice", "NSD"], rows, si=True))
    T += posthoc_tables(N)
    return T


# ----------------------------------------------------------------------------------------- figures
def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    PREV.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(PREV / f"{name}.png", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


def mk(ax, run, x, y, **kw):
    st = STYLE[run]
    face = "white" if st.get("hollow") else st["color"]
    ax.plot(x, y, ls="none", marker=st["marker"], mfc=face, mec=st["color"] if st.get("hollow") else INK, mew=0.6,
            color=st["color"], **kw)


def fig_pipeline(plt):
    """Schematic in inch coordinates on a full-figure axes (no tight_layout), boxes sized to their text."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    W, H = FULL_W, 3.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f3f0", ls="-", fs=6.3, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05", fc=fc, ec=INK2, lw=0.6, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.2, fontweight=weight)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.6, color=INK2,
                                     shrinkA=0, shrinkB=0))

    ax.text(W / 2, 2.93, "All trained models: BCE + soft Dice · Adam, learning rate 1e-4 · batch 4 · early stopping on "
            "validation Dice at 2048 · 3 seeds · soft targets (CNNs, whole-image SAM) or binary tile targets (tiled SAM)",
            ha="center", va="center", fontsize=5.8, color=INK2)
    ax.text(W / 2, 2.77, "Analysis decisions recorded before test evaluation (commit 0ee7360)", ha="center", va="center",
            fontsize=6.0, color=INK2)
    box(0.05, 0.95, 0.95, 0.85, "FIVES image\n2048 × 2048\n\nCLAHE on LAB\nlightness")
    rows = [(2.18, "resize to 512", "CNNs: U-Net ± attention\n± multi-scale (2 × 2)", "512 map", "-"),
            (1.55, "4 native\n1024 crops", "SAM ViT-B + LoRA, tiled\n(overlap: 9 crops, stride 512)", "512 map\n(4 × 256)", "-"),
            (0.92, "resize to 1024", "SAM ViT-B + LoRA,\nwhole image", "256 → 1024", "-"),
            (0.29, "resize to 1024", "SAM zero-shot, 5 + 5 oracle\npoints from own ground truth", "256 → 1024", (0, (3, 2)))]
    h = 0.46
    for y, prep, model, out, ls in rows:
        box(1.25, y, 0.85, h, prep, ls=ls)
        box(2.3, y, 1.85, h, model, fc="#ffffff", ls=ls)
        box(4.35, y, 0.8, h, out, ls=ls)
        arrow(1.0, 1.375, 1.25, y + h / 2)
        arrow(2.1, y + h / 2, 2.3, y + h / 2)
        arrow(4.15, y + h / 2, 4.35, y + h / 2)
        arrow(5.15, y + h / 2, 5.38, 1.45)
    box(5.38, 0.45, 1.07, 2.0, "Bilinear to 2048\n\nThreshold chosen\non validation\n\nScored inside\nthe FOV: Dice,\nIoU, P, R, thin/\nmedium/thick\ncentreline recall,\nclDice, NSD")
    ax.text(2.3, 0.16, "dashed: oracle experiment, reported separately", fontsize=5.6, color=INK2, va="center")
    return fig


def fig_threshold(plt, split="test"):
    d = csv("results/tables/thin_recall_vs_threshold.csv")
    d = d[d.split == split]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.55), sharey=True)
    groups = [("A  CNNs", CNN), ("B  Fine-tuned SAM vs U-Net", ["unet", "sam_lora_tiled", "sam_lora_tiled_overlap", "sam_lora_whole"])]
    for ax, (title, runs) in zip(axes, groups):
        for r in runs:
            s = d[d.run == r].sort_values("threshold")
            st = STYLE[r]
            ax.fill_between(s.threshold, s.seed_min, s.seed_max, color=st["color"], alpha=0.16, lw=0)
            ax.plot(s.threshold, s.cl_recall_thin, color=st["color"], ls=st["ls"], lw=1.3,
                    marker=st["marker"], markevery=3, mfc="white" if st.get("hollow") else st["color"],
                    mec=st["color"] if st.get("hollow") else INK, mew=0.5, ms=3.8, label=seedlab(r, int(s.n_seeds.iloc[0])))
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.set_xlabel("Threshold")
        ax.set_xlim(0.05, 0.95)
        ax.grid(axis="y")
        ax.legend(loc="lower left", fontsize=6.3, handlelength=2.4)
    axes[0].set_ylabel(f"Thin-vessel centreline recall ({split})")
    fig.tight_layout(pad=0.3, w_pad=1.0)
    return fig


def fig_precision_thin(plt):
    """Post hoc (D2): thin-vessel recall against overall precision on the test set, one point per threshold of the fine
    grid (0.05-0.95 shown), seed-averaged."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True, sharex=True)
    groups = [("A  CNNs", CNN), ("B  Fine-tuned SAM vs U-Net", ["unet", "sam_lora_tiled", "sam_lora_tiled_overlap", "sam_lora_whole"])]
    for ax, (title, runs) in zip(axes, groups):
        for r in runs:
            f = pd.concat([csv(f"results/eval/{r}/seed{s}/test_sweep_fine.csv") for s in SEEDS])
            per_seed = f.groupby(["seed", "threshold"])[["precision", "cl_recall_thin"]].mean().reset_index()
            c = per_seed.groupby("threshold")[["precision", "cl_recall_thin"]].mean()
            c = c[(c.index >= 0.05 - 1e-9) & (c.index <= 0.95 + 1e-9)]
            st = STYLE[r]
            ax.plot(c.precision, c.cl_recall_thin, color=st["color"], ls=st["ls"], lw=1.3, marker=st["marker"],
                    markevery=[int(np.argmin(np.abs(c.index.to_numpy() - 0.5)))], ms=4.2,
                    mfc="white" if st.get("hollow") else st["color"], mec=st["color"] if st.get("hollow") else INK,
                    mew=0.5, label=seedlab(r, per_seed.seed.nunique()))
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.set_xlabel("Overall precision (test)")
        ax.grid(axis="both")
        ax.legend(loc="lower left", fontsize=6.3, handlelength=2.4)
    axes[0].set_ylabel("Thin-vessel centreline recall (test)")
    axes[0].set_xlim(0.80, 0.99)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    return fig


def per_seed_test(run, metric, thr=None):
    vals = []
    for s in SEEDS:
        if thr is None:
            vals.append(csv(f"results/eval/{run}/seed{s}/test_metrics.csv")[metric].mean())
        else:
            sw = csv(f"results/eval/{run}/seed{s}/test_sweep.csv")
            vals.append(sw[np.isclose(sw.threshold, thr)][metric].mean())
    return np.array(vals)


def fig_ablation(plt):
    a = csv("results/tables/test_ablation_effects.csv")
    f = csv("results/tables/test_ablation_effects_fixed_t0.5.csv")
    metrics = ["dice", "cldice", "nsd", "cl_recall_thin", "precision", "recall"]
    fig = plt.figure(figsize=(FULL_W, 4.35))
    top = fig.add_gridspec(1, 3, left=0.11, right=0.99, top=0.88, bottom=0.55, wspace=0.08)
    bot = fig.add_gridspec(1, 2, left=0.11, right=0.99, top=0.40, bottom=0.10, wspace=0.12)
    axes = [fig.add_subplot(top[0, 0])]
    axes += [fig.add_subplot(top[0, k], sharex=axes[0]) for k in (1, 2)]
    for j, (ax, eff, title) in enumerate(zip(axes, ("attention", "multi-scale", "interaction"),
                                             ("A  Attention main effect", "B  Multi-scale main effect", "C  Interaction"))):
        for i, m in enumerate(metrics):
            for dfr, off, mk_, face, lab in ((a, -0.15, "o", INK, "val-selected thresholds"), (f, 0.15, "s", "white", "fixed threshold 0.5")):
                row = dfr[(dfr.metric == m) & (dfr.effect == eff)]
                if not len(row):
                    continue
                row = row.iloc[0]
                y = len(metrics) - 1 - i + off
                ax.plot([row.ci_low, row.ci_high], [y, y], color=INK, lw=0.9)
                ax.plot(row["mean"], y, marker=mk_, mfc=face, mec=INK, mew=0.7, ms=4, ls="none", label=lab)
        ax.axvline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([("Thin recall" if m == "cl_recall_thin" else MLABEL[m]) for m in metrics[::-1]] if j == 0 else [])
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.set_xlabel("Effect (95% CI over images)")
        ax.grid(axis="x")
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="upper right", ncol=2, fontsize=6.6, bbox_to_anchor=(0.99, 0.995))
    b0 = fig.add_subplot(bot[0, 0])
    b1 = fig.add_subplot(bot[0, 1], sharey=b0)
    for ax, thr, title in ((b0, None, "D  Thin recall per seed, val-selected thresholds"), (b1, 0.5, "E  Thin recall per seed, threshold 0.5")):
        for k, r in enumerate(CNN):
            v = per_seed_test(r, "cl_recall_thin", thr)
            st = STYLE[r]
            ax.plot([k - 0.24, k + 0.24], [v.mean()] * 2, color=INK, lw=1.0)
            ax.plot(np.full(3, k) + np.array([-0.08, 0, 0.08]), v, ls="none", marker=st["marker"], mfc=st["color"], mec=INK,
                    mew=0.5, ms=4.4)
        ax.set_xticks(range(len(CNN)))
        ax.set_xticklabels([SHORT[r] for r in CNN])
        ax.set_title(title, loc="left", fontsize=7.6)
        ax.grid(axis="y")
    b0.set_ylabel("Test thin recall\n(marker: seed; bar: mean)")
    plt.setp(b1.get_yticklabels(), visible=False)
    return fig


def fig_sam(plt):
    runs = ["unet", "sam_lora_whole", "sam_lora_tiled", "sam_lora_tiled_overlap"]
    ref = csv("results/tables/test_references.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    widths = [("cl_recall_thin", "Thin (r < 3)"), ("cl_recall_medium", "Medium (3–7)"), ("cl_recall_thick", "Thick (r > 7)")]
    for g, (m, lab) in enumerate(widths):
        for k, r in enumerate(runs):
            x = g + (k - 1.5) * 0.19
            v = per_seed_test(r, m)
            st = STYLE[r]
            ax.plot([x - 0.07, x + 0.07], [v.mean()] * 2, color=INK, lw=1.0)
            ax.plot(np.full(3, x), v, ls="none", marker=st["marker"], ms=4, mew=0.5,
                    mfc="white" if st.get("hollow") else st["color"], mec=st["color"] if st.get("hollow") else INK,
                    label=seedlab(r) if g == 0 else None)
        for rr, ls_, rl in (("reference_soft_512", (0, (1, 1.2)), "GT through 512 grid"), ("reference_soft_256", (0, (4, 1.5, 1, 1.5)), "GT through 256 grid")):
            y = ref[(ref.run == rr) & (ref.metric == m)]["mean"].iloc[0]
            ax.plot([g - 0.42, g + 0.42], [y, y], color=INK2, lw=0.9, ls=ls_, label=rl if g == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels([w[1] for w in widths])
    ax.set_ylabel("Test centreline recall (per seed)")
    ax.set_ylim(0.3, 1.02)
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=6.1, ncol=1)
    ax.set_title("A  Centreline recall by vessel width", loc="left", fontsize=7.8)
    ax = axes[1]
    for k, (m, lab) in enumerate((("dice", "Dice"), ("nsd", "NSD"))):
        for j, r in enumerate(runs):
            x = k + (j - 1.5) * 0.19
            v = per_seed_test(r, m)
            st = STYLE[r]
            ax.plot([x - 0.07, x + 0.07], [v.mean()] * 2, color=INK, lw=1.0)
            ax.plot(np.full(3, x), v, ls="none", marker=st["marker"], ms=4, mew=0.5,
                    mfc="white" if st.get("hollow") else st["color"], mec=st["color"] if st.get("hollow") else INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Dice", "NSD"])
    ax.set_title("B  Overlap and boundary", loc="left", fontsize=7.8)
    ax.grid(axis="y")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    return fig


def fig_tradeoff(plt):
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.4))
    fam = [("CNNs", CNN), ("Fine-tuned SAM vs U-Net", ["unet", "sam_lora_tiled", "sam_lora_tiled_overlap", "sam_lora_whole"])]
    letters = (("A", "C"), ("B", "D"))
    for j, (title, runs) in enumerate(fam):
        ax = axes[0, j]
        for r in runs:
            mk(ax, r, per_seed_test(r, "dice"), per_seed_test(r, "cl_recall_thin"), ms=4.5, label=seedlab(r))
        ax.set_title(f"{letters[j][0]}  Test: {title}", loc="left", fontsize=7.8)
        ax.set_xlabel("Test Dice (seed mean over 200 images)")
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        ax = axes[1, j]
        for r in [x for x in runs if x != "sam_lora_tiled_overlap"]:
            xs = [js(f"results/train/{r}/seed{s}/summary.json")["best_val_dice_at_0.5"] for s in SEEDS]
            ys = [csv(f"results/eval/{r}/seed{s}/val_metrics.csv")["cl_recall_thin"].mean() for s in SEEDS]
            mk(ax, r, xs, ys, ms=4.5)
        ax.set_title(f"{letters[j][1]}  Validation: {title}", loc="left", fontsize=7.8)
        ax.set_xlabel("Best validation Dice at 0.5 (checkpoint criterion)")
        ax.grid(True)
        keep = [(h, l) for h, l in zip(handles, labels) if "overlapping" not in l]
        ax.legend([h for h, _ in keep], [l for _, l in keep], fontsize=6.0, loc="upper left" if j == 0 else "upper center",
                  bbox_to_anchor=None if j == 0 else (0.48, 1.0))
    for i in range(2):
        lo = min(a.get_xlim()[0] for a in axes[i]); hi = max(a.get_xlim()[1] for a in axes[i])
        ylo = min(a.get_ylim()[0] for a in axes[i]); yhi = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_xlim(lo, hi)
            a.set_ylim(ylo, yhi)
    axes[0, 0].set_ylabel("Test thin recall")
    axes[1, 0].set_ylabel("Validation thin recall\n(at the chosen threshold)")
    fig.tight_layout(pad=0.3, h_pad=1.0, w_pad=1.0)
    return fig


def fig_cost(plt):
    from matplotlib.ticker import FuncFormatter
    cost = csv("results/tables/cost.csv").set_index("run")
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 2.6))
    offsets = {"unet": (-7, 0, "right"), "attention_unet": (6, 7, "left"), "multiscale_unet": (-7, -7, "right"),
               "ms_attention_unet": (7, 0, "left"), "sam_lora_tiled": (-7, 0, "right"),
               "sam_lora_tiled_overlap": (7, 0, "left"), "sam_lora_whole": (7, 0, "left")}
    for r in RUNS:
        v = per_seed_test(r, "cl_recall_thin")
        x = cost.loc[r, "inference_ms_mean"]
        hollow = STYLE[r].get("hollow")
        ax.plot([x, x], [v.min(), v.max()], color=INK2, lw=0.8, zorder=1)
        ax.plot(x, v.mean(), marker="o" if FAMILY[r] == "cnn" else "v", ms=4.8, mfc="white" if hollow else INK, mec=INK,
                mew=0.7, ls="none", zorder=2)
        dx, dy, ha = offsets[r]
        ax.annotate(SHORT[r], (x, v.mean()), xytext=(dx, dy), textcoords="offset points", fontsize=6.3, ha=ha,
                    va="center", color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(2, 2500)
    ax.set_xlabel("Inference time per image (ms, log scale; one L40S)")
    ax.set_ylabel("Test thin recall\n(seed mean; bar: seed range)")
    ax.grid(True, which="major")
    fig.tight_layout(pad=0.3)
    return fig


def fig_training(plt):
    """Validation Dice (2048, threshold 0.5, FOV) per epoch from the saved training logs."""
    groups = [("A  CNNs", CNN), ("B  Fine-tuned SAM", ["sam_lora_tiled", "sam_lora_whole"])]
    from matplotlib.lines import Line2D
    seed_ls = ("-", (0, (4, 1.5)), (0, (1, 1.2)))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), sharey=True)
    for ax, (title, runs) in zip(axes, groups):
        for r in runs:
            st = STYLE[r]
            for sd in SEEDS:
                lg = csv(f"results/train/{r}/seed{sd}/train_log.csv")
                ax.plot(lg["epoch"], lg["val_dice"], color=st["color"], lw=0.9, alpha=0.9, ls=seed_ls[sd])
        ax.axvline(150, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.set_xlabel("Epoch")
        ax.set_xlim(-3, 155)
        ax.set_ylim(0.86, 0.92)
        ax.grid(True)
        # Identity by colour (model) and line style (seed), in two compact columns clear of the early, steep epochs.
        handles = [Line2D([], [], color=STYLE[r]["color"], lw=1.2, label=NAME[r].replace(" (main SAM)", "")) for r in runs]
        handles += [Line2D([], [], lw=0, label=" ")] * max(0, len(SEEDS) - len(runs))   # keep the seeds in one column
        handles += [Line2D([], [], color=INK2, lw=0.9, ls=seed_ls[sd], label=f"seed {sd}") for sd in SEEDS]
        ax.legend(handles=handles, fontsize=5.6, loc="lower right", ncol=2, bbox_to_anchor=(0.99, 0.02),
                  handlelength=2.6, columnspacing=1.0, framealpha=0.92)
    axes[0].set_ylabel("Validation Dice (2048, threshold 0.5)")
    fig.tight_layout(pad=0.3, w_pad=1.0)
    return fig


def fig_disease(plt):
    d = csv("results/tables/test_disease.csv")
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 3.6), sharey="row")
    for i, m in enumerate(("dice", "cl_recall_thin")):
        for j, (c, cn) in enumerate(CATS):
            ax = axes[i, j]
            for k, r in enumerate(RUNS):
                row = d[(d.run == r) & (d.category == c) & (d.metric == m)].iloc[0]
                st = STYLE[r]
                ax.errorbar(k, row["mean"], yerr=row["sd_seeds"], fmt=st["marker"], ms=3.8, mfc="white" if st.get("hollow") else st["color"],
                            mec=st["color"] if st.get("hollow") else INK, mew=0.5, ecolor=INK2, elinewidth=0.7, capsize=0)
            ax.set_xticks(range(len(RUNS)))
            ax.set_xticklabels([SHORT[r] for r in RUNS] if i == 1 else [], rotation=60, ha="right", fontsize=6)
            if i == 0:
                ax.set_title(f"{cn} ({c}), n = {int(d[(d.category == c)].n_images.max())}", loc="left", fontsize=7.4)
            ax.grid(axis="y")
        axes[i, 0].set_ylabel("Dice" if m == "dice" else "Thin recall")
    fig.tight_layout(pad=0.3)
    return fig


def fig_bland_altman(plt, n_img_text):
    frames = [csv(f"results/eval/{r}/seed{s}/test_metrics.csv").assign(run=r) for r in RUNS for s in SEEDS]
    df = pd.concat(frames)
    pim = df.groupby(["run", "image_id"])[["density_pred", "density_gt", "fd_pred", "fd_gt"]].mean().reset_index()
    fig, axes = plt.subplots(2, len(RUNS), figsize=(FULL_W, 2.9), sharey="row", sharex="row")
    for j, r in enumerate(RUNS):
        s = pim[pim.run == r]
        for i, (k, lab) in enumerate((("density", "Vessel density"), ("fd", "Fractal dimension"))):
            ax = axes[i, j]
            mean = (s[f"{k}_pred"] + s[f"{k}_gt"]) / 2
            diff = s[f"{k}_pred"] - s[f"{k}_gt"]
            ax.scatter(mean, diff, s=2.5, color=STYLE[r]["color"], lw=0)
            b, sd = diff.mean(), diff.std(ddof=1)
            for y, ls_ in ((b, "-"), (b - 1.96 * sd, (0, (2, 2))), (b + 1.96 * sd, (0, (2, 2)))):
                ax.axhline(y, color=INK, lw=0.6, ls=ls_)
            ax.axhline(0, color=INK2, lw=0.4)
            if i == 0:
                ax.set_title(SHORT[r], fontsize=6.5)
            if j == 0:
                ax.set_ylabel(f"{lab}\npred − GT", fontsize=6.5)
            ax.tick_params(labelsize=5.5)
    axes[0, len(RUNS) // 2].set_xlabel("Mean of predicted and ground-truth vessel density", fontsize=6.5)
    axes[1, len(RUNS) // 2].set_xlabel("Mean of predicted and ground-truth fractal dimension", fontsize=6.5)
    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.5)
    return fig


# ----------------------------------------------------------------------------------------- image figures (job)
def image_figures(plt, N):
    import cv2

    from src.data import _redacted, raw_image_path, raw_mask_path, read_npz_member, read_rgb, read_unchanged
    from src.evaluate import eval_settings, load_fov
    from src.metrics import prepare_gt
    from src.report import qualitative_ids, thin_window
    from src.resample import binarize, upsample_prob
    from src.utils import load_config
    cfg = load_config("configs/base.yaml")
    s = eval_settings(cfg)

    def err(pred, gt, fov):
        out = np.empty(pred.shape + (3,), np.uint8)
        out[:] = ERR["bg"]
        p, g = pred & fov, gt & fov
        out[p & g], out[p & ~g], out[~p & g] = ERR["tp"], ERR["fp"], ERR["fn"]
        return out

    def small(a, n=600):
        return cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)

    def clean(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.4)
            sp.set_color(INK2)

    # ---- data audit (all fundus panels pixelated outside the FOV)
    from matplotlib.patches import Rectangle as _Rect
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 3.55))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.08, wspace=0.06, hspace=0.10)
    ig = js("results/checks/integrity.json")
    dup = next(g for g in ig["identical_images"]["groups"] if g["splits"] == ["train", "train"])
    a, b = dup["images"]
    ga, gb = read_unchanged(raw_mask_path(cfg, a)) > 0, read_unchanged(raw_mask_path(cfg, b)) > 0
    one = (ga ^ gb).astype(np.float64)                      # 512 x 512 native window with the most disagreement
    ii = np.pad(one.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    W = 512
    best, (cy, cx) = -1, (0, 0)
    for y in range(0, 2048 - W + 1, 16):
        for x in range(0, 2048 - W + 1, 16):
            v = ii[y + W, x + W] - ii[y, x + W] - ii[y + W, x] + ii[y, x]
            if v > best:
                best, (cy, cx) = v, (y, x)
    dis = np.full(ga.shape + (3,), 255, np.uint8)
    dis[ga & gb], dis[ga & ~gb], dis[gb & ~ga] = ERR["tp"], ERR["fp"], ERR["fn"]
    axes[0, 0].imshow(small(_redacted(cfg, read_rgb(raw_image_path(cfg, a)))))
    sc = 600 / 2048
    axes[0, 0].add_patch(_Rect((cx * sc, cy * sc), W * sc, W * sc, fill=False, ec="#ffffff", lw=0.9))
    axes[1, 0].imshow(dis[cy:cy + W, cx:cx + W], interpolation="nearest")
    axes[0, 0].set_title("A  Same image file,\n     two masks", loc="left", fontsize=7)
    axes[1, 0].set_xlabel(f"native crop; mask Dice {dup['mask_dice']:.3f}", fontsize=5.8)
    e = ig["empty_masks"][0]["image"]
    rgb_e, fov_e = read_rgb(raw_image_path(cfg, e)), load_fov(cfg, e)
    axes[0, 1].imshow(small(_redacted(cfg, rgb_e)))
    # inset (referee minor 8): the whole eye, green channel contrast-stretched inside the FOV (1st-99.5th percentile);
    # everything outside the FOV is set to black, so no marking can appear
    gsm = cv2.resize(rgb_e[..., 1].astype(np.float32), (300, 300), interpolation=cv2.INTER_AREA)
    fsm = cv2.resize(fov_e.astype(np.uint8), (300, 300), interpolation=cv2.INTER_NEAREST) > 0
    lo_, hi_ = np.percentile(gsm[fsm], [1, 99.5])
    st_ = (np.clip((gsm - lo_) / max(hi_ - lo_, 1.0), 0, 1) * 255).astype(np.uint8)
    st_[~fsm] = 0
    ins = axes[0, 1].inset_axes([0.53, 0.01, 0.46, 0.46])
    ins.imshow(st_, cmap="gray", vmin=0, vmax=255)
    clean(ins)
    for sp in ins.spines.values():
        sp.set_color("#ffffff")
        sp.set_linewidth(0.6)
    axes[1, 1].imshow(small(np.where(read_unchanged(raw_mask_path(cfg, e))[..., None] > 0, 0, 255).repeat(3, 2).astype(np.uint8)))
    axes[0, 1].set_title("B  Empty ground\n     truth", loc="left", fontsize=7)
    axes[1, 1].set_xlabel("mask: no vessel pixels", fontsize=5.8)
    tr, te = "train_174_D", "test_54_D"
    gtr, gte = read_unchanged(raw_mask_path(cfg, tr)) > 0, read_unchanged(raw_mask_path(cfg, te)) > 0
    ov = np.full(gtr.shape + (3,), 255, np.uint8)
    ov[gtr & gte], ov[gtr & ~gte], ov[gte & ~gtr] = ERR["tp"], ERR["fp"], ERR["fn"]
    pair = np.concatenate([small(_redacted(cfg, read_rgb(raw_image_path(cfg, tr))), 300),
                           small(_redacted(cfg, read_rgb(raw_image_path(cfg, te))), 300)], 1)
    axes[0, 2].imshow(pair)
    axes[1, 2].imshow(small(ov))
    axes[0, 2].set_title("C  Same eye in train\n     and test", loc="left", fontsize=7)
    axes[1, 2].set_xlabel(f"overlaid masks; layout IoU {plain(N.text['SameEyeIoU'])}", fontsize=5.8)
    bl = ig["gt_outside_fov"]["images_by_kind"]["border_line"][0]
    gbl = read_unchanged(raw_mask_path(cfg, bl)) > 0
    fbl = load_fov(cfg, bl)
    pad = 90
    mvis = np.full((2048 + 2 * pad, 2048 + 2 * pad, 3), 255, np.uint8)
    core = mvis[pad:-pad, pad:-pad]
    core[gbl & fbl] = ERR["tp"]
    edge = cv2.morphologyEx(fbl.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)) > 0
    core[edge] = (11, 11, 11)
    outside = np.zeros(mvis.shape[:2], np.uint8)
    outside[pad:-pad, pad:-pad] = gbl & ~fbl
    mvis[cv2.dilate(outside, np.ones((29, 29), np.uint8)) > 0] = ERR["fp"]
    axes[0, 3].imshow(small(_redacted(cfg, read_rgb(raw_image_path(cfg, bl)))))
    axes[1, 3].imshow(small(mvis))
    axes[0, 3].set_title("D  Annotation outside\n     the eye", loc="left", fontsize=7)
    axes[1, 3].set_xlabel("orange (thickened): mask\noutside the FOV", fontsize=5.8)
    for ax in axes.ravel():
        clean(ax)
    fig.savefig(FIG / "fig_audit_raster.png", dpi=300)
    fig.savefig(PREV / "fig_audit_raster.png", dpi=150)
    plt.close(fig)

    # ---- qualitative (seed-0 runs; crops at native resolution; windows chosen automatically)
    thr = {r: json.loads((EVAL / r / "seed0" / "val_threshold.json").read_text())["threshold"] for r in RUNS}
    ids = qualitative_ids(cfg)

    def panels_for(iid, runs):
        gt = read_unchanged(raw_mask_path(cfg, iid)) > 0
        fov = load_fov(cfg, iid)
        g = prepare_gt(gt, fov)
        y0, x0 = thin_window(g, s["cutoffs"])
        crop = (slice(y0, y0 + 256), slice(x0, x0 + 256))
        img = _redacted(cfg, read_rgb(raw_image_path(cfg, iid)))
        out = [("image", img, crop), ("ground truth", err(gt, gt, fov)[crop], None)]
        for r in runs:
            prob = read_npz_member(ROOT / "outputs" / r / "seed0" / "test" / "probs.npz", iid).astype(np.float32)
            pred = binarize(upsample_prob(prob, s["native"]), thr[r])
            out.append((SHORT[r], err(pred, gt, fov)[crop], None))
        return out, (y0, x0), img[crop]

    from matplotlib.patches import Patch, Rectangle
    main_ids = [ids[i] for i in (0, 2, 4, 6)]          # first draw per category (draws are seeded, not hand-picked)
    main_runs = ["unet", "multiscale_unet", "sam_lora_tiled", "sam_lora_whole"]
    fig, axes = plt.subplots(len(main_ids), 2 + 1 + len(main_runs), figsize=(FULL_W, 4.45))
    for i, iid in enumerate(main_ids):
        pan, (y0, x0), icrop = panels_for(iid, main_runs)
        axes[i, 0].imshow(small(pan[0][1], 400))
        sc = 400 / s["native"]
        axes[i, 0].add_patch(Rectangle((x0 * sc, y0 * sc), 256 * sc, 256 * sc, fill=False, ec="#ffffff", lw=0.9))
        axes[i, 1].imshow(icrop, interpolation="nearest")
        for j, (title, arr, _) in enumerate(pan[1:]):
            axes[i, 2 + j].imshow(arr, interpolation="nearest")
            if i == 0:
                axes[i, 2 + j].set_title(title, fontsize=6.4)
        if i == 0:
            axes[i, 0].set_title("image", fontsize=6.4)
            axes[i, 1].set_title("crop", fontsize=6.4)
        axes[i, 0].set_ylabel("test {} ({})".format(*iid.split("_")[1:]), fontsize=6)
        for ax in axes[i]:
            clean(ax)
    handles = [Patch(color=np.array(ERR[k]) / 255, label=lab) for k, lab in (("tp", "true positive"), ("fp", "false positive"), ("fn", "false negative"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.3, frameon=False)
    fig.tight_layout(pad=0.2, rect=(0, 0.04, 1, 1), w_pad=0.15, h_pad=0.25)
    fig.savefig(FIG / "fig_qualitative_raster.png", dpi=300)
    fig.savefig(PREV / "fig_qualitative_raster.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(ids), 2 + len(RUNS), figsize=(FULL_W, 7.2))
    for i, iid in enumerate(ids):
        pan, (y0, x0), icrop = panels_for(iid, RUNS)
        axes[i, 0].imshow(icrop, interpolation="nearest")
        for j, (title, arr, _) in enumerate(pan[1:]):
            axes[i, 1 + j].imshow(arr, interpolation="nearest")
            if i == 0:
                axes[i, 1 + j].set_title(title, fontsize=5.6)
        if i == 0:
            axes[i, 0].set_title("crop", fontsize=5.6)
        axes[i, 0].set_ylabel("test {} ({})".format(*iid.split("_")[1:]), fontsize=5.6)
        for ax in axes[i]:
            clean(ax)
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.tight_layout(pad=0.2, rect=(0, 0.03, 1, 1), w_pad=0.1, h_pad=0.15)
    fig.savefig(FIG / "fig_si_qualitative_raster.png", dpi=300)
    fig.savefig(PREV / "fig_si_qualitative_raster.png", dpi=110)
    plt.close(fig)
    return ids, main_ids


# ----------------------------------------------------------------------------------------- captions + review
FIGURES = [
    ("fig_pipeline", "Study design.",
     "Each 2048 × 2048 FIVES image is contrast-enhanced (CLAHE) and passed to one of four model families. All probability "
     "maps are enlarged to 2048 by bilinear interpolation, thresholded at a value chosen on the validation set, and scored "
     "against the original masks inside a field-of-view (FOV) mask. Oracle-prompted SAM uses prompts from each image's own "
     "ground truth and is reported separately. LoRA: low-rank adaptation; BCE: binary cross-entropy; P, R: precision, "
     "recall; NSD: normalized surface Dice."),
    ("fig_audit_raster", "Data audit examples.",
     "(A) Two byte-identical training images with different masks: 512 × 512 native-resolution window with the most "
     "disagreement (white box), chosen automatically (grey: vessel in both masks; orange, blue: vessel in one mask only). "
     "(B) One of <<NEmptyMasks>> training masks with no vessel pixels; inset: the same image with its green channel "
     "contrast-stretched inside the eye (outside set to black), in which the optic disc is visible but the vessels "
     "barely are. (C) A training and a test image of the same eye, "
     "with their masks overlaid (colours as in A). (D) A mask with an annotation line along the image border, outside "
     "the eye (orange, thickened for visibility), and the fitted FOV boundary (black). In every image panel the area "
     "outside the FOV is pixelated, so burned-in text markings are not legible."),
    ("fig_threshold", "Thin-vessel recall against the threshold (test set).",
     "Centreline recall of thin vessels (radius < 3 px at 2048) for thresholds 0.05--0.95. Lines: mean over <<NSeeds>> "
     "training seeds of each seed's mean over the <<NTestImages>> test images; bands: range across seeds. (A) The four "
     "CNNs. (B) Fine-tuned SAM with U-Net as reference; the overlapping-tile curve (dashed, hollow markers) uses the same "
     "three models as the tiled curve."),
    ("fig_ablation", "2 × 2 ablation of attention gates and multi-scale blocks (test set).",
     "(A--C) Main effects and interaction for six metrics, with 95% bootstrap confidence intervals over the "
     "<<NTestImages>> test images (each image averaged over seeds), at the validation-selected thresholds (filled circles) "
     "and at a fixed threshold of 0.5 (open squares); clDice and NSD exist only at the validation-selected thresholds. "
     "(D, E) Thin-vessel recall of each training seed (markers) and the seed mean (bar) under each threshold rule. The "
     "intervals in A--C resample images and do not include training-seed variability, which D and E show."),
    ("fig_sam", "Whole-image and tiled SAM (test set).",
     "(A) Centreline recall by vessel width for each training seed (markers; bar: seed mean). Dotted and dash-dot lines: "
     "the test ground truth pushed through the 512 and 256 output grids and back to 2048 (mask-resampling references; "
     "context, not upper bounds). (B) Dice and NSD per seed. Hollow markers: overlapping-tile inference with the tiled "
     "models. U-Net is shown for reference."),
    ("fig_tradeoff", "Dice does not track thin-vessel recall.",
     "(A, B) Test Dice against test thin-vessel recall, one marker per training seed. (C, D) The checkpoint-selection "
     "criterion (best validation Dice at threshold 0.5) against validation thin-vessel recall at the chosen threshold, "
     "one marker per training run; overlapping-tile inference has no checkpoint of its own and is omitted from C and D. "
     "Axes are shared within each row. Dice spans a much narrower range than thin-vessel recall."),
    ("fig_qualitative_raster", "Qualitative examples (test set, seed-0 models).",
     "One test image per disease category, drawn at random with a fixed seed. Left: original image (not contrast-enhanced) "
     "with the analysed window (white box; area outside the FOV pixelated); crop: the 256 × 256 native-resolution window of the original image "
     "with the most thin-vessel ground-truth centreline, chosen automatically. Error maps (grey true positive, orange false "
     "positive, blue false negative) at each model's validation-selected threshold for the U-Net, the Multi-Scale U-Net "
     "(lowest CNN thin-vessel recall) and the two SAM configurations; all seven configurations are shown in S3 Fig."),
]
SI_FIGURES = [
    ("fig_cost", "Inference cost and thin-vessel recall.",
     "Inference time per image on one NVIDIA L40S (seed-0 models, batch of one, log scale) against test thin-vessel "
     "recall (marker: seed mean; vertical bar: range across <<NSeeds>> seeds). Circles: CNNs; triangles: SAM (hollow: "
     "overlapping tiles)."),
    ("fig_threshold_val", "Thin-vessel recall against the threshold (validation set).", "As in the test-set figure, on the <<NVal>> validation images."),
    ("fig_precision_thin", "Thin-vessel recall against overall precision (test set).",
     "Post hoc. Each curve joins the operating points of one configuration at thresholds 0.05–0.95 (step 0.01; mean over "
     "<<NSeeds>> seeds of the means over <<NTestImages>> images); marker: threshold 0.5. Curves further up at "
     "the same precision recover more thin-vessel centreline. (A) The four CNNs. (B) The SAM configurations and the U-Net."),
    ("fig_disease", "Results by disease category (test set).", "Dice (top) and thin-vessel recall (bottom) per category; marker: seed mean; bar: standard deviation across <<NSeeds>> seeds (other figures show the range across seeds)."),
    ("fig_training", "Validation Dice during training.", "Validation Dice (2048, threshold 0.5, inside the FOV) per epoch for every training run (colour: model; line style: seed), from the saved training logs; dashed vertical line: the 150-epoch limit. (A) The four CNNs. (B) Tiled and whole-image SAM; <<NSAMTiledAtCap>> tiled and <<NSAMWholeAtCap>> whole-image runs reached the limit."),
    ("fig_bland_altman", "Image-derived vessel measures.", "Bland--Altman plots of vessel density and fractal dimension, predicted minus ground truth against their mean (per-image values averaged over seeds); solid line: mean difference; dashed: 95% limits of agreement."),
    ("fig_si_qualitative_raster", "All qualitative test images.", "The eight test images (two per category, drawn at random with a fixed seed), native-resolution crops with the most thin-vessel centreline, error maps of every model (seed 0) at its validation-selected threshold. Colours as in the qualitative figure."),
]


def resolve(text: str, N: Numbers, tex: bool) -> str:
    """Caption template -> LaTeX (macros, LaTeX-safe symbols) or plain text (values)."""
    import re
    parts = re.split(r"(<<\w+>>)", text)
    out = []
    for p in parts:
        m = re.fullmatch(r"<<(\w+)>>", p)
        if m:
            out.append(f"\\{m.group(1)}{{}}" if tex else plain(N.text[m.group(1)]))
        else:
            out.append(texsafe(p) if tex else p.replace("--", "–"))
    return "".join(out)


def write_captions(N, tables):
    lines = ["% captions.tex -- generated by paper/scripts/make_assets.py (Step 2 drafts). Do not edit by hand.", ""]
    for key, title, cap in FIGURES + SI_FIGURES:
        nm = "Cap" + "".join(w.title() for w in key.replace("fig_", "").replace("_raster", "").split("_"))
        body = resolve(cap, N, True)
        lines.append(f"\\newcommand{{\\{nm}}}{{{{\\bf {texsafe(title)}}} {body}}}")
    (PAPER / "captions.tex").write_text("\n".join(lines) + "\n")


def write_review(N, tables, have_images):
    out = ["# Figures and tables for review (Step 2)", "",
           "Generated by `paper/scripts/make_assets.py`; every number comes from `results/` (see `numbers.tex` and "
           "`check_numbers.py`). Captions are drafts. Previews are PNG; the paper uses the PDF (charts) or 300-dpi PNG "
           "(image figures).", ""]
    out.append("## Main figures\n")
    for key, title, cap in FIGURES:
        exists = (PREV / f"{key}.png").exists()
        out += [f"### {key}", f"**{title}** {resolve(cap, N, False)}", "",
                f"![{key}](figures/preview/{key}.png)" if exists else "*(image figure: generated by the `--images` job)*", ""]
    out.append("## Main tables\n")
    for tb in [x for x in tables if not x.si]:
        out += [f"### Table: {tb.key}", f"**{tb.title}** {resolve(tb.caption, N, False).replace(chr(92) + 'textsuperscript{-4}', '⁻⁴')}", "", tb.md(), ""]
    out.append("## Supporting figures\n")
    for key, title, cap in SI_FIGURES:
        exists = (PREV / f"{key}.png").exists()
        out += [f"### {key}", f"**{title}** {resolve(cap, N, False)}", "",
                f"![{key}](figures/preview/{key}.png)" if exists else "*(image figure: generated by the `--images` job)*", ""]
    out.append("## Supporting tables\n")
    for tb in [x for x in tables if x.si]:
        out += [f"### Table: {tb.key}", f"**{tb.title}** {resolve(tb.caption, N, False)}", "", tb.md(), ""]
    txt = "\n".join(out)
    for macro in sorted(N.text, key=len, reverse=True):          # captions in tables may hold \Macro{} calls
        txt = txt.replace(f"\\{macro}{{}}", plain(N.text[macro]))
    (PAPER / "assets_review.md").write_text(txt.replace("$\\pm$", "±").replace("\\%", "%"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true", help="also build image figures (reads fundus images; run as a job)")
    a = ap.parse_args()
    N = build_numbers()
    N.write(PAPER / "numbers.tex")
    tables = build_tables(N)
    TAB.mkdir(parents=True, exist_ok=True)
    for tb in tables:
        (TAB / f"{tb.key}.tex").write_text(tb.tex(resolve(tb.caption, N, True)))
    plt = set_style()
    save(fig_pipeline(plt), "fig_pipeline")
    save(fig_threshold(plt, "test"), "fig_threshold")
    save(fig_threshold(plt, "val"), "fig_threshold_val")
    save(fig_precision_thin(plt), "fig_precision_thin")
    save(fig_ablation(plt), "fig_ablation")
    save(fig_sam(plt), "fig_sam")
    save(fig_tradeoff(plt), "fig_tradeoff")
    save(fig_cost(plt), "fig_cost")
    save(fig_disease(plt), "fig_disease")
    save(fig_training(plt), "fig_training")
    save(fig_bland_altman(plt, plain(N.text["NTestImages"])), "fig_bland_altman")
    if a.images:
        image_figures(plt, N)
    write_captions(N, tables)
    write_review(N, tables, a.images)
    print(f"[assets] {len(N.items)} numbers, {len(tables)} tables, figures in {FIG}")


if __name__ == "__main__":
    main()
