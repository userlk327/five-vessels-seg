"""Independently re-derive every number in paper/numbers.tex from its stated source file.

    python paper/scripts/check_numbers.py

Written separately from make_assets.py (standard library only: csv, json, statistics; no pandas and no
shared code), so that a lookup or aggregation error in make_assets.py shows up as a mismatch. For each
macro it reads the provenance line, recomputes the raw value, formats it with its own formatter and
compares the text exactly. Exit code 1 on any mismatch.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_rows: dict[str, list[dict]] = {}


def rows(rel):
    if rel not in _rows:
        with open(ROOT / rel, newline="") as f:
            _rows[rel] = list(csv.DictReader(f))
    return _rows[rel]


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def match(row, where):
    for k, v in where.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            if str(row[k]) != str(v):
                return False
        else:
            x = num(row[k])
            if x is None or abs(x - float(v)) > 1e-9 * max(1.0, abs(float(v))):
                return False
    return True


def jget(rel, path):
    x = json.loads((ROOT / rel).read_text())
    for k in path:
        x = x[k]
    return x


def compute(p, raw):
    op = p["op"]
    if op in ("cell", "mean", "agg"):
        sel = [r for r in rows(p["file"]) if match(r, p.get("where", {}))]
        if op == "cell":
            assert len(sel) == 1, f"{len(sel)} rows"
            return float(sel[0][p["col"]])
        vals = [v for v in (num(r[p["col"]]) for r in sel) if v is not None]
        if op == "mean":
            return sum(vals) / len(vals)
        fn = p["fn"]
        if fn == "count":
            return float(len(sel))
        if fn == "min":
            return min(vals)
        if fn == "max":
            return max(vals)
        if fn == "sum":
            return sum(vals)
        raise ValueError(fn)
    if op == "json":
        return float(jget(p["file"], p["path"]))
    if op == "json_list":
        lst = [e for e in jget(p["file"], p["path"]) if all(e.get(k) == v for k, v in p.get("match", {}).items())]
        if p["fn"] == "count":
            return float(len(lst))
        vals = [float(e[p["field"]]) for e in lst]
        if p["fn"] == "one":
            assert len(vals) == 1
            return vals[0]
        return min(vals) if p["fn"] == "min" else max(vals)
    if op == "curve":
        tab = {}
        for r in rows(p["file"]):
            if r["split"] == p["split"]:
                tab.setdefault(float(r["threshold"]), {})[r["run"]] = (float(r["cl_recall_thin"]), float(r["seed_min"]), float(r["seed_max"]))
        subj, refs, kind = p["subject"], p["refs"], p["kind"]
        if kind == "overlap_count":
            return float(sum(1 for t in tab.values() if t[subj][1] <= t[refs[0]][2] and t[refs[0]][1] <= t[subj][2]))
        if kind == "above_count":
            return float(sum(1 for t in tab.values() if t[subj][0] - max(t[r][0] for r in refs) > 0))
        if kind == "gap_best":
            vals = [t[subj][0] - max(t[r][0] for r in refs) for t in tab.values()]
        else:
            vals = [t[subj][0] - t[refs[0]][0] for t in tab.values()]
        return min(vals) if p["fn"] == "min" else max(vals)
    if op == "expr":
        env = {k: raw[v] for k, v in p["args"].items()}
        return float(eval(p["expr"], {"min": min, "max": max, "abs": abs, "__builtins__": {}}, env))
    raise ValueError(op)


def fmt(x, kind):
    if kind == "int":
        s = format(int(round(x)), ",")
    elif kind.startswith("sf"):
        d = int(kind[2:])
        body = "%.*f" % (d, abs(x))
        s = body if float(body) == 0 else ("+" if x > 0 else "-") + body
    elif kind.startswith("pct"):
        s = "%.*f" % (int(kind[3:]), 100 * x)
    elif kind.startswith("f"):
        s = "%.*f" % (int(kind[1:]), x)
    elif kind == "sci":
        if x >= 0.001:
            s = "%.3f" % x
        else:
            mant, ex = ("%.1e" % x).split("e")
            return "$%s\\times10^{%d}$" % (mant, int(ex))
    else:
        raise ValueError(kind)
    return "\\textminus{}" + s[1:] if s.startswith("-") else s


def main():
    text = (ROOT / "paper" / "numbers.tex").read_text().splitlines()
    raw, bad, n = {}, [], 0
    for i, line in enumerate(text):
        if not line.startswith("% PROV "):
            continue
        p = json.loads(line[len("% PROV "):])
        m = re.fullmatch(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text[i + 1])
        name, got = m.group(1), m.group(2)
        assert name == p["name"], (name, p["name"])
        try:
            val = compute(p, raw)
        except Exception as e:                       # noqa: BLE001
            bad.append((name, f"error: {e}"))
            continue
        raw[name] = val
        want = fmt(val, p["fmt"])
        n += 1
        if want != got:
            bad.append((name, f"numbers.tex has {got!r}, source gives {want!r}"))
    print(f"checked {n} macros against their sources: {len(bad)} mismatches")
    for b in bad[:40]:
        print("  MISMATCH", *b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
