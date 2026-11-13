"""Turn config ranges into concrete, buildable parameter sets.

Ranges in the config are [min, max] and sampled uniformly. Values ending in
_rel are multiplied by the body's mount-face width, so the same config gives
sensibly-sized buttons on a different body.
"""
from __future__ import annotations

import math

import numpy as np

from .shapes import Footprint


def _u(rng, rng_range):
    if isinstance(rng_range, (int, float)):
        return float(rng_range)
    lo, hi = rng_range
    return float(rng.uniform(lo, hi))


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def family_config(cfg: dict, family: str) -> dict:
    return _merge(cfg.get("defaults", {}), cfg["families"][family])


def sample_footprint(rng, family: str, fc: dict, ref: float) -> dict:
    f = fc["footprint"]
    if family == "round":
        return {"diameter": round(_u(rng, f["diameter_rel"]) * ref, 2)}
    if family == "rect":
        w = _u(rng, f["width_rel"]) * ref
        l = w * _u(rng, f.get("aspect", [1, 1]))
        if rng.random() < 0.5:
            w, l = l, w
        cr = min(_u(rng, f.get("corner_r_mm", [0.5, 3])), min(w, l) / 2 - 0.5)
        return {"width": round(w, 2), "length": round(l, 2), "corner_r": round(max(cr, 0), 2),
                "yaw_deg": round(_u(rng, f.get("yaw_deg", 0)), 1)}
    if family == "tri":
        s = _u(rng, f["side_rel"]) * ref
        cr = min(_u(rng, f.get("corner_r_mm", [2, 5])), s / (2 * math.sqrt(3)) - 0.5)
        return {"side": round(s, 2), "corner_r": round(cr, 2),
                "yaw_deg": round(_u(rng, f.get("yaw_deg", 0)), 1)}
    if family == "oval":
        w = _u(rng, f["width_rel"]) * ref
        l = w * _u(rng, f.get("aspect", [1.6, 2.5]))
        return {"width": round(w, 2), "length": round(l, 2),
                "yaw_deg": round(_u(rng, f.get("yaw_deg", 0)), 1)}
    raise ValueError(family)


def sample_top(rng, fc: dict, fp: Footprint) -> dict:
    tops = fc["tops"]
    names = [k for k, v in tops.items() if v.get("weight", 1) > 0]
    w = np.array([tops[k].get("weight", 1) for k in names], float)
    style = names[rng.choice(len(names), p=w / w.sum())]
    t = tops[style]
    base = _u(rng, t.get("base_mm", fc["base_mm"]))
    a_out, a_in = fp.r_out, fp.r_in
    top = {"style": style}
    if style in ("dome", "flattop", "dimple"):
        h = min(_u(rng, t["h_rel"]) * a_out, a_out)  # never more than a hemisphere
        top["h"] = round(h, 2)
        if style == "flattop":
            top["h_cut"] = round(h * _u(rng, t["cut_frac"]), 2)
        if style == "dimple":
            top["dimple_d"] = round(_u(rng, t["dimple_d_rel"]) * 2 * a_in, 2)
            top["dimple_depth"] = round(min(_u(rng, t["depth_mm"]), 0.6 * h), 2)
    elif style == "dish":
        d = _u(rng, t["depth_mm"])
        top["depth"] = round(d, 2)
        top["rim"] = round(min(_u(rng, t["rim_mm"]), 0.3 * a_in), 2)
        base = max(base, d + _u(rng, t.get("floor_mm", [1.5, 3.0])))
    elif style == "pillow":
        r = _u(rng, t["edge_r_mm"])
        base = max(base, r + 1.0)
        top["edge_r"] = round(min(r, 0.45 * a_in), 2)
    top["base"] = round(base, 2)
    return top


def sample_neck(rng, family: str, fc: dict, footprint: dict, ref: float):
    if family != "round":
        return None
    m = fc.get("mushroom", {})
    if not m or footprint["diameter"] <= m.get("above_rel", 1.0) * ref:
        return None
    nd = min(_u(rng, m["neck_d_rel"]) * ref, footprint["diameter"] - 4)
    return {"diameter": round(nd, 2), "height": round(_u(rng, m["neck_h_mm"]), 2)}


def sample_color(rng, palette: dict):
    name = list(palette)[rng.integers(len(palette))]
    return name, tuple(palette[name])
