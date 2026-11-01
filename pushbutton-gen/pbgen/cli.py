"""Command line entry point:  python -m pbgen --config config.yaml"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import yaml

from . import sampling as S
from .body import interferes, load_body
from .shapes import FAMILIES, Footprint, build_button


def _flatten(meta: dict) -> dict:
    row = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                row[f"{k}_{kk}"] = vv
        else:
            row[k] = v
    return row


def make_one(rng, family, fc, body, max_tries=25):
    ref = body.top_diameter
    for attempt in range(max_tries):
        fpp = S.sample_footprint(rng, family, fc, ref)
        fp = Footprint(family, fpp)
        top = S.sample_top(rng, fc, fp)
        neck = S.sample_neck(rng, family, fc, fpp, ref)
        try:
            btn = build_button(family, fpp, top, neck)
        except Exception:
            continue
        bb = btn.BoundingBox()
        if bb.zmax > fc.get("max_height_mm", 1e9):
            continue
        placed = btn.translate(cq.Vector(body.center[0], body.center[1], body.mount_z))
        if interferes(placed, body):
            continue
        return btn, fpp, top, neck
    raise RuntimeError(f"could not build a valid {family} button in {max_tries} tries; "
                       f"check the ranges in config.yaml")


def save_assembly(path, name, body, btn):
    a = cq.Assembly(name=f"{name}_assembly")
    for s, n in zip(body.solids, body.solid_names):
        a.add(s, name=n, color=cq.Color(0.55, 0.55, 0.57))
    a.add(btn, name=f"{name}_button",
          loc=cq.Location((body.center[0], body.center[1], body.mount_z)),
          color=cq.Color(0.8, 0.1, 0.1))
    a.save(str(path))


def run(cfg: dict, cfg_dir: Path, only_body=None, only_family=None, count=None,
        preview=False):
    out_root = (cfg_dir / cfg.get("output_dir", "out")).resolve()
    outs = cfg.get("outputs", {})
    phys = cfg.get("physics") if cfg.get("physics", {}).get("enabled", True) else None
    palette = cfg.get("colors", {"red": [0.8, 0.1, 0.1]})
    seed = cfg.get("seed", 0)

    for bcfg in cfg["bodies"]:
        if only_body and bcfg["name"] != only_body:
            continue
        body = load_body(bcfg["name"], cfg_dir / bcfg["step"], bcfg.get("mount"))
        print(body.summary())
        bdir = out_root / body.name
        bdir.mkdir(parents=True, exist_ok=True)
        body_usd = bdir / "_body" / f"{body.name}.usda"
        if outs.get("usd", True):
            from .usd_export import write_body_usd
            body_usd.parent.mkdir(exist_ok=True)
            write_body_usd(str(body_usd), body, phys)

        rows = []
        fams = [f for f in cfg["families"] if f in FAMILIES]
        for family in fams:
            if only_family and family != only_family:
                continue
            fc = S.family_config(cfg, family)
            n = count if count is not None else fc.get("count", 10)
            if n <= 0:
                continue
            prefix = fc.get("prefix", family)
            start = fc.get("start_index", 1)
            # each (seed, body, family) gets its own stream -> reproducible runs
            rng = np.random.default_rng([seed, sum(map(ord, body.name)),
                                         FAMILIES.index(family), start])
            for i in range(start, start + n):
                name = f"{prefix}_{i}"
                t0 = time.time()
                btn, fpp, top, neck = make_one(rng, family, fc, body)
                cname, rgb = S.sample_color(rng, palette)
                vdir = bdir / name
                vdir.mkdir(exist_ok=True)
                bb = btn.BoundingBox()
                meta = {"name": name, "body": body.name, "family": family, "top": top["style"],
                        "footprint": fpp, "top_params": {k: v for k, v in top.items() if k != "style"},
                        "neck_d": neck["diameter"] if neck else "",
                        "neck_h": neck["height"] if neck else "",
                        "span_x_mm": round(bb.xlen, 2), "span_y_mm": round(bb.ylen, 2),
                        "height_mm": round(bb.zmax, 2), "volume_mm3": round(btn.Volume(), 1),
                        "color": cname}
                if phys:
                    meta["mass_g"] = round(btn.Volume() * 1e-9 * phys["density_kg_m3"] * 1000, 2)
                if outs.get("step_button", True):
                    cq.exporters.export(btn, str(vdir / f"{name}_button.step"))
                if outs.get("step_assembly", True):
                    save_assembly(vdir / f"{name}_assembly.step", name, body, btn)
                if outs.get("usd", True):
                    from .usd_export import write_variant_usd
                    write_variant_usd(str(vdir / f"{name}.usda"), name, body, str(body_usd),
                                      btn, meta, rgb, phys)
                (vdir / f"{name}_params.json").write_text(json.dumps(meta, indent=2))
                rows.append(_flatten(meta))
                desc = f"{top['style']}" + (" +neck" if neck else "")
                print(f"  {name:<12} {desc:<14} {bb.xlen:5.1f} x {bb.ylen:5.1f} x {bb.zmax:4.1f} mm"
                      f"  ({time.time() - t0:.1f}s)")

        if rows:
            keys = []
            for r in rows:
                keys += [k for k in r if k not in keys]
            csv_path = bdir / "parameters.csv"
            old = []
            if csv_path.exists():  # keep rows from earlier runs of other families
                with open(csv_path, newline="") as f:
                    old = [r for r in csv.DictReader(f) if r["name"] not in {x["name"] for x in rows}]
                for r in old:
                    keys += [k for k in r if k not in keys]
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, keys)
                w.writeheader()
                w.writerows(sorted(old + rows, key=lambda r: (r["family"], int(r["name"].rsplit("_", 1)[1]))))
            print(f"  -> {bdir}  ({len(rows)} buttons, parameters.csv updated)")
            if preview:
                from .preview import contact_sheet
                p = contact_sheet(bdir, [r["name"] for r in rows])
                print(f"  -> preview: {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate pushbutton cap variations for Isaac Sim.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--body", help="only this body (name from config)")
    ap.add_argument("--family", choices=FAMILIES, help="only this family")
    ap.add_argument("--count", type=int, help="override count for every family")
    ap.add_argument("--preview", action="store_true", help="also render a PNG contact sheet")
    a = ap.parse_args(argv)
    cfg_path = Path(a.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    t0 = time.time()
    run(cfg, cfg_path.parent, a.body, a.family, a.count, a.preview)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
