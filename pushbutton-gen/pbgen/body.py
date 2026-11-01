"""Load a pushbutton body STEP and find where the button mounts.

This assumes that the STEP file is aligned with the z-axis so it seems like this would only work for the pushbuttons so far unfortunately lol

The width reference is set in this file by picking the highest flat plane with the normal pointing up.

Anything auto-detected can be overridden per body in config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq


@dataclass
class Body:
    name: str
    step_path: Path
    solids: list
    mount_z: float          # mm, Z of the face the button sits on
    center: tuple           # mm, (x, y) of the press axis
    top_diameter: float     # mm, width of the mount face (size reference)
    solid_names: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.name}: {len(self.solids)} solid(s), mount face Z={self.mount_z:.3f} mm, "
                f"axis=({self.center[0]:.3f}, {self.center[1]:.3f}), "
                f"mount face width={self.top_diameter:.2f} mm")


def detect_mount(solids):
    best = None
    for s in solids:
        for f in s.Faces():
            if f.geomType() != "PLANE":
                continue
            if f.normalAt().z < 0.999:
                continue
            bb = f.BoundingBox()
            z = bb.zmax
            if best is None or z > best[0] + 1e-6:
                best = (z, [f])
            elif abs(z - best[0]) <= 1e-6:
                best[1].append(f)
    if best is None:
        raise RuntimeError("no flat upward-facing face found; set 'mount' in config")
    z, faces = best
    bb = cq.Compound.makeCompound(faces).BoundingBox()
    return z, ((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2), max(bb.xlen, bb.ylen)


def load_body(name: str, step_path: str | Path, mount: dict | None = None) -> Body:
    step_path = Path(step_path)
    shape = cq.importers.importStep(str(step_path)).val()
    solids = shape.Solids()
    if not solids:
        raise RuntimeError(f"{step_path} has no solids")
    z, c, d = detect_mount(solids)
    mount = mount or {}
    z = mount.get("z", z)
    c = tuple(mount.get("center", c))
    d = mount.get("top_diameter", d)
    names = [f"{name}_part{i + 1}" for i in range(len(solids))]
    return Body(name, step_path, solids, float(z), (float(c[0]), float(c[1])), float(d), names)


def interferes(button_placed: cq.Solid, body: Body, tol=1e-3) -> bool:
    for s in body.solids:
        try:
            if button_placed.intersect(s).Volume() > tol:
                return True
        except Exception:
            return True
    return False
