"""PNG contact sheet of generated buttons (shaded 3D view + cross-section)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cadquery as cq
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])


def contact_sheet(body_dir: Path, names, cols=5, out_name=None):
    body_dir = Path(body_dir)
    rows = math.ceil(len(names) / cols)
    fig = plt.figure(figsize=(4 * cols, 3.6 * rows))
    for i, n in enumerate(names):
        meta = json.loads((body_dir / n / f"{n}_params.json").read_text())
        shp = cq.importers.importStep(str(body_dir / n / f"{n}_button.step")).val()
        v, t = shp.tessellate(0.05, 0.2)
        V = np.array([[p.x, p.y, p.z] for p in v])
        T = np.array(t)
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        sh = np.clip(fn @ LIGHT, 0, 1) * 0.7 + 0.3
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(V[T], facecolors=np.c_[0.85 * sh, 0.2 * sh, 0.2 * sh],
                                             edgecolor="none"))
        r = max(np.abs(V[:, :2]).max(), 12)
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-2, r)
        ax.set_box_aspect((1, 1, (r + 2) / (2 * r)))
        ax.view_init(28, -60); ax.axis("off")
        # section through the axis
        sec = shp.intersect(cq.Solid.makeBox(4 * r, 0.02, 4 * r, cq.Vector(-2 * r, -0.01, -r)))
        ins = ax.inset_axes([0.6, 0.0, 0.4, 0.3])
        for e in sec.Edges():
            pts = np.array([[p.x, p.z] for p in (e.positionAt(s) for s in np.linspace(0, 1, 20))])
            ins.plot(pts[:, 0], pts[:, 1], "k", lw=0.7)
        ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([])
        ax.set_title(f"{n}  {meta['top']}  {meta['span_x_mm']}x{meta['span_y_mm']}x{meta['height_mm']}",
                     fontsize=9)
    plt.tight_layout()
    out = body_dir / (out_name or f"preview_{names[0]}-{names[-1]}.png")
    plt.savefig(out, dpi=70)
    plt.close(fig)
    return out
