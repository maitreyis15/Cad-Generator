"""Look at a generated button (or try out parameters) in VS Code.

Needs the "OCP CAD Viewer" extension and `pip install ocp_vscode`.
Open the viewer (Ctrl+K V, or the OCP icon in the side bar), then run this file.

  python view.py                      # shows the example button below
  python view.py out/rockwell/dome_3  # shows a generated button on its body
"""
import json
import sys
from pathlib import Path

import cadquery as cq
from ocp_vscode import show

from pbgen.body import load_body
from pbgen.shapes import build_button

if len(sys.argv) > 1:
    d = Path(sys.argv[1])
    meta = json.loads((d / f"{d.name}_params.json").read_text())
    btn = cq.importers.importStep(str(d / f"{d.name}_button.step")).val()
    asm = cq.importers.importStep(str(d / f"{d.name}_assembly.step"))
    show(asm, names=[d.name])
else:
    # edit these numbers and re-run to try a shape out
    btn = build_button(
        "round",
        {"diameter": 24},
        {"style": "dimple", "base": 2.5, "h": 6, "dimple_d": 10, "dimple_depth": 1.2},
    )
    body = load_body("rockwell", "bodies/Rockwell_Body.STEP")
    placed = btn.translate(cq.Vector(*body.center, body.mount_z))
    show(*body.solids, placed, names=body.solid_names + ["button"],
         colors=["gray"] * len(body.solids) + ["red"])
