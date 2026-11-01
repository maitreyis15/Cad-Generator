"""Parametric pushbutton cap geometry.

Every button is built in its own local frame:
  * origin at the centre of the bottom face (the face that sits on the bezel)
  * +Z is the press axis (up, away from the body)
  * units are millimetres

A button = footprint (rect / round / tri) + top style (flat, dome, flattop,
dimple, dish, pillow). Round buttons wider than the bezel get a neck so they
become mushroom heads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cadquery as cq
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

FAMILIES = ("rect", "round", "tri")
TOP_STYLES = ("flat", "dome", "flattop", "dimple", "dish", "pillow")


@dataclass
class Footprint:
    """A 2D outline that can be extruded to any height."""
    family: str
    params: dict

    # radius of the smallest circle (about the origin) that contains the outline
    @property
    def r_out(self) -> float:
        p = self.params
        if self.family == "round":
            return p["diameter"] / 2
        if self.family == "rect":
            cr = p["corner_r"]
            return math.hypot(p["width"] / 2 - cr, p["length"] / 2 - cr) + cr
        if self.family == "tri":
            return p["side"] / math.sqrt(3) - p["corner_r"]
        raise ValueError(self.family)

    # radius of the largest circle (about the origin) inside the outline
    @property
    def r_in(self) -> float:
        p = self.params
        if self.family == "round":
            return p["diameter"] / 2
        if self.family == "rect":
            return min(p["width"], p["length"]) / 2
        if self.family == "tri":
            return p["side"] / (2 * math.sqrt(3))
        raise ValueError(self.family)

    def extrude(self, h: float) -> cq.Solid:
        p = self.params
        wp = cq.Workplane("XY")
        if self.family == "round":
            s = wp.circle(p["diameter"] / 2).extrude(h)
        elif self.family == "rect":
            s = wp.rect(p["width"], p["length"]).extrude(h)
            if p["corner_r"] > 0:
                s = s.edges("|Z").fillet(p["corner_r"])
        elif self.family == "tri":
            R = p["side"] / math.sqrt(3)  # circumradius, centroid at origin
            pts = [(R * math.cos(math.radians(90 + 120 * k)),
                    R * math.sin(math.radians(90 + 120 * k))) for k in range(3)]
            s = wp.polyline(pts).close().extrude(h)
            if p["corner_r"] > 0:
                s = s.edges("|Z").fillet(p["corner_r"])
        else:
            raise ValueError(self.family)
        solid = s.val()
        yaw = p.get("yaw_deg", 0.0)
        if yaw:
            solid = solid.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), yaw)
        return solid


def _sphere(z_center: float, radius: float) -> cq.Solid:
    return cq.Solid.makeSphere(radius, cq.Vector(0, 0, z_center),
                               angleDegrees1=-90, angleDegrees2=90)


def _one_solid(shape: cq.Shape) -> cq.Solid:
    shape = shape.clean()
    solids = shape.Solids()
    if len(solids) != 1:
        raise RuntimeError(f"expected one solid, got {len(solids)}")
    return solids[0]


def _dome_sphere(base: float, h: float, a: float):
    """Sphere whose surface passes through radius a at z=base and apex z=base+h."""
    R = (a * a + h * h) / (2 * h)
    return base + h - R, R


def _build_top(fp: Footprint, top: dict) -> cq.Solid:
    style, base = top["style"], top["base"]
    a_out, a_in = fp.r_out, fp.r_in
    if style == "flat":
        return fp.extrude(base)
    if style == "pillow":
        s = cq.Workplane().add(fp.extrude(base))
        return s.faces(">Z").edges().fillet(top["edge_r"]).val()
    if style == "dish":
        ri = a_in - top["rim"]
        d = top["depth"]
        Rd = (ri * ri + d * d) / (2 * d)
        return _one_solid(fp.extrude(base).cut(_sphere(base - d + Rd, Rd)))
    if style in ("dome", "flattop", "dimple"):
        h = top["h"]
        zc, R = _dome_sphere(base, h, a_out)
        cap_h = top["h_cut"] if style == "flattop" else h
        body = fp.extrude(base).fuse(fp.extrude(base + cap_h).intersect(_sphere(zc, R)))
        body = _one_solid(body)
        if style == "dimple":
            rd = top["dimple_d"] / 2
            dd = top["dimple_depth"]
            zd = zc + math.sqrt(R * R - rd * rd)
            Rd = (rd * rd + dd * dd) / (2 * dd)
            body = _one_solid(body.cut(_sphere(zd - dd + Rd, Rd)))
        return body
    raise ValueError(f"unknown top style {style}")


# ---------------------------------------------------------------- fillets ---
def _edge_face_map(shape: cq.Shape):
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape.wrapped, TopAbs_EDGE, TopAbs_FACE, m)
    out = {}
    for i in range(1, m.Extent() + 1):
        faces = [cq.Face(f) for f in m.FindFromIndex(i)]
        out[cq.Edge(m.FindKey(i))] = faces
    return out


def _sharp_edges(shape: cq.Solid, zmin: float, zmax: float):
    """Non-tangent edges between two distinct faces, inside a Z band."""
    picked = []
    for e, faces in _edge_face_map(shape).items():
        bb = e.BoundingBox()
        if bb.zmin < zmin or bb.zmax > zmax:
            continue
        uniq = []
        for f in faces:
            if not any(f.isSame(u) for u in uniq):
                uniq.append(f)
        if len(uniq) != 2:  # seam edge
            continue
        if e.geomType() == "LINE" and abs(e.tangentAt(0.5).z) > 0.99:
            continue  # vertical corner seams
        p = e.positionAt(0.5)
        n0, n1 = uniq[0].normalAt(p), uniq[1].normalAt(p)
        if abs(n0.dot(n1)) > 0.995:  # already tangent
            continue
        picked.append(e)
    return picked


def _try_fillet(shape: cq.Solid, edges, radii):
    if not edges:
        return shape
    for r in radii:
        try:
            out = shape.fillet(r, edges)
            if out.isValid():
                return out
        except Exception:
            pass
    return shape


def soften(shape: cq.Solid, bottom_r=0.4, crease_r=0.8) -> cq.Solid:
    """Round the bottom edge and the creases on top, as a moulded part would be."""
    shape = _try_fillet(shape, _sharp_edges(shape, -1e-3, 1e-3),
                        [bottom_r, bottom_r / 2])
    bb = shape.BoundingBox()
    shape = _try_fillet(shape, _sharp_edges(shape, 0.05, bb.zmax + 1),
                        [crease_r, crease_r / 2, 0.25])
    return shape


# ------------------------------------------------------------------ build ---
def build_button(family: str, footprint: dict, top: dict, neck: dict | None = None) -> cq.Solid:
    """Build one button cap.

    family    : "rect" | "round" | "tri"
    footprint : rect  -> width, length, corner_r, yaw_deg
                round -> diameter
                tri   -> side, corner_r, yaw_deg
    top       : style + base (straight-side height) + style-specific values
                dome    -> h
                flattop -> h, h_cut
                dimple  -> h, dimple_d, dimple_depth
                dish    -> depth, rim
                pillow  -> edge_r
    neck      : optional {"diameter", "height"} -> mushroom head on a stem
    """
    fp = Footprint(family, footprint)
    head = _build_top(fp, top)
    if neck:
        head = head.translate(cq.Vector(0, 0, neck["height"]))
        stem = cq.Workplane().circle(neck["diameter"] / 2).extrude(neck["height"] + 0.01).val()
        head = _one_solid(head.fuse(stem))
    head = soften(head)
    if not head.isValid():
        raise RuntimeError("invalid solid")
    return head
