"""Write Isaac Sim-ready USDA files.

Layout (metres, Z up, as Isaac Sim expects):

  <out>/<body>/_body/<body>.usda            body meshes + static collision (written once)
  <out>/<body>/<name>/<name>.usda
      /<name>                       (defaultPrim)
          body        -> references ../_body/<body>.usda, kinematic rigid body
          button      rigid body, sits on the mount face
              geom    mesh + collider
          press_joint prismatic joint along Z, limit [-travel, 0], spring drive back to 0
          Looks       display + physics materials

The body is kinematic, so you can place / move the whole asset in a scene and
it stays put; the button is free to slide down the joint when pressed.
"""
from __future__ import annotations

import json
import os

import cadquery as cq
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

MM = 0.001  # mm -> m


def _mesh_vertex_normals(P, T):
    """Area-weighted vertex normals from triangle winding (fallback)."""
    N = np.zeros_like(P)
    tn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3):
        np.add.at(N, T[:, k], tn)
    L = np.linalg.norm(N, axis=1, keepdims=True)
    L[L == 0] = 1.0
    return N / L


def _face_normals(f, vs, P, T):
    """Exact surface normals, with a mesh-based fallback at singular points
    (sphere poles, cone tips, degenerate fillet edges) where OCC can't define one."""
    out = np.empty_like(P)
    fallback = None
    for i, v in enumerate(vs):
        try:
            n = f.normalAt(v)
            out[i] = (n.x, n.y, n.z)
        except Exception:
            if fallback is None:
                fallback = _mesh_vertex_normals(P, T)
            out[i] = fallback[i]
    return out


def tessellate(shape: cq.Shape, tol=0.02, ang=0.15):
    """Per-face tessellation so normals stay sharp at real edges and smooth on curves."""
    pts, idx, nrm, count = [], [], [], 0
    for f in shape.Faces():
        vs, tris = f.tessellate(tol, ang)
        if not tris:
            continue
        P = np.array([[v.x, v.y, v.z] for v in vs])
        T = np.array(tris)
        N = _face_normals(f, vs, P, T)
        # flip shading normals per triangle-winding if they disagree (keeps them outward)
        tn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        vn = N[T].mean(1)
        if (tn * vn).sum() < 0:
            N = -N
        pts.append(P)
        nrm.append(N)
        idx.append(T + count)
        count += len(P)
    P = np.vstack(pts)
    N = np.vstack(nrm)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return P, np.vstack(idx), N

def _new_stage(path: str):
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    return stage


def _mesh(stage, path, shape, offset=(0, 0, 0), tol=0.02):
    P, T, N = tessellate(shape, tol=tol)
    P = (P + np.array(offset)) * MM
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(T)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(T.flatten().tolist()))
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.astype(np.float32)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr([Gf.Vec3f(*P.min(0)), Gf.Vec3f(*P.max(0))])
    return m


def _display_material(stage, path, rgb, roughness=0.45, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path, static_f, dynamic_f, restitution=0.0):
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(static_f)
    pm.CreateDynamicFrictionAttr(dynamic_f)
    pm.CreateRestitutionAttr(restitution)
    return mat


def _bind(prim, display=None, physics=None):
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if display is not None:
        api.Bind(display)
    if physics is not None:
        api.Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")


def write_body_usd(path: str, body, phys: dict | None, color=(0.55, 0.55, 0.57)):
    stage = _new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/body")
    stage.SetDefaultPrim(root.GetPrim())
    look = _display_material(stage, "/body/Looks/body_mat", color, roughness=0.6)
    pmat = _physics_material(stage, "/body/Looks/body_physics",
                             phys["friction_static"], phys["friction_dynamic"]) if phys else None
    for s, n in zip(body.solids, body.solid_names):
        m = _mesh(stage, f"/body/{n}", s)
        _bind(m.GetPrim(), look, pmat)
        if phys:
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            # exact triangle mesh is allowed because the body is kinematic/static
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(
                phys.get("body_collision", "none"))
    stage.GetRootLayer().Save()


def write_variant_usd(path: str, name: str, body, body_usd_path: str, button: cq.Solid,
                      meta: dict, color_rgb, phys: dict | None):
    stage = _new_stage(path)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root.GetPrim()).SetKind("component")

    # body (shared file, referenced)
    bprim = UsdGeom.Xform.Define(stage, f"/{name}/body").GetPrim()
    rel = os.path.relpath(body_usd_path, os.path.dirname(path)).replace(os.sep, "/")
    bprim.GetReferences().AddReference("./" + rel if not rel.startswith(".") else rel)

    # button
    cx, cy = body.center
    mount = (cx * MM, cy * MM, body.mount_z * MM)
    btn = UsdGeom.Xform.Define(stage, f"/{name}/button")
    btn.AddTranslateOp().Set(Gf.Vec3d(*mount))
    geom = _mesh(stage, f"/{name}/button/geom", button)
    look = _display_material(stage, f"/{name}/Looks/button_mat", color_rgb)

    concave = meta["top"] in ("dish", "dimple") or meta.get("neck_d")
    if phys:
        pmat = _physics_material(stage, f"/{name}/Looks/button_physics",
                                 phys["friction_static"], phys["friction_dynamic"])
        _bind(geom.GetPrim(), look, pmat)

        rb = UsdPhysics.RigidBodyAPI.Apply(bprim)
        rb.CreateKinematicEnabledAttr(False)

        UsdPhysics.RigidBodyAPI.Apply(btn.GetPrim())
        UsdPhysics.MassAPI.Apply(btn.GetPrim()).CreateDensityAttr(phys["density_kg_m3"])
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr(
            "convexDecomposition" if concave else "convexHull")

        travel = phys["travel_mm"] * MM
        j = UsdPhysics.PrismaticJoint.Define(stage, f"/{name}/press_joint")
        j.CreateAxisAttr("Z")
        j.CreateBody0Rel().SetTargets([bprim.GetPath()])
        j.CreateBody1Rel().SetTargets([btn.GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(*mount))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(-travel)
        j.CreateUpperLimitAttr(0.0)
        j.CreateCollisionEnabledAttr(False)  # button may sink into the bezel
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateTargetPositionAttr(0.0)
        d.CreateStiffnessAttr(phys["spring_stiffness_n_per_m"])
        d.CreateDampingAttr(phys["damping_ns_per_m"])
        if phys.get("max_force_n"):
            d.CreateMaxForceAttr(phys["max_force_n"])

        p = root.GetPrim()
        p.CreateAttribute("pb:travel_m", Sdf.ValueTypeNames.Float).Set(travel)
        p.CreateAttribute("pb:press_threshold_m", Sdf.ValueTypeNames.Float).Set(
            travel * phys["press_fraction"])
        p.CreateAttribute("pb:joint_path", Sdf.ValueTypeNames.String).Set(str(j.GetPath()))
    else:
        _bind(geom.GetPrim(), look)

    p = root.GetPrim()
    p.CreateAttribute("pb:family", Sdf.ValueTypeNames.String).Set(meta["family"])
    p.CreateAttribute("pb:top", Sdf.ValueTypeNames.String).Set(meta["top"])
    p.CreateAttribute("pb:body", Sdf.ValueTypeNames.String).Set(body.name)
    p.CreateAttribute("pb:params_json", Sdf.ValueTypeNames.String).Set(json.dumps(meta))
    stage.GetRootLayer().Save()
