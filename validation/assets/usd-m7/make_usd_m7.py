"""M7 USD test assets (docs/decisions/m7-api.md §3; PLAN §7.2 E2E-USD, (viii-L)): hand-authored with OpenUSD (pxr).

  /Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 validation/assets/usd-m7/make_usd_m7.py

Writes (next to this script, deterministic):
  m7_textured.usda + textures/*.png  UsdUVTexture coverage: diffuse (sRGB, repeat / mirror), roughness + metallic from ONE
                                     texture (G / B) and from separate textures (synthesised pair), normal maps (scale 2,
                                     bias −1), an emissive texture, a cut-out leaf card (opacity = diffuse alpha,
                                     opacityThreshold 0.5); a smooth UV sphere; RectLight; camera. Y-up, metres.
  m7_instancing.usda                 instanceable references to a class prototype (4 instances, non-uniform scale and
                                     rotation), a PointInstancer under a rotated / translated parent (2 prototypes, 16
                                     instances: positions, quath orientations, scales, protoIndices), DistantLight,
                                     SphereLight treatAsPoint; camera. Z-up, metres.
  m7_e2e_yup.usda / m7_e2e_zup.usda  the PLAN §7.2 E2E-USD hand file: upAxis Y with metersPerUnit 0.01 (centimetres) and
                                     upAxis Z with metersPerUnit 1; RectLight normalize 1, DiskLight normalize 0, SphereLight
                                     treatAsPoint, SphereLight radius 0.5, SphereLight + ShapingAPI (spot), DistantLight; an
                                     asymmetric texture for the UV flip; a smooth sphere; camera.
"""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

HERE = Path(__file__).resolve().parent


# ---- PNG (RGBA8, deterministic) ---------------------------------------------------------------------------------------

def write_png(path: Path, w: int, h: int, f) -> None:
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes(int(max(0, min(255, round(c)))) for c in f(x, y))
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    path.write_bytes(data)


def enc(n):
    l = math.sqrt(sum(c * c for c in n))
    return [(c / l + 1) / 2 * 255 for c in n] + [255]


def textures() -> None:
    t = HERE / "textures"
    t.mkdir(exist_ok=True)
    write_png(t / "checker.png", 64, 64, lambda x, y: ([200, 190, 170, 255] if ((x // 8) + (y // 8)) % 2 else [60, 80, 120, 255]))
    write_png(t / "orm.png", 64, 64, lambda x, y: [255, 40 + 3 * x, 255 if (y // 16) % 2 else 0, 255])          # G rough, B metal
    write_png(t / "rough.png", 64, 64, lambda x, y: [60 + 2 * y, 0, 0, 255])                                    # R channel used
    write_png(t / "plaster.png", 64, 64, lambda x, y: [180 + 40 * math.sin(x / 5.0), 170, 150 + 30 * math.cos(y / 7.0), 255])
    def tiles(x, y):
        u, v = (x + 0.5) / 64 * 4 % 1, (y + 0.5) / 64 * 4 % 1
        nx = -0.55 if u < 0.12 else 0.55 if u > 0.88 else 0
        ny = 0.55 if v < 0.12 else -0.55 if v > 0.88 else 0
        return enc([nx, ny, 1])
    write_png(t / "tiles_n.png", 64, 64, tiles)
    def bumps(x, y):
        u, v = (x + 0.5) / 64 * 6 % 1 - 0.5, (y + 0.5) / 64 * 6 % 1 - 0.5
        d2 = u * u + v * v
        if d2 >= 0.16:
            return enc([0, 0, 1])
        h = math.sqrt(0.16 - d2)
        return enc([0.8 * u / h, -0.8 * v / h, 1])
    write_png(t / "bumps_n.png", 64, 64, bumps)
    write_png(t / "emit.png", 32, 32, lambda x, y: [255, 180, 60, 255] if (x // 4 + y // 4) % 2 else [40, 120, 255, 255])
    def leaf(x, y):
        u, v = (x + 0.5) / 64 - 0.5, (y + 0.5) / 64 - 0.5
        inside = (u / 0.42) ** 2 + (v / 0.3) ** 2 <= 1 or (abs(u) < 0.02 and v > 0)
        return [50 + x, 140 + y // 2, 40, 255 if inside else 0]
    write_png(t / "leaf.png", 64, 64, leaf)
    def letter_f(x, y):   # asymmetric in u and v: the UV flip test
        on = (8 <= x <= 14 and 8 <= y <= 56) or (8 <= x <= 44 and 8 <= y <= 14) or (8 <= x <= 32 and 28 <= y <= 34)
        return [230, 40, 30, 255] if on else [235, 235, 225, 255]
    write_png(t / "letter_f.png", 64, 64, letter_f)


# ---- USD helpers ------------------------------------------------------------------------------------------------------

def new_stage(name: str, up: str, mpu: float, doc: str) -> Usd.Stage:
    path = HERE / name
    if path.exists():
        path.unlink()
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y if up == "Y" else UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, mpu)
    st.GetRootLayer().documentation = doc
    world = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(world.GetPrim())
    return st


def mesh(st, path, points, counts, indices, st_uv=None, normals=None, binding=None, xform=None):
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if normals is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if st_uv is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(*u) for u in st_uv]))
    if xform:
        xf = UsdGeom.Xformable(m)
        for op, v in xform:
            if op == "t":
                xf.AddTranslateOp().Set(Gf.Vec3d(*v))
            elif op == "r":
                xf.AddRotateXYZOp().Set(Gf.Vec3f(*v))
            elif op == "s":
                xf.AddScaleOp().Set(Gf.Vec3f(*v))
    if binding is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(binding)
    return m


def quad(st, path, c, ex, ey, uv_scale=(1, 1), binding=None, mirror_u=False):
    """Quad centre c, half-edge vectors ex (u) and ey (v); normal = ex × ey; st (0..su, 0..sv)."""
    pts = [[c[k] - ex[k] - ey[k] for k in range(3)], [c[k] + ex[k] - ey[k] for k in range(3)],
           [c[k] + ex[k] + ey[k] for k in range(3)], [c[k] - ex[k] + ey[k] for k in range(3)]]
    su, sv = uv_scale
    uv = [(0, 0), (su, 0), (su, sv), (0, sv)]
    if mirror_u:
        uv = [(su - u, v) for u, v in uv]
    n = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]]
    l = math.sqrt(sum(x * x for x in n))
    n = [x / l for x in n]
    return mesh(st, path, pts, [4], [0, 1, 2, 3], uv, [n] * 4, binding)


def box(st, path, c, h, binding=None, xform=None):
    """Axis-aligned box (outward faces, per-face vertices), half sizes h."""
    P, N, U, I = [], [], [], []
    faces = [((1, 0, 0), (0, 0, -1), (0, 1, 0)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
             ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0))]
    for n, a, b in faces:
        base = len(P)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            P.append([c[k] + h[k] * (n[k] + su * a[k] + sv * b[k]) for k in range(3)])
            N.append(list(n))
            U.append(((su + 1) / 2, (sv + 1) / 2))
        I += [base, base + 1, base + 2, base + 3]
    return mesh(st, path, P, [4] * 6, I, U, N, binding, xform)


def uv_sphere(st, path, c, r, nlat, nlon, binding=None, uv_scale=(1, 1)):
    P, N, U, I, C = [], [], [], [], []
    th0, th1 = 0.06, math.pi - 0.06
    for i in range(nlat + 1):
        th = th0 + (th1 - th0) * i / nlat
        for j in range(nlon + 1):
            ph = 2 * math.pi * j / nlon
            n = (math.sin(th) * math.cos(ph), math.cos(th), -math.sin(th) * math.sin(ph))
            P.append([c[k] + r * n[k] for k in range(3)])
            N.append(n)
            U.append((uv_scale[0] * j / nlon, uv_scale[1] * (1 - i / nlat)))
    at = lambda i, j: i * (nlon + 1) + j
    for i in range(nlat):
        for j in range(nlon):
            I += [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]
            C.append(4)
    return mesh(st, path, P, C, I, U, N, binding)


def preview(st, path, *, diffuse=(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0, emissive=None, opacity=None, threshold=None, tex=None):
    """UsdPreviewSurface material; tex: {input: (file, output channel, extra dict)} → UsdUVTexture + st reader."""
    mat = UsdShade.Material.Define(st, path)
    ps = UsdShade.Shader.Define(st, f"{path}/PS")
    ps.CreateIdAttr("UsdPreviewSurface")
    ps.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    ps.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if emissive is not None:
        ps.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    if opacity is not None:
        ps.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if threshold is not None:
        ps.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(threshold)
    mat.CreateSurfaceOutput().ConnectToSource(ps.ConnectableAPI(), "surface")
    if tex:
        rd = UsdShade.Shader.Define(st, f"{path}/StReader")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        rd_out = rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        types = {"diffuseColor": Sdf.ValueTypeNames.Color3f, "emissiveColor": Sdf.ValueTypeNames.Color3f, "normal": Sdf.ValueTypeNames.Normal3f,
                 "roughness": Sdf.ValueTypeNames.Float, "metallic": Sdf.ValueTypeNames.Float, "opacity": Sdf.ValueTypeNames.Float}
        made = {}
        for inp, (file, ch, extra) in tex.items():
            key = (file, tuple(sorted(extra.items())))
            if key not in made:
                t = UsdShade.Shader.Define(st, f"{path}/Tex{len(made)}")
                t.CreateIdAttr("UsdUVTexture")
                t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"./textures/{file}"))
                t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd_out)
                for k, v in extra.items():
                    if k in ("scale", "bias"):
                        t.CreateInput(k, Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*v))
                    else:
                        t.CreateInput(k, Sdf.ValueTypeNames.Token).Set(v)
                made[key] = t
            t = made[key]
            out = t.CreateOutput(ch, Sdf.ValueTypeNames.Float3 if ch == "rgb" else Sdf.ValueTypeNames.Float)
            ps.CreateInput(inp, types[inp]).ConnectToSource(out)
    return mat


def camera(st, path, eye, target, up_axis, vfov_deg=40.0, units=1.0):
    cam = UsdGeom.Camera.Define(st, path)
    va = 24.0
    cam.CreateVerticalApertureAttr(va)
    cam.CreateHorizontalApertureAttr(va)
    cam.CreateFocalLengthAttr(va / (2 * math.tan(math.radians(vfov_deg) / 2)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01 * units, 1000 * units))
    up = Gf.Vec3d(0, 1, 0) if up_axis == "Y" else Gf.Vec3d(0, 0, 1)
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse()
    UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    return cam


def place(prim, m: Gf.Matrix4d):
    UsdGeom.Xformable(prim).AddTransformOp().Set(m)


def aim(pos, direction, up_axis) -> Gf.Matrix4d:
    """Light frame at pos whose −Z points along direction."""
    d = Gf.Vec3d(*direction).GetNormalized()
    up = Gf.Vec3d(0, 1, 0) if up_axis == "Y" else Gf.Vec3d(0, 0, 1)
    if abs(Gf.Dot(d, up)) > 0.95:
        up = Gf.Vec3d(1, 0, 0)
    eye = Gf.Vec3d(*pos)
    return Gf.Matrix4d().SetLookAt(eye, eye + d, up).GetInverse()


# ---- assets -----------------------------------------------------------------------------------------------------------

def make_textured() -> None:
    st = new_stage("m7_textured.usda", "Y", 1.0, "M7 UsdUVTexture coverage (make_usd_m7.py)")
    L = "/World/Looks"
    floor = preview(st, f"{L}/Floor", tex={
        "diffuseColor": ("checker.png", "rgb", {"sourceColorSpace": "sRGB", "wrapS": "repeat", "wrapT": "repeat"}),
        "roughness": ("orm.png", "g", {"sourceColorSpace": "raw", "wrapS": "repeat", "wrapT": "repeat"}),
        "metallic": ("orm.png", "b", {"sourceColorSpace": "raw", "wrapS": "repeat", "wrapT": "repeat"}),
        "normal": ("tiles_n.png", "rgb", {"sourceColorSpace": "raw", "wrapS": "repeat", "wrapT": "repeat", "scale": (2, 2, 2, 1), "bias": (-1, -1, -1, 0)}),
    })
    wall = preview(st, f"{L}/Wall", metallic=0.0, tex={
        "diffuseColor": ("plaster.png", "rgb", {"sourceColorSpace": "sRGB", "wrapS": "mirror", "wrapT": "mirror"}),
        "roughness": ("rough.png", "r", {"sourceColorSpace": "raw", "wrapS": "mirror", "wrapT": "mirror"}),
        "normal": ("bumps_n.png", "rgb", {"sourceColorSpace": "raw", "wrapS": "mirror", "wrapT": "mirror", "scale": (2, 2, 2, 1), "bias": (-1, -1, -1, 0)}),
    })
    ball = preview(st, f"{L}/Ball", roughness=0.35, tex={
        "diffuseColor": ("checker.png", "rgb", {"sourceColorSpace": "sRGB", "wrapS": "repeat", "wrapT": "repeat"}),
        "normal": ("bumps_n.png", "rgb", {"sourceColorSpace": "raw", "wrapS": "repeat", "wrapT": "repeat", "scale": (2, 2, 2, 1), "bias": (-1, -1, -1, 0)}),
    })
    leaf = preview(st, f"{L}/Leaf", roughness=0.6, threshold=0.5, tex={
        "diffuseColor": ("leaf.png", "rgb", {"sourceColorSpace": "sRGB", "wrapS": "clamp", "wrapT": "clamp"}),
        "opacity": ("leaf.png", "a", {"sourceColorSpace": "sRGB", "wrapS": "clamp", "wrapT": "clamp"}),
    })
    glow = preview(st, f"{L}/Glow", diffuse=(0.05, 0.05, 0.05), tex={
        "emissiveColor": ("emit.png", "rgb", {"sourceColorSpace": "sRGB", "wrapS": "repeat", "wrapT": "repeat"}),
    })
    quad(st, "/World/Floor", (0, 0, 0), (2, 0, 0), (0, 0, -2), (4, 4), floor)
    quad(st, "/World/Wall", (0, 1.25, -2), (2, 0, 0), (0, 1.25, 0), (3, 2), wall)
    uv_sphere(st, "/World/Ball", (-0.6, 0.55, -0.6), 0.55, 24, 48, ball, (3, 2))
    quad(st, "/World/LeafCard", (0.9, 0.7, -0.7), (0.5, 0, 0.15), (0, 0.6, 0), (1, 1), leaf)
    quad(st, "/World/GlowPanel", (1.4, 1.6, -1.98), (0.35, 0, 0), (0, 0.25, 0), (2, 1), glow)
    rect = UsdLux.RectLight.Define(st, "/World/Key")
    rect.CreateWidthAttr(1.2); rect.CreateHeightAttr(0.8); rect.CreateIntensityAttr(12.0); rect.CreateNormalizeAttr(False)
    place(rect.GetPrim(), aim((0.4, 2.9, 0.6), (0, -1, -0.2), "Y"))
    camera(st, "/World/Camera", (0.3, 1.6, 3.6), (0, 0.7, -0.8), "Y", 45)
    st.GetRootLayer().Save()


def make_instancing() -> None:
    st = new_stage("m7_instancing.usda", "Z", 1.0, "M7 instancing coverage (make_usd_m7.py)")
    L = "/World/Looks"
    crate_m = preview(st, f"{L}/Crate", diffuse=(0.6, 0.45, 0.25), roughness=0.55)
    gem_m = preview(st, f"{L}/Gem", diffuse=(0.15, 0.4, 0.8), roughness=0.25, metallic=0.3)
    pyr_m = preview(st, f"{L}/Pyramid", diffuse=(0.8, 0.2, 0.15), roughness=0.4)
    floor_m = preview(st, f"{L}/Floor", diffuse=(0.7, 0.7, 0.68), roughness=0.6)
    # class prototype (abstract: never drawn directly) and instanceable references to it
    proto = st.CreateClassPrim("/Proto")
    box(st, "/Proto/Crate", (0, 0, 0.25), (0.25, 0.25, 0.25), crate_m)
    crates = UsdGeom.Xform.Define(st, "/World/Crates")
    specs = [((1.2, 0.4, 0.0), 15, (1, 1, 1)), ((1.9, -0.5, 0.0), -30, (1.4, 0.7, 1.0)), ((0.9, -1.2, 0.0), 60, (0.8, 0.8, 1.6)),
             ((2.1, 0.9, 0.5), 5, (0.6, 0.6, 0.6))]
    for i, (t, rz, s) in enumerate(specs):
        p = st.DefinePrim(f"/World/Crates/C{i}", "Xform")
        p.GetReferences().AddInternalReference("/Proto")
        p.SetInstanceable(True)
        xf = UsdGeom.Xformable(p)
        xf.AddTranslateOp().Set(Gf.Vec3d(*t)); xf.AddRotateZOp().Set(rz); xf.AddScaleOp().Set(Gf.Vec3f(*s))
    # PointInstancer under a rotated, translated parent
    parent = UsdGeom.Xform.Define(st, "/World/Scatter")
    parent.AddTranslateOp().Set(Gf.Vec3d(-1.2, 0.3, 0.0)); parent.AddRotateZOp().Set(20.0)
    pi = UsdGeom.PointInstancer.Define(st, "/World/Scatter/PI")
    mesh(st, "/World/Scatter/PI/Protos/Pyr", [(-0.15, -0.15, 0), (0.15, -0.15, 0), (0.15, 0.15, 0), (-0.15, 0.15, 0), (0, 0, 0.35)],
         [4, 3, 3, 3, 3], [3, 2, 1, 0, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], binding=pyr_m)
    mesh(st, "/World/Scatter/PI/Protos/Gem", [(0, 0, 0), (0.12, 0, 0.15), (0, 0.12, 0.15), (-0.12, 0, 0.15), (0, -0.12, 0.15), (0, 0, 0.3)],
         [3] * 8, [0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4, 5, 1, 2, 5, 2, 3, 5, 3, 4, 5, 4, 1], binding=gem_m)
    pi.CreatePrototypesRel().SetTargets([Sdf.Path("/World/Scatter/PI/Protos/Pyr"), Sdf.Path("/World/Scatter/PI/Protos/Gem")])
    pos, ori, scl, idx = [], [], [], []
    for k in range(16):
        a = 2 * math.pi * k / 16
        pos.append(Gf.Vec3f(0.8 * math.cos(a), 0.8 * math.sin(a), 0.0))
        q = Gf.Rotation(Gf.Vec3d(0.3, 0.2, 1).GetNormalized(), 25 * k).GetQuat()
        ori.append(Gf.Quath(float(q.GetReal()), Gf.Vec3h(*[float(x) for x in q.GetImaginary()])))
        scl.append(Gf.Vec3f(1 + 0.05 * k, 1.0, 1.2 - 0.03 * k))
        idx.append(k % 2)
    pi.CreatePositionsAttr(Vt.Vec3fArray(pos)); pi.CreateOrientationsAttr(Vt.QuathArray(ori))
    pi.CreateScalesAttr(Vt.Vec3fArray(scl)); pi.CreateProtoIndicesAttr(Vt.IntArray(idx))
    quad(st, "/World/Floor", (0.4, 0, 0), (2.6, 0, 0), (0, 2.2, 0), (1, 1), floor_m)
    sun = UsdLux.DistantLight.Define(st, "/World/Sun")
    sun.CreateIntensityAttr(2.5); sun.CreateAngleAttr(0.53)
    place(sun.GetPrim(), aim((0, 0, 5), (0.3, 0.4, -1), "Z"))
    bulb = UsdLux.SphereLight.Define(st, "/World/Bulb")
    bulb.CreateIntensityAttr(40.0); bulb.CreateTreatAsPointAttr(True); bulb.CreateRadiusAttr(0.05)
    place(bulb.GetPrim(), Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.3, -0.4, 1.4)))
    camera(st, "/World/Camera", (0.6, -4.2, 2.6), (0.4, 0, 0.2), "Z", 40)
    st.GetRootLayer().Save()


def make_e2e(name: str, up: str, mpu: float) -> None:
    """PLAN §7.2 E2E-USD hand file. Geometry in metres × u (u = 1/mpu), upAxis `up`."""
    u = 1.0 / mpu
    st = new_stage(name, up, mpu, f"M7 E2E-USD hand file, upAxis {up}, metersPerUnit {mpu} (make_usd_m7.py)")
    # canonical (Y-up metres) → stage coordinates
    def P(x, y, z):
        return (x * u, y * u, z * u) if up == "Y" else (x * u, -z * u, y * u)
    def V(x, y, z):
        return (x, y, z) if up == "Y" else (x, -z, y)
    L = "/World/Looks"
    white = preview(st, f"{L}/White", diffuse=(0.75, 0.75, 0.72), roughness=0.6)
    red = preview(st, f"{L}/Red", diffuse=(0.7, 0.12, 0.1), roughness=0.4)
    sign = preview(st, f"{L}/Sign", roughness=0.5, tex={"diffuseColor": ("letter_f.png", "rgb", {"sourceColorSpace": "sRGB", "wrapS": "clamp", "wrapT": "clamp"})})
    blue = preview(st, f"{L}/Blue", diffuse=(0.15, 0.3, 0.7), roughness=0.3, metallic=0.5)
    def Q(path, c, ex, ey, mat, uv=(1, 1)):
        quad(st, path, P(*c), [x * u for x in V(*ex)], [x * u for x in V(*ey)], uv, mat)
    Q("/World/Floor", (0, 0, 0), (1.5, 0, 0), (0, 0, -1.5), white)
    Q("/World/Back", (0, 1.0, -1.5), (1.5, 0, 0), (0, 1.0, 0), white)
    Q("/World/Left", (-1.5, 1.0, 0), (0, 0, -1.5), (0, 1.0, 0), red)
    Q("/World/Sign", (0.5, 1.0, -1.49), (0.4, 0, 0), (0, 0.4, 0), sign)
    box(st, "/World/Box", P(-0.6, 0.3, -0.6), [0.3 * u if up == "Y" else 0.3 * u] * 3, white)
    s = uv_sphere(st, "/World/Ball", P(0.55, 0.35, 0.1), 0.35 * u, 20, 40, blue)
    if up == "Z":   # uv_sphere builds around +Y: rotate the generated points into the Z-up frame
        pts = s.GetPointsAttr().Get()
        c = Gf.Vec3f(*P(0.55, 0.35, 0.1))
        s.GetPointsAttr().Set(Vt.Vec3fArray([c + Gf.Vec3f((p - c)[0], -(p - c)[2], (p - c)[1]) for p in pts]))
        ns = s.GetNormalsAttr().Get()
        s.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(n[0], -n[2], n[1]) for n in ns]))
    def at(prim, pos, d=None):
        place(prim, aim(P(*pos), V(*d), up) if d else Gf.Matrix4d().SetTranslate(Gf.Vec3d(*P(*pos))))
    rect = UsdLux.RectLight.Define(st, "/World/Rect")       # normalize 1
    rect.CreateWidthAttr(0.6 * u); rect.CreateHeightAttr(0.4 * u); rect.CreateIntensityAttr(6.0); rect.CreateNormalizeAttr(True)
    at(rect.GetPrim(), (0, 1.95, -0.2), (0, -1, 0))
    disk = UsdLux.DiskLight.Define(st, "/World/Disk")       # normalize 0
    disk.CreateRadiusAttr(0.12 * u); disk.CreateIntensityAttr(15.0); disk.CreateNormalizeAttr(False)
    at(disk.GetPrim(), (1.3, 1.2, 0.4), (-1, -0.4, -0.5))
    bulb = UsdLux.SphereLight.Define(st, "/World/Bulb")     # treatAsPoint
    bulb.CreateIntensityAttr(8.0); bulb.CreateTreatAsPointAttr(True)
    at(bulb.GetPrim(), (-0.9, 1.6, 0.6))
    ball = UsdLux.SphereLight.Define(st, "/World/BallLight")  # radius 0.5 (Blender / v1 rule: point, r := 0)
    ball.CreateIntensityAttr(0.4); ball.CreateRadiusAttr(0.5 * u)
    at(ball.GetPrim(), (0.2, 2.6, 1.8))
    spot = UsdLux.SphereLight.Define(st, "/World/Spot")     # ShapingAPI
    spot.CreateIntensityAttr(60.0); spot.CreateRadiusAttr(0.02 * u)
    sh = UsdLux.ShapingAPI.Apply(spot.GetPrim())
    sh.CreateShapingConeAngleAttr(25.0); sh.CreateShapingConeSoftnessAttr(0.3)
    at(spot.GetPrim(), (0.9, 1.7, 1.0), (-0.4, -0.8, -1.0))
    sun = UsdLux.DistantLight.Define(st, "/World/Sun")
    sun.CreateIntensityAttr(1.5); sun.CreateAngleAttr(0.53)
    at(sun.GetPrim(), (0, 5, 0), (0.4, -1, -0.5))
    camera(st, "/World/Camera", P(0.2, 1.2, 3.4), P(0, 0.8, -0.5), up, 45, u)
    st.GetRootLayer().Save()


if __name__ == "__main__":
    textures()
    make_textured()
    make_instancing()
    make_e2e("m7_e2e_yup.usda", "Y", 0.01)
    make_e2e("m7_e2e_zup.usda", "Z", 1.0)
    print("wrote", sorted(p.name for p in HERE.glob("*.usda")), sorted(p.name for p in (HERE / "textures").glob("*.png")))
