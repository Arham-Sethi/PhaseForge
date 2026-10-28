# qfmc_nn_storyboard.py
# QFM-CNN Explanatory Film (≈55 seconds), fully 3D with HUD-safe overlays.
#
# Manim Community 0.19.x, OpenGL renderer recommended
#   manim -pqh --renderer=opengl qfmc_nn_storyboard.py QFM_Film
#
# Structure follows the requested storyboard to the letter:
#   0–4s   Title Card
#   4–10s  Input → Network
#   10–18s Branching
#   18–26s Kernel Morphology
#   26–34s Quantum Interference Fusion
#   34–44s Output Panel
#   44–55s Outro
#
# All HUD text is fixed to the screen plane (no tilt).
# Camera motion is gentle and continuous. All colors, sizes, and timings match the prompt.
# Robust to OpenGL quirks: no transforms across incompatible mobject types,
# guarded .play calls, safe unfixing of HUD mobjects, per-node reveals, etc.

from __future__ import annotations
from manim import *
import numpy as np
import os
import random


# --------------------- Global Styling ---------------------

BG           = "#0a0d14"
TEXT         = "#e6edf3"
TEXT_MUTED   = "#9ca3af"
ACCENT_A     = "#22d3ee"   # cyan
ACCENT_B     = "#6ee7b7"   # teal
ACCENT_C     = "#a78bfa"   # violet/purple
SPATIAL_C    = "#7dd3fc"   # cyan-ish for spatial branch
FREQ_C       = "#fca5a5"   # red-ish for frequency branch
MORPH_C      = "#34d399"   # green
NET_NODE     = "#93c5fd"   # node spheres
NET_EDGE     = "#334155"   # edge default
PANEL_BG     = "#0f172a"
SAFE_FONT    = "Segoe UI"  # Present on Windows; easy fallback if custom fonts missing

random.seed(7)
np.random.seed(7)

# --------------------- Utilities ---------------------

def SafeTex(expr: str, **kwargs) -> Mobject:
    """
    Try MathTex; if LaTeX not present, fall back to Text with simple unicode replacements.
    """
    try:
        return MathTex(expr, **kwargs)
    except Exception:
        # Replace common LaTeX tokens to keep some fidelity
        txt = (expr
               .replace(r"\psi", "ψ")
               .replace(r"\theta", "θ")
               .replace(r"\omega", "ω")
               .replace(r"|", "∣")
               .replace(r"\cdot", "·")
               .replace(r"\mathcal{F}", "F")
               .replace(r"^{-1}", "^{-1}")
               )
        fs = kwargs.pop("font_size", 40)
        return Text(txt, font_size=fs, color=TEXT, font=SAFE_FONT, **kwargs)

def banner(text: str, sub: str | None = None, width=6.2, height=1.1) -> VGroup:
    r = RoundedRectangle(corner_radius=0.2, width=width, height=height, color=WHITE, stroke_opacity=0.35)
    r.set_fill(BG, opacity=0.45)
    t = Text(text, font=SAFE_FONT, font_size=34, color=WHITE)
    grp = VGroup(r, t)
    if sub:
        s = Text(sub, font=SAFE_FONT, font_size=24, color=TEXT_MUTED).next_to(t, DOWN, buff=0.3)
        grp.add(s)
    return grp

def make_image_plane(size=3.5, img_path="crop.jpg") -> VGroup:
    """
    Left input panel, slight 15° Z tilt per spec.
    If image missing, draws a cyan→teal gradient stripes.
    """
    rect = Rectangle(width=size, height=size * 0.7, stroke_color=WHITE, stroke_opacity=0.25)
    rect.set_fill("#0a1722", opacity=1.0)
    plane = VGroup(rect)

    if os.path.exists(img_path):
        try:
            im = ImageMobject(img_path)
            im.width, im.height = rect.width, rect.height
            im.move_to(rect.get_center())
            im.set_opacity(1.0)
            plane.add(im)
        except Exception:
            pass

    if len(plane) == 1:  # no image added, fallback
        grads = color_gradient([ACCENT_A, ACCENT_B], 20)
        stripes = VGroup()
        for i in range(20):
            r = Rectangle(width=rect.width, height=rect.height / 20, stroke_opacity=0)
            r.set_fill(grads[i], opacity=0.30 + 0.15 * (i % 2))
            r.align_to(rect, LEFT)
            r.move_to(rect.get_center() + UP * (rect.height / 20 * (9.5 - i)))
            stripes.add(r)
        plane.add(stripes)

    # Tilt 15° about Z
    plane.rotate(15 * DEGREES, axis=OUT)
    return plane

def flow_dot_and_path(points: list[np.ndarray], color=YELLOW_E, radius=0.045) -> tuple[Sphere, VMobject]:
    """
    A glowing sphere following a smooth VMobject path through 3D points.
    """
    dot = Sphere(radius=radius, resolution=(16, 16), color=color)
    path = VMobject(stroke_color=color, stroke_opacity=0.0)
    path.set_points_smoothly(points)
    return dot, path

# --- DROP-IN FIX for v0.19 (Cairo) ---
from manim import *
import numpy as np

# ---------------- wireframe helper (no SurfaceMesh, no surf.uv_func) ----------------
def wireframe_from_surface_param(
    param,                     # callable: param(u, v) -> np.array([x,y,z])
    u_range=(-1.8, 1.8),
    v_range=(-1.8, 1.8),
    nu: int = 12,
    nv: int = 12,
    color=WHITE,
    opacity=0.08,
    width=0.5,
) -> VGroup:
    u0, u1 = map(float, u_range)
    v0, v1 = map(float, v_range)

    # constant-u lines (vary v)
    u_vals = np.linspace(u0, u1, max(2, nu))
    v_lines = VGroup()
    for u in u_vals:
        f = ParametricFunction(lambda v, u=u: param(u, v), t_range=(v0, v1))
        f.set_stroke(color, width, opacity=opacity)
        v_lines.add(f)

    # constant-v lines (vary u)
    v_vals = np.linspace(v0, v1, max(2, nv))
    u_lines = VGroup()
    for v in v_vals:
        f = ParametricFunction(lambda u, v=v: param(u, v), t_range=(u0, u1))
        f.set_stroke(color, width, opacity=opacity)
        u_lines.add(f)

    mesh = VGroup(u_lines, v_lines)
    try:
        mesh.always_sort_to_camera = True
    except Exception:
        pass
    return mesh


# ---------------- height surface builder (use Surface in Manim 0.19) ----------------
def surface_from_height(
    fn,                         # callable: fn(x, y) -> z (height)
    x_range=(-1.8, 1.8),
    y_range=(-1.8, 1.8),
    step=0.12,
    color="#a78bfa",
    z_scale=0.6,
) -> VGroup:
    """
    Build a 3D height field as a Surface + light wireframe overlay.
    Works on Manim Community v0.19 (Cairo or OpenGL).
    """
    x0, x1 = x_range
    y0, y1 = y_range
    nu = max(3, int((x1 - x0) / step))
    nv = max(3, int((y1 - y0) / step))

    def param(u, v):
        return np.array([u, v, z_scale * fn(u, v)])

    # NOTE: use Surface (ParametricSurface doesn't exist in 0.19)
    surf = Surface(
        lambda u, v: param(u, v),
        u_range=(x0, x1),
        v_range=(y0, y1),
        resolution=(nu, nv),
        checkerboard_colors=[color, color],  # solid fill
        fill_opacity=0.90,
        stroke_opacity=0.0,                  # hide polygon outlines
    )

    mesh = wireframe_from_surface_param(
        param,
        u_range=(x0, x1),
        v_range=(y0, y1),
        nu=12,
        nv=12,
        color=WHITE,
        opacity=0.08,
        width=0.5,
    )

    try:
        surf.always_sort_to_camera = True
        mesh.always_sort_to_camera = True
    except Exception:
        pass

    return VGroup(surf, mesh)


def gabor_height(x, y, sigma=0.9, freq=0.45, theta=0.4):
    xr =  np.cos(theta)*x + np.sin(theta)*y
    yr = -np.sin(theta)*x + np.cos(theta)*y
    return np.exp(-(xr**2 + yr**2)/(2*sigma**2)) * np.cos(2*np.pi*freq*xr)

def laplacian_height(x, y, sigma=1.0):
    r2 = x**2 + y**2
    return (1 - r2/sigma**2) * np.exp(-r2/(2*sigma**2))

def wave_x(u, v, amp=0.55, f=0.45, phase=0.0):
    return amp * np.sin(2*np.pi*f*u + phase)

def wave_y(u, v, amp=0.55, f=0.45, phase=0.0):
    return amp * np.sin(2*np.pi*f*v + phase)

def tint_edges_split(edges: VGroup, y_thresh=0.15, top=SPATIAL_C, bottom=FREQ_C, mid_opacity=0.25):
    """
    Color edges by midpoint y to convey split; used in branching act.
    """
    for e in edges:
        ym = 0.5 * (e.get_start()[1] + e.get_end()[1])
        if ym > y_thresh:
            e.set_color(top).set_opacity(0.75)
        elif ym < -y_thresh:
            e.set_color(bottom).set_opacity(0.75)
        else:
            e.set_color(NET_EDGE).set_opacity(mid_opacity)

def set_edges_opacity(edges: VGroup, opacity: float):
    for e in edges:
        e.set_opacity(opacity)

def safe_play(scene: Scene, *anims: Animation, run_time: float | None = None, **kwargs):
    """
    Only call scene.play if there's at least one animation.
    Prevents ValueError: Called Scene.play with no animations.
    """
    if not anims:
        return
    # Filter out Nones (in case of optional anims)
    anims = tuple(a for a in anims if a is not None)
    if len(anims) == 0:
        return
    if run_time is not None:
        scene.play(*anims, run_time=run_time, **kwargs)
    else:
        scene.play(*anims, **kwargs)

# --------------------- Network Builder ---------------------

def make_nn(layers=(7, 10, 10, 7), layer_spacing=1.9, node_radius=0.05,
            z_jitter=0.12) -> tuple[VGroup, VGroup]:
    """
    Returns (nodes_group, edges_group) for a 3D feed-forward net.
    - Nodes: spheres with slight z jitter to avoid perfect coplanarity
    - Edges: Line3D segments, initial opacity modest
    """
    nodes_layers = []
    all_nodes = VGroup()
    x0 = - (len(layers) - 1) * layer_spacing / 2.0
    for li, n_nodes in enumerate(layers):
        gl = VGroup()
        for ni in range(n_nodes):
            y = (n_nodes - 1) / 2 - ni
            s = Sphere(radius=node_radius, resolution=(16, 16), color=NET_NODE)
            s.set_fill(color=NET_NODE, opacity=0.0)  # reveal later; avoids FadeIn(Groups) issues
            z = z_jitter * np.sin(ni * 0.45)
            s.move_to([x0 + li * layer_spacing, 0.35 * y, z])
            gl.add(s)
        nodes_layers.append(gl)
        all_nodes.add(gl)

    edges = VGroup()
    for li in range(len(layers) - 1):
        left_layer = nodes_layers[li]
        right_layer = nodes_layers[li + 1]
        for a in left_layer:
            for b in right_layer:
                e = Line3D(a.get_center(), b.get_center(), thickness=0.01, color=NET_EDGE)
                e.set_opacity(0.0)  # reveal edges first in a controlled manner
                edges.add(e)
    return all_nodes, edges

def reveal_edges_then_nodes(scene: Scene, edges: VGroup, nodes: VGroup,
                            edge_rt=1.0, node_rt=0.8):
    """
    Safely reveals edges first (opacity tween), then nodes per-sphere opacity tween.
    """
    # Edges
    safe_play(scene, *[e.animate.set_opacity(0.6) for e in edges], run_time=edge_rt)

    # Nodes
    # Reveal in small batches to avoid huge single animation
    for layer in nodes:
        batch_anims = []
        for s in layer:
            batch_anims.append(s.animate.set_fill(opacity=0.95))
        safe_play(scene, *batch_anims, run_time=max(0.2, node_rt * 0.6))

# --------------------- Layer Manager (World vs HUD) ---------------------

class LayerManager:
    """
    Keeps strict control over what is on screen.
    - world_layers: 3D objects (nodes, edges, surfaces, planes, etc.)
    - hud_layers  : 2D overlay objects fixed to frame
    Provides clean fade/removal between acts without 'play with no animations' errors.
    """
    def __init__(self, scene: ThreeDScene):
        self.scene = scene
        self.world_layers: list[VGroup] = []
        self.hud_layers:   list[VGroup] = []

    def push_world(self, *mobs: Mobject) -> VGroup:
        g = VGroup(*mobs)
        self.scene.add(g)
        self.world_layers.append(g)
        return g

    def push_hud(self, *mobs: Mobject) -> VGroup:
        g = VGroup(*mobs)
        self.scene.add_fixed_in_frame_mobjects(g)
        self.hud_layers.append(g)
        return g

    def _fade_remove_group(self, group: VGroup, run_time=0.6, shift=OUT*0.15):
        sc = self.scene
        # Stop updaters first
        for m in group:
            try:
                m.clear_updaters()
            except Exception:
                pass
        anims = [FadeOut(m, shift=shift) for m in group]
        safe_play(sc, *anims, run_time=run_time)
        try:
            sc.remove(group)
        except Exception:
            pass

    def clear_world(self, run_time=0.6):
        while self.world_layers:
            g = self.world_layers.pop()
            self._fade_remove_group(g, run_time=run_time)

    def clear_hud(self, run_time=0.4):
        sc = self.scene
        while self.hud_layers:
            g = self.hud_layers.pop()
            # Unfix safely
            for m in g:
                m.clear_updaters()
                try:
                    sc.remove_fixed_in_frame_mobjects(m)
                except Exception:
                    pass
            anims = [FadeOut(m, shift=UP*0.1) for m in g]
            safe_play(sc, *anims, run_time=run_time)
            try:
                sc.remove(g)
            except Exception:
                pass

    def clean_slate(self, world_time=0.6, hud_time=0.4):
        self.clear_world(run_time=world_time)
        self.clear_hud(run_time=hud_time)

# =========================================================
#                        SCENES
# =========================================================

# --------------------- 0–4s Title Card ---------------------

class QFM_Title(ThreeDScene):
    """
    0–4s Title Card
    - BG deep navy
    - Title "QFM‑CNN" gradient cyan→teal, 84pt, y=+2.5
    - Subtitle white 28pt below
    - Byline grey 22pt below
    - Animations: fade up, write subtitle, fade byline upward
    - Camera static
    """
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)  # locked

        title = Text("QFM‑CNN", font=SAFE_FONT, font_size=84, gradient=(ACCENT_A, ACCENT_B))
        title.move_to([0, 2.5, 0])

        subtitle = Text("Quantum Frequency Morphological Convolution",
                        font=SAFE_FONT, font_size=28, color=WHITE)
        subtitle.next_to(title, DOWN, buff=0.4)

        byline = Text("PhaseForge Technology", font=SAFE_FONT, font_size=22, color=TEXT_MUTED)
        byline.next_to(subtitle, DOWN, buff=0.25)

        # Fixed to frame
        hud = VGroup(title, subtitle, byline)
        self.add_fixed_in_frame_mobjects(hud)

        # Animations
        safe_play(self, FadeIn(title, shift=UP*0.3), run_time=1.1)
        safe_play(self, Write(subtitle), run_time=1.0)
        safe_play(self, FadeIn(byline, shift=UP*0.2), run_time=0.8)
        self.wait(0.9)  # ≈4s total


# --------------------- 4–10s Input → Network ---------------------

class QFM_InputToNetwork(ThreeDScene):
    """
    4–10s Input → Network
    - Input plane tilted 15° Z, left side, width≈3.5
    - 3D network (7,10,10,7) spheres + translucent edges
    - Edges reveal first, then nodes
    - Glowing yellow dots arc from plane to first layer
    - Camera pan: theta -60→-40°, phi 50→55°
    """
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=50*DEGREES, theta=-60*DEGREES)

        L = LayerManager(self)

        # Build network
        nodes, edges = make_nn(layers=(7,10,10,7), layer_spacing=1.9, node_radius=0.05)
        net = VGroup(edges, nodes)

        # Input plane on far left of net
        image_plane = make_image_plane(size=3.5, img_path="crop.jpg")
        # Place left of net
        left_anchor = net.get_left() if len(net) > 0 else ORIGIN + LEFT*3.0
        image_plane.move_to(left_anchor + LEFT*1.8 + OUT*0.0)

        # HUD label (fixed orientation: sits near plane but doesn't tilt)
        input_lbl = Text("Input Image", font=SAFE_FONT, font_size=26, color=WHITE)
        self.add_fixed_orientation_mobjects(input_lbl)
        input_lbl.next_to(image_plane, DOWN, buff=0.2)

        L.push_world(net, image_plane, input_lbl)

        # Edges then nodes reveal
        reveal_edges_then_nodes(self, edges, nodes, edge_rt=1.0, node_rt=0.8)

        # Pixel flows from plane to first layer
        centers_left = [n.get_center() for n in nodes[0]]
        spawn = image_plane.get_center() + LEFT * 0.1

        for _ in range(6):
            target = centers_left[np.random.randint(0, len(centers_left))]
            jitter_y = 0.3 - 0.6*np.random.random()
            mid = spawn + np.array([0.7, jitter_y, 0.05])
            path_pts = [spawn + np.array([0, 0, 0.05]), mid, target + np.array([0, 0, 0.0])]
            dot, path = flow_dot_and_path(path_pts, color=YELLOW_E, radius=0.045)
            self.add(dot)
            safe_play(self, MoveAlongPath(dot, path), run_time=0.6, rate_func=rate_functions.ease_in_out_sine)

        # Camera slow pan
        self.move_camera(theta=-40*DEGREES, phi=55*DEGREES, run_time=1.4, rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.2)


# --------------------- 10–18s Branching ---------------------

class QFM_Branching(ThreeDScene):
    """
    10–18s Branching
    - Splitter (rectangular box) after first layer
    - Upper edges cyan, lower edges red, middle faint
    - HUD banners (top-left cyan; bottom-left red)
    - Two streams split with pulse (opacity osc), 0.8 Hz
    - Camera: slight zoom + move up to emphasize bifurcation
    """
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=55*DEGREES, theta=-42*DEGREES)

        L = LayerManager(self)

        nodes, edges = make_nn(layers=(7,10,10,7), layer_spacing=1.9, node_radius=0.05)
        net = VGroup(edges, nodes).scale(1.02)
        L.push_world(net)

        # Reveal quickly
        reveal_edges_then_nodes(self, edges, nodes, edge_rt=0.8, node_rt=0.6)

        # Splitter = cube scaled thin, white semi-transparent
        splitter = Cube(side_length=0.4)
        splitter.set_fill(WHITE, opacity=0.2).set_stroke(WHITE, opacity=0.35, width=1)
        # Place near between first and second layer centers
        left_layer_center = nodes[0].get_center()
        right_layer_center = nodes[1].get_center()
        mid_x = 0.5 * (left_layer_center[0] + right_layer_center[0])
        splitter.move_to([mid_x - 0.1, 0.0, 0.0])
        L.push_world(splitter)

        # Tint edges by y
        tint_edges_split(edges, y_thresh=0.15, top=SPATIAL_C, bottom=FREQ_C, mid_opacity=0.25)

        # HUD banners
        sp_banner = banner("Spatial Branch — edges, textures")
        fr_banner = banner("Frequency Branch — spectral patterns")
        # Fixed to frame positions
        sp_banner.to_corner(UL).shift(DOWN*0.3 + RIGHT*0.2)
        fr_banner.to_corner(DL).shift(UP*0.3 + RIGHT*0.2)
        L.push_hud(sp_banner, fr_banner)

        # Streams split (cyan up, red down). We'll spawn dots near splitter, route to later layer nodes
        top_targets = [n.get_center() for n in nodes[2] if n.get_center()[1] > 0]
        bot_targets = [n.get_center() for n in nodes[2] if n.get_center()[1] < 0]
        start_base = splitter.get_center() + RIGHT*0.2

        # Pulse at 0.8 Hz: change opacity scale via tracker + updater
        pulse = ValueTracker(0.0)

        def make_pulsing_dot(color_hex, radius=0.04):
            d = Sphere(radius=radius, resolution=(16,16), color=color_hex)
            def upd(m, dt):
                pulse.increment_value(dt)
                m.set_opacity(0.5 + 0.5*np.sin(2*np.pi*0.8*pulse.get_value()))
            d.add_updater(upd)
            return d

        # Launch few dots up (cyan) and down (red)
        for _ in range(5):
            c = make_pulsing_dot(SPATIAL_C)
            start = start_base + UP * 0.4
            if top_targets:
                t = random.choice(top_targets)
            else:
                t = nodes[2][0].get_center() + UP*0.4
            mid = start + RIGHT*1.4 + UP*0.4
            path_pts = [start, mid, t]
            path = VMobject().set_points_smoothly(path_pts)
            self.add(c)
            safe_play(self, MoveAlongPath(c, path), run_time=0.6)

        for _ in range(5):
            c = make_pulsing_dot(FREQ_C)
            start = start_base + DOWN * 0.4
            if bot_targets:
                t = random.choice(bot_targets)
            else:
                t = nodes[2][-1].get_center() + DOWN*0.4
            mid = start + RIGHT*1.4 + DOWN*0.4
            path_pts = [start, mid, t]
            path = VMobject().set_points_smoothly(path_pts)
            self.add(c)
            safe_play(self, MoveAlongPath(c, path), run_time=0.6)

        # Camera: subtle zoom & upshift
        self.move_camera(phi=58*DEGREES, theta=-40*DEGREES, zoom=1.06,
                         frame_center=self.camera.frame_center + UP*0.15,
                         run_time=1.6)
        self.wait(0.2)


# --------------------- 18–26s Kernel Morphology ---------------------

class QFM_KernelMorphology(ThreeDScene):
    """
    18–26s Kernel Morphology
    - Gabor 3D surface (violet) fades in with wave motion
    - Morph into Laplacian surface (same color) over 2s
    - Overlay green skeleton curves oscillating
    - HUD caption top: "Kernel Superposition + Morphology-guided Warping"
    - Camera orbit theta -40→-20°
    """
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=55*DEGREES, theta=-40*DEGREES)

        L = LayerManager(self)

        # Position kernel surfaces at center-right
        pos = np.array([0.8, 0.0, 0.0])

        gabor = surface_from_height(lambda u,v: gabor_height(u,v, sigma=0.9, freq=0.45, theta=0.4),
                                    color=ACCENT_C, z_scale=0.7).scale(0.95)
        gabor.move_to(pos)

        lapla = surface_from_height(lambda u,v: laplacian_height(u,v, sigma=1.0),
                                    color=ACCENT_C, z_scale=0.7).scale(0.95)
        lapla.move_to(pos)

        # Skeleton overlay
        skel = VGroup()
        for a in np.linspace(-1.2, 1.2, 12):
            curve = ParametricFunction(lambda t, a=a: np.array([t, 0.35*np.sin(2*t + a), 0]),
                                       t_range=[-1.6, 1.6], color=MORPH_C)
            curve.set_opacity(0.7)
            curve.move_to(pos)
            skel.add(curve)

        # HUD caption
        caption = Text("Kernel Superposition + Morphology-guided Warping",
                       font=SAFE_FONT, font_size=28, color=WHITE)
        self.add_fixed_in_frame_mobjects(caption)
        caption.to_edge(UP).shift(DOWN*0.2)

        L.push_world(gabor, skel)

        # Fade in gabor with subtle oscillation
        for sub in gabor:
            sub.set_opacity(0.0)
        safe_play(self, *[sub.animate.set_opacity(1.0) for sub in gabor], run_time=0.7)

        # Oscillate skeleton additions one-by-one
        for c in skel:
            safe_play(self, Create(c), run_time=0.2)

        # Gentle animation on skeleton (up-down + slight scale)
        safe_play(self, skel.animate.shift(UP*0.06).scale(1.02), run_time=1.0)

        # Morph gabor → laplacian: avoid Transform (OpenGL mismatch risks).
        # Instead crossfade: fade out gabor, fade in lapla
        for sub in lapla:
            sub.set_opacity(0.0)
        L.push_world(lapla)
        safe_play(self,
                  *[sub.animate.set_opacity(0.0) for sub in gabor],
                  *[sub.animate.set_opacity(1.0) for sub in lapla],
                  run_time=2.0,
                  rate_func=rate_functions.ease_in_out_cubic)

        # Camera orbit theta -40→-20
        self.move_camera(theta=-20*DEGREES, run_time=1.6, rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.2)


# --------------------- 26–34s Quantum Interference Fusion ---------------------

class QFM_InterferenceFusion(ThreeDScene):
    """
    26–34s Quantum Interference Fusion
    - Left cyan sine surface (S), right red sine surface (F)
    - Slide toward center, then show fused interference surface (teal/green)
    - HUD equation at bottom center:
        y(x) = | A_s(x) S(x) + A_f(x) e^{i θ(x)} F(x) |^2
    - Camera rotates ~60° around Y during fusion
    """
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=50*DEGREES, theta=-50*DEGREES)

        L = LayerManager(self)

        left_wave = surface_from_height(lambda u,v: wave_x(u,v, amp=0.55, f=0.45, phase=0.0),
                                        color=SPATIAL_C, z_scale=0.8).scale(1.05)
        right_wave = surface_from_height(lambda u,v: wave_y(u,v, amp=0.55, f=0.45, phase=0.8),
                                         color=FREQ_C, z_scale=0.8).scale(1.05)
        left_wave.move_to([-1.4, 0, 0.0])
        right_wave.move_to([ 1.4, 0, 0.0])

        inter = surface_from_height(lambda u,v: wave_x(u,v, amp=0.55, f=0.45, phase=0.0) +
                                                 wave_y(u,v, amp=0.55, f=0.45, phase=0.8),
                                    color=ACCENT_B, z_scale=0.8).scale(1.15)
        inter.move_to([0,0,0])

        L.push_world(left_wave, right_wave)

        # HUD equation
        eq = SafeTex(r"y(x) = \left| A_s(x)\,S(x) + A_f(x)\,e^{i\theta(x)}\,F(x) \right|^2", font_size=40)
        self.add_fixed_in_frame_mobjects(eq)
        eq.to_edge(DOWN).shift(UP*0.35)

        # Slide toward center (without transforming incompatible types)
        safe_play(self,
                  left_wave.animate.shift(RIGHT*1.1),
                  right_wave.animate.shift(LEFT*1.1),
                  run_time=1.2)

        # Camera rotate ~60° around Y (theta)
        self.move_camera(theta=-110*DEGREES, run_time=1.6, rate_func=rate_functions.ease_in_out_sine)

        # Cross-fade: left/right wave out, interference in
        for sub in inter:
            sub.set_opacity(0.0)
        L.push_world(inter)
        safe_play(self,
                  *[sub.animate.set_opacity(0.0) for sub in left_wave],
                  *[sub.animate.set_opacity(0.0) for sub in right_wave],
                  *[sub.animate.set_opacity(1.0) for sub in inter],
                  run_time=1.6)

        self.wait(0.4)


# --------------------- 34–44s Output Panel ---------------------

class QFM_OutputsPanel(ThreeDScene):
    """
    34–44s Output Panel
    - Rounded rectangle HUD panel, dark navy fill, white border
    - Text lines fade sequentially with glow:
        "Crop: Wheat 🌾" (white)
        "Moisture: 32% 💧" (cyan)
        "Particle size: 1.8 cm" (teal)
    - Panel rises from center of interference surface (conceptually)
    - Camera: slight zoom out, keep panel centered
    """
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=50*DEGREES, theta=-60*DEGREES)

        L = LayerManager(self)

        # World filler: a faint interference-looking surface under panel (purely aesthetic)
        inter = surface_from_height(lambda u,v: wave_x(u,v, amp=0.35, f=0.35) + wave_y(u,v, amp=0.35, f=0.35),
                                    color=ACCENT_B, z_scale=0.6).scale(1.0)
        inter.set_opacity(0.25)
        inter.move_to([0,0,0])
        L.push_world(inter)

        # HUD Panel
        panel = RoundedRectangle(corner_radius=0.2, width=5.6, height=2.4, color=WHITE, stroke_opacity=0.95)
        panel.set_fill(PANEL_BG, opacity=0.95)

        lbl1 = Text("Crop: Wheat 🌾", font=SAFE_FONT, font_size=30, color=WHITE)
        lbl2 = Text("Moisture: 32% 💧", font=SAFE_FONT, font_size=30, color=ACCENT_A)
        lbl3 = Text("Particle size: 1.8 cm", font=SAFE_FONT, font_size=30, color=ACCENT_B)
        lines = VGroup(lbl1, lbl2, lbl3).arrange(DOWN, buff=0.2)
        # Make them HUD-fixed and centered inside panel
        hud = VGroup(panel, lines)
        self.add_fixed_in_frame_mobjects(hud)
        hud.move_to(ORIGIN)

        # Rise from center
        panel.shift(DOWN*0.2)
        safe_play(self, FadeIn(panel, shift=UP*0.2), run_time=0.7)

        # Each line fades in with subtle pulse
        for lbl in [lbl1, lbl2, lbl3]:
            lbl.set_opacity(0.0)
        for lbl in [lbl1, lbl2, lbl3]:
            safe_play(self, lbl.animate.set_opacity(1.0).scale(1.02), run_time=0.35)
            safe_play(self, lbl.animate.scale(1/1.02), run_time=0.15)

        # Camera: slight zoom out, keep centered
        self.move_camera(zoom=1.05, run_time=1.2)
        self.wait(0.4)


# --------------------- 44–55s Outro ---------------------

class QFM_Outro(ThreeDScene):
    """
    44–55s Outro
    - Fade out world; final text HUD:
        "QFM-CNN • Morphing the Future of AI" (white)
        "PhaseForge — Built by Arham Sethi" (grey)
      plus pulsing cyan dot (logo heartbeat)
    - Camera gentle pull back; fade to black
    """
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=55*DEGREES, theta=-45*DEGREES)

        # Final HUD
        title = Text("QFM‑CNN • Morphing the Future of AI", font=SAFE_FONT, font_size=44, color=WHITE)
        sub   = Text("PhaseForge — Built by Arham Sethi", font=SAFE_FONT, font_size=28, color=TEXT_MUTED)
        grp   = VGroup(title, sub).arrange(DOWN, buff=0.25).move_to(ORIGIN)

        dot = Dot(color=ACCENT_A).scale(1.8)
        dot.move_to(ORIGIN + DOWN*1.8)

        self.add_fixed_in_frame_mobjects(grp, dot)

        safe_play(self, FadeIn(title, shift=UP*0.3), run_time=0.7)
        safe_play(self, FadeIn(sub, shift=UP*0.2), run_time=0.6)

        # Pulse (heartbeat)
        for _ in range(3):
            safe_play(self, dot.animate.scale(1.25), run_time=0.25)
            safe_play(self, dot.animate.scale(1/1.25), run_time=0.25)

        # Pull back + fade to black
        self.move_camera(zoom=0.95, run_time=1.2)
        # Fade
        safe_play(self, FadeOut(title), FadeOut(sub), FadeOut(dot), run_time=0.8)
        self.wait(0.2)


# --------------------- Film Stitch (Single Scene) ---------------------

class QFM_Film(ThreeDScene):
    """
    One-pass version chaining the seven acts in ~55 seconds.
    This is provided for convenience. Editors usually render each
    scene separately, but this lets you preview the flow end-to-end.

    IMPORTANT:
    - To keep OpenGL happy and avoid massive state carryover,
      we clean between acts with LayerManager-like logic, but
      kept local here to reduce overhead.
    """
    def construct(self):
        self.camera.background_color = BG

        # === ACT 0: Title (≈4s)
        self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
        title = Text("QFM‑CNN", font=SAFE_FONT, font_size=84, gradient=(ACCENT_A, ACCENT_B)).move_to([0, 2.5, 0])
        subtitle = Text("Quantum Frequency Morphological Convolution", font=SAFE_FONT, font_size=28, color=WHITE).next_to(title, DOWN, buff=0.4)
        byline   = Text("PhaseForge Technology", font=SAFE_FONT, font_size=22, color=TEXT_MUTED).next_to(subtitle, DOWN, buff=0.25)
        hud0 = VGroup(title, subtitle, byline)
        self.add_fixed_in_frame_mobjects(hud0)
        safe_play(self, FadeIn(title, shift=UP*0.3), run_time=1.1)
        safe_play(self, Write(subtitle), run_time=1.0)
        safe_play(self, FadeIn(byline, shift=UP*0.2), run_time=0.8)
        self.wait(0.9)
        # Clear HUD 0
        safe_play(self, *[FadeOut(m) for m in hud0], run_time=0.5)

        # === ACT 1: Input → Network (≈6s)
        self.set_camera_orientation(phi=50*DEGREES, theta=-60*DEGREES)
        nodes, edges = make_nn(layers=(7,10,10,7), layer_spacing=1.9, node_radius=0.05)
        net = VGroup(edges, nodes)
        image_plane = make_image_plane(size=3.5, img_path="crop.jpg")
        left_anchor = net.get_left() if len(net) > 0 else ORIGIN + LEFT*3.0
        image_plane.move_to(left_anchor + LEFT*1.8)
        input_lbl = Text("Input Image", font=SAFE_FONT, font_size=26, color=WHITE)
        self.add_fixed_orientation_mobjects(input_lbl)
        input_lbl.next_to(image_plane, DOWN, buff=0.2)
        self.add(net, image_plane, input_lbl)
        reveal_edges_then_nodes(self, edges, nodes, edge_rt=1.0, node_rt=0.8)

        centers_left = [n.get_center() for n in nodes[0]]
        spawn = image_plane.get_center() + LEFT * 0.1
        for _ in range(6):
            target = centers_left[np.random.randint(0, len(centers_left))]
            jitter_y = 0.3 - 0.6*np.random.random()
            mid = spawn + np.array([0.7, jitter_y, 0.05])
            path_pts = [spawn + np.array([0, 0, 0.05]), mid, target + np.array([0, 0, 0.0])]
            dot, path = flow_dot_and_path(path_pts, color=YELLOW_E, radius=0.045)
            self.add(dot)
            safe_play(self, MoveAlongPath(dot, path), run_time=0.6)
        self.move_camera(theta=-40*DEGREES, phi=55*DEGREES, run_time=1.4)
        # Clean world for next act
        safe_play(self, FadeOut(net), FadeOut(image_plane), FadeOut(input_lbl), run_time=0.5)

        # === ACT 2: Branching (≈8s)
        self.set_camera_orientation(phi=55*DEGREES, theta=-42*DEGREES)
        nodes2, edges2 = make_nn(layers=(7,10,10,7), layer_spacing=1.9, node_radius=0.05)
        net2 = VGroup(edges2, nodes2).scale(1.02)
        self.add(net2)
        reveal_edges_then_nodes(self, edges2, nodes2, edge_rt=0.8, node_rt=0.6)
        splitter = Cube(side_length=0.4).set_fill(WHITE, opacity=0.2).set_stroke(WHITE, opacity=0.35, width=1)
        left_c = nodes2[0].get_center()
        right_c = nodes2[1].get_center()
        mid_x = 0.5*(left_c[0] + right_c[0])
        splitter.move_to([mid_x - 0.1, 0.0, 0.0])
        self.add(splitter)
        tint_edges_split(edges2, y_thresh=0.15, top=SPATIAL_C, bottom=FREQ_C, mid_opacity=0.25)

        sp_banner = banner("Spatial Branch — edges, textures")
        fr_banner = banner("Frequency Branch — spectral patterns")
        sp_banner.to_corner(UL).shift(DOWN*0.3 + RIGHT*0.2)
        fr_banner.to_corner(DL).shift(UP*0.3 + RIGHT*0.2)
        self.add_fixed_in_frame_mobjects(sp_banner, fr_banner)

        top_targets = [n.get_center() for n in nodes2[2] if n.get_center()[1] > 0]
        bot_targets = [n.get_center() for n in nodes2[2] if n.get_center()[1] < 0]
        start_base = splitter.get_center() + RIGHT*0.2

        pulse = ValueTracker(0.0)
        def make_pulse(color_hex):
            d = Sphere(radius=0.04, resolution=(16,16), color=color_hex)
            def upd(m,dt):
                pulse.increment_value(dt)
                m.set_opacity(0.5+0.5*np.sin(2*np.pi*0.8*pulse.get_value()))
            d.add_updater(upd)
            return d

        for _ in range(5):
            d = make_pulse(SPATIAL_C)
            start = start_base + UP * 0.4
            t = random.choice(top_targets) if top_targets else nodes2[2][0].get_center()+UP*0.4
            mid = start + RIGHT*1.4 + UP*0.4
            path = VMobject().set_points_smoothly([start, mid, t])
            self.add(d)
            safe_play(self, MoveAlongPath(d, path), run_time=0.6)

        for _ in range(5):
            d = make_pulse(FREQ_C)
            start = start_base + DOWN * 0.4
            t = random.choice(bot_targets) if bot_targets else nodes2[2][-1].get_center()+DOWN*0.4
            mid = start + RIGHT*1.4 + DOWN*0.4
            path = VMobject().set_points_smoothly([start, mid, t])
            self.add(d)
            safe_play(self, MoveAlongPath(d, path), run_time=0.6)

        self.move_camera(phi=58*DEGREES, theta=-40*DEGREES, zoom=1.06,
                         frame_center=self.camera.frame_center + UP*0.15,
                         run_time=1.6)
        # Clean
        safe_play(self, FadeOut(net2), FadeOut(splitter), run_time=0.5)
        safe_play(self, FadeOut(sp_banner), FadeOut(fr_banner), run_time=0.4)

        # === ACT 3: Kernel Morphology (≈8s)
        self.set_camera_orientation(phi=55*DEGREES, theta=-40*DEGREES)
        pos = np.array([0.8, 0.0, 0.0])
        gabor = surface_from_height(lambda u,v: gabor_height(u,v, 0.9, 0.45, 0.4),
                                    color=ACCENT_C, z_scale=0.7).scale(0.95).move_to(pos)
        lapla = surface_from_height(lambda u,v: laplacian_height(u,v, 1.0),
                                    color=ACCENT_C, z_scale=0.7).scale(0.95).move_to(pos)
        skel = VGroup(*[
            ParametricFunction(lambda t, a=a: np.array([t, 0.35*np.sin(2*t+a), 0]),
                               t_range=[-1.6,1.6], color=MORPH_C).set_opacity(0.7).move_to(pos)
            for a in np.linspace(-1.2, 1.2, 12)
        ])
        cap = Text("Kernel Superposition + Morphology-guided Warping",
                   font=SAFE_FONT, font_size=28, color=WHITE)
        self.add_fixed_in_frame_mobjects(cap)
        cap.to_edge(UP).shift(DOWN*0.2)

        self.add(gabor, skel)
        for sub in gabor:
            sub.set_opacity(0.0)
        safe_play(self, *[sub.animate.set_opacity(1.0) for sub in gabor], run_time=0.7)
        for c in skel:
            safe_play(self, Create(c), run_time=0.2)
        safe_play(self, skel.animate.shift(UP*0.06).scale(1.02), run_time=1.0)

        for sub in lapla:
            sub.set_opacity(0.0)
        self.add(lapla)
        safe_play(self, *[sub.animate.set_opacity(0.0) for sub in gabor],
                         *[sub.animate.set_opacity(1.0) for sub in lapla],
                  run_time=2.0)
        self.move_camera(theta=-20*DEGREES, run_time=1.6)
        # Clean
        safe_play(self, FadeOut(gabor), FadeOut(lapla), FadeOut(skel), FadeOut(cap), run_time=0.6)

        # === ACT 4: Interference Fusion (≈8s)
        self.set_camera_orientation(phi=50*DEGREES, theta=-50*DEGREES)
        left_wave = surface_from_height(lambda u,v: wave_x(u,v, 0.55, 0.45, 0.0),
                                        color=SPATIAL_C, z_scale=0.8).scale(1.05).move_to([-1.4,0,0])
        right_wave = surface_from_height(lambda u,v: wave_y(u,v, 0.55, 0.45, 0.8),
                                         color=FREQ_C, z_scale=0.8).scale(1.05).move_to([1.4,0,0])
        inter = surface_from_height(lambda u,v: wave_x(u,v,0.55,0.45,0.0)+wave_y(u,v,0.55,0.45,0.8),
                                    color=ACCENT_B, z_scale=0.8).scale(1.15).move_to([0,0,0])
        self.add(left_wave, right_wave)

        eq = SafeTex(r"y(x) = \left| A_s(x)\,S(x) + A_f(x)\,e^{i\theta(x)}\,F(x) \right|^2", font_size=40)
        self.add_fixed_in_frame_mobjects(eq)
        eq.to_edge(DOWN).shift(UP*0.35)

        safe_play(self, left_wave.animate.shift(RIGHT*1.1),
                         right_wave.animate.shift(LEFT*1.1),
                  run_time=1.2)
        self.move_camera(theta=-110*DEGREES, run_time=1.6)

        for sub in inter:
            sub.set_opacity(0.0)
        self.add(inter)
        safe_play(self, *[sub.animate.set_opacity(0.0) for sub in left_wave],
                         *[sub.animate.set_opacity(0.0) for sub in right_wave],
                         *[sub.animate.set_opacity(1.0) for sub in inter],
                  run_time=1.6)
        # Clean eq later with panel

        # === ACT 5: Outputs Panel (≈10s)
        inter.set_opacity(0.25)
        panel = RoundedRectangle(corner_radius=0.2, width=5.6, height=2.4, color=WHITE, stroke_opacity=0.95)
        panel.set_fill(PANEL_BG, opacity=0.95)
        lbl1 = Text("Crop: Wheat 🌾", font=SAFE_FONT, font_size=30, color=WHITE)
        lbl2 = Text("Moisture: 32% 💧", font=SAFE_FONT, font_size=30, color=ACCENT_A)
        lbl3 = Text("Particle size: 1.8 cm", font=SAFE_FONT, font_size=30, color=ACCENT_B)
        lines = VGroup(lbl1, lbl2, lbl3).arrange(DOWN, buff=0.2)

        hudp = VGroup(panel, lines)
        self.add_fixed_in_frame_mobjects(hudp)
        hudp.move_to(ORIGIN)

        panel.shift(DOWN*0.2)
        safe_play(self, FadeIn(panel, shift=UP*0.2), run_time=0.7)
        for lbl in [lbl1, lbl2, lbl3]:
            lbl.set_opacity(0.0)
        for lbl in [lbl1, lbl2, lbl3]:
            safe_play(self, lbl.animate.set_opacity(1.0).scale(1.02), run_time=0.35)
            safe_play(self, lbl.animate.scale(1/1.02), run_time=0.15)
        self.move_camera(zoom=1.05, run_time=1.2)
        self.wait(0.2)
        # Remove equation now (clean HUD bottom)
        safe_play(self, FadeOut(eq), run_time=0.4)

        # === ACT 6: Outro (≈11s)
        safe_play(self, FadeOut(panel), FadeOut(lines), run_time=0.6)
        safe_play(self, FadeOut(inter), run_time=0.4)

        title = Text("QFM‑CNN • Morphing the Future of AI", font=SAFE_FONT, font_size=44, color=WHITE)
        sub   = Text("PhaseForge — Built by Arham Sethi", font=SAFE_FONT, font_size=28, color=TEXT_MUTED)
        grp   = VGroup(title, sub).arrange(DOWN, buff=0.25).move_to(ORIGIN)
        dot   = Dot(color=ACCENT_A).scale(1.8).move_to(ORIGIN + DOWN*1.8)
        self.add_fixed_in_frame_mobjects(grp, dot)

        safe_play(self, FadeIn(title, shift=UP*0.3), run_time=0.7)
        safe_play(self, FadeIn(sub, shift=UP*0.2), run_time=0.6)

        for _ in range(3):
            safe_play(self, dot.animate.scale(1.25), run_time=0.25)
            safe_play(self, dot.animate.scale(1/1.25), run_time=0.25)

        self.move_camera(zoom=0.95, run_time=1.2)
        safe_play(self, FadeOut(title), FadeOut(sub), FadeOut(dot), run_time=0.8)
        self.wait(0.2)

