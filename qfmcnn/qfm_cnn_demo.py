from manim import *
import numpy as np
import os

# ---------- Look & Feel ----------
BG = "#0a0d14"
ACCENT_A = "#22d3ee"
ACCENT_B = "#6ee7b7"
ACCENT_C = "#a78bfa"
SPATIAL_C = "#7dd3fc"   # light blue
FREQ_C    = "#fca5a5"   # soft red
MORPH_C   = "#34d399"   # green
NET_NODE  = "#93c5fd"
NET_EDGE  = "#334155"

def SafeTex(expr, **kwargs):
    """Try LaTeX; if missing packages, fall back to plain text with unicode."""
    try:
        return MathTex(expr, **kwargs)
    except Exception:
        txt = (expr
               .replace(r"\psi", "ψ")
               .replace(r"\theta", "θ")
               .replace("|", "∣")
               .replace(r"_{spatial}", "_spatial")
               .replace(r"_{freq}", "_freq"))
        return Text(txt, **{k: v for k, v in kwargs.items() if k != "font_size"})

def make_nn(layers=(6, 8, 8, 6), layer_spacing=2.2, node_radius=0.06, z_shift=0.0):
    """3D feedforward network with spheres and line3D edges."""
    nodes_layers = []
    all_nodes = VGroup()
    x0 = - (len(layers) - 1) * layer_spacing / 2
    for li, n_nodes in enumerate(layers):
        g = VGroup()
        for ni in range(n_nodes):
            y = (n_nodes - 1) / 2 - ni
            s = Sphere(radius=node_radius, resolution=(16, 16), color=NET_NODE)
            s.set_fill(color=NET_NODE, opacity=0.9)
            s.move_to([x0 + li * layer_spacing, y * 0.35, z_shift])
            g.add(s)
        nodes_layers.append(g)
        all_nodes.add(g)

    edges = VGroup()
    for li in range(len(layers) - 1):
        left_layer = nodes_layers[li]
        right_layer = nodes_layers[li + 1]
        for a in left_layer:
            for b in right_layer:
                e = Line3D(a.get_center(), b.get_center(), thickness=0.01, color=NET_EDGE)
                e.set_opacity(0.6)
                edges.add(e)
    return all_nodes, edges

def flow_pixel_path(points, color=YELLOW, radius=0.04):
    """Glowing sphere moved along a 3D VMobject path."""
    dot = Sphere(radius=radius, resolution=(16, 16), color=color)
    # VMobject supports 3D points
    path = VMobject(stroke_color=color, stroke_opacity=0.0)
    path.set_points_smoothly(points)
    return dot, path

def make_image_plane(size=3.6, img_path="parali.jpg"):
    """A thin plane; use image if available, else gradient stripes."""
    rect = Rectangle(width=size, height=size * 0.7, color=WHITE, stroke_opacity=0.35)
    rect.set_fill(color="#0a1722", opacity=1.0)
    plane = VGroup(rect)

    if os.path.exists(img_path):
        try:
            im = ImageMobject(img_path)
            im.height = rect.height
            im.width = rect.width
            im.move_to(rect.get_center())
            im.set_opacity(1.0)
            plane.add(im)
            return plane
        except Exception:
            pass

    # fallback gradient with color_gradient (no Color() constructor)
    grads = color_gradient([ACCENT_A, ACCENT_B], 22)
    stripes = VGroup()
    for i in range(22):
        r = Rectangle(width=rect.width, height=rect.height / 22, stroke_opacity=0)
        r.set_fill(grads[i], opacity=0.25 + 0.35 * (i % 2))
        r.align_to(rect, LEFT)
        r.move_to(rect.get_center() + UP * (rect.height / 22 * (10 - i)))
        stripes.add(r)
    plane.add(stripes)
    return plane

def gabor_height(x, y, sigma=0.8, freq=2.0, theta=0.0):
    xr =  np.cos(theta)*x + np.sin(theta)*y
    yr = -np.sin(theta)*x + np.cos(theta)*y
    return np.exp(-(xr**2+yr**2)/(2*sigma**2))*np.cos(2*np.pi*freq*xr)

def laplacian_height(x, y, sigma=0.6):
    r2 = x**2 + y**2
    return (1 - r2/sigma**2) * np.exp(-r2/(2*sigma**2))

def surface_from_height(fn, x_range=(-1.8, 1.8), y_range=(-1.8, 1.8), step=0.12, color=ACCENT_C):
    def param(u, v):
        return np.array([u, v, 0.6 * fn(u, v)])
    surf = Surface(
        lambda u, v: param(u, v),
        u_range=x_range, v_range=y_range,
        resolution=(max(3, int((x_range[1]-x_range[0]) / step)),
                    max(3, int((y_range[1]-y_range[0]) / step))),
        fill_opacity=0.85,
        checkerboard_colors=[color, color]
    )
    mesh = always_redraw(lambda: surf.copy().set_stroke(WHITE, opacity=0.08, width=0.5).set_fill(opacity=0))

    return VGroup(surf, mesh)

def make_interference_surface(phi=0.0, amp1=0.55, amp2=0.55, color="#9ae6b4"):
    def h(u, v):
        return amp1*np.sin(2*np.pi*0.5*u) + amp2*np.sin(2*np.pi*0.5*v + phi)
    return surface_from_height(h, color=color)

def banner(text, sub=None):
    r = RoundedRectangle(corner_radius=0.2, width=5.8, height=1.1, color=WHITE)
    r.set_fill(BG, opacity=0.4)
    t = Text(text, font_size=36, color=WHITE)
    grp = VGroup(r, t)
    if sub:
        s = Text(sub, font_size=24, color=GRAY_B).next_to(t, DOWN * 0.7)
        grp.add(s)
    return grp

class QFMCinematicScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=50*DEGREES, theta=-60*DEGREES, zoom=1.0)

        # --- ACT 0: Title (0-4s) ---
        title = Text("QFM‑CNN", gradient=(ACCENT_A, ACCENT_B), font_size=84)
        subtitle = Text("Quantum Frequency Morphological Convolution", font_size=28).next_to(title, DOWN * 0.8)
        byline = Text("AgroDry • Moisture & Grading", font_size=22, color=GRAY_C).next_to(subtitle, DOWN * 0.5)
        header = VGroup(title, subtitle, byline).move_to([0, 2.7, 0])
        self.play(FadeIn(title, shift=UP * 0.4), run_time=1.2)
        self.play(Write(subtitle), run_time=1.2)
        self.play(FadeIn(byline, shift=UP * 0.2), run_time=0.8)
        self.wait(0.6)
        self.play(header.animate.to_edge(UP).shift(UP * 0.2), run_time=0.8)

        # --- ACT 1: Network (4-10s) ---
        nodes, edges = make_nn(layers=(7, 10, 10, 7), layer_spacing=2.0, node_radius=0.05)
        net = VGroup(edges, nodes)
        self.play(LaggedStart(Create(edges), FadeIn(nodes, scale=0.8), lag_ratio=0.02, run_time=2.2))
        self.move_camera(theta=-40*DEGREES, phi=60*DEGREES, zoom=1.15, run_time=2.5, rate_func=rate_functions.ease_in_out_cubic)

        # Input plane at far left
        image_plane = make_image_plane(size=3.6)
        image_plane.rotate(axis=UP, angle=90 * DEGREES)
        image_plane.move_to(net.get_left() + LEFT * 1.8 + OUT * 0.2)
        label_input = Text("Input Image", font_size=26, color=WHITE).next_to(image_plane, DOWN).rotate(90 * DEGREES, axis=UP)
        self.play(FadeIn(image_plane, shift=IN * 0.2), FadeIn(label_input), run_time=1.2)

        # Pixel flows
        centers_left = [n.get_center() for n in nodes[0]]
        spawn = image_plane.get_center() + LEFT * 0.1
        for _ in range(8):
            target = centers_left[np.random.randint(0, len(centers_left))]
            path_pts = [
                spawn + np.array([0, 0, 0.1]),
                spawn + np.array([0.7, 0.3 - 0.6*np.random.random(), 0.1]),
                target + np.array([0, 0, 0.0]),
            ]
            dot, path = flow_pixel_path(path_pts, color=YELLOW_E, radius=0.045)
            self.add(dot)
            self.play(MoveAlongPath(dot, path), run_time=0.6, rate_func=rate_functions.ease_in_out_sine)

        # --- ACT 2: Split branches (10-18s) ---
        splitter = Prism(dimensions=[0.15, 0.5, 0.6], fill_opacity=1.0, fill_color=WHITE, stroke_opacity=0.0)
        splitter.move_to(nodes[0][len(nodes[0]) // 2].get_center() + RIGHT * 0.8)
        self.play(FadeIn(splitter, scale=0.5), run_time=0.6)
        
        self.play(FadeOut(image_plane), FadeOut(label_input), run_time=0.5)
        fo_g1 = VGroup(nodes[0], nodes[1])

        sp_banner = banner("Spatial Branch", "local edges, textures")
        sp_banner.rotate(90 * DEGREES, axis=UP).move_to([splitter.get_center()[0] + 2.7, 1.7, 0.0])
        fr_banner = banner("Frequency Branch", "adaptive spectral patterns")
        fr_banner.rotate(90 * DEGREES, axis=UP).move_to([splitter.get_center()[0] + 2.7, -1.7, 0.0])
        self.play(FadeIn(sp_banner, shift=UP * 0.2), FadeIn(fr_banner, shift=DOWN * 0.2), run_time=0.9)

        # tint edges to suggest path
        for e in edges:
            ymid = (e.get_start()[1] + e.get_end()[1]) / 2
            if ymid > 0.15:
                e.set_color(SPATIAL_C).set_opacity(0.7)
            elif ymid < -0.15:
                e.set_color(FREQ_C).set_opacity(0.7)
            else:
                e.set_opacity(0.25)

        def spawn_stream(delta_y, color_hex):
            for _ in range(6):
                start = splitter.get_center() + RIGHT * 0.2 + UP * delta_y
                end   = start + RIGHT * 1.7 + UP * (delta_y * 0.8)
                mid   = start + RIGHT * 0.7 + UP * (delta_y * 1.2)
                dot, path = flow_pixel_path([start, mid, end], color=color_hex, radius=0.04)
                self.add(dot)
                self.play(MoveAlongPath(dot, path), run_time=0.5, rate_func=rate_functions.ease_in_out_sine)

        spawn_stream(0.5, SPATIAL_C)
        spawn_stream(-0.5, FREQ_C)

        # --- ACT 3: Kernel morphing & morphology (18-28s) ---
        kernel_pos = net.get_center() + RIGHT * 1.8 + OUT * 0.3
        gabor = surface_from_height(lambda u, v: gabor_height(u, v, sigma=0.9, freq=0.45, theta=0.4), color=ACCENT_C).scale(0.9)
        gabor.move_to(kernel_pos)
        lapla = surface_from_height(lambda u, v: laplacian_height(u, v, sigma=1.0), color=ACCENT_C).scale(0.9)
        lapla.move_to(kernel_pos)

        skel = VGroup()
        for a in np.linspace(-1.2, 1.2, 10):
            curve = ParametricFunction(lambda t: np.array([t, 0.35*np.sin(2*t + a), 0]), t_range=[-1.6, 1.6], color=MORPH_C)
            curve.move_to(kernel_pos)
            curve.set_opacity(0.7)
            skel.add(curve)

        morph_caption = Text("Kernel Superposition + Morphology‑guided Warping", font_size=28, color=WHITE)
        morph_caption.rotate(90 * DEGREES, axis=UP).move_to(kernel_pos + UP * 2.2)

        self.play(FadeIn(gabor, shift=OUT * 0.2), FadeIn(morph_caption), LaggedStart(*[Create(c) for c in skel], lag_ratio=0.05, run_time=1.2))
        self.play(Transform(gabor, lapla), skel.animate.shift(UP * 0.1).scale(1.02), run_time=2.2, rate_func=rate_functions.ease_in_out_cubic)
        self.wait(0.4)
        self.play(FadeOut(gabor), FadeOut(lapla), FadeOut(skel), FadeOut(morph_caption), FadeOut(splitter), FadeOut(sp_banner), FadeOut(fr_banner), run_time=0.8)

        # --- ACT 4: Interference fusion (28-38s) ---
        left_wave  = surface_from_height(lambda u, v: 0.55*np.sin(2*np.pi*0.45*u), color=SPATIAL_C).scale(1.0).move_to([-1.4, 0, 0.0])
        right_wave = surface_from_height(lambda u, v: 0.55*np.sin(2*np.pi*0.45*v + 0.8), color=FREQ_C).scale(1.0).move_to([ 1.4, 0, 0.0])
        self.play(FadeIn(left_wave, shift=RIGHT * 0.3), FadeIn(right_wave, shift=LEFT * 0.3), run_time=1.0)
        self.move_camera(theta=-30*DEGREES, phi=55*DEGREES, zoom=1.2, run_time=1.6)

        inter = make_interference_surface(phi=0.8, color=ACCENT_B).scale(1.2).move_to([0, 0, 0.0])
        eq = SafeTex(r"y = \left|\psi_{spatial} + e^{i\theta}\psi_{freq}\right|^2", font_size=40)
        eq.rotate(90 * DEGREES, axis=UP).move_to([0, -2.3, 0])

        self.play(Transform(VGroup(left_wave, right_wave), inter), Write(eq), run_time=2.2)
        self.wait(0.8)
        self.move_camera(theta=-75*DEGREES, phi=50*DEGREES, run_time=2.3, rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeOut(inter), FadeOut(eq), run_time=0.8)

        # --- ACT 5: Output panel (38-50s) ---
        panel = RoundedRectangle(corner_radius=0.2, width=5.6, height=2.4, color=WHITE)
        panel.set_fill(color="#0f172a", opacity=0.85)
        panel.rotate(90 * DEGREES, axis=UP).move_to([0, 0, 0.2])
        lbl = Text("QFM‑CNN Outputs", font_size=30, color=WHITE).rotate(90 * DEGREES, axis=UP).next_to(panel, UP * 1.0)
        crop = Text("Crop: Wheat 🌾", font_size=28, color=WHITE).rotate(90 * DEGREES, axis=UP).move_to(panel.get_center() + UP * 0.6)
        moist = Text("Moisture: 32% 💧", font_size=28, color=ACCENT_B).rotate(90 * DEGREES, axis=UP).move_to(panel.get_center())
        psize = Text("Particle Size: 1.8 cm", font_size=28, color=ACCENT_A).rotate(90 * DEGREES, axis=UP).move_to(panel.get_center() + DOWN * 0.6)

        self.play(FadeIn(panel, shift=OUT * 0.2), FadeIn(lbl, shift=UP * 0.2), run_time=0.8)
        self.play(LaggedStart(Write(crop), Write(moist), Write(psize), lag_ratio=0.2, run_time=1.2))
        self.move_camera(theta=-40*DEGREES, phi=60*DEGREES, zoom=1.05, run_time=1.8)
        self.play(FadeOut(panel), FadeOut(lbl), FadeOut(crop), FadeOut(moist), FadeOut(psize), run_time=0.7)

        # --- ACT 6: Outro (50-60s) ---
        outro  = Text("QFM‑CNN • Morphing the Future of AI", font_size=44, color=WHITE)
        outro2 = Text("AgroDry — Built by Arham Sethi", font_size=28, color=GRAY_C).next_to(outro, DOWN * 0.6)
        logo = Dot(color=ACCENT_B).scale(3).move_to([0, 0, 0])
        self.play(FadeIn(logo, scale=2), run_time=0.6)
        self.play(Transform(logo, VGroup(outro)), run_time=0.8)
        self.play(FadeIn(outro2, shift=UP * 0.2), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(outro), FadeOut(outro2), FadeOut(logo), FadeOut(net), FadeOut(image_plane), FadeOut(header), run_time=1.0)
        self.wait(0.2)
