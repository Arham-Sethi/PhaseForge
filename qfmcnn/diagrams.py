# qfm_cnn_workflow_diagram.py
# Minimal-word, left→right workflow for Refined QFM-CNN.
# Outputs: PNG, SVG, PDF (light and dark themes).

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

def rbox(ax, x, y, w, h, text, fc="#ffffff", ec="#111827", fs=10, rs=0.06, lw=1.2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.012,rounding_size={rs}",
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, color=ec)
    return box

def arrow(ax, xy1, xy2, col="#111827", txt=None, fs=9, rad=0.0, lw=1.4):
    a = FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=11,
                        linewidth=lw, color=col,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if txt:
        mx, my = (xy1[0]+xy2[0])/2, (xy1[1]+xy2[1])/2
        ax.text(mx, my+0.02, txt, fontsize=fs, ha="center", va="bottom", color=col)

def draw(theme="light"):
    # Colors
    if theme == "light":
        BG, EDGE = "#ffffff", "#0b1020"
        SPAT, FREQ, MORPH = "#eaf3ff", "#eef7ff", "#e8f7ed"
        CTRL, FUSION, OUT = "#fff3e6", "#f1f5f9", "#f7fee7"
        BLUE, GREEN, ORANGE, PURPLE = "#2563eb", "#10b981", "#f59e0b", "#7c3aed"
        FILT, KM, MASK = "#fefce8", "#fff7ed", "#eafaf0"
    else:  # dark
        BG, EDGE = "#0b1220", "#dbe5ff"
        SPAT, FREQ, MORPH = "#0f1a30", "#13203a", "#0f2a20"
        CTRL, FUSION, OUT = "#2a1f10", "#121a25", "#182012"
        BLUE, GREEN, ORANGE, PURPLE = "#8db0ff", "#6fe6b7", "#ffbe6a", "#c1a4ff"
        FILT, KM, MASK = "#29260f", "#2a2316", "#10271a"

    fig = plt.figure(figsize=(16, 8))
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    # Title
    ax.text(0.02, 0.955, "Refined QFM-CNN — End-to-End Workflow",
            fontsize=16, fontweight="bold", color=EDGE)
    ax.text(0.02, 0.925, "Input → Morphology → Spatial & Frequency → Controller → Phase-aware Fusion → Task Heads",
            fontsize=10.5, color=EDGE)

    # y-levels
    y_top, y_mid, y_bot = 0.74, 0.50, 0.26

    # 1) Input
    inp = rbox(ax, 0.04, y_mid-0.06, 0.10, 0.12, "Input\nx", fc=FREQ, ec=EDGE)

    # 2) Morphology
    morph = rbox(ax, 0.20, y_mid-0.10, 0.13, 0.20, "Morphology\nM(x)", fc=MORPH, ec=EDGE)
    arrow(ax, (0.14, y_mid), (0.20, y_mid), col=BLUE)  # x → M(x)

    # 3) Branches
    # Spatial
    sp = rbox(ax, 0.38, y_top-0.06, 0.22, 0.12, "Spatial\nS(x)", fc=SPAT, ec=EDGE)
    # Kernel morphing + base filters (above)
    bf = rbox(ax, 0.29, y_top+0.08, 0.08, 0.08, "Base\n{B_k}", fc=FILT, ec=EDGE, fs=9)
    km = rbox(ax, 0.38, y_top+0.08, 0.22, 0.08, "Kernel morphing\nU_m (unitary)", fc=KM, ec=EDGE, fs=9)
    arrow(ax, (0.37, y_top+0.12), (0.38, y_top+0.12), col=EDGE)   # {B_k} → morph
    arrow(ax, (0.49, y_top+0.08), (0.49, y_top+0.02), col=PURPLE) # U_m unitary

    # Frequency
    freq = rbox(ax, 0.38, y_bot-0.06, 0.22, 0.12, "Frequency\nF(x)", fc=FREQ, ec=EDGE)
    fft  = rbox(ax, 0.385, y_bot+0.05, 0.05, 0.05, "FFT", fc=FREQ, ec=EDGE, fs=9)
    uc   = rbox(ax, 0.445, y_bot+0.05, 0.09, 0.05, "1×1  U_c", fc=KM, ec=EDGE, fs=9)
    ifft = rbox(ax, 0.545, y_bot+0.05, 0.05, 0.05, "IFFT", fc=FREQ, ec=EDGE, fs=9)
    arrow(ax, (0.435, y_bot+0.075), (0.445, y_bot+0.075), col=EDGE)
    arrow(ax, (0.535, y_bot+0.075), (0.545, y_bot+0.075), col=EDGE)
    ax.text(0.49, y_bot+0.12, "U_c unitary", fontsize=9, color=PURPLE, ha="center")
    mask = rbox(ax, 0.445, y_bot-0.01, 0.15, 0.04, "mask  g(ω|M)", fc=MASK, ec=EDGE, fs=9)

    # Input → branches
    arrow(ax, (0.14, y_mid+0.07), (0.38, y_top), col=BLUE)  # x → S
    arrow(ax, (0.14, y_mid-0.07), (0.38, y_bot), col=BLUE)  # x → F
    # Morph → branches
    arrow(ax, (0.33, y_mid+0.05), (0.38, y_top+0.02), col=GREEN, txt="M(x)")
    arrow(ax, (0.33, y_mid-0.05), (0.445, y_bot+0.01), col=GREEN, txt="M(x)")

    # 4) Controller
    ctrl = rbox(ax, 0.66, y_mid-0.03, 0.12, 0.12, "Controller\n[αs, αf, θ]", fc=CTRL, ec=EDGE)
    arrow(ax, (0.60, y_top), (0.66, y_mid+0.03), col=BLUE,  txt="S(x)")
    arrow(ax, (0.60, y_bot), (0.66, y_mid-0.01), col=BLUE,  txt="F(x)")

    # 5) Fusion
    fusion = rbox(ax, 0.80, y_mid-0.05, 0.16, 0.16, "Fusion\n|αs S + αf e^{iθ} F|^2", fc=FUSION, ec=EDGE)
    arrow(ax, (0.72, y_mid+0.01), (0.80, y_mid+0.03), col=ORANGE, txt="αs, αf, θ")
    arrow(ax, (0.60, y_top-0.02), (0.80, y_mid+0.06), col=BLUE)   # S → fusion
    arrow(ax, (0.60, y_bot+0.02), (0.80, y_mid-0.02), col=BLUE)   # F → fusion

    # 6) Outputs
    out = rbox(ax, 0.82, y_bot-0.22, 0.14, 0.16, "Task heads\nclass / seg / reg", fc=OUT, ec=EDGE)
    arrow(ax, (0.88, y_mid-0.05), (0.89, y_bot-0.06), col=EDGE)

    # Legend
    l1 = Line2D([0],[0], color=BLUE,   lw=2)
    l2 = Line2D([0],[0], color=GREEN,  lw=2)
    l3 = Line2D([0],[0], color=ORANGE, lw=2)
    l4 = Line2D([0],[0], color=PURPLE, lw=2)
    ax.legend([l1,l2,l3,l4], ["feature", "morph", "gates/phase", "unitary"],
              loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=4,
              frameon=False, fontsize=9)

    return fig

if __name__ == "__main__":
    for theme in ("light", "dark"):
        fig = draw(theme)
        fig.savefig(f"qfm_cnn_workflow_{theme}.png",  bbox_inches="tight", dpi=300)
        fig.savefig(f"qfm_cnn_workflow_{theme}.svg",  bbox_inches="tight")
        fig.savefig(f"qfm_cnn_workflow_{theme}.pdf",  bbox_inches="tight")
        plt.close(fig)
    print("Saved: qfm_cnn_workflow_light/ dark .png .svg .pdf")
