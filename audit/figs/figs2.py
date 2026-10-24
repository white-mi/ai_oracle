import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
BASE = "."
for f in ["PT_Sans-Web-Regular.ttf", "PT_Sans-Web-Bold.ttf", "PT_Sans-Web-Italic.ttf"]:
    fm.fontManager.addfont(f"{BASE}/fonts/{f}")
plt.rcParams.update({"font.family": "PT Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
C_A, C_R1, C_R2 = "#2a78d6", "#eb6834", "#1baf7a"
def ru(x, nd=1): return f"{x:.{nd}f}".replace(".", ",")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.42})
rows = [("GPT-5 nano", 82.34, 75.19, 74.35), ("DeepSeek V3", 82.42, 76.20, 76.70), ("Llama 3.3 70B", 81.24, 76.20, 76.37), ("Большинство трёх", 82.93, 76.62, 76.11)]
ys = [3, 2, 1, -0.2]
for (name, a, r1, r2), y in zip(rows, ys):
    ax1.plot([min(r1, r2), a], [y, y], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
    for v, c, z in [(a, C_A, 3), (r1, C_R1, 4), (r2, C_R2, 5)]:
        ax1.scatter([v], [y], s=70, color=c, edgecolor="white", linewidth=1.3, zorder=z)
    ax1.text(a + 0.55, y, ru(a), va="center", ha="left", color=INK2, fontsize=9)
    ax1.text(min(r1, r2) - 0.55, y, f"{ru(r1)} / {ru(r2)}", va="center", ha="right", color=INK2, fontsize=9)
ax1.axhline(0.4, color=GRID, lw=0.8)
ax1.set_yticks(ys); ax1.set_yticklabels([r[0] for r in rows], color=INK); ax1.get_yticklabels()[-1].set_fontweight("bold")
ax1.set_xlim(68.6, 85.8); ax1.set_ylim(-0.9, 3.6); ax1.set_xticks([72, 76, 80, 84]); ax1.set_xticklabels(["72%", "76%", "80%", "84%"], color=MUTED)
ax1.tick_params(axis="both", length=0); ax1.grid(axis="x", color=GRID, lw=0.7); ax1.set_axisbelow(True)
for s in ("left", "bottom"): ax1.spines[s].set_visible(False)
ax1.set_title("Точность: падение происходит до дебатов", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
groups = [("Исход YES\n(488 строк)", (67.3, 49.4, 47.7)), ("Исход NO\n(701 строка)", (93.7, 95.6, 95.9))]
h, gap = 0.24, 0.035
for gi, (gname, vals) in enumerate(groups):
    y0 = 1 - gi
    for k, (v, c) in enumerate(zip(vals, (C_A, C_R1, C_R2))):
        y = y0 + (1 - k) * (h + gap)
        ax2.add_patch(FancyBboxPatch((0, y - h / 2), v, h, boxstyle="round,pad=0,rounding_size=0.06", mutation_aspect=0.012, fc=c, ec="none", zorder=2))
        ax2.text(v + 1.5, y, ru(v) + "%", va="center", ha="left", color=INK2, fontsize=9)
ax2.set_yticks([1, 0]); ax2.set_yticklabels([g[0] for g in groups], color=INK)
ax2.set_xlim(0, 112); ax2.set_ylim(-0.48, 1.48); ax2.set_xticks([0, 25, 50, 75, 100]); ax2.set_xticklabels(["0", "25%", "50%", "75%", "100%"], color=MUTED)
ax2.tick_params(axis="both", length=0); ax2.grid(axis="x", color=GRID, lw=0.7); ax2.set_axisbelow(True)
for s in ("left", "bottom"): ax2.spines[s].set_visible(False)
ax2.set_title("Верные ответы большинства по истинному исходу", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
handles = [Line2D([], [], marker="o", ls="", color=c, markersize=7, markeredgecolor="white") for c in (C_A, C_R1, C_R2)]
fig.legend(handles, ["Архитектура A (независимо)", "B, раунд 1 (до обмена аргументами)", "B, раунд 2 (после дебатов)"], loc="lower center", ncol=3, frameon=False, fontsize=9,
           handletextpad=0.3, columnspacing=1.8, bbox_to_anchor=(0.5, 0.0), labelcolor=INK2)
fig.subplots_adjust(left=0.15, right=0.985, top=0.89, bottom=0.2)
fig.savefig("fig_drop.png", dpi=200); plt.close(fig); print("ok")
