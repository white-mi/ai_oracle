import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
BASE = "."
for f in ["PT_Sans-Web-Regular.ttf", "PT_Sans-Web-Bold.ttf", "PT_Sans-Web-Italic.ttf"]:
    fm.fontManager.addfont(f"{BASE}/fonts/{f}")
plt.rcParams.update({"font.family": "PT Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
C_A, C_B, C_G = "#2a78d6", "#eb6834", "#1baf7a"
def ru(x, nd=1): return f"{x:.{nd}f}".replace(".", ",")
def style(ax, xgrid=False, ygrid=True):
    ax.tick_params(axis="both", length=0, colors=INK2)
    if ygrid: ax.grid(axis="y", color=GRID, lw=0.7)
    if xgrid: ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)

# ---- Fig B: where quotations come from
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.28})
ax = axs[0]
labels = ["GPT-5 nano\n3 из 19 / 0 из 32", "DeepSeek V3\n48 из 142 / 2 из 128", "Llama 3.3 70B\n17 из 40 / 1 из 107", "Claude Haiku 4.5\n(мой прогон)\n102 из 391 / 0 из 477"]
a = [3/19*100, 48/142*100, 17/40*100, 102/391*100]
b = [0/32*100, 2/128*100, 1/107*100, 0/477*100]
na = ["27 из 114", "232 из 760", "85 из 292", "102 из 391"]; nb = ["2 из 134", "5 из 617", "2 из 570", "0 из 477"]
w = 0.36
for i in range(4):
    ax.bar(i - w/2 - 0.02, a[i], w, color=C_A, zorder=3); ax.bar(i + w/2 + 0.02, b[i], w, color=C_B, zorder=3)
    ax.text(i - w/2 - 0.02, a[i] + 0.8, ru(a[i]) + "%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(i + w/2 + 0.02, b[i] + 0.8, ru(b[i]) + "%", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.3, color=INK)
ax.set_ylim(0, 48); ax.set_yticks([0, 10, 20, 30, 40]); ax.set_yticklabels(["0", "10%", "20%", "30%", "40%"])
ax.axvline(2.5, color=GRID, lw=1, ls=(0, (3, 3)))
style(ax)
ax.set_title("Доля прямых цитат, взятых из текста источников", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
fig.legend(handles=[Patch(color=C_A, label="архитектура A (у Haiku: условие с текстом источников)"), Patch(color=C_B, label="архитектура B, раунд 1 (у Haiku: условие «только заголовки»)")],
          frameon=False, fontsize=8.8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), labelcolor=INK2, handlelength=1.2, columnspacing=1.8)
ax = axs[1]
labs = ["хотя бы один агент пишет,\nчто текста источника нет", "Llama: уверенность\nровно 0"]
va = [3/1189*100, 63/1189*100]; vb = [98/1189*100, 362/1189*100]; ta = ["3", "63"]; tb = ["98", "362"]
for i in range(2):
    ax.bar(i - w/2 - 0.02, va[i], w, color=C_A, zorder=3); ax.bar(i + w/2 + 0.02, vb[i], w, color=C_B, zorder=3)
    ax.text(i - w/2 - 0.02, va[i] + 0.8, ta[i], ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(i + w/2 + 0.02, vb[i] + 0.8, tb[i], ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(range(2)); ax.set_xticklabels(labs, fontsize=9, color=INK)
ax.set_ylim(0, 38); ax.set_yticks([0, 10, 20, 30]); ax.set_yticklabels(["0", "10%", "20%", "30%"]); style(ax)
ax.set_title("Доля строк из 1 189 (над столбцом число строк)", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.31)
fig.savefig("fig_quotes2.png", dpi=200); plt.close(fig)


print('ok')
