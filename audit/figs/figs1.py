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
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.28})
ax = axs[0]
labels = ["GPT-5 nano", "DeepSeek V3", "Llama 3.3 70B", "Claude Haiku 4.5\n(мой прогон)"]
a = [27/114*100, 232/760*100, 85/292*100, 102/391*100]
b = [2/134*100, 5/617*100, 2/570*100, 0/477*100]
na = ["27 из 114", "232 из 760", "85 из 292", "102 из 391"]; nb = ["2 из 134", "5 из 617", "2 из 570", "0 из 477"]
w = 0.36
for i in range(4):
    ax.bar(i - w/2 - 0.02, a[i], w, color=C_A, zorder=3); ax.bar(i + w/2 + 0.02, b[i], w, color=C_B, zorder=3)
    ax.text(i - w/2 - 0.02, a[i] + 0.8, ru(a[i]) + "%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(i + w/2 + 0.02, b[i] + 0.8, ru(b[i]) + "%", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9, color=INK)
ax.set_ylim(0, 38); ax.set_yticks([0, 10, 20, 30]); ax.set_yticklabels(["0", "10%", "20%", "30%"])
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
fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.27)
fig.savefig("fig_quotes.png", dpi=200); plt.close(fig)

# ---- Fig C: E1 2x2 with CIs + published models on the same rows
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.3})
ax = axs[0]
conds = [("с текстом\nпромпт A", 84.2, 80.6, 87.6, C_A), ("с текстом\nпромпт B", 84.0, 80.4, 87.4, C_A),
         ("заголовки\nпромпт A", 77.0, 72.9, 80.9, C_B), ("заголовки\nпромпт B", 79.3, 75.2, 83.1, C_B)]
for i, (l, v, lo, hi, c) in enumerate(conds):
    ax.errorbar([i], [v], yerr=[[v - lo], [hi - v]], fmt="o", color=c, ms=8, lw=1.6, capsize=4, zorder=3)
    ax.text(i + 0.13, v, ru(v), va="center", ha="left", fontsize=9.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in conds], fontsize=9, color=INK)
ax.set_xlim(-0.5, 3.7); ax.set_ylim(70, 90); ax.set_yticks([70, 75, 80, 85, 90]); ax.set_yticklabels(["70%", "75%", "80%", "85%", "90%"]); style(ax)
ax.set_title("Claude Haiku 4.5, 430 вопросов: точность и 95% интервал", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
ax = axs[1]
rows = [("GPT-5 nano", 79.1, 74.4), ("DeepSeek V3", 81.6, 76.0), ("Llama 3.3 70B", 78.8, 75.8), ("Claude Haiku 4.5\n(мой прогон)", 84.0, 79.3)]
ys = [3, 2, 1, -0.15]
for (n, a_, b_), y in zip(rows, ys):
    ax.plot([b_, a_], [y, y], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
    ax.scatter([a_], [y], s=70, color=C_A, edgecolor="white", linewidth=1.3, zorder=3)
    ax.scatter([b_], [y], s=70, color=C_B, edgecolor="white", linewidth=1.3, zorder=3)
    ax.text(a_ + 0.5, y, ru(a_), va="center", ha="left", fontsize=9, color=INK2)
    ax.text(b_ - 0.5, y, ru(b_), va="center", ha="right", fontsize=9, color=INK2)
    ax.text(87.6, y, "−" + ru(a_ - b_) + " п.п.", va="center", ha="left", fontsize=9, color=INK, fontweight="bold")
ax.axhline(0.45, color=GRID, lw=0.8)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=9)
ax.set_xlim(70, 91.5); ax.set_ylim(-0.8, 3.6); ax.set_xticks([72, 76, 80, 84]); ax.set_xticklabels(["72%", "76%", "80%", "84%"])
ax.tick_params(axis="both", length=0, colors=INK2); ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ("left", "bottom"): ax.spines[s].set_visible(False)
ax.set_title("Те же 430 вопросов: с текстом и без, дебатов нет", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], marker="o", ls="", color=C_A, ms=7, markeredgecolor="white"), Line2D([], [], marker="o", ls="", color=C_B, ms=7, markeredgecolor="white")],
           ["агент видит текст источников (у статьи: архитектура A)", "агент видит только заголовки, ссылки и даты (у статьи: B, раунд 1)"],
           loc="lower center", ncol=2, frameon=False, fontsize=8.8, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.6)
fig.subplots_adjust(left=0.07, right=0.985, top=0.89, bottom=0.25)
fig.savefig("fig_e1.png", dpi=200); plt.close(fig)
print("ok")
