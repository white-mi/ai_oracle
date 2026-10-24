import sys, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
BASE = "."
sys.path.insert(0, f"{BASE}/analysis")
from kota import load, M
for f in ["PT_Sans-Web-Regular.ttf", "PT_Sans-Web-Bold.ttf", "PT_Sans-Web-Italic.ttf"]:
    fm.fontManager.addfont(f"{BASE}/fonts/{f}")
plt.rcParams.update({"font.family": "PT Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983"
def ru(x, nd=1): return f"{x:.{nd}f}".replace(".", ",")
P = pickle.load(open(f"{BASE}/exp/e2_signals.pkl", "rb")); SIG, y, keys = P["SIG"], P["y"], P["keys"]
A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").loc[keys]
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"wspace": 0.2, "width_ratios": [1, 1.05]})
ax = axs[0]
rows = [("Постоянный ответ NO", 59.5, None, None, C4), ("Haiku 4.5 без свидетельств", 63.7, 59.1, 68.3, C2), ("Большинство трёх моделей статьи,\nсо свидетельствами", 81.6, None, None, C4),
        ("Haiku 4.5 со свидетельствами", 84.2, 80.6, 87.6, C1), ("Sonnet 5.5 без свидетельств\n(знает исходы из обучения)", 86.0, 82.7, 89.4, C2)]
for i, (lab, v, lo, hi, c) in enumerate(rows):
    yy = len(rows) - 1 - i
    ax.barh(yy, v, height=0.56, color=c, zorder=2)
    if lo is not None: ax.plot([lo, hi], [yy, yy], color=INK, lw=1.3, zorder=3)
    ax.text((hi if hi else v) + 1.2, yy, ru(v) + "%", va="center", fontsize=9.5, color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color=INK)
ax.set_xlim(0, 104); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"]); ax.tick_params(length=0, colors=INK2); ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.set_title("Точность на 430 вопросах", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
ax = axs[1]
def curve(score, ok):
    o = np.argsort(-score, kind="stable"); acc = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1); cov = np.arange(1, len(ok) + 1) / len(ok); return cov, acc
v = SIG["verbalized confidence (decision + confidence, with reasoning)"]; b = SIG["bare P(YES), no reasoning"]; e = SIG["bare P(evidence states it occurred), no reasoning"]
series = [("три модели статьи: единогласие + уверенность", Au.unan.values + Au.mean_conf.values, Au.maj_ok.values.astype(float), C4, 1.8),
          ("Haiku: названная уверенность, один вызов", np.nan_to_num(v["conf"]), np.nan_to_num(v["ok"]), C1, 2.3),
          ("Haiku: только вероятность YES, без рассуждения", np.nan_to_num(b["conf"], nan=0.5), np.nan_to_num(b["ok"]), C3, 1.8),
          ("Haiku: вероятность «свидетельства сообщают об исходе»", np.nan_to_num(e["conf"], nan=0.5), np.nan_to_num(e["ok"]), C2, 1.5)]
for lab, s, ok, c, lw in series:
    cov, acc = curve(s, ok); mk = cov >= 0.08
    ax.plot(100 * cov[mk], 100 * acc[mk], color=c, lw=lw, label=lab)
ax.scatter([70.0], [90.7], s=60, color="white", edgecolor=INK, linewidth=1.5, zorder=5); ax.annotate("третий ответ «недостаточно»:\n90,7% при покрытии 70%", (70.0, 90.7), xytext=(63, 97.6), textcoords="data", fontsize=8.3, color=INK, linespacing=1.15, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlim(8, 100); ax.set_ylim(80, 100.6); ax.set_xticks([20, 40, 60, 80, 100]); ax.set_xticklabels(["20%", "40%", "60%", "80%", "100%"]); ax.set_yticks([80, 85, 90, 95, 100]); ax.set_yticklabels(["80%", "85%", "90%", "95%", "100%"])
ax.tick_params(length=0, colors=INK2); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.set_xlabel("Доля вопросов, решённых автоматически", color=INK2)
ax.set_title("Точность на автоматически решённой части", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
ax.legend(frameon=False, fontsize=8.2, loc="lower left", labelcolor=INK2, handlelength=1.5)
fig.subplots_adjust(left=0.235, right=0.97, top=0.9, bottom=0.14)
fig.savefig("fig_e2.png", dpi=200); print("ok")
