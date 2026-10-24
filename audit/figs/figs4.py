import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import norm as N01
from scipy import integrate
BASE = "."
sys.path.insert(0, f"{BASE}/analysis")
from kota import load, M
for f in ["PT_Sans-Web-Regular.ttf", "PT_Sans-Web-Bold.ttf", "PT_Sans-Web-Italic.ttf"]:
    fm.fontManager.addfont(f"{BASE}/fonts/{f}")
plt.rcParams.update({"font.family": "PT Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983"
A, B, D = load(); Au = A.drop_duplicates("key")
ok = Au.maj_ok.values.astype(float)
def curve(score, ok):
    o = np.argsort(-score, kind="stable"); acc = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1); cov = np.arange(1, len(ok) + 1) / len(ok); return cov, acc
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"wspace": 0.2, "width_ratios": [1, 1.12]})
ax = axs[0]
sigs = [("единогласие + средняя уверенность (правило статьи)", Au.unan.values + Au.mean_conf.values, ok, C1, 2.4),
        ("только средняя уверенность трёх моделей", Au.mean_conf.values, ok, C3, 1.6),
        ("уверенность GPT-5 nano, его же ответ", Au.gpt4o_confidence.values.astype(float), Au.gpt4o_ok.values.astype(float), C2, 1.6),
        ("уверенность DeepSeek V3, его же ответ", Au.deepseek_confidence.values.astype(float), Au.deepseek_ok.values.astype(float), C4, 1.3)]
for lab, s, o_, c, lw in sigs:
    cov, acc = curve(s, o_); mk = cov >= 0.05
    ax.plot(100 * cov[mk], 100 * acc[mk], color=c, lw=lw, label=lab)
# honest CV points
for cv, ac, lab in [(42.5, 98.57, "цель 99%"), (54.4, 96.81, "цель 97%"), (62.8, 94.75, "цель 95%")]:
    ax.scatter([cv], [ac], s=55, color="white", edgecolor=INK, linewidth=1.5, zorder=5)
    ax.annotate(lab, (cv, ac), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=INK)
ax.set_xlim(5, 100); ax.set_ylim(80, 100.5); ax.set_xlabel("Доля вопросов, решённых автоматически", color=INK2); ax.set_xticks([20, 40, 60, 80, 100]); ax.set_xticklabels(["20%", "40%", "60%", "80%", "100%"])
ax.set_yticks([80, 85, 90, 95, 100]); ax.set_yticklabels(["80%", "85%", "90%", "95%", "100%"]); ax.tick_params(length=0, colors=INK2); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.set_title("Точность на автоматически решённой части, 1 153 вопроса", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
ax.legend(frameon=False, fontsize=8.3, loc="lower left", labelcolor=INK2, handlelength=1.6)
ax.text(64, 99.3, "кружки: порог подобран\nна других рынках (5 блоков)", fontsize=8.3, color=INK2, va="center")
# right: majority of N under correlated errors
ax = axs[1]
p = 0.819; t = N01.ppf(p)
def maj(Nv, rho):
    if rho == 0:
        from scipy.stats import binom
        return 1 - binom.cdf(Nv // 2, Nv, p)
    f = lambda z: (1 - __import__("scipy.stats").stats.binom.cdf(Nv // 2, Nv, N01.cdf((t - np.sqrt(rho) * z) / np.sqrt(1 - rho)))) * N01.pdf(z)
    return integrate.quad(f, -8, 8)[0]
Ns = [1, 3, 5, 7, 9, 15, 25, 51, 101]
for rho, c, lab in [(0.0, C4, "независимые ошибки"), (0.2, C3, "корреляция 0,2"), (0.4, C2, "корреляция 0,4"), (0.845, C1, "корреляция 0,845,\nоценка по данным статьи")]:
    ys = [100 * maj(n, rho) for n in Ns]
    ax.plot(range(len(Ns)), ys, color=c, lw=2.2 if rho == 0.845 else 1.6, marker="o", ms=4.5, label=lab)
    ax.text(len(Ns) - 0.8, ys[-1] - (0.9 if rho == 0.845 else 0), f"{ys[-1]:.1f}".replace(".", ",") + "%  " + lab, va="center", fontsize=8.6, color=c, linespacing=1.2)
ax.scatter([1], [82.74], s=70, color="white", edgecolor=INK, linewidth=1.6, zorder=6); ax.annotate("наблюдалось: 82,7%", (1, 82.74), xytext=(10, -14), textcoords="offset points", fontsize=8.5, color=INK)
ax.set_xticks(range(len(Ns))); ax.set_xticklabels([str(n) for n in Ns]); ax.set_xlim(-0.3, len(Ns) + 3.6); ax.set_ylim(80, 101)
ax.set_yticks([80, 85, 90, 95, 100]); ax.set_yticklabels(["80%", "85%", "90%", "95%", "100%"]); ax.tick_params(length=0, colors=INK2); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.set_xlabel("Число моделей в голосовании", color=INK2)
ax.set_title("Точность большинства при коррелированных ошибках", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
fig.subplots_adjust(left=0.065, right=0.985, top=0.9, bottom=0.14)
fig.savefig("fig_newcalc.png", dpi=200); print("ok")
