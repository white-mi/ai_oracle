import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BASE = "."
for f in ["PT_Sans-Web-Regular.ttf", "PT_Sans-Web-Bold.ttf", "PT_Sans-Web-Italic.ttf"]:
    fm.fontManager.addfont(f"{BASE}/fonts/{f}")
plt.rcParams.update({"font.family": "PT Sans", "font.size": 10})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
C_A, C_B, C_G = "#2a78d6", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(figsize=(9.6, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
def box(x, y, w, h, text, fc="#ffffff", ec=GRID, tc=INK, fs=9.5, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=3, linespacing=1.25)
def arr(x1, y1, x2, y2, c=MUTED, ls="-", lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=c, lw=lw, ls=ls, zorder=1, shrinkA=0, shrinkB=0))
# left column: shared part
box(1, 34, 17, 9, "Вопрос рынка\nи правила\n(KalshiBench v2)", fc="#f6f5f1")
box(1, 19, 17, 9, "Поиск Exa\nдо 10 источников,\nдата ≤ закрытие + 1 день", fc="#f6f5f1")
box(1, 4, 17, 9, "Пакет свидетельств:\nзаголовок, ссылка, дата,\nвыдержки до 2 000 знаков", fc="#f6f5f1")
arr(9.5, 34, 9.5, 28.6); arr(9.5, 19, 9.5, 13.6)
# Architecture A
ax.text(26, 45.5, "Архитектура A: независимое голосование", fontsize=10.5, fontweight="bold", color=C_A, va="center")
for i, m in enumerate(["GPT-5 nano", "DeepSeek V3", "Llama 3.3 70B"]):
    box(26, 37.5 - i * 4.6, 17, 3.6, m, ec=C_A, fs=9)
    arr(43.6, 39.3 - i * 4.6, 49.5, 34.7, c=C_A)
box(50, 31.5, 19, 6.4, "Голосование:\nбольшинство или\nвес по уверенности", ec=C_A, fs=9)
box(76, 31.5, 22, 6.4, "Итог YES или NO\n82,9% и 83,4%", fc=C_A, ec=C_A, tc="white", fs=9.5, bold=True)
arr(69.6, 34.7, 75.6, 34.7, c=C_A)
arr(18.6, 8.5, 25.5, 34.7, c=C_A)
# Architecture B
ax.text(26, 22.5, "Архитектура B: дебаты в два раунда", fontsize=10.5, fontweight="bold", color=C_B, va="center")
box(26, 11, 17, 8, "Раунд 1:\nтри модели\nотвечают отдельно", ec=C_B, fs=9)
box(50, 11, 19, 8, "Раунд 2:\nкаждая видит ответы\nдвух других\nи может передумать", ec=C_B, fs=9)
box(76, 11, 22, 8, "Большинство раунда 2\n76,1%", fc=C_B, ec=C_B, tc="white", fs=9.5, bold=True)
arr(43.6, 15, 49.6, 15, c=C_B); arr(69.6, 15, 75.6, 15, c=C_B)
arr(18.6, 8.5, 25.5, 15, c=C_B, ls=(0, (4, 3)))
ax.text(26, 5.2, "Пунктир: по тексту статьи агенты B получают тот же пакет.\nПо коду к ним доходят только заголовок, ссылка и дата (вкладка «Проблемы статьи»).", fontsize=8.8, color=INK2, va="center", linespacing=1.3)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig("fig_pipeline.png", dpi=200); print("ok")
