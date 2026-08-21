import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


unstable_counts = np.arange(33)
n_systems = np.array([
    790, 101, 64, 48, 34, 38, 38, 40, 40, 33, 32,
    36, 36, 40, 37, 44, 42, 33, 48, 34, 36, 51,
    45, 58, 66, 86, 86, 103, 138, 178, 294, 557, 1692,
])

total_systems = int(n_systems.sum())
assert total_systems == 4998
percentages = n_systems / total_systems * 100.0

if sns is not None:
    sns.set_theme(style="ticks", context="paper")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(5.0, 2.8))

base_color = "#7A9CC6"
highlight_color = "#D55E00"
colors = [base_color] * 33
colors[0] = highlight_color
colors[32] = highlight_color

ax.bar(
    unstable_counts,
    percentages,
    width=0.82,
    color=colors,
    edgecolor="none",
)

for x in (0, 32):
    ax.text(
        x,
        percentages[x] + 0.7,
        f"{percentages[x]:.2f}%",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
    )

ax.text(
    0.52,
    0.94,
    "Overall unstable variations: 68.49%",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=7,
)

ax.set_xlabel("Number of unstable variations")
ax.set_ylabel("Systems (%)")
ax.set_xlim(-0.7, 32.7)
ax.set_ylim(0, 37)
ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32])
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
ax.grid(axis="x", visible=False)

if sns is not None:
    sns.despine(ax=ax)
else:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("unstable_variations_distribution.pdf", bbox_inches="tight")
fig.savefig("unstable_variations_distribution.png", dpi=400, bbox_inches="tight")
plt.close(fig)

print(f"total systems: {total_systems}")
print(f"all-stable systems: {n_systems[0]} ({percentages[0]:.2f}%)")
print(f"all-unstable systems: {n_systems[32]} ({percentages[32]:.2f}%)")
