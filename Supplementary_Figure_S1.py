"""Supplementary Figure S1. Pain change according to count-based MRI progression."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from common import INK, INK2, W2, load, setup, save, panel, boxes, trajectories

setup()
df = load()
df['Status'] = np.where(df.MRI_Progression == 1, 'Progression', 'No progression')
COL = {'No progression': '#56a0d3', 'Progression': '#e8743b'}

fig = plt.figure(figsize=(W2, W2 * 0.42))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.35)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharey=axA)
axC = fig.add_subplot(gs[2])

# a, b: individual trajectories with group mean and 95% CI
for ax, c, let in [(axA, 'No progression', 'a'), (axB, 'Progression', 'b')]:
    x = df[df.Status == c]
    col = COL[c]
    trajectories(ax, x)
    m = [x.I_VAS.mean(), x.F_VAS.mean()]
    tq = stats.t.ppf(0.975, len(x) - 1)
    ci = [tq * stats.sem(x.I_VAS), tq * stats.sem(x.F_VAS)]
    ax.plot([0, 1], m, color=col, lw=2.4, zorder=4)
    ax.errorbar([0, 1], m, yerr=ci, fmt='o', color=col, ms=5.5, mec='white', mew=1, elinewidth=1.4, capsize=0, zorder=5)
    p = stats.ttest_rel(x.I_VAS, x.F_VAS).pvalue
    ax.set_title(f'{c} (n = {len(x)})', loc='left', color=INK)
    ax.text(
        0.5,
        1.0,
        f'ΔVAS {x.Delta_VAS.mean():.2f} ± {x.Delta_VAS.std():.2f}; p = {p:.4f}'.replace('-', '−'),
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=7,
        color=INK2,
        bbox=dict(fc='white', ec='none', pad=1),
    )
    ax.set_xticks([0, 1], ['Initial', 'Final'], fontweight='bold', color=INK, fontsize=8)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.5, 11)
    ax.set_yticks(range(0, 11, 2))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    panel(ax, let, x=-0.2 if c == 'No progression' else -0.12)
axA.set_ylabel('Pain intensity (VAS, 0–10)')
plt.setp(axB.get_yticklabels(), visible=False)

# c: change in VAS by chronicity
rng = np.random.default_rng(3)
groups = ['No progression', 'Progression']
boxes(axC, [df[df.Status == c].Delta_VAS.values for c in groups], [0, 1], [COL[c] for c in groups])
for i, c in enumerate(groups):
    v = df[df.Status == c].Delta_VAS
    axC.scatter(
        i + rng.uniform(-0.15, 0.15, len(v)),
        v + rng.uniform(-0.08, 0.08, len(v)),
        s=10,
        color=COL[c],
        alpha=0.75,
        lw=0.4,
        edgecolor='white',
        zorder=3,
    )
a = df[df.Status == 'Progression'].Delta_VAS
b = df[df.Status == 'No progression'].Delta_VAS
r = stats.ttest_ind(a, b)
ci = r.confidence_interval()
axC.axhline(0, color=INK2, lw=0.7, ls=(0, (3, 2)))
axC.set_xticks([0, 1], ['No\nprogression', 'Progression'], fontweight='bold', color=INK, fontsize=8)
axC.set_xlim(-0.55, 1.55)
axC.set_ylabel('Change in VAS (final − initial)')
axC.yaxis.grid(True)
axC.set_axisbelow(True)
axC.set_title('Pain change by MRI progression', loc='left', color=INK)
axC.text(
    0.5,
    0.015,
    f'Difference (progression − no progression)\n{a.mean() - b.mean():.2f} (95% CI {ci[0]:.2f} to {ci[1]:.2f}); p = {r.pvalue:.3f}'.replace(
        '-', '−'
    ),
    transform=axC.transAxes,
    ha='center',
    va='bottom',
    fontsize=6.6,
    color=INK2,
    bbox=dict(fc='white', ec='none', pad=1),
)
axC.set_ylim(-11, 9)
axC.set_yticks(range(-8, 9, 2))
panel(axC, 'c', x=-0.36)

fig.legend(
    handles=[
        Line2D([], [], color=INK2, lw=2.4, marker='o', ms=5, mec='white', label='Group mean (95% CI)'),
        Line2D([], [], marker='D', ls='', mfc='white', mec=INK, ms=5, label='Mean (box: median and IQR)'),
        Line2D(
            [], [], color=INK2, lw=0.8, alpha=0.5, label='Patients (thicker line = more patients with the same values)'
        ),
    ],
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(0.5, -0.08),
)
save(fig, 'Supplementary_Figure_S1')
