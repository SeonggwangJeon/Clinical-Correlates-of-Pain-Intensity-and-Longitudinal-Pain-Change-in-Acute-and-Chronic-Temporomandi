"""Figure 4. Pain trajectories according to the direction of MRI change."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import INK, CHRON, GCOL, GROUPS, LINE, W2, load, setup, save, panel, trajectories

setup()
df = load()
df['Group'] = df.MRI_Change_Group

fig, axs = plt.subplots(1, 4, figsize=(W2, W2 * 0.38), sharey=True)
fig.subplots_adjust(wspace=0.16)
for ax, g, let in zip(axs, GROUPS, 'abcd'):
    x = df[df.Group == g]
    trajectories(ax, x)
    for c in ['Acute', 'Chronic']:
        xc = x[x.Chron == c]
        ax.plot(
            [0, 1],
            [xc.I_VAS.mean(), xc.F_VAS.mean()],
            color=CHRON[c],
            lw=2.2,
            marker='o',
            ms=5,
            mec='white',
            mew=1,
            zorder=3,
        )
    ax.set_xticks([0, 1], ['Initial', 'Final'], fontweight='bold', color=INK, fontsize=8)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.5, 10.5)
    ax.set_title(
        f'{g} (n = {len(x)})\nΔVAS {x.Delta_VAS.mean():.2f} ± {x.Delta_VAS.std():.2f}'.replace('-', '−'),
        loc='center',
        color=INK,
        fontsize=7.8,
        pad=9,
    )
    ax.add_patch(plt.Rectangle((0, 1.005), 1, 0.022, transform=ax.transAxes, color=GCOL[g], clip_on=False, lw=0))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    panel(ax, let, x=-0.06 if g != 'Stable' else -0.32, y=1.2)
axs[0].set_ylabel('Pain intensity (VAS, 0–10)')
fig.legend(
    handles=[
        Line2D([], [], color=CHRON[c], lw=2.2, marker='o', ms=5, mec='white', label=f'{c} (group mean)') for c in CHRON
    ]
    + [
        Line2D(
            [], [], color=LINE, lw=0.8, alpha=0.7, label='Patients (thicker line = more patients with the same values)'
        )
    ],
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(0.5, -0.1),
)
save(fig, 'Figure4')
