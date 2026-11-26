"""Figure 5. Pain change according to MRI change."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy import stats
import statsmodels.formula.api as smf
from common import (
    INK,
    INK2,
    CHRON,
    FIND,
    GCOL,
    GROUPS,
    SIDES,
    FINDINGS,
    W2,
    load,
    setup,
    save,
    panel,
    boxes,
    disc_position,
)

setup()
df = load()
df['Group'] = pd.Categorical(df.MRI_Change_Group, GROUPS)
rng = np.random.default_rng(7)

fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.62), gridspec_kw={'width_ratios': [1.25, 1]})
fig.subplots_adjust(wspace=1.32, left=0.08, right=0.66)

# a: change in VAS by direction group
ax = axs[0]
bp = boxes(ax, [df[df.Group == g].Delta_VAS.values for g in GROUPS], range(4), ['#f0f0f0'] * 4, width=0.55, alpha=1)
for i, g in enumerate(GROUPS):
    x = df[df.Group == g]
    for c in ['Acute', 'Chronic']:
        v = x[x.Chron == c].Delta_VAS
        ax.scatter(
            i + rng.uniform(-0.17, 0.17, len(v)),
            v + rng.uniform(-0.1, 0.1, len(v)),
            s=10,
            color=CHRON[c],
            alpha=0.8,
            lw=0.4,
            edgecolor='white',
            zorder=3,
        )
ax.axhline(0, color=INK2, lw=0.7, ls=(0, (3, 2)))
ax.set_xticks(
    range(4), GROUPS, fontsize=7.5, fontweight='bold', color=INK, rotation=30, ha='right', rotation_mode='anchor'
)
ax.set_ylabel('Change in VAS (final − initial)')
kw = stats.kruskal(*[df[df.Group == g].Delta_VAS for g in GROUPS]).pvalue
ax.set_ylim(-9, 10.2)
ax.set_yticks(range(-8, 9, 2))
ax.text(
    0.03, 0.985, f'Kruskal–Wallis p = {kw:.2f}', transform=ax.transAxes, va='top', ha='left', fontsize=7, color=INK2
)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.legend(
    handles=[Line2D([], [], marker='o', ls='', color=CHRON[c], ms=4.5, label=c) for c in CHRON]
    + [Line2D([], [], marker='D', ls='', mfc='white', mec=INK, ms=5, label='Mean')],
    loc='center',
    bbox_to_anchor=(0.5, 0.06),
    bbox_transform=fig.transFigure,
    ncol=3,
    handletextpad=0.2,
    columnspacing=0.8,
)
ax.set_title('Pain change by group', loc='left', color=INK, x=-0.05)
panel(ax, 'a', x=-0.36, y=1.02)

# b: differences in change in VAS
rows = []


def diff(mask, label, colour, section):
    a = df[mask].Delta_VAS
    b = df[~mask].Delta_VAS
    r = stats.ttest_ind(a, b)
    ci = r.confidence_interval()
    rows.append((section, label, a.mean() - b.mean(), ci[0], ci[1], f'{len(a)} / {len(b)}', colour))


adj = smf.ols(
    "Delta_VAS ~ C(Group, Treatment('Stable')) + I_VAS + Treatment_Duration + Chronicity + Age", data=df
).fit()
for g in GROUPS[1:]:
    k = f"C(Group, Treatment('Stable'))[T.{g}]"
    lo, hi = adj.conf_int().loc[k]
    rows.append(
        (
            'Direction of change vs stable (adjusted)',
            g,
            adj.params[k],
            lo,
            hi,
            f'{(df.Group == g).sum()} / {(df.Group == "Stable").sum()}',
            GCOL[g],
        )
    )
disc_worse = sum(disc_position(df, 'F', s) > disc_position(df, 'I', s) for s in SIDES) > 0
sec = 'Individual findings (either joint)'
for k, code in FINDINGS.items():
    if k == 'DDwR':
        continue  # DDwR transitions are included in the disc-position row
    new = sum(((df[f'I_{s}_{code}'] == 0) & (df[f'F_{s}_{code}'] == 1)) for s in SIDES) > 0
    res = sum(((df[f'I_{s}_{code}'] == 1) & (df[f'F_{s}_{code}'] == 0)) for s in SIDES) > 0
    diff(new, f'New {k}', FIND[k], sec)
    diff(res, f'Resolved {k}', FIND[k], sec)
    if k == 'DDwoR':
        diff(disc_worse, 'Worsened disc position', FIND['DDwoR'], sec)
diff(df.MRI_Progression == 1, 'Count increase in either joint', INK2, 'Count-based definition')
R = pd.DataFrame(rows, columns=['section', 'label', 'est', 'lo', 'hi', 'n', 'col'])

ax = axs[1]
bt = blended_transform_factory(ax.transAxes, ax.transData)
yv = 0
for s in R.section.unique():
    ax.text(
        -0.95,
        yv,
        s,
        fontsize=7.8,
        fontweight='bold',
        color=INK,
        va='center',
        clip_on=False,
        transform=bt,
        zorder=5,
        bbox=dict(fc='white', ec='none', pad=0.5),
    )
    yv += 1
    for _, r in R[R.section == s].iterrows():
        filled = not r.label.startswith('Resolved')
        ax.plot([r.lo, r.hi], [yv, yv], color=r.col, lw=1.4, solid_capstyle='round')
        ax.scatter(
            r.est,
            yv,
            s=26,
            marker='s' if 'adjusted' in s else 'o',
            facecolor=r.col if filled else 'white',
            edgecolor=r.col,
            lw=1.3,
            zorder=3,
        )
        ax.text(-0.04, yv, r.label, fontsize=7, color=INK, va='center', ha='right', clip_on=False, transform=bt)
        ax.text(
            1.04,
            yv,
            f'{r.est:+.2f} ({r.lo:.2f}, {r.hi:.2f})'.replace('-', '−'),
            fontsize=6.6,
            color=INK2,
            va='center',
            clip_on=False,
            transform=bt,
        )
        ax.text(2.12, yv, r.n, fontsize=6.6, color=INK2, va='center', ha='right', clip_on=False, transform=bt)
        yv += 1
    yv += 0.4
ax.text(1.04, -0.9, 'Diff. (95% CI)', fontsize=6.8, color=INK, clip_on=False, transform=bt)
ax.text(2.12, -0.9, 'n', fontsize=6.8, color=INK, ha='right', clip_on=False, transform=bt)
ax.axvline(0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=0)
ax.set_xlim(-4.2, 4.4)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_ylim(yv - 0.2, -1.8)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Difference in ΔVAS (points)\n← greater pain reduction     less pain reduction →', fontsize=7.2)
ax.text(
    -0.95, 1.02, 'Difference in pain change by MRI change', transform=ax.transAxes, fontsize=8.5, color=INK, va='bottom'
)
panel(ax, 'b', x=-1.07, y=1.02)

key = [
    Line2D([], [], marker='s', ls='-', color=INK2, mfc=INK2, ms=5, lw=1.3, label='Group vs stable (adjusted)'),
    Line2D([], [], marker='o', ls='-', color=INK2, mfc=INK2, ms=5, lw=1.3, label='New or worsened finding'),
    Line2D([], [], marker='o', ls='-', color=INK2, mfc='white', mew=1.3, ms=5, lw=1.3, label='Resolved finding'),
]
YL = -0.09
pa = axs[0].get_position()
pb = axs[1].get_position()
ax.legend(
    handles=key,
    loc='center',
    bbox_to_anchor=(pb.x0 + pb.width * 0.66, YL),
    bbox_transform=fig.transFigure,
    ncol=3,
    frameon=False,
    handlelength=1.5,
    handletextpad=0.3,
    columnspacing=0.7,
)
axs[0].get_legend().set_bbox_to_anchor((pa.x0 + pa.width * 0.5, YL), transform=fig.transFigure)
save(fig, 'Figure5')
R.drop(columns='col').round(2).to_csv('output/Figure5b_values.csv', index=False)
