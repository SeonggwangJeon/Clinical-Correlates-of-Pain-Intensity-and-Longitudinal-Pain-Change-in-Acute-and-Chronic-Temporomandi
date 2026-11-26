"""Figure 3. MRI findings at the initial and final examinations and changes between them."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from common import (
    INK,
    INK2,
    CHRON,
    FIND,
    DISC,
    FINDINGS,
    GROUPS,
    SIDES,
    W2,
    load,
    setup,
    save,
    panel,
    joint,
    disc_position,
)

setup()
df = load()
df['Group'] = df.MRI_Change_Group

fig, axs = plt.subplots(2, 2, figsize=(W2, W2 * 0.78))
fig.subplots_adjust(wspace=0.78, hspace=0.55)
names = list(FINDINGS)[::-1]
y = np.arange(len(names))

# a: prevalence at the two examinations (178 joints)
ax = axs[0, 0]
for i, k in enumerate(names):
    a, b = joint(df, 'I', FINDINGS[k]).mean() * 100, joint(df, 'F', FINDINGS[k]).mean() * 100
    ax.plot([a, b], [i, i], color=FIND[k], lw=2, solid_capstyle='round', zorder=1)
    ax.scatter(a, i, s=42, facecolor='white', edgecolor=FIND[k], lw=1.6, zorder=3)
    ax.scatter(b, i, s=42, color=FIND[k], edgecolor='white', lw=1, zorder=3)
    ax.text(63, i, f'{a:.0f}% → {b:.0f}%', va='center', ha='left', fontsize=7, color=INK2, clip_on=False)
ax.set_yticks(y, names, fontweight='bold', color=INK, fontsize=8)
ax.set_xlim(0, 80)
ax.set_ylim(-0.6, len(names) - 0.4)
ax.set_xticks([0, 20, 40, 60])
ax.set_xlabel('Joints with finding (%, n = 178)')
ax.xaxis.grid(True)
ax.set_axisbelow(True)
ax.legend(
    handles=[
        Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, ms=5.5, label='Initial MRI'),
        Line2D([], [], marker='o', ls='', mfc=INK2, mec='white', ms=6, label='Final MRI'),
    ],
    loc='lower left',
    bbox_to_anchor=(0, 1.0),
    ncol=2,
    handletextpad=0.2,
    columnspacing=1,
)
ax.set_title('Prevalence at initial and final MRI', loc='left', color=INK, pad=16)
panel(ax, 'a', x=-0.42, y=1.13)

# b: new, resolved and persistent findings
ax = axs[0, 1]
for i, k in enumerate(names):
    a, b = joint(df, 'I', FINDINGS[k]), joint(df, 'F', FINDINGS[k])
    new = ((a == 0) & (b == 1)).sum()
    res = ((a == 1) & (b == 0)).sum()
    per = ((a == 1) & (b == 1)).sum()
    ax.barh(i, new, height=0.56, color=FIND[k], edgecolor='white', lw=1)
    ax.barh(i, -res, height=0.56, color=FIND[k], alpha=0.38, edgecolor='white', lw=1, hatch='////')
    ax.text(new + 1, i, str(new), va='center', fontsize=7, color=INK)
    ax.text(-res - 1, i, str(res), va='center', ha='right', fontsize=7, color=INK)
    ax.text(54, i, str(per), va='center', ha='center', fontsize=7, color=INK2, clip_on=False)
ax.text(54, len(names) - 0.45, 'Persistent', ha='center', fontsize=7, color=INK2, clip_on=False)
ax.axvline(0, color=INK2, lw=0.6)
ax.set_yticks(y, names, fontweight='bold', color=INK, fontsize=8)
ax.set_xlim(-28, 44)
ax.set_ylim(-0.6, len(names) - 0.1)
ax.set_xticks([-20, -10, 0, 10, 20, 30], ['20', '10', '0', '10', '20', '30'])
ax.text(-2, len(names) - 0.45, '← Resolved', ha='right', fontsize=7, color=INK2)
ax.text(2, len(names) - 0.45, 'New →', ha='left', fontsize=7, color=INK2)
ax.set_xlabel('Joints (n)')
ax.set_title('Joint-level changes between examinations', loc='left', color=INK, pad=16)
panel(ax, 'b', x=-0.42, y=1.13)

# c: disc position transitions
ax = axs[1, 0]
states = ['DDwoR', 'DDwR', 'Normal']
lab = {0: 'Normal', 1: 'DDwR', 2: 'DDwoR'}
d1 = np.concatenate([[lab[v] for v in disc_position(df, 'I', s)] for s in SIDES])
d2 = np.concatenate([[lab[v] for v in disc_position(df, 'F', s)] for s in SIDES])
gap = 4
w = 0.16


def stack(d):
    pos = {}
    y0 = 0
    for s in states:
        n = (d == s).sum()
        pos[s] = [y0, y0 + n]
        y0 += n + gap
    return pos


P1, P2 = stack(d1), stack(d2)
c1 = {k: v[0] for k, v in P1.items()}
c2 = {k: v[0] for k, v in P2.items()}
xs = np.linspace(w, 1 - w, 40)
smooth = lambda t: 3 * t**2 - 2 * t**3
labels = []
for a in states:
    for b in states:
        n = ((d1 == a) & (d2 == b)).sum()
        if n == 0:
            continue
        y1, y2 = c1[a], c2[b]
        c1[a] += n
        c2[b] += n
        top = y1 + (y2 - y1) * smooth((xs - w) / (1 - 2 * w))
        poly = np.vstack([np.c_[xs, top], np.c_[xs[::-1], (top + n)[::-1]]])
        ax.add_patch(Polygon(poly, color=DISC[a], alpha=0.5 if a != b else 0.28, lw=0))
        if a != b:
            labels.append((y1 + n / 2, f'→ {b} {n}', a))
for P, x0, title in [(P1, 0, 'Initial'), (P2, 1 - w, 'Final')]:
    for s, (lo, hi) in P.items():
        ax.add_patch(Rectangle((x0, lo), w, hi - lo, color=DISC[s], lw=0))
        ax.text(
            x0 + w / 2,
            (lo + hi) / 2,
            f'{s}\n{hi - lo}',
            ha='center',
            va='center',
            fontsize=6.5,
            color='white' if s != 'Normal' else INK,
            fontweight='bold',
        )
    ax.text(x0 + w / 2, -4, title, ha='center', va='bottom', fontsize=7.5, color=INK2)
for a in states:
    items = sorted([l for l in labels if l[2] == a])
    lo, hi = P1[a]
    for k, (yy, txt, _) in enumerate(items):
        y_lab = max(yy, lo + 6) if len(items) == 1 else lo + (hi - lo) * (k + 1) / (len(items) + 1)
        if a == 'DDwoR':
            y_lab = hi - 22 + 14 * k
        ax.annotate(
            txt,
            xy=(w + 0.02, yy),
            xytext=(w + 0.05, y_lab),
            fontsize=6.3,
            color=INK,
            va='center',
            arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5),
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.9),
        )
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(max(P1['Normal'][1], P2['Normal'][1]) + 2, -12)
ax.axis('off')
ax.set_title('Disc position transitions (joints, n = 178)', loc='left', color=INK)
panel(ax, 'c', x=-0.08, y=1.02)

# d: direction of MRI change by chronicity
ax = axs[1, 1]
yy = np.arange(4)[::-1]
h = 0.36
for c, off in [('Acute', h / 2), ('Chronic', -h / 2)]:
    n = [((df.Group == g) & (df.Chron == c)).sum() for g in GROUPS]
    tot = (df.Chron == c).sum()
    ax.barh(yy + off, n, height=h, color=CHRON[c], edgecolor='white', lw=1, label=f'{c} (n = {tot})')
    for yi, v in zip(yy + off, n):
        ax.text(v + 0.4, yi, f'{v} ({v / tot * 100:.0f}%)', va='center', fontsize=6.5, color=INK2)
ax.set_yticks(yy, [f'{g}\n(n = {(df.Group == g).sum()})' for g in GROUPS], fontweight='bold', color=INK, fontsize=8)
ax.set_xlim(0, 34)
ax.set_xticks([0, 5, 10, 15, 20, 25])
ax.set_xlabel('Patients (n)')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=2, columnspacing=1.2, handlelength=1.2)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
ax.set_title('Direction of MRI change by chronicity', loc='left', color=INK, pad=16)
panel(ax, 'd', x=-0.42, y=1.02)

save(fig, 'Figure3')
