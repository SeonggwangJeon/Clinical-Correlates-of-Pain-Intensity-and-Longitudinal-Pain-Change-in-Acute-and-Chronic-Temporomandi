import numpy as np
import pandas as pd
import matplotlib as mpl

DATA = 'data/20261119_TMD_MRI_89patients_CLEAN_deidentified.xlsx'
OUT = 'output'

SIDES = ['Rt', 'Lt']
FINDINGS = {
    'DDwR': 'DDwR',
    'DDwoR': 'DDwoR',
    'TMJ OA': 'TMJOA',
    'Disc degeneration': 'DiscDegeneration',
    'Effusion': 'Effusion',
}
GROUPS = ['Stable', 'Worsened only', 'Improved only', 'Mixed']

# colours
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e6e5e0'
LINE = '#9e9d97'
CHRON = {'Acute': '#1f3a93', 'Chronic': '#d1495b'}
FIND = {
    'DDwR': '#7fbf3f',
    'DDwoR': '#1b7837',
    'TMJ OA': '#e87ba4',
    'Disc degeneration': '#b8860b',
    'Effusion': '#7b3294',
}
DISC = {'Normal': '#b9b8b1', 'DDwR': FIND['DDwR'], 'DDwoR': FIND['DDwoR']}
GCOL = {'Stable': '#56a0d3', 'Worsened only': '#e8743b', 'Improved only': '#e2b33c', 'Mixed': '#c25ba6'}
MM = 1 / 25.4
W2 = 174 * MM  # full page width


def load(path=DATA):
    df = pd.read_excel(path)
    df['Chron'] = np.where(df['Chronicity'] == 1, 'Chronic', 'Acute')
    return df


def joint(df, visit, code):
    """Values of one finding for all 178 joints (right joints, then left)."""
    return pd.concat([df[f'{visit}_{s}_{code}'] for s in SIDES]).values


def disc_position(df, visit, side):
    """0 = normal, 1 = DDwR, 2 = DDwoR."""
    return np.where(df[f'{visit}_{side}_DDwoR'] == 1, 2, np.where(df[f'{visit}_{side}_DDwR'] == 1, 1, 0))


def change_events(df, sides=None):
    """Number of worsening and improving events per patient.

    sides: optional array of side lists per patient (painful-joint analysis).
    """
    n = len(df)
    worse = np.zeros(n, int)
    better = np.zeros(n, int)
    for s in SIDES:
        use = np.ones(n, bool) if sides is None else np.array([s in x for x in sides])
        d1, d2 = disc_position(df, 'I', s), disc_position(df, 'F', s)
        worse += use & (d2 > d1)
        better += use & (d2 < d1)
        for code in ['TMJOA', 'DiscDegeneration', 'Effusion']:
            a, b = df[f'I_{s}_{code}'], df[f'F_{s}_{code}']
            worse += use & ((a == 0) & (b == 1)).values
            better += use & ((a == 1) & (b == 0)).values
    return worse, better


def direction_group(worse, better):
    g = np.select(
        [(worse == 0) & (better == 0), (worse > 0) & (better == 0), (worse == 0) & (better > 0)], GROUPS[:3], 'Mixed'
    )
    return pd.Categorical(g, GROUPS)


def painful_sides(df):
    code = df['I_PainSide_Rt1Lt2Both3']
    return [['Rt'] if c == 1 else ['Lt'] if c == 2 else ['Rt', 'Lt'] for c in code]


def setup():
    mpl.rcParams.update(
        {
            'font.family': 'Liberation Sans',
            'font.size': 8,
            'axes.titlesize': 8.5,
            'axes.labelsize': 8,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 7.5,
            'ytick.labelsize': 7.5,
            'legend.fontsize': 7.5,
            'axes.edgecolor': INK2,
            'axes.labelcolor': INK,
            'xtick.color': INK2,
            'ytick.color': INK2,
            'axes.linewidth': 0.6,
            'xtick.major.width': 0.6,
            'ytick.major.width': 0.6,
            'xtick.major.size': 2.5,
            'ytick.major.size': 2.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.color': GRID,
            'grid.linewidth': 0.5,
            'legend.frameon': False,
            'savefig.dpi': 600,
            'pdf.fonttype': 42,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
        }
    )


def panel(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left', color=INK)


def save(fig, name):
    import os

    os.makedirs(OUT, exist_ok=True)
    for ext in ['png', 'tiff', 'pdf']:
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tiff' else {}
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight', **kw)


def boxes(ax, data, positions, fills, width=0.5, alpha=0.28):
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=INK, lw=1.6),
        whiskerprops=dict(color=INK2, lw=0.9),
        capprops=dict(color=INK2, lw=0.9),
        boxprops=dict(lw=0.9, edgecolor=INK2),
        zorder=1,
    )
    for b, c in zip(bp['boxes'], fills):
        b.set_facecolor(c)
        b.set_alpha(alpha)
    for d, p in zip(data, positions):
        ax.scatter(p, np.mean(d), marker='D', s=34, facecolor='white', edgecolor=INK, lw=1.1, zorder=5)
    return bp


def trajectories(ax, x):
    """Grey patient lines; width and point size scale with the number of identical pairs."""
    pairs = x.groupby(['I_VAS', 'F_VAS']).size().reset_index(name='n')
    for _, r in pairs.iterrows():
        ax.plot(
            [0, 1],
            [r.I_VAS, r.F_VAS],
            color=LINE,
            alpha=0.45,
            lw=0.6 + 0.65 * (r.n - 1),
            solid_capstyle='round',
            zorder=1,
        )
    for k, v in [(0, x.I_VAS), (1, x.F_VAS)]:
        cnt = v.value_counts()
        ax.scatter([k] * len(cnt), cnt.index, s=6 + 9 * cnt.values, color=LINE, alpha=0.55, lw=0, zorder=2)
