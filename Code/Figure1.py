"""Figure 1. Study flow and analysis plan."""

import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Ellipse, Circle
from common import INK, INK2, CHRON, FIND, W2, setup, save

setup()

# profile icon with the TMJ region marked
SKIN = '#f1cfb4'
SKINE = '#c99a7a'
HAIR = '#5a463c'
SHIRT = '#8fa7bd'
RED = '#d7191c'


def smooth(pts, n=200):
    p = np.array(pts + [pts[0]]).T
    tck, u = splprep(p, s=0, per=1)
    x, y = splev(np.linspace(0, 1, n), tck)
    return np.c_[x, y]


def draw_icon(ax):
    ax.add_patch(Ellipse((5.0, 0.2), 6.6, 3.4, fc=SHIRT, ec='none', zorder=1))
    ax.add_patch(Polygon([(4.15, 4.0), (5.85, 3.2), (6.0, 0.9), (4.0, 0.9)], fc=SKIN, ec=SKINE, lw=0.6, zorder=2))
    hair_back = smooth(
        [
            (5.6, 9.3),
            (4.2, 9.2),
            (3.0, 8.3),
            (2.5, 6.8),
            (2.6, 5.4),
            (2.2, 4.4),
            (2.7, 3.4),
            (3.8, 3.2),
            (4.6, 3.9),
            (4.5, 5.6),
            (5.6, 8.0),
            (6.6, 8.1),
            (6.7, 8.9),
        ]
    )
    face = smooth(
        [
            (5.3, 8.7),
            (6.4, 8.2),
            (6.85, 7.3),
            (6.9, 6.7),
            (7.6, 5.75),
            (7.0, 5.45),
            (7.05, 5.05),
            (6.85, 4.8),
            (6.95, 4.45),
            (6.75, 3.75),
            (6.2, 3.35),
            (5.4, 3.3),
            (4.6, 3.9),
            (4.3, 5.2),
            (4.6, 6.9),
        ]
    )
    ax.add_patch(Polygon(hair_back, fc=HAIR, ec='none', zorder=3))
    ax.add_patch(Polygon(face, fc=SKIN, ec=SKINE, lw=0.6, zorder=4))
    ax.add_patch(
        Polygon(
            smooth([(4.5, 8.9), (5.8, 9.0), (6.7, 8.3), (5.6, 8.0), (4.7, 7.2), (4.4, 6.2), (4.0, 7.6)]),
            fc=HAIR,
            ec='none',
            zorder=5,
        )
    )
    ax.add_patch(Ellipse((4.78, 5.55), 0.75, 1.15, fc=SKIN, ec=SKINE, lw=0.6, zorder=6))
    ax.plot([5.95, 6.35], [6.35, 6.3], color=SKINE, lw=0.7, zorder=6, solid_capstyle='round')
    ax.add_patch(Circle((5.42, 5.3), 0.62, fc=RED, alpha=0.22, ec='none', zorder=7))
    ax.add_patch(Circle((5.42, 5.3), 0.27, fc=RED, ec='none', zorder=8))
    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(0, 9.8)
    ax.set_aspect('equal')
    ax.axis('off')


fig = plt.figure(figsize=(W2, W2 * 0.74))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(4, 86)
ax.axis('off')
LINE = '#9a998f'
BG = '#f7f7f5'


def box(x, y, w, h, fc='white', ec=LINE, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec=ec, lw=lw))


def t(x, y, s, **k):
    k.setdefault('fontsize', 7.3)
    k.setdefault('color', INK)
    k.setdefault('va', 'top')
    ax.text(x, y, s, **k)


def arr(x1, y1, x2, y2, c=INK2):
    ax.add_patch(
        FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, lw=0.8, color=c, shrinkA=0, shrinkB=0)
    )


# row1
box(2, 76, 44, 8.5)
t(4, 83.4, 'Patients with TMD assessed (n = 94)', fontweight='bold', fontsize=8)
t(4, 80.4, 'Clinical assessment and bilateral TMJ MRI\nat both the initial and the final visit', fontsize=7)
box(54, 72.3, 44, 11, fc=BG)
t(56, 82.3, 'Excluded (n = 5)', fontweight='bold', fontsize=7.8)
t(
    56,
    79.5,
    '• Facial pain attributable to causes other than TMD\n• Insufficient clinical or imaging information\n• Conditions confounding pain or imaging assessment',
    fontsize=6.6,
    linespacing=1.35,
)
arr(24, 76, 24, 71.5)
arr(24, 74.2, 54, 74.2)
# row2 cohort
box(2, 62, 96, 9.5)
t(4, 69.6, 'Analysed cohort (n = 89)', fontweight='bold', fontsize=8)
t(4, 66.4, 'Retrospective longitudinal cohort\nKyung Hee University Dental Hospital', fontsize=6.8, color=INK2)
box(40, 63.3, 27, 6.8, fc='#e9edf7', ec=CHRON['Acute'])
t(42, 68.8, 'Acute TMD  ·  n = 51', fontweight='bold', fontsize=7.6)
t(42, 66.0, 'Symptom duration < 6 months', fontsize=6.8)
box(69.5, 63.3, 27, 6.8, fc='#fbeaed', ec=CHRON['Chronic'])
t(71.5, 68.8, 'Chronic TMD  ·  n = 38', fontweight='bold', fontsize=7.6)
t(71.5, 66.0, 'Symptom duration ≥ 6 months', fontsize=6.8)
arr(50, 62, 50, 58.5)
# row3 timeline
box(2, 40.5, 96, 18, fc=BG, ec=BG)
t(4, 57.2, 'Repeated assessment design', fontweight='bold', fontsize=7.8)
for x0, lab in [(4, 'Initial visit'), (70, 'Final visit')]:
    box(x0, 42, 26, 12.6)
    t(x0 + 2, 53.4, lab, fontweight='bold', fontsize=7.6)
    t(
        x0 + 2,
        50.3,
        'Pain intensity (VAS, 0–10)\nCMI, mouth opening, symptoms\nBilateral TMJ MRI',
        fontsize=6.7,
        linespacing=1.4,
    )
arr(31, 48.3, 69, 48.3, c=INK)
t(50, 52.2, 'Interval between MRIs', ha='center', fontsize=7, color=INK2)
t(50, 47.3, '31.5 ± 20.4 months', ha='center', fontsize=8, fontweight='bold')
t(50, 44.3, 'Treatment as clinically indicated', ha='center', fontsize=6.6, color=INK2)
# row4 analyses
arr(17, 40.5, 17, 37)
arr(83, 40.5, 83, 37)
box(2, 15.5, 44, 21.5)
t(4, 35.6, 'Pain outcomes', fontweight='bold', fontsize=7.8)
t(
    4,
    32.4,
    '• Correlates of initial pain\n• Chronicity interaction tests\n• Concurrent correlates of final pain\n• Pain change (ΔVAS = final − initial)\n• Acute vs chronic comparison\n• Response: reduction ≥ 2 points or ≥ 50%',
    fontsize=6.7,
    linespacing=1.45,
)
box(54, 15.5, 44, 21.5)
t(56, 35.6, 'MRI change between examinations', fontweight='bold', fontsize=7.8)
t(56, 32.5, 'Five findings per joint (178 joints)', fontsize=6.7)
chips = [
    ('DDwR', FIND['DDwR']),
    ('DDwoR', FIND['DDwoR']),
    ('TMJ OA', FIND['TMJ OA']),
    ('Disc degeneration', FIND['Disc degeneration']),
    ('Effusion', FIND['Effusion']),
]
pos = [(56.3, 29.5), (64.6, 29.5), (73.6, 29.5), (56.3, 27.2), (72.5, 27.2)]
for (n, c), (xx, yy) in zip(chips, pos):
    ax.add_patch(FancyBboxPatch((xx, yy - 0.95), 1.5, 1.5, boxstyle='round,pad=0,rounding_size=0.3', fc=c, ec='none'))
    ax.text(xx + 2.1, yy - 0.2, n, fontsize=6.6, va='center', color=INK)
t(
    56,
    25.2,
    '• Per-finding transitions (new / resolved)\n• Four MRI direction groups:\n   Stable, Worsened only, Improved only, Mixed\n• Count-based definition',
    fontsize=6.7,
    linespacing=1.45,
)
# key comparison
arr(24, 15.5, 40, 11.7)
arr(76, 15.5, 60, 11.7)
box(28, 4.8, 44, 6.9, fc='#f0efec', ec=INK2, lw=1)
t(50, 10.4, 'Pain change according to MRI change', ha='center', fontweight='bold', fontsize=7.8)
t(50, 7.6, 'MRI direction groups · individual findings · count rule', ha='center', fontsize=6.7, color=INK2)
# exploratory
box(76, 4.8, 22, 6.9, fc=BG, ec=BG)
t(87, 10.9, 'Exploratory (Supplementary)', ha='center', fontsize=6.6, fontweight='bold', color=INK2)
t(87, 8.6, 'Random forest with SHAP;\n3D visualizations', ha='center', fontsize=6.2, color=INK2, linespacing=1.3)
Y = lambda y: (y - 4) / 82
iax = fig.add_axes([0.385, Y(76.2), 0.07, Y(84.6) - Y(76.2)])
draw_icon(iax)
save(fig, 'Figure1')
