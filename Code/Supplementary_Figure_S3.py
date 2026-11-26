"""Supplementary Figure S3. SHAP analysis of the baseline random forest model for final pain intensity.

Also writes the bootstrap stability table (mean |SHAP|, 95% interval, % of resamples ranked first).
The 200 bootstrap refits take several minutes.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from common import INK, INK2, W2, load, setup, save, panel

setup()
df = load()
y = df.F_VAS.values
CORE = ['I_VAS', 'Onset_to_InitialVisit', 'Age', 'Sex']
EXPANDED = CORE + [
    'I_CMI',
    'I_CMO',
    'I_TMJpain',
    'I_Bruxism',
    'I_SleepProblems',
    'I_Headache',
    'Any_DDwoR',
    'Any_Effusion',
    'Any_TMJOA',
]
LABEL = {
    'I_VAS': 'Initial VAS',
    'Onset_to_InitialVisit': 'Symptom duration',
    'Age': 'Age',
    'Sex': 'Sex',
    'I_CMI': 'Initial CMI',
    'I_CMO': 'Initial CMO',
    'I_TMJpain': 'TMJ pain',
    'I_Bruxism': 'Bruxism',
    'I_SleepProblems': 'Sleep problems',
    'I_Headache': 'Headache',
    'Any_DDwoR': 'DDwoR',
    'Any_Effusion': 'Effusion',
    'Any_TMJOA': 'TMJ OA',
}
N_BOOT = 200


def forest(seed=42):
    return RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, max_features='sqrt', random_state=seed, n_jobs=-1
    )


def stability(X):
    model = forest().fit(df[X].values, y)
    sv = shap.TreeExplainer(model).shap_values(df[X].values)
    rng = np.random.default_rng(42)
    imp = []
    first = []
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        m = forest(b).fit(df.iloc[idx][X].values, y[idx])
        mi = np.abs(shap.TreeExplainer(m).shap_values(df[X].values)).mean(0)
        imp.append(mi)
        first.append(np.argmax(mi))
    imp = np.array(imp)
    T = pd.DataFrame(
        {
            'Feature': [LABEL[x] for x in X],
            'mean_abs_SHAP': np.abs(sv).mean(0),
            'boot_2.5': np.percentile(imp, 2.5, 0),
            'boot_97.5': np.percentile(imp, 97.5, 0),
            'ranked_first_pct': np.bincount(first, minlength=len(X)) / N_BOOT * 100,
        }
    )
    return sv, T.sort_values('mean_abs_SHAP', ascending=False)


sv, T = stability(CORE)
T.round(3).to_csv('output/SHAP_stability_core.csv', index=False)
_, T2 = stability(EXPANDED)
T2.round(3).to_csv('output/SHAP_stability_expanded.csv', index=False)
print(T.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.42), gridspec_kw={'width_ratios': [1.3, 1]})
fig.subplots_adjust(wspace=0.95)

# a: SHAP values per patient
ax = axs[0]
rng = np.random.default_rng(1)
cmap = plt.get_cmap('coolwarm')
order = list(np.argsort(np.abs(sv).mean(0))[::-1])
for row, j in enumerate(order):
    v = df[CORE[j]].values.astype(float)
    vn = (v - v.min()) / (v.max() - v.min() + 1e-9)
    ax.scatter(
        sv[:, j],
        len(order) - 1 - row + rng.uniform(-0.18, 0.18, len(v)),
        c=cmap(vn),
        s=12,
        lw=0.3,
        edgecolor='white',
        zorder=3,
    )
ax.axvline(0, color=INK2, lw=0.7, ls=(0, (3, 2)))
ax.set_yticks(range(len(order)), [LABEL[CORE[j]] for j in order][::-1], fontweight='bold', color=INK, fontsize=8)
ax.set_xlabel('SHAP value (VAS points)')
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.02, ticks=[0, 1])
cb.ax.set_yticklabels(['Low', 'High'])
cb.set_label('Feature value', fontsize=7)
cb.outline.set_visible(False)
ax.set_title('SHAP values, baseline model (n = 89)', loc='left', color=INK)
panel(ax, 'a', x=-0.32)

# b: importance with bootstrap interval
ax = axs[1]
Tt = T.iloc[::-1]
yv = np.arange(len(Tt))
ax.hlines(yv, Tt['boot_2.5'], Tt['boot_97.5'], color=INK2, lw=1.6)
ax.scatter(Tt.mean_abs_SHAP, yv, s=34, color=INK, zorder=3)
for yi, (_, r) in zip(yv, Tt.iterrows()):
    ax.text(r['boot_97.5'] + 0.02, yi, f"{r.ranked_first_pct:.0f}%", va='center', fontsize=7, color=INK2)
ax.set_yticks(yv, Tt.Feature, fontweight='bold', color=INK, fontsize=8)
ax.set_xlabel('Mean |SHAP| (VAS points)')
ax.set_xlim(0, max(Tt['boot_97.5']) * 1.35)
ax.legend(
    handles=[
        Line2D([], [], marker='o', ls='', color=INK, ms=5, label='Full-data estimate'),
        Line2D([], [], color=INK2, lw=1.6, label=f'Bootstrap 95% interval ({N_BOOT} resamples)'),
        Line2D([], [], ls='', label='%: resamples ranked first'),
    ],
    loc='upper center',
    bbox_to_anchor=(0.35, -0.2),
    fontsize=6.5,
    handlelength=1.2,
    frameon=False,
)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
ax.set_title('Importance and bootstrap stability', loc='left', color=INK)
panel(ax, 'b', x=-0.32)
save(fig, 'Supplementary_Figure_S3')
