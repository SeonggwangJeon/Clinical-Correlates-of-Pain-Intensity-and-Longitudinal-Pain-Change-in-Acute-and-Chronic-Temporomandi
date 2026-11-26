"""Pain change according to the direction of MRI change (Results; Figures 3-5).

Includes the per-finding comparisons, the painful-joint analysis and the sensitivity
analysis excluding patients with initial VAS <= 2.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from common import load, GROUPS, SIDES, FINDINGS, joint, disc_position, change_events, direction_group, painful_sides

df = load()
df['Group'] = pd.Categorical(df.MRI_Change_Group, GROUPS)
COV = ' + I_VAS + Treatment_Duration + Chronicity + Age'
TERMS = [f"C(Group, Treatment('Stable'))[T.{g}]" for g in GROUPS[1:]]


def group_tests(d, label):
    kw = stats.kruskal(*[d[d.Group == g].Delta_VAS for g in GROUPS]).pvalue
    m = smf.ols("Delta_VAS ~ C(Group, Treatment('Stable'))" + COV, d).fit()
    overall = m.f_test(', '.join(f'{t} = 0' for t in TERMS)).pvalue
    print(f'{label}: Kruskal-Wallis p = {kw:.3f}; adjusted overall p = {overall:.3f}')
    return m


# joint-level changes
print('Joint-level findings (178 joints)')
for k, code in FINDINGS.items():
    a, b = joint(df, 'I', code), joint(df, 'F', code)
    print(
        f'  {k:18s} initial {a.mean() * 100:5.1f}%  final {b.mean() * 100:5.1f}%  '
        f'new {((a == 0) & (b == 1)).sum():3d}  resolved {((a == 1) & (b == 0)).sum():3d}  persistent {((a == 1) & (b == 1)).sum():3d}'
    )
lab = {0: 'Normal', 1: 'DDwR', 2: 'DDwoR'}
d1 = pd.Series(np.concatenate([disc_position(df, 'I', s) for s in SIDES])).map(lab)
d2 = pd.Series(np.concatenate([disc_position(df, 'F', s) for s in SIDES])).map(lab)
print('\nDisc position (rows: initial, columns: final)')
print(pd.crosstab(d1, d2))

# patient-level groups
print('\nDirection groups')
for g in GROUPS:
    x = df[df.Group == g]
    p = stats.ttest_rel(x.I_VAS, x.F_VAS).pvalue
    print(
        f'  {g:14s} n = {len(x):2d}  initial VAS {x.I_VAS.mean():.2f}  change {x.Delta_VAS.mean():.2f} ± {x.Delta_VAS.std():.2f}  '
        f'paired p = {p:.4f}'
    )
print(
    f'Initial VAS across groups: Kruskal-Wallis p = {stats.kruskal(*[df[df.Group == g].I_VAS for g in GROUPS]).pvalue:.3f}'
)
print(
    f'Interval between MRIs: Kruskal-Wallis p = {stats.kruskal(*[df[df.Group == g].Treatment_Duration for g in GROUPS]).pvalue:.3f}'
)
print(f'Chronicity: chi-square p = {stats.chi2_contingency(pd.crosstab(df.Group, df.Chronicity))[1]:.3f}\n')
m = group_tests(df, 'All patients')
for t, g in zip(TERMS, GROUPS[1:]):
    lo, hi = m.conf_int().loc[t]
    print(f'  {g:14s} vs stable: {m.params[t]:.2f} [{lo:.2f}, {hi:.2f}]')

# individual findings
print('\nIndividual findings (either joint): difference in change in VAS, with vs without')


def compare(mask, label):
    a, b = df[mask].Delta_VAS, df[~mask].Delta_VAS
    r = stats.ttest_ind(a, b)
    ci = r.confidence_interval()
    print(
        f'  {label:28s} n = {len(a):2d}/{len(b):2d}  {a.mean() - b.mean():5.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  '
        f'Mann-Whitney p = {stats.mannwhitneyu(a, b).pvalue:.3f}'
    )


for k, code in FINDINGS.items():
    if k == 'DDwR':
        continue
    new = sum(((df[f'I_{s}_{code}'] == 0) & (df[f'F_{s}_{code}'] == 1)) for s in SIDES) > 0
    res = sum(((df[f'I_{s}_{code}'] == 1) & (df[f'F_{s}_{code}'] == 0)) for s in SIDES) > 0
    compare(new, f'New {k}')
    compare(res, f'Resolved {k}')
compare(sum(disc_position(df, 'F', s) > disc_position(df, 'I', s) for s in SIDES) > 0, 'Worsened disc position')

# painful joint or joints only
print()
w, b = change_events(df, painful_sides(df))
pj = df.copy()
pj['Group'] = direction_group(w, b)
group_tests(pj, 'Painful joint only')
mask = w > 0
a, c = df[mask].Delta_VAS, df[~mask].Delta_VAS
r = stats.ttest_ind(a, c)
ci = r.confidence_interval()
print(f'  any worsening on the painful side: {a.mean() - c.mean():.2f} [{ci[0]:.2f}, {ci[1]:.2f}]')

# sensitivity: initial VAS > 2
print()
s = df[df.I_VAS > 2]
group_tests(s, f'Initial VAS > 2 (n = {len(s)})')
ch, ac = s[s.Chronicity == 1].Delta_VAS, s[s.Chronicity == 0].Delta_VAS
r = stats.ttest_ind(ch, ac)
ci = r.confidence_interval()
print(f'  chronic - acute change in VAS: {ch.mean() - ac.mean():.2f} [{ci[0]:.2f}, {ci[1]:.2f}], p = {r.pvalue:.3f}')
resp = pd.crosstab(s.Chronicity, s.VAS_Response_2pt_or_50pct)
print(
    f'  reduction >=2 points or >=50%: acute {s[s.Chronicity == 0].VAS_Response_2pt_or_50pct.mean() * 100:.1f}%, '
    f'chronic {s[s.Chronicity == 1].VAS_Response_2pt_or_50pct.mean() * 100:.1f}%, '
    f'p = {stats.chi2_contingency(resp, correction=True)[1]:.3f}'
)
low = df[df.I_VAS <= 3]
only50 = low[(low.VAS_Response_2pt_or_50pct == 1) & (low.I_VAS - low.F_VAS < 2)]
print(
    f'  initial VAS <= 3: {low.VAS_Response_2pt_or_50pct.sum()} responders, {len(only50)} only through the 50% criterion'
)
