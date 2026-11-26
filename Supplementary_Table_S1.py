"""Supplementary Table S1. Count-based MRI progression and pain change.

Progression: a higher count of the five findings (0-5 per joint) at the final than at
the initial examination in either joint (MRI_Progression).
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from common import load

df = load()
df['P'] = df.MRI_Progression
prog, none = df[df.P == 1], df[df.P == 0]


def msd(x):
    return f'{x.mean():.2f} ± {x.std():.2f}'


print(
    f'Progression: {len(prog)} ({len(prog) / len(df) * 100:.1f}%); '
    f'acute {(prog.Chronicity == 0).sum()}, chronic {(prog.Chronicity == 1).sum()}; '
    f'chi-square (Yates) p = {stats.chi2_contingency(pd.crosstab(df.P, df.Chronicity), correction=True)[1]:.3f}\n'
)
for label, x in [('No progression', none), ('Progression', prog)]:
    print(
        f'{label:15s} initial {msd(x.I_VAS)}  final {msd(x.F_VAS)}  change {msd(x.Delta_VAS)}  '
        f'paired t p = {stats.ttest_rel(x.I_VAS, x.F_VAS).pvalue:.4f}  Wilcoxon p = {stats.wilcoxon(x.I_VAS, x.F_VAS).pvalue:.4f}'
    )


def beta(formula, term='P'):
    m = smf.ols(formula, df).fit()
    lo, hi = m.conf_int().loc[term]
    return f'{m.params[term]:.2f} ({lo:.2f} to {hi:.2f}); p = {m.pvalues[term]:.3f}'


print('\nFinal VAS:', beta('F_VAS ~ P'))
print('Change in VAS:', beta('Delta_VAS ~ P'))
print('Change in VAS, adjusted for age, sex and chronicity:', beta('Delta_VAS ~ P + Age + Sex + Chronicity'))
for cov, lab in [('', 'unadjusted'), (' + Age + Sex', 'adjusted for age and sex')]:
    m = smf.ols(f'Delta_VAS ~ P * Chronicity{cov}', df).fit()
    print(f'Progression-by-chronicity interaction, {lab}: p = {m.pvalues["P:Chronicity"]:.3f}')
m = smf.ols('Delta_VAS ~ P * Chronicity + Age + Sex + I_VAS', df).fit()
print(f'  after additional adjustment for initial VAS: p = {m.pvalues["P:Chronicity"]:.3f}')
for c, lab in [(0, 'acute'), (1, 'chronic')]:
    d = df[df.Chronicity == c]
    a, b = d[d.P == 1].Delta_VAS, d[d.P == 0].Delta_VAS
    r = stats.ttest_ind(a, b)
    ci = r.confidence_interval()
    print(f'  {lab}: {a.mean() - b.mean():.2f} ({ci[0]:.2f} to {ci[1]:.2f}); p = {r.pvalue:.3f}')
print(f'Mann-Whitney U: p = {stats.mannwhitneyu(prog.Delta_VAS, none.Delta_VAS).pvalue:.3f}')
rng = np.random.default_rng(42)
boot = [
    rng.choice(prog.Delta_VAS.values, len(prog)).mean() - rng.choice(none.Delta_VAS.values, len(none)).mean()
    for _ in range(10000)
]
print(
    f'Bootstrap difference: {prog.Delta_VAS.mean() - none.Delta_VAS.mean():.2f} '
    f'({np.percentile(boot, 2.5):.2f} to {np.percentile(boot, 97.5):.2f})'
)
excl = df[df.I_VAS > 2]
a, b = excl[excl.P == 1].Delta_VAS, excl[excl.P == 0].Delta_VAS
r = stats.ttest_ind(a, b)
ci = r.confidence_interval()
print(f'Excluding initial VAS <= 2: {a.mean() - b.mean():.2f} ({ci[0]:.2f} to {ci[1]:.2f})')
for v, lab in [
    ('VAS_Improvement', 'Any VAS reduction'),
    ('VAS_Response_2pt_or_50pct', 'Reduction >=2 points or >=50%'),
]:
    p = stats.chi2_contingency(pd.crosstab(df.P, df[v]), correction=True)[1]
    print(
        f'{lab}: no progression {int(none[v].sum())} ({none[v].mean() * 100:.1f}%), '
        f'progression {int(prog[v].sum())} ({prog[v].mean() * 100:.1f}%); chi-square (Yates) p = {p:.3f}'
    )
