"""Table 2. Pain intensity, clinical measures and longitudinal change by chronicity."""

import pandas as pd
from scipy import stats
from common import load, GROUPS, OUT

df = load()
ac, ch = df[df.Chron == 'Acute'], df[df.Chron == 'Chronic']
df['Delta_CMI'] = df.F_CMI - df.I_CMI
ac, ch = df[df.Chron == 'Acute'], df[df.Chron == 'Chronic']


def msd(x):
    return f'{x.mean():.2f} ± {x.std():.2f}'


def npct(x):
    return f'{int(x.sum())} ({x.mean() * 100:.1f}%)'


rows = []
for lab, v in [('Initial pain intensity', 'I_VAS'), ('Final pain intensity', 'F_VAS')]:
    rows.append([lab, msd(df[v]), msd(ac[v]), msd(ch[v]), '', stats.mannwhitneyu(ac[v], ch[v]).pvalue])
for lab, v in [
    ('Change in pain intensity (F_VAS − I_VAS)', 'Delta_VAS'),
    ('Change in CMI (F_CMI − I_CMI)', 'Delta_CMI'),
]:
    r = stats.ttest_ind(ch[v], ac[v])
    ci = r.confidence_interval()
    rows.append(
        [
            lab,
            msd(df[v]),
            msd(ac[v]),
            msd(ch[v]),
            f'{ch[v].mean() - ac[v].mean():.2f} [{ci[0]:.2f}, {ci[1]:.2f}]',
            r.pvalue,
        ]
    )
    if v == 'Delta_VAS':
        rows.append(
            [
                'Within-group change in pain intensity',
                '',
                f'p = {stats.ttest_rel(ac.I_VAS, ac.F_VAS).pvalue:.4f}',
                f'p = {stats.ttest_rel(ch.I_VAS, ch.F_VAS).pvalue:.4f}',
                '',
                None,
            ]
        )
        for lab2, v2 in [
            ('Any reduction in VAS', 'VAS_Improvement'),
            ('Reduction of ≥2 VAS points or ≥50%', 'VAS_Response_2pt_or_50pct'),
        ]:
            p = stats.chi2_contingency(pd.crosstab(df.Chron, df[v2]), correction=True)[1]
            rows.append([lab2, npct(df[v2]), npct(ac[v2]), npct(ch[v2]), '', p])
for lab, v in [('Initial CMI', 'I_CMI'), ('Final CMI', 'F_CMI')]:
    rows.append([lab, msd(df[v]), msd(ac[v]), msd(ch[v]), '', stats.mannwhitneyu(ac[v], ch[v]).pvalue])
for lab, v in [
    ('Treatment duration, months', 'Treatment_Duration'),
    ('Onset to completion, months', 'Onset_to_Completion'),
]:
    rows.append([lab, msd(df[v]), msd(ac[v]), msd(ch[v]), '', stats.mannwhitneyu(ac[v], ch[v]).pvalue])
p = stats.chi2_contingency(pd.crosstab(df.Chron, df.MRI_Change_Group))[1]
rows.append(['MRI change between examinations (direction-based)', '', '', '', '', p])
for g in GROUPS:
    rows.append(
        [
            f'   {g}',
            npct(df.MRI_Change_Group == g),
            npct(ac.MRI_Change_Group == g),
            npct(ch.MRI_Change_Group == g),
            '',
            None,
        ]
    )
p = stats.chi2_contingency(pd.crosstab(df.Chron, df.MRI_Progression), correction=True)[1]
rows.append(
    [
        'MRI progression (count-based)',
        npct(df.MRI_Progression),
        npct(ac.MRI_Progression),
        npct(ch.MRI_Progression),
        '',
        p,
    ]
)

T = pd.DataFrame(rows, columns=['Outcome', 'Total', 'Acute', 'Chronic', 'Difference [95% CI]', 'p'])
T['p'] = T.p.map(lambda p: '' if p is None or pd.isna(p) else ('<0.001' if p < 0.001 else f'{p:.4f}'))
print(T.to_string(index=False))
T.to_csv(f'{OUT}/Table2.csv', index=False)
