"""Table 1. Baseline characteristics of acute and chronic TMD."""

import numpy as np
import pandas as pd
from scipy import stats
from common import load, OUT

df = load()
ac, ch = df[df.Chron == 'Acute'], df[df.Chron == 'Chronic']

CONTINUOUS = {
    'Age, years': 'Age',
    'Onset to initial visit, months': 'Onset_to_InitialVisit',
    'Initial VAS': 'I_VAS',
    'Initial CMI': 'I_CMI',
    'Comfortable mouth opening, mm': 'I_CMO',
}
BINARY = {
    'Male sex': None,
    'TMJ pain': 'I_TMJpain',
    'TMJ noise': 'I_TMJnoise',
    'TMJ locking': 'I_TMJlocking',
    'Muscle stiffness': 'I_MuscleStiffness',
    'Tinnitus': 'I_Tinnitus',
    'Bruxism': 'I_Bruxism',
    'Psychological stress': 'I_PsychologicalStress',
    'Sleep problems': 'I_SleepProblems',
    'Headache': 'I_Headache',
    'Caffeine': 'I_Caffeine',
}
for side, name in [('Rt', 'right'), ('Lt', 'left')]:
    for code, lab in [
        ('DDwR', 'DDwR'),
        ('DDwoR', 'DDwoR'),
        ('TMJOA', 'TMJ OA'),
        ('DiscDegeneration', 'Disc degeneration'),
        ('Effusion', 'Effusion'),
    ]:
        BINARY[f'{lab}, {name}'] = f'I_{side}_{code}'


def msd(x):
    return f'{x.mean():.2f} ± {x.std():.2f} ({x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}])'


def npct(x):
    return f'{int(x.sum())} ({x.mean() * 100:.1f}%)'


rows = []
for lab, v in CONTINUOUS.items():
    p = stats.mannwhitneyu(ac[v], ch[v]).pvalue
    sd = np.sqrt(((len(ac) - 1) * ac[v].var() + (len(ch) - 1) * ch[v].var()) / (len(df) - 2))
    rows.append([lab, msd(df[v]), msd(ac[v]), msd(ch[v]), f'd = {(ac[v].mean() - ch[v].mean()) / sd:.2f}', p])
for lab, v in BINARY.items():
    x = (df.Sex == 1).astype(int) if v is None else df[v]
    t = pd.crosstab(df.Chron, x)
    chi2, p, _, exp = stats.chi2_contingency(t, correction=True)
    if (exp < 5).any():
        p = stats.fisher_exact(t.values)[1]
    v_ = np.sqrt(stats.chi2_contingency(t, correction=False)[0] / len(df))
    xa = x[df.Chron == 'Acute']
    xc = x[df.Chron == 'Chronic']
    rows.append([lab, npct(x), npct(xa), npct(xc), f'V = {v_:.2f}', p])
T = pd.DataFrame(
    rows,
    columns=[
        'Variable',
        f'Total (N = {len(df)})',
        f'Acute (n = {len(ac)})',
        f'Chronic (n = {len(ch)})',
        'Effect size',
        'p',
    ],
)
T['p'] = T.p.map(lambda p: '<0.001' if p < 0.001 else f'{p:.4f}')
print(T.to_string(index=False))
T.to_csv(f'{OUT}/Table1.csv', index=False)
