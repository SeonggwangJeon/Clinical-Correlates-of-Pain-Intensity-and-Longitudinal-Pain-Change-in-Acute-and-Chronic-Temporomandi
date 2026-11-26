"""Table 3. Associations with initial pain intensity and predictor-by-chronicity interactions."""

import pandas as pd
import statsmodels.formula.api as smf
from common import load, OUT

df = load()
for code in ['DDwR', 'DDwoR', 'TMJOA', 'DiscDegeneration', 'Effusion']:
    df[f'Base_{code}'] = ((df[f'I_Rt_{code}'] == 1) | (df[f'I_Lt_{code}'] == 1)).astype(int)

PREDICTORS = {
    'Initial CMI': 'I_CMI',
    'Initial CMO': 'I_CMO',
    'TMJ pain': 'I_TMJpain',
    'TMJ noise': 'I_TMJnoise',
    'TMJ locking': 'I_TMJlocking',
    'Muscle stiffness': 'I_MuscleStiffness',
    'Tinnitus': 'I_Tinnitus',
    'Bruxism': 'I_Bruxism',
    'Psychological stress': 'I_PsychologicalStress',
    'Sleep problems': 'I_SleepProblems',
    'Headache': 'I_Headache',
    'Age': 'Age',
    'Sex': 'Sex',
    'Baseline DDwR': 'Base_DDwR',
    'Baseline DDwoR': 'Base_DDwoR',
    'Baseline effusion': 'Base_Effusion',
    'Baseline TMJ OA': 'Base_TMJOA',
    'Baseline disc degeneration': 'Base_DiscDegeneration',
}


def est(model, x):
    lo, hi = model.conf_int().loc[x]
    return f'{model.params[x]:.2f} [{lo:.2f}, {hi:.2f}]', model.pvalues[x]


def covariates(x):
    # age and sex are adjusted for each other; other predictors for both
    return {'Age': ' + Sex', 'Sex': ' + Age'}.get(x, ' + Age + Sex')


rows = []
for lab, x in PREDICTORS.items():
    cov = covariates(x)
    uni, p_uni = est(smf.ols(f'I_VAS ~ {x}', df).fit(), x)
    adj, p_adj = est(smf.ols(f'I_VAS ~ {x}{cov}', df).fit(), x)
    acute, _ = est(smf.ols(f'I_VAS ~ {x}{cov}', df[df.Chronicity == 0]).fit(), x)
    chronic, _ = est(smf.ols(f'I_VAS ~ {x}{cov}', df[df.Chronicity == 1]).fit(), x)
    p_int = smf.ols(f'I_VAS ~ {x} * Chronicity{cov}', df).fit().pvalues[f'{x}:Chronicity']
    rows.append([lab, uni, p_uni, adj, p_adj, acute, chronic, p_int])

fmt = lambda p: '<0.001' if p < 0.0005 else f'{p:.3f}'
T = pd.DataFrame(
    rows,
    columns=[
        'Predictor',
        'Univariable β [95% CI]',
        'p',
        'Adjusted β [95% CI]',
        'p ',
        'Acute β [95% CI]',
        'Chronic β [95% CI]',
        'p for interaction',
    ],
)
for c in ['p', 'p ', 'p for interaction']:
    T[c] = T[c].map(fmt)
print(T.to_string(index=False))
T.to_csv(f'{OUT}/Table3.csv', index=False)
