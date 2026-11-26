"""Table 4. Models of final pain intensity.

Linear models and random forests are compared with a mean-only reference by five-fold
cross-validation repeated 20 times on identical splits. Random forests use baseline
information only and are tested with a permutation test (500 permutations).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, KFold, permutation_test_score
from common import load, OUT

df = load()
y = df.F_VAS.values

M1 = ['Age', 'Sex', 'Chronicity', 'I_VAS']
M2 = M1 + ['I_CMI', 'I_CMO', 'I_TMJpain', 'I_SleepProblems', 'Any_Effusion']
M3 = M2 + ['Treatment_Duration']
RF_CORE = ['I_VAS', 'Onset_to_InitialVisit', 'Age', 'Sex']
RF_EXPANDED = RF_CORE + [
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
MODELS = {
    'Mean-only reference': (None, None),
    'Model 1': ('linear', M1),
    'Model 2': ('linear', M2),
    'Model 3': ('linear', M3),
    'Random forest, core': ('forest', RF_CORE),
    'Random forest, expanded': ('forest', RF_EXPANDED),
}


def forest(trees=500):
    return RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1
    )


# concurrent association with final CMI
m = smf.ols('F_VAS ~ F_CMI', df).fit()
lo, hi = m.conf_int().loc['F_CMI']
print(
    f'Final CMI: beta {m.params.F_CMI:.2f} [{lo:.2f}, {hi:.2f}], p {m.pvalues.F_CMI:.3g}, '
    f'R2 {m.rsquared:.3f}, adj R2 {m.rsquared_adj:.3f}, RMSE {np.sqrt(m.mse_resid * (len(y) - 2) / len(y)):.3f}\n'
)

splits = list(RepeatedKFold(n_splits=5, n_repeats=20, random_state=42).split(df))
pred = {k: np.zeros((20, len(y))) for k in MODELS}
for s, (tr, te) in enumerate(splits):
    for k, (kind, X) in MODELS.items():
        if kind is None:
            p = np.full(len(te), y[tr].mean())
        else:
            mdl = LinearRegression() if kind == 'linear' else forest()
            mdl.fit(df.iloc[tr][X].values, y[tr])
            p = mdl.predict(df.iloc[te][X].values)
        pred[k][s // 5, te] = p

rows = []
for k, (kind, X) in MODELS.items():
    P = pred[k]
    rmse = np.sqrt(((P - y) ** 2).mean(1))
    mae = np.abs(P - y).mean(1)
    r2 = 1 - ((P - y) ** 2).sum(1) / ((y - y.mean()) ** 2).sum()
    row = {
        'Model': k,
        'CV RMSE': f'{rmse.mean():.3f} ({rmse.std(ddof=1):.3f})',
        'CV MAE': f'{mae.mean():.3f} ({mae.std(ddof=1):.3f})',
        'CV R2': f'{r2.mean():.3f} ({r2.std(ddof=1):.3f})',
    }
    if kind == 'linear':
        f = sm.OLS(y, sm.add_constant(df[X].values)).fit()
        row.update({'R2': f'{f.rsquared:.3f}', 'adj R2': f'{f.rsquared_adj:.3f}', 'model p': f'{f.f_pvalue:.3f}'})
    if kind == 'forest':
        fit = forest().fit(df[X].values, y)
        p = fit.predict(df[X].values)
        row['apparent R2'] = f'{1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}'
        _, _, pv = permutation_test_score(
            forest(200),
            df[X].values,
            y,
            cv=KFold(5, shuffle=True, random_state=42),
            scoring='neg_root_mean_squared_error',
            n_permutations=500,
            random_state=42,
            n_jobs=-1,
        )
        row['permutation p'] = f'{pv:.3f}'
    rows.append(row)
T = pd.DataFrame(rows).fillna('')
print(T.to_string(index=False))
T.to_csv(f'{OUT}/Table4.csv', index=False)
