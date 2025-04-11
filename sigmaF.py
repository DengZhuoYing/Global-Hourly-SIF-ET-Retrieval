import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import shap
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score
import seaborn as sns


# TYPE = '0-1'
#
# scopefolder = fr'D:\Global_SIF_Simulate\scope模拟结果\2verification_run_2024-10-27-1043_0-1\\'
#
# dfout = pd.DataFrame(columns=['sigmaF', 'RED', 'NIR', 'SZA', 'LAI'])
#
# dfsigmaF = pd.read_csv(scopefolder + f'2sigmaF{TYPE}.csv', header=None, skiprows=2)
# dfout['sigmaF'] = dfsigmaF.iloc[:, 118] / np.pi
#
# dfrednir = pd.read_csv(scopefolder + f'2rso{TYPE}.csv', header=None, skiprows=2)
# dfout['RED'] = dfrednir.iloc[:, 249]
# dfout['NIR'] = dfrednir.iloc[:, 459]
#
# dflai = pd.read_csv(scopefolder + f'2pars_and_input_short{TYPE}.csv', header=None, skiprows=2)
# dfout['SZA'] = dflai.iloc[:, 6]
# dfout['LAI'] = dflai.iloc[:, 3]
#
# dfout.to_csv(f'scope/2scope{TYPE}.csv', index=False)
#
# df00 = pd.read_csv('scope/scope00.csv')
# df01 = pd.read_csv('scope/scope01.csv')
# df035015 = pd.read_csv('scope/scope035015.csv')
# df0_1 = pd.read_csv('scope/scope0-1.csv')
# df_10 = pd.read_csv('scope/scope-10.csv')
# df10 = pd.read_csv('scope/scope10.csv')
#
# df00_2 = pd.read_csv('scope/2scope00.csv')
# df01_2 = pd.read_csv('scope/2scope01.csv')
# df035015_2 = pd.read_csv('scope/2scope035015.csv')
# df0_1_2 = pd.read_csv('scope/2scope0-1.csv')
# df_10_2 = pd.read_csv('scope/2scope-10.csv')
# df10_2 = pd.read_csv('scope/2scope10.csv')
# dfout = pd.concat([df00, df01, df035015, df0_1, df_10, df10, df00_2, df01_2, df035015_2, df0_1_2, df_10_2, df10_2], axis=0)
#
# dfout.to_csv('scope/scope-all.csv', index=False)


def objective(trial, x, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 32, 128),
        # 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
        # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        # 'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1),
        # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
        # 'max_depth': trial.suggest_int('max_depth', 5, 20),
        # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        # 'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        # 'min_sum_hessian_in_leaf': trial.suggest_loguniform('min_sum_hessian_in_leaf', 1e-3, 100),
        # 'max_bin': trial.suggest_int('max_bin', 255, 1000),
        # 'metric': trial.suggest_categorical('metric', ['rmse', 'mae', 'huber', 'fair']),
        # 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42
    }

    model = lgb.LGBMRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, x, y, cv=cv, scoring=make_scorer(r2_score)).mean()

    return score


df = pd.read_csv('scope/scope-all.csv')
df['NIRv'] = ((df['NIR'] - df['RED']) / (df['NIR'] + df['NIR'])) * df['NIR']
df = df.drop_duplicates()
df = df.dropna()
df = df.reset_index(drop=True)
y = df['sigmaF']
x = df[['NIR', 'RED', 'NIRv', 'SZA', 'LAI']]

# sampler = optuna.samplers.TPESampler(seed=42)
# model = optuna.create_study(direction='maximize', sampler=sampler)
# func = lambda trial: objective(trial, x.values, y.values)
# model.optimize(func, n_trials=50)
# trial = model.best_trial

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.3, random_state=42)
# lgbm = lgb.LGBMRegressor(verbosity=-1, n_jobs=-1, random_state=42, n_estimators=trial.params['n_estimators'], learning_rate=trial.params['learning_rate'], num_leaves=trial.params['num_leaves'])
#
# lgbm.fit(x_train, y_train)
# joblib.dump(lgbm, 'scope/scope_lgbm_5.pkl')
lgbm = joblib.load(r"scope/scope_lgbm_5.pkl")
print(lgbm.get_params())

y_predicted = lgbm.predict(x_test, n_jobs=-1)
r2 = r2_score(y_test, y_predicted)
rmse = np.sqrt(np.mean((y_predicted - y_test) ** 2))
print(f'R2: {r2}')
print(f'RMSE: {rmse}')
lr = LinearRegression()
lr.fit(y_test.reshape(-1, 1), y_predicted.reshape(-1, 1))
print(f'回归系数a: {lr.coef_[0][0]}')
print(f'回归系数b: {lr.intercept_[0]}')

r2 = stats.pearsonr(y_test, y_predicted)[0] ** 2
lr = LinearRegression()
lr.fit(y_test.reshape(-1, 1), y_predicted.reshape(-1, 1))
rmse = np.sqrt(np.mean((y_predicted - y_test) ** 2))
sample_size = 20000  # 设置要显示的散点数量
indices = np.random.choice(len(y_test), size=sample_size, replace=False)
sampled_y_test = y_test[indices]
sampled_y_predicted = y_predicted[indices]
lr_y_predict = lr.predict(y_test.reshape(-1, 1))
xy = np.vstack([sampled_y_test, sampled_y_predicted])
z = np.abs(scipy.stats.gaussian_kde(xy)(xy))
plt.scatter(sampled_y_test, sampled_y_predicted, c=z / 1000, cmap='viridis', s=np.sqrt(z) / 2)
plt.plot(y_test, lr_y_predict, color='red')
plt.plot([0, 10], [0, 10], '--', color='black')
plt.xlabel('Escape ratio by SCOPE', fontsize=14)
plt.ylabel('Escape ratio by LightGBM', fontsize=14)
plt.annotate(f'y = {lr.coef_[0][0]:.2f}x+{lr.intercept_[0]:.2f}', xy=(0.06, 0.93), xycoords='axes fraction',
             fontsize=14)
plt.annotate(f'R\u00b2 = {r2:.3f}', xy=(0.06, 0.86), xycoords='axes fraction', fontsize=14)
plt.annotate(f'RMSE={rmse:.3f}', xy=(0.06, 0.79), xycoords='axes fraction', fontsize=14)
# plt.annotate(f'n=20000', xy=(0.08, 0.78), xycoords='axes fraction', fontsize=14,)
plt.xlim(0, 0.25)
plt.ylim(0, 0.25)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(np.arange(0, 0.25, 0.05), fontsize=14)
plt.yticks(np.arange(0, 0.25, 0.05), fontsize=14)
# plt.title('LightGBM Regressor', fontsize=14, fontweight='bold')
# cb = plt.colorbar()
# cb.set_label('Density', fontsize=14)
# cb.ax.tick_params(labelsize=14, width=2, length=5, color='black')
plt.clim(0, 3)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig(fr'TOTAL COVERT.png', dpi=300)
plt.show()