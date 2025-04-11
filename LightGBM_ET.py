import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")


def objective(trial, x, y, x2, y2):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 90, 110),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 100, 1000),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.2, 1),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.2, 1),
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 1000),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 1000),
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42
    }

    modeltrain = lgb.LGBMRegressor(**params)
    modeltrain.fit(x, y)
    y_model = modeltrain.predict(x2, n_jobs=-1)
    score = np.sqrt(np.mean((y2 - y_model) ** 2))

    return score


df = pd.read_csv(r"ET_global_final_clean.csv")

dftrain, dfvalidationandtest = train_test_split(df, test_size=0.3, random_state=42)
dfvalidation, dftest = train_test_split(dfvalidationandtest, test_size=0.5, random_state=42)
print(dftrain.shape, dfvalidation.shape, dftest.shape)

driven_features = ['LANDCOVER', 'APAR', 'T2M', 'SM', 'VPD', 'DEM', 'PAR', 'FPAR']
predict_features = 'ETobs'

# x = dftrain[driven_features].values
# y = dftrain[predict_features].values
# x2 = dfvalidation[driven_features].values
# y2 = dfvalidation[predict_features].values
# sampler = optuna.samplers.TPESampler(seed=42)
# model = optuna.create_study(direction='minimize', sampler=sampler)
# func = lambda trial: objective(trial, x, y, x2, y2)
# model.optimize(func, n_trials=500)
# trial = model.best_trial

x_train = dftrain[driven_features].values
y_train = dftrain[predict_features].values
x_test = dftest[driven_features].values
y_test = dftest[predict_features].values

# lgbm = lgb.LGBMRegressor(verbosity=-1, n_jobs=-1, random_state=42, n_estimators=trial.params['n_estimators'],
#                          learning_rate=trial.params['learning_rate'], num_leaves=trial.params['num_leaves'],
#                          bagging_fraction=trial.params['bagging_fraction'],
#                          bagging_freq=trial.params['bagging_freq'],
#                          feature_fraction=trial.params['feature_fraction'],
#                          min_sum_hessian_in_leaf=trial.params['min_sum_hessian_in_leaf'],
#                          min_data_in_leaf=trial.params['min_data_in_leaf'])
# joblib.dump(lgbm, 'ET_lgbm.pkl')
lgbm = joblib.load(r"..\生产\ET_lgbm.pkl")
print(lgbm.get_params())

start = time.time()
y_predicted = lgbm.predict(x_test, n_jobs=-1)
end = time.time()
print(f'运行时间：{end - start}')
r2 = stats.pearsonr(y_test, y_predicted)[0] ** 2
rmse = np.sqrt(np.mean((y_predicted - y_test) ** 2))
print(f'R2: {r2}')
print(f'RMSE: {rmse}')
lr = LinearRegression()
lr.fit(y_test.reshape(-1, 1), y_predicted.reshape(-1, 1))
print(f'回归系数a: {lr.coef_[0][0]}')
print(f'回归系数b: {lr.intercept_[0]}')

# r2 = stats.pearsonr(y_test, y_predicted)[0] ** 2
# lr = LinearRegression()
# lr.fit(y_test.reshape(-1, 1), y_predicted.reshape(-1, 1))
# rmse = np.sqrt(np.mean((y_predicted - y_test) ** 2))
# sample_size = 20000  # 设置要显示的散点数量
# indices = np.random.choice(len(y_test), size=sample_size, replace=False)
# sampled_y_test = y_test[indices]
# sampled_y_predicted = y_predicted[indices]
# lr_y_predict = lr.predict(y_test.reshape(-1, 1))
# xy = np.vstack([sampled_y_test, sampled_y_predicted])
# z = np.abs(scipy.stats.gaussian_kde(xy)(xy))
# plt.scatter(sampled_y_test, sampled_y_predicted, c=z * 40000, cmap='viridis', s=20 * (z + 1))
# plt.plot(y_test, lr_y_predict, color='red')
# plt.plot([0, 550], [0, 550], '--', color='black')
# plt.xlabel('Observed ET (W/m\u00b2)', fontsize=14)
# plt.ylabel('Predicted ET (W/m\u00b2)', fontsize=14)
# plt.annotate(f'y = {lr.coef_[0][0]:.2f}x+{lr.intercept_[0]:.2f}', xy=(0.56, 0.16), xycoords='axes fraction',
#              fontsize=14)
# plt.annotate(f'R\u00b2 = {r2:.3f}', xy=(0.56, 0.09), xycoords='axes fraction', fontsize=14)
# plt.annotate(f'RMSE={rmse:.3f}', xy=(0.56, 0.02), xycoords='axes fraction', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.clim(0, 3)
# plt.gca().set_aspect('equal')
# plt.tight_layout()
# plt.savefig(fr'ET.png', dpi=300)
# plt.show()
