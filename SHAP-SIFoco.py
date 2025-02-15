import lightgbm as lgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from scipy.stats import pearsonr

# df = pd.read_csv(r"OCO_global_final_SIFobs.csv")
# df = df.sample(n=100000, random_state=0)
# df.to_csv(r"OCO_global_final_SIFobs_10w.csv", index=False)

# df = pd.read_csv(r"OCO_global_final_SIFobs_10w.csv")
# driven_features = ['LANDCOVER', 'COSSZA', 'APAR', 'T2M', 'SM', 'VPD', 'LON', 'LAT', 'DOY', 'DEM']
# predict_features = 'SIFobs'
# x = df[driven_features].values
# y = df[predict_features].values
# model = joblib.load('OCO_lgbm.pkl')
# y_predicted = model.predict(x, n_jobs=-1)
# print(pearsonr(y, y_predicted)[0] ** 2)

# explainer = shap.Explainer(model)
# shap_values = explainer(x)
# shap_values = shap_values.values
# shap_values = np.abs(shap_values).mean(0)
# shap_values = shap_values / shap_values.sum()
# print(df.columns)
# print(shap_values)

shap_values = [0.05083338, 0.04769681, 0.46113693, 0.09260134, 0.02379483, 0.13322546, 0.02804426, 0.0708913,
               0.06129851, 0.03047719]
shap_values = np.array(shap_values) * 100
categories = ['LC', 'cosSZA', 'APAR', r"$\mathrm{T}_{\mathrm{2m}}$", 'SM', 'VPD', 'LON', 'LAT', 'DOY', 'DEM']

categories = ['Solar', 'Temp', 'Water', 'Location', 'Others']
shap_values = [shap_values[2] + shap_values[1], shap_values[3], shap_values[5] + shap_values[4],
               shap_values[6] + shap_values[7],
               shap_values[0] + shap_values[8] + shap_values[9]]

categories = [x for _, x in sorted(zip(shap_values, categories), reverse=True)]
shap_values = sorted(shap_values, reverse=True)

values = shap_values

colors = [(251 / 255, 180 / 255, 99 / 255), (128 / 255, 177 / 255, 211 / 255), (189 / 255, 186 / 255, 219 / 255),
          (251 / 255, 248 / 255, 180 / 255), (244 / 255, 127 / 255, 114 / 255)]

plt.figure(figsize=(6, 6))
bars = plt.bar(categories, values, color=colors, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f'{height:.1f}%',
             ha='center', va='bottom', fontsize=18, color='black')

plt.xlabel('Input variables', fontsize=18)
plt.ylabel('Relative importance (%)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 55)
plt.xlim(-0.6, 4.6)
plt.tight_layout()
plt.savefig('SHAP-SIFoco.png', dpi=300)
plt.show()
