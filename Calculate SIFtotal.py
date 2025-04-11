import pandas as pd
import numpy as np
import joblib
from scipy.stats import pearsonr

lgbm = joblib.load(r'D:\bishe\code\扩展至总SIF\scope\scope_lgbm_5.pkl')

df2 = pd.read_csv(r"OCO2_global_final.csv")
df3 = pd.read_csv(r"OCO3_global_final.csv")

df = pd.concat([df2, df3], axis=0)

SZA = df['SZA']
LAI = df['LAI']
CI = df['CI']
NIR = df['NIR']
RED = df['RED']
SIFobs = df['SIFobs']
L = df['L']

NIRV = ((NIR - RED) / (NIR + RED)) * NIR
fai1 = 0.5 - 0.663 * L - 0.33 * L ** 2
fai2 = 0.877 * (1 - 2 * fai1)
G = fai1 + fai2 * np.cos(SZA * np.pi / 180)
i0 = 1 - np.exp(-G * LAI * CI / np.cos(SZA * np.pi / 180))
fesc = NIRV / (np.pi * i0 * 1.2)
SIFtotal1 = SIFobs / fesc

x2 = pd.DataFrame({'NIR': NIRV, 'RED': RED, 'NIRv': NIRV, 'SZA': SZA, 'LAI': LAI})
SIFtotal2 = SIFobs / lgbm.predict(x2)

df['SIFtotal1'] = SIFtotal1
df['SIFtotal2'] = SIFtotal2

SIFtotal1=df['SIFtotal1'].values
SIFtotal2=df['SIFtotal2'].values

print(pearsonr(SIFtotal1, SIFtotal2))

# df.to_csv(r'OCO_global_final_total.csv', index=False)
