import pandas as pd
import os, sys

sys.path.append(os.path.abspath('../utils'))
import numpy as np
import surd as surd
from joblib import Parallel, delayed


def create_pfm(s, a, dt, bins):
    V = np.vstack([s[dt:], [a[i][:-dt] for i in range(len(a))]]).T
    h, _ = np.histogramdd(V, bins=bins)
    return h / h.sum()


def mysurd(q1, q2, q3, q4, target, nbins):
    nlags_range = range(1, 61, 1)
    agents = (q1, q2, q3, q4)
    unique_lag = []
    for dt in nlags_range:
        hist = create_pfm(target, agents, dt, nbins)
        Rd, Sy, MI, info_leak = surd.surd(hist)
        single_digit_keys = [key for key in Rd.keys() if len(key) == 1]
        sum_causalities = sum(Rd[key] for key in single_digit_keys)
        unique_lag.append(sum_causalities)
    dt_best = np.argmax(unique_lag) + 1
    # print(f"Best lag: {dt_best}")
    hist = create_pfm(target, agents, dt_best, nbins)
    Rd, Sy, MI, info_leak = surd.surd(hist)
    r_ = {key: value / max(MI.values()) for key, value in Rd.items()}
    # surd.nice_print(Rd, Sy, MI, info_leak)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for key in r_.keys():
        if 1 in key:
            sum1 += r_[key]
        if 2 in key:
            sum2 += r_[key]
        if 3 in key:
            sum3 += r_[key]
        if 4 in key:
            sum4 += r_[key]

    if sum1 > sum2 * 1.5 and sum1 > sum3 * 1.5 and sum1 > sum4 * 1.5:
        return 1
    elif sum2 > sum1 * 1.5 and sum2 > sum3 * 1.5 and sum2 > sum4 * 1.5:
        return 2
    elif sum3 > sum1 * 1.5 and sum3 > sum2 * 1.5 and sum3 > sum4 * 1.5:
        return 3
    elif sum4 > sum1 * 1.5 and sum4 > sum2 * 1.5 and sum4 > sum3 * 1.5:
        return 4
    else:
        return 0


dfSIF = pd.read_csv(r'SIFoco-daily.csv')
dfpar = pd.read_csv(r'par-daily.csv')
dfvpd = pd.read_csv(r'vpd-daily.csv')
dfsm = pd.read_csv(r'sm-daily.csv')
dft2m = pd.read_csv(r't2m-daily.csv')


def calculate_surd(i):
    try:
        q1 = np.nan_to_num(dfpar.iloc[i].values)
        q2 = np.nan_to_num(dfvpd.iloc[i].values)
        q3 = np.nan_to_num(dfsm.iloc[i].values)
        q4 = np.nan_to_num(dft2m.iloc[i].values)
        target = np.nan_to_num(dfSIF.iloc[i].values)
        print(f"Calculating index {i}")
        return mysurd(q1, q2, q3, q4, target, 10)
    except Exception as e:
        print(f"Error at index {i}: {e}")
        return 0


surdlist = Parallel(n_jobs=-1)(delayed(calculate_surd)(i) for i in range(len(dfSIF)))
dfSIFout = pd.DataFrame({'surd': surdlist})
dfSIFout.to_csv(r'surd-SIFoco.csv', index=False)
