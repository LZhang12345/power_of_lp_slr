#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 19:29:39 2025

@author: zhangluyang
"""

import numpy as np
import pandas as pd
import gzip
import os
from tqdm import tqdm
import numpy.lib.recfunctions as rfn
from IPython.display import clear_output
from multiprocessing import Pool
import timeit
import random
import json
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

# --- 1) Read raw CSV as strings so we can clean reliably ---
names = ['LogPrice',
         'WeightedLogPrice',
         'TransactionPrice',
         'PurchaseLabel',
         'ClickLabel',
         'Size',
         'AveragePrice',
         'WeightedPrice',
         'Revenue',
         'WeightedPTR',
         'AverageTranRanking',
         '90',
         '95',
         'DARWO',
         'Date']

df = pd.read_csv(
    "./synthetic_data_online.csv",
    names=names,
    header=None,
    dtype=str,                         # read everything as string first
    na_values=['', 'NA', 'NaN', 'null', 'None']
)

# --- 2) Trim whitespace in all string columns ---
for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = df[c].str.strip()

# --- 3) Convert numeric columns (strip commas/$ first, then to_numeric) ---
num_cols = ['LogPrice',
             'WeightedLogPrice',
             'TransactionPrice',
             'PurchaseLabel',
             'ClickLabel',
             'Size',
             'AveragePrice',
             'WeightedPrice',
             'Revenue',
             'WeightedPTR',
             'AverageTranRanking',
             '90',
             '95',
             'DARWO']

for c in num_cols:
    df[c] = pd.to_numeric(
        df[c].str.replace(r'[,\$]', '', regex=True),  # remove thousand separators or $
        errors='coerce'
    )

# --- 4) Create syntactic treatment columns used by the formula ---
df['T90'] = pd.to_numeric(df['90'], errors='coerce')
df['T95'] = pd.to_numeric(df['95'], errors='coerce')

# --- 5) Ensure Date is string for C(Date) (categorical fixed effects) ---
df['Date'] = df['Date'].astype(str)

# --- 6) (Optional but recommended) drop rows missing any key model vars ---
key_vars = ['Revenue','LogPrice','WeightedPTR','T90','T95','Date']
df = df.dropna(subset=key_vars)

df_90_vs_control = df[df['T95'] != 1]
model = smf.ols(
    formula="""
        Revenue ~ T90
        + LogPrice + WeightedPTR 
        + T90:LogPrice
        + C(Date)
    """,
    data=df_90_vs_control
).fit(cov_type='HC1')

print(model.summary())

df_95_vs_control = df[df['T90'] != 1]
model = smf.ols(
    formula="""
        Revenue ~ T95 
        + LogPrice + WeightedPTR 
        + T95:LogPrice
        + C(Date)
    """,
    data=df_95_vs_control
).fit(cov_type='HC1')

print(model.summary())