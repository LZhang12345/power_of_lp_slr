import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import gzip
from tqdm import tqdm
import re
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import pearsonr
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import seaborn as sns
import matplotlib.pyplot as plt
# ===================================================================
# 1. DEFINE RE-RANKING FUNCTIONS
# (Paste your existing 'solve_by_LP', 'apply_binary_reranking', 
#  and 'perform_reranking_on_combined_data' functions here)
# ===================================================================

def find_x(p, r, v, mu): # given p r v mu, find the optimal solution
    m = len(p)
    n = len(r)
    X = np.zeros((m,n))
    deri = v + mu*r # get the deriviative
    sort = np.argsort(deri)[::-1][:m] # rank the deriviative and get the index
    #print("index", sort)
    for i in range(len(sort)): # we get an integral solution 
        X[i, sort[i]] = 1
    # by proof, we can convert the fractional solution to integer ones 
    # therefore, we can still deal with the situation with repeating p same as before
    return X
    
def search_mu(p,r,v,lamb,a, ite, eps): # search for the range of the optimal mu
    # ite: iteration number
    # eps: epslion
    m = len(p)
    n = len(r)
    sorted_v = np.sort(v)[::-1]
    sorted_r = np.sort(r)[::-1]
    mu_temp = max(1,abs((sorted_v[-1] - sorted_v[0])/(sorted_r[-1] - sorted_r[0])))
    mu_high = mu_temp
    mu_low = 0
    
    i = 1
    #print("p@x", p@find_x(p, r, v, mu_high))
    # binary search for the range of mu
    while np.sign(p@find_x(p, r, v, mu_high)@r.T - lamb*a) == np.sign(p@find_x(p, r, v, mu_low)@r.T - lamb*a) and i < 10:
        #print(i, "-----")
        #print(mu_high, mu_low)
        #print(p@find_x(p, r, v, mu_high)@r.T - lamb*a, p@find_x(p, r, v, mu_low)@r.T - lamb*a)
        mu_temp = 0.5*(mu_high + mu_low)
        if p@find_x(p, r, v, mu_temp)@r.T - lamb*a >=0:
            mu_high = mu_temp
            mu_low = 0.5*mu_low
        else:
            mu_low = mu_temp
            mu_high = 2*mu_high
        i += 1
    mu_temp = 0.5*(mu_high + mu_low)
    #print(i, "-----")
    #print(mu_high, mu_low)
    #print(p@find_x(p, r, v, mu_high)@r.T - lamb*a, p@find_x(p, r, v, mu_low)@r.T - lamb*a)
    i = 1
    # narrow down the range 
    while i <= ite and abs(p@find_x(p, r, v, mu_temp)@r.T-lamb*a)>eps:
        #print(i, "-----")
        #print(mu_high, mu_low)
        #print(p@find_x(p, r, v, mu_high)@r.T - lamb*a, p@find_x(p, r, v, mu_low)@r.T - lamb*a)
        mu_temp = 0.5*(mu_high + mu_low)
        if p@find_x(p, r, v, mu_temp)@r.T-lamb*a >=0:
            mu_high = mu_temp
        else:
            mu_low = mu_temp
        i += 1
    mu_temp = 0.5*(mu_high + mu_low)
    #print(i, "-----")
    #print(mu_high, mu_low)
    #print(p@find_x(p, r, v, mu_high)@r.T - lamb*a, p@find_x(p, r, v, mu_low)@r.T - lamb*a)
    return mu_high, mu_low

def solve_by_LP(r, v, p, m, n, lamb):
    a = np.dot(p, np.sort(r)[::-1][:m])
    # find the ranking with the largest revenue
    sort = np.argsort(v)[::-1][:m]
    X = np.zeros((m,n))
    for i in range(len(sort)): # we get an integral solution 
        X[i, sort[i]] = 1
    #check if the max revenue ranking is feasible 
    if p@X@r.T >= lamb*a:
        return X #(0,0) # optimal 
    mu_high, mu_low = search_mu(p,r,v,lamb,a, ite=10, eps=1e-4)
    return find_x(p, r, v, mu_high)
    #X_approx = convertion(p,r,v,lamb,a, mu_high, mu_low)
    #if p@find_x(p, r, v, mu_high)@r.T >= lamb*a:
    #    #print(p@find_x(p, r, v, mu_high)@v.T, p@find_x(p, r, v, mu_high)@v.T + mu_high*(p@find_x(p, r, v, mu_high)@r.T - lamb*a))
    #    # return the gap
    #    # return ((mu_high*(p@find_x(p, r, v, mu_high)@r.T - lamb*a))/(p@find_x(p, r, v, mu_high)@v.T), 0)
    #else:
    #    return (0, (p@find_x(p, r, v, mu_high)@r.T - lamb*a)/lamb*a)

def apply_binary_reranking(group_df, sol):
    # Make sure sol is a NumPy array
    sol = np.array(sol, dtype=int)

    # Convert DataFrame to numpy array
    original_array = group_df.to_numpy()

    # Apply the binary re-ranking
    reranked_array = sol @ original_array

    # Convert back to DataFrame
    reranked_df = pd.DataFrame(reranked_array, columns=group_df.columns)

    return reranked_df

def perform_reranking_on_combined_data(df_combined):
    """
    Perform re-ranking on the entire combined dataset, grouped by meid
    (This is your function from the prompt)
    """
    print('Starting re-ranking on combined data...')

    reranked_dfs_85 = []
    reranked_dfs_90 = []
    reranked_dfs_95 = []
    score_ranked_dfs = []
    
    click_propensity = [0.0112, 0.0113, 0.0106, 0.0095, 0.0094, 0.0091, 0.002, 0.0022, 0.002, 0.002, 0.0021, 0.0023]
    lenC = len(click_propensity)
    
    # Group by meid across the entire dataset
    grouped = df_combined.groupby('meid')
    
    for meid, group_df in tqdm(grouped, desc="Re-ranking impressions"):
        # we want to solve for the lp-based using the same impression data
        # v is revenue, r is relevance
        v = group_df['Expected_Revenue'].to_numpy()
        r = group_df['mlrModelScore'].to_numpy()
        p = np.array(click_propensity[:min(lenC, len(group_df))])

        m = min(lenC, len(group_df))
        n = len(group_df)

        # Solve the problem
        sol_85 = solve_by_LP(r, v, p, m, n, lamb = 0.85)
        sol_90 = solve_by_LP(r, v, p, m, n, lamb = 0.90)
        sol_95 = solve_by_LP(r, v, p, m, n, lamb = 0.95)

        reranked_df_85 = apply_binary_reranking(group_df, sol_85)
        reranked_dfs_85.append(reranked_df_85)
        reranked_df_90 = apply_binary_reranking(group_df, sol_90)
        reranked_dfs_90.append(reranked_df_90)
        reranked_df_95 = apply_binary_reranking(group_df, sol_95)
        reranked_dfs_95.append(reranked_df_95)
        
        score_ranked_dfs.append(group_df.iloc[:m])
    
    print(f'Re-ranking completed. Processed {len(grouped)} impressions.')
    return reranked_dfs_85, reranked_dfs_90, reranked_dfs_95, score_ranked_dfs

# ===================================================================
# 2. THE NEW, SIMPLE SCRIPT
# (This replaces your complex file-loading script)
# ===================================================================

print(" Starting data processing for public analysis...")

# Define the single file to read
synthetic_data_file = "./synthetic_offline_data.csv"

try:
    # --- Step 1: Load the single synthetic CSV file ---
    # This one line replaces your entire parallel-loading logic
    print(f" Loading synthetic data from '{synthetic_data_file}'...")
    df_combined = pd.read_csv(synthetic_data_file)
    
    print(f" Synthetic data loaded. Shape: {df_combined.shape}")
    print(f" Number of unique meids: {df_combined['meid'].nunique()}")

    # --- Step 2: Perform re-ranking on the big combined dataset ---
    print(" Starting re-ranking on synthetic dataset...")
    
    # Call your existing function (defined in Step 1)
    reranked_85, reranked_90, reranked_95, score_ranked_all = \
        perform_reranking_on_combined_data(df_combined)
    
    # --- Step 3: Convert to numpy arrays (as in your original script) ---
    print(" Converting to numpy arrays...")
    
    all_reranked_85 = [df.to_numpy() for df in reranked_85]
    all_reranked_90 = [df.to_numpy() for df in reranked_90]
    all_reranked_95 = [df.to_numpy() for df in reranked_95]
    all_score_ranked = [df.to_numpy() for df in score_ranked_all]
        
    print(f" Re-ranking completed!")
    print(f" Processed {len(all_reranked_85)} re-ranked impressions")
    print(f" Processed {len(all_reranked_90)} re-ranked impressions")
    print(f" Processed {len(all_reranked_95)} re-ranked impressions")
    print(f" Processed {len(all_score_ranked)} score-ranked impressions")

except FileNotFoundError:
    print(f" ERROR: Cannot find the file '{synthetic_data_file}'.")
    print("Please make sure the file is in the same directory as your script.")
except Exception as e:
    print(f"An error occurred: {e}")

print("🎉 Process completed!")


# --- Step 1: Get Clean Sets of Selected itemIds for ALL groups ---
print(" Step 1: Extracting clean sets of selected itemIds...")

# Extract itemIds for all three algorithm outputs
lp85_selected_ids = {str(item_id) for arr in all_reranked_85 for item_id in arr[:, 4]}
lp90_selected_ids = {str(item_id) for arr in all_reranked_90 for item_id in arr[:, 4]}
lp95_selected_ids = {str(item_id) for arr in all_reranked_95 for item_id in arr[:, 4]}
score_selected_ids = {str(item_id) for arr in all_score_ranked for item_id in arr[:, 4]}

# The universe is now the union of ALL items selected by any algorithm
universe_ids = lp90_selected_ids.union(lp95_selected_ids).union(score_selected_ids)
print(f" Found {len(universe_ids)} unique items in the total universe.")


# --- Step 2: Build the SINGLE Master Item Features Table ---
# This expensive step is now only done ONCE.
print("\n Step 2: Building the master features table...")

# Filter df_combined before using it to build features
print(f"Original shape of df_combined: {df_combined.shape}")
df_combined = df_combined[df_combined['Item_NCurrentPrice'] > 0].copy()
print(f"New shape after filtering Item_NCurrentPrice > 0: {df_combined.shape}")

# Ensure itemId in the source DataFrame is a string
df_combined['itemId'] = df_combined['itemId'].astype(str)

# Calculate appearance counts from the full candidate set
item_counts = df_combined['itemId'].value_counts().to_dict()

# Aggregate core features
feature_cols = ['itemId', 'adRate', 'mlrModelScore', 'Item_NCurrentPrice', 'Expected_Revenue', 'Organic_Revenue', 'Ads_Revenue']
df_features_agg = df_combined[feature_cols].groupby('itemId').mean().reset_index()

# Create appearance DataFrame
df_appearance = pd.DataFrame(item_counts.items(), columns=['itemId', 'appearance'])
df_appearance['itemId'] = df_appearance['itemId'].astype(str)

# Merge to create the final, complete features table
df_features = pd.merge(df_features_agg, df_appearance, on='itemId', how='left')
print(f" Master features table created with {len(df_features)} items.")


# --- Step 3: Create the UNIFIED Final Model DataFrame ---
print("\n Step 3: Creating the final model-ready DataFrame...")

# Filter the master features table to only items in our universe
df_model = df_features[df_features['itemId'].isin(universe_ids)].copy()

# Add BOTH dependent variables to this single DataFrame
df_model['Selected_by_LP90'] = df_model['itemId'].isin(lp90_selected_ids).astype('int8')
df_model['Selected_by_LP95'] = df_model['itemId'].isin(lp95_selected_ids).astype('int8')

df_model['LogPrice'] = np.log(df_model['Item_NCurrentPrice'])

print(f" Final model-ready DataFrame shape: {df_model.shape}")
print("\nDataFrame ready for all selection regressions:")
print(df_model.head())
print("\nDistribution of LP90 selection:")
print(df_model['Selected_by_LP90'].value_counts(normalize=True))
print("\nDistribution of LP95 selection:")
print(df_model['Selected_by_LP95'].value_counts(normalize=True))


# --- Run Regressions for LP90 ---
print("\n--- Fitting Selection Models for LP90 ---")
formula_platform_90 = "Selected_by_LP90 ~ Expected_Revenue + appearance"
results_platform_90 = smf.logit(formula=formula_platform_90, data=df_model).fit()
print(results_platform_90.summary())

formula_platform_90 = "Selected_by_LP90 ~ Organic_Revenue + Ads_Revenue + appearance"
results_platform_90 = smf.logit(formula=formula_platform_90, data=df_model).fit()
print(results_platform_90.summary())

# --- Run Regressions for LP95 ---
print("\n\n--- Fitting Selection Models for LP95 ---")
formula_platform_95 = "Selected_by_LP95 ~ Expected_Revenue + appearance"
results_platform_95 = smf.logit(formula=formula_platform_95, data=df_model).fit()
print(results_platform_95.summary())

formula_platform_95 = "Selected_by_LP95 ~ Organic_Revenue + Ads_Revenue + appearance"
results_platform_95 = smf.logit(formula=formula_platform_95, data=df_model).fit()
print(results_platform_95.summary())

print("Kicking off Task 2: The Conditional Ranking Model...")

# --- Step 1: Calculate Appearance Counts from the Full Candidate Set ---
# This mirrors the successful logic from your reference code.
print("Calculating item appearance counts from df_combined...")
df_combined['itemId'] = df_combined['itemId'].astype(str)
# We keep this as a pandas Series, as the .map() method is very efficient
item_counts = df_combined['itemId'].value_counts()


# --- Step 2: Reconstruct DataFrames from the ORIGINAL 11-column arrays ---
print("Reconstructing ranked data...")
# CORRECT: We use the 11-column list because 'appearance' will be added next.
COLUMNS_ORIGINAL = [
    'meid', 'adRate', 'userId', 'sellerId', 'itemId', 'mlrModelScore',
    'Item_NCurrentPrice', 'Seed_Revenue', 'Expected_Revenue', 'Organic_Revenue', 'Ads_Revenue',
    'labelPurchase', 'labelClick'
]
df_lp_ranked_85 = pd.DataFrame(np.vstack(all_reranked_85), columns=COLUMNS_ORIGINAL)
df_lp_ranked_90 = pd.DataFrame(np.vstack(all_reranked_90), columns=COLUMNS_ORIGINAL)
df_lp_ranked_95 = pd.DataFrame(np.vstack(all_reranked_95), columns=COLUMNS_ORIGINAL)
df_score_ranked = pd.DataFrame(np.vstack(all_score_ranked), columns=COLUMNS_ORIGINAL)


# --- Step 3: Add 'appearance' Column and Standardize Data Types ---
print(" Adding 'appearance' column and standardizing types...")
for df in [df_score_ranked, df_lp_ranked_85, df_lp_ranked_90, df_lp_ranked_95]:
    # Standardize all ID columns to strings FIRST for safe mapping and merging
    for col in ['meid', 'userId', 'sellerId', 'itemId']:
        df[col] = df[col].astype(str)
    # Add the appearance column using the robust .map() method
    df['appearance'] = df['itemId'].map(item_counts)


# --- Step 4: Add Rank Columns ---
df_score_ranked['score_rank'] = df_score_ranked.groupby('meid').cumcount() + 1
df_lp_ranked_85['lp_rank'] = df_lp_ranked_85.groupby('meid').cumcount() + 1
df_lp_ranked_90['lp_rank'] = df_lp_ranked_90.groupby('meid').cumcount() + 1
df_lp_ranked_95['lp_rank'] = df_lp_ranked_95.groupby('meid').cumcount() + 1

# Convert Item_NCurrentPrice to numeric BEFORE taking log
for df in [df_score_ranked, df_lp_ranked_85, df_lp_ranked_90, df_lp_ranked_95]:
    df['Item_NCurrentPrice'] = pd.to_numeric(df['Item_NCurrentPrice'], errors='coerce')

df_score_ranked['LogPrice'] = np.log(df_score_ranked['Item_NCurrentPrice'])
df_lp_ranked_85['LogPrice'] = np.log(df_lp_ranked_85['Item_NCurrentPrice'])
df_lp_ranked_90['LogPrice'] = np.log(df_lp_ranked_90['Item_NCurrentPrice'])
df_lp_ranked_95['LogPrice'] = np.log(df_lp_ranked_95['Item_NCurrentPrice'])

# --- Step 5: Create Merged DataFrames for Each Comparison ---
print("Merging to find overlapping items...")
df_merged_85 = pd.merge(df_score_ranked, df_lp_ranked_85[['meid', 'itemId', 'lp_rank']], on=['meid', 'itemId'], how='inner')
df_merged_90 = pd.merge(df_score_ranked, df_lp_ranked_90[['meid', 'itemId', 'lp_rank']], on=['meid', 'itemId'], how='inner')
df_merged_95 = pd.merge(df_score_ranked, df_lp_ranked_95[['meid', 'itemId', 'lp_rank']], on=['meid', 'itemId'], how='inner')


# --- Step 6: Create the 'Promoted' Dependent Variable ---
df_merged_85['Promoted'] = (df_merged_85['lp_rank'] < df_merged_85['score_rank']).astype(int)
df_merged_90['Promoted'] = (df_merged_90['lp_rank'] < df_merged_90['score_rank']).astype(int)
df_merged_95['Promoted'] = (df_merged_95['lp_rank'] < df_merged_95['score_rank']).astype(int)


# --- Step 7: Optimize Memory for Both DataFrames ---
print("\n⚙️ Optimizing memory for both merged DataFrames...")
for df in [df_merged_85, df_merged_90, df_merged_95]:
    # First, ensure all feature columns are numeric, converting from 'object' if needed
    for col in ['adRate', 'mlrModelScore', 'LogPrice', 'Expected_Revenue', 'Organic_Revenue', 'Ads_Revenue', 'appearance']:
         df[col] = pd.to_numeric(df[col], errors='coerce')
    # Now, downcast to more efficient types
    for col in ['adRate', 'mlrModelScore', 'LogPrice', 'Expected_Revenue', 'Organic_Revenue', 'Ads_Revenue']:
        df[col] = df[col].astype('float32')
    for col in ['score_rank', 'lp_rank', 'appearance', 'Promoted']:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in ['meid', 'itemId', 'userId', 'sellerId']:
        df[col] = df[col].astype('category')


# --- Run Regressions for LP90 Ranking ---
print("\n--- Fitting Ranking Models for LP90 ---")

formula_rank_user = "Promoted ~ mlrModelScore + LogPrice + adRate + appearance"
results_rank_user_90 = smf.logit(formula=formula_rank_user, data=df_merged_90).fit()
print(results_rank_user_90.summary())

formula_rank_platform = "Promoted ~ Expected_Revenue + appearance"
results_rank_platform_90 = smf.logit(formula=formula_rank_platform, data=df_merged_90).fit()
print(results_rank_platform_90.summary())

formula_rank_revenue = "Promoted ~ Organic_Revenue + Ads_Revenue + appearance"
results_rank_revenue_90 = smf.logit(formula=formula_rank_revenue, data=df_merged_90).fit()
print(results_rank_revenue_90.summary())


# --- Run Regressions for LP95 Ranking ---
print("\n\n---  Fitting Ranking Models for LP95 ---")

results_rank_user_95 = smf.logit(formula=formula_rank_user, data=df_merged_95).fit()
print(results_rank_user_95.summary())

results_rank_platform_95 = smf.logit(formula=formula_rank_platform, data=df_merged_95).fit()
print(results_rank_platform_95.summary())

results_rank_revenue_95 = smf.logit(formula=formula_rank_revenue, data=df_merged_95).fit()
print(results_rank_revenue_95.summary())





