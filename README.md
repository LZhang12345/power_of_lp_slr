# **Why This Synthetic Data Method?**
The primary motivation for this synthetic data generator is to create a high-fidelity, privacy-preserving dataset that enables public, reproducible research. Our original data is proprietary and cannot be shared, but our analysis and re-ranking algorithms are valuable to the community.

To allow others to run, validate, and build upon our code, we must provide a dataset that mirrors the distributions of the original data without containing any of the real, sensitive information.

# **What Is Our Method?**

**Offline data**: our method is a Two-Stage, Non-Parametric, Hybrid Generator. It models each feature's distribution by building a precise "lookup table" from the original data, rather than assuming a parametric disitbrution family. 

- **Stage 1: Impression-Level Generation**

  First, we model the top-level structure of the data: the impressions.

  - Model Impression Size: We learn the empirical Probability Mass Function (PMF) of the impression sizes (i.e., the count of items per impression). This gives us a discrete probability $P(k)$ for an impression having $k$ items.
  - Generate Impressions: We generate $N$ new impressions by sampling from this PMF.
  - Generate Impression Features: We use our generator to create $N$ samples of the impression-level features (like Seed Revenue and Sedd Price) — one for each new impression.

- **Stage 2: Item-Level Generation and Stitching**

  Second, we generate all the item-level features and stitch the data together.

  - Model Item Features: We fit our generator on the full, item-level dataset to learn the distributions of item-level features (like adRate, mlrModelScore, etc.).
  
  - Generate Items: We calculate the total number of items needed ($T = \sum_{i=1}^{N} k_i$) and generate $T$ samples of each item-level feature.
  
  - Stitch Data: We construct the final dataframe by "stretching" the $N$ impression-level features to match their corresponding item counts and concatenating them with the $T$ item-level features. We also generate new, unique meid, userId, sellerId, and itemId values.

**Online Data**: 
For the online dataset, the challenge is slightly different. The data is flat (one row per impression), removing the need for hierarchical stitching, but the statistical challenges -- extreme skew and zero-inflation -- remain just as critical.

Our method for the online data is a Single-Stage, Tuned Non-Parametric Generator. Unlike the offline method, which requires two stages to handle nesting, this method generates all impression-level features simultaneously but independently.

## **Mathematical Foundation: The Tuned Empirical Generator**

Both the online and offline pipelines rely on the same core engine to generate individual features: a Tuned Non-Parametric Generator. This engine addresses extreme skew, zero-inflation, and high variance through three integrated mathematical steps.

**1. Log-Space Stabilizing Transformation**

To handle extreme right-skew (where standard deviations are much larger than means), we do not model the raw feature $X$. Instead, we model its log-transformation $Y$, which stabilizes variance and creates a more tractable distribution:  
$$Y = \log(1 + X)$$  
After generation, synthetic values are transformed back, ensuring strict non-negativity and restoring the original skew:  
$$X_{\text{synth}} = \exp(Y_{\text{synth}}) - 1$$

**2. Zero-Inflated Mixture Model**

Features with frequent zeros (e.g., Revenue) are modeled as a two-part mixture process.

First, we determine if a value is zero using a Bernoulli trial based on the empirical zero probability $p_0$:  
$$b \sim \text{Bernoulli}(p_0), \quad \text{where } p_0 = P(Y=0)$$  
Second, if non-zero, we generate a value using the empirical quantile function $\hat{F}^{-1}_{nz}$ of the non-zero data:  
$$Y_{\text{synth}} = \begin{cases} 0 & \text{if } b = 1   
\\ \hat{F}_{nz}^{-1}(u) & \text{if } b = 0 \end{cases}$$  

**3. Tuned Stratified Sampling**

Standard inverse transform sampling uses a uniform random variable $u \sim U(0, 1)$. To precisely match the extreme tail variance of the original data, we replace standard uniform sampling with stratified sampling controlled by tuning parameters $q$ (cutoff, e.g., 0.995) and $r$ (tail rate, e.g., 0.005):

$$
u \sim \begin{cases}  
U(0, q) & \text{with probability } 1-r \quad \text{(Bulk)} \\  
U(q, 1) & \text{with probability } r \quad \text{(Extreme Tail)}  
\end{cases}  
$$
By manually tuning $q$ and $r$ for highly skewed features, we can over- or under-sample the extreme tails in log-space, providing precise control over the final standard deviation of $X_{\text{synth}}$.
