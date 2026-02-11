# Exposure-Aware Reach Estimation and Dynamic Engagement Weight Learning (Master/ISI-ready)

## Goal (What this method achieves)

We want to do two things in a scientifically defensible way:

1. **Estimate Reach** (how much a post is shown / exposed).
2. **Learn data-driven engagement weights** (how important a Like vs. a Retweet is) **after removing exposure bias**.
3. Make these weights **dynamic** (they can change with context like Hour or Klout).

This document explains the full pipeline step by step so that even a reader with limited ML background can follow it.

---

## 1) Why raw engagement cannot directly give you valid weights

On social media, large engagement can happen for two different reasons:

- **High exposure**: the post was shown to many people (high Reach).
- **High reaction strength**: people reacted strongly _given that they saw it_.

If we learn weights from raw Likes/Retweets without controlling exposure, we confuse the two mechanisms.

Example:

- A celebrity tweet gets many Likes mainly because it was shown to many people.
- A normal user tweet may get fewer Likes, but can be _very strong relative to its small exposure_.

So we must separate:

- **Visibility / Exposure process** → determines Reach
- **Reaction process** → determines Likes/Retweets _conditional on exposure_

This separation is the foundation of a defensible academic method.

---

## 2) Dataset columns and what each one is used for

Your dataset columns:

**Context at posting time**

- `Weekday`, `Hour`, `Day`, `Lang`, `LocationID`, `IsReshare`

**Author information**

- `Klout`, `UserID`

**Content**

- `text`, `Sentiment`

**Observed outcomes**

- `Reach`, `Likes`, `RetweetCount`

We will use them in two stages:

- **Stage A (Exposure model)**: estimate Reach from variables known at posting time.
- **Stage B (Debiased popularity + weights)**: learn Like/Retweet weights using the exposure estimate.

---

# Stage A — Reach (Exposure) Estimation

## A1) Definition and notation

For each tweet \(i\):

- Observed reach: \(R_i\)
- Observed likes: \(L_i\)
- Observed retweets: \(T_i\)

We interpret \(R_i\) as a proxy for **exposure** (how many users potentially saw the tweet).

Exposure is influenced by:

- author authority (Klout)
- time of posting
- language
- reshare status
- platform ranking/algorithm

**Stage A goal:** estimate the exposure that would be expected from _only pre-reaction information_.

---

## A2) Why log-transform Reach

Reach is typically extremely skewed: many small values, few extremely large values.

To stabilize training we define:

\[
Z_i = \log(1 + R_i)
\]

Why \(+1\)?

- Because \(\log(0)\) is undefined.

Benefits:

- reduces the dominance of outliers
- improves numerical stability
- makes errors comparable between small and large reach

---

## A3) Exposure features (inputs to Stage A)

Define a feature vector \(\mathbf{x}\_i\) using only information known before users react:

- \(K_i\): Klout
- \(H_i\): Hour
- \(W_i\): Weekday
- \(D_i\): Day (day-of-month)
- \(Lang_i\): language
- \(S_i\): IsReshare
- \(Loc_i\): LocationID (optional if it is clean and useful)

**Important:** Do NOT use Likes or Retweets in Stage A.  
They happen after exposure and create circular reasoning.

---

## A4) Exposure model

Train a regression model \(f(\cdot)\) such that:

\[
\hat{Z}\_i = f(\mathbf{x}\_i)
\]

Then convert back to predicted Reach:

\[
\widehat{R}\_i = \exp(\hat{Z}\_i) - 1
\]

Interpretation:

- \(\widehat{R}\_i\) is the **expected exposure** for tweet \(i\) given author/time/context.
- It approximates “how much the platform would show this tweet” before seeing how users reacted.

This is what you mean by “estimating Reach.”

---

## A5) Evaluation of Stage A (scientific defensibility)

Evaluate on held-out data (test set):

### MAE on log-Reach

\[
\text{MAE}_Z = \frac{1}{n} \sum_{i=1}^{n} |Z_i - \hat{Z}\_i|
\]

### RMSE on log-Reach

\[
\text{RMSE}_Z = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (Z_i - \hat{Z}\_i)^2}
\]

### R²

\[
R^2 = 1 - \frac{\sum*{i}(Z_i - \hat{Z}\_i)^2}{\sum*{i}(Z_i - \bar{Z})^2}
\]

### Critical splitting rule (avoid leakage)

Use **grouped split by UserID**:

- Tweets from the same user must not appear in both train and test.

Reason:

- Otherwise the model can memorize each user’s typical exposure pattern and performance becomes unrealistically high.

---

# Stage B — Debiased Popularity and Engagement Weight Learning

## B1) Why we build a debiased popularity target

If we use raw engagement counts as the target, we mostly learn exposure effects.  
We instead want:

> “How strong was the reaction relative to expected exposure?”

So we use \(\widehat{R}\_i\) to correct exposure bias.

---

## B2) Exposure-normalized reaction rates (optional but intuitive)

Define:

\[
\text{like_rate}\_i = \frac{L_i}{1 + \widehat{R}\_i}
\]
\[
\text{rt_rate}\_i = \frac{T_i}{1 + \widehat{R}\_i}
\]

Interpretation:

- likes per expected view
- retweets per expected view

This makes comparisons fair across users with very different exposure.

---

## B3) Debiased popularity score (the Stage B target)

We want a single scalar target \(Y_i\). A simple, defensible definition is:

\[
Y_i = \log(1 + L_i + T_i) - \log(1 + \widehat{R}\_i)
\]

Why this works:

- \(\log(1 + L_i + T_i)\) measures reaction intensity.
- subtracting \(\log(1+\widehat{R}\_i)\) penalizes posts with huge expected exposure.

So:

- high engagement with low exposure → high \(Y_i\)
- high engagement with massive exposure → smaller \(Y_i\)

This \(Y_i\) is an exposure-aware (debiased) popularity signal.

---

## B4) Learning engagement weights (interpretable coefficients)

Now we learn a model that predicts \(Y_i\) using Likes and Retweets as inputs.

A linear (interpretable) model:

\[
Y_i = \beta_0 + \beta_1 L_i + \beta_2 T_i + \beta_3\,\text{Sentiment}\_i + \beta_4\,\text{TextFeat}\_i + \beta_5 K_i + \beta_6\,\text{TimeFeat}\_i + \varepsilon_i
\]

Interpretation:

- \(\beta_1\) is the learned “weight” of Likes.
- \(\beta_2\) is the learned “weight” of Retweets.

Because the target \(Y_i\) already corrected for exposure, these weights have a much clearer meaning.

---

## B5) Why regularization is needed (for ISI-level rigor)

Real data has correlated variables. Without regularization, coefficients can become unstable.

Use **ElasticNet**:

\[
\min*{\beta} \sum*{i=1}^n (Y_i - \hat{Y}\_i)^2 + \lambda\Big(\alpha \|\beta\|\_1 + (1-\alpha)\|\beta\|\_2^2\Big)
\]

Where:

- \(\|\beta\|\_1\): L1 penalty → sparsity / feature selection
- \(\|\beta\|\_2^2\): L2 penalty → stabilizes correlated coefficients
- \(\lambda\): strength of regularization
- \(\alpha\): mix of L1 and L2

Regularization makes your learned weights reproducible and defensible.

---

## B6) Making weights dynamic (context-dependent coefficients)

You want weights to change with context (e.g., Hour, Klout).  
We do this with **interaction terms**.

### Dynamic weights by Hour

Add:

- \(L_i \times H_i\)
- \(T_i \times H_i\)

Model:

\[
Y_i = \beta_0 + \beta_1 L_i + \beta_2 T_i + \gamma_1(L_i H_i) + \gamma_2(T_i H_i) + \dots
\]

Then the effective weights become:

\[
w*{like}(H_i) = \beta_1 + \gamma_1 H_i
\]
\[
w*{rt}(H_i) = \beta_2 + \gamma_2 H_i
\]

Meaning:

- at different hours, the same Like/Retweet can imply different popularity.

### Dynamic weights by Klout

Add:

- \(L_i \times K_i\)
- \(T_i \times K_i\)

Effective weights:

\[
w*{like}(K_i) = \beta_1 + \delta_1 K_i
\]
\[
w*{rt}(K_i) = \beta_2 + \delta_2 K_i
\]

Meaning:

- likes/retweets may carry different meaning depending on author authority.

---

## B7) Simple text features (easy to compute and defend)

To keep things simple and understandable:

- \(len_i\): length of text (characters/words)
- \(hash_i\): number of hashtags
- \(mention_i\): number of mentions

You can add TF-IDF later, but these simple features are already defensible.

---

## B8) Evaluation of Stage B

Use the same evaluation metrics (on \(Y\)):

\[
\text{MAE} = \frac{1}{n} \sum*i |Y_i - \hat{Y}\_i|
\]
\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_i (Y_i - \hat{Y}\_i)^2}
\]
\[
R^2 = 1 - \frac{\sum*{i}(Y*i - \hat{Y}\_i)^2}{\sum*{i}(Y_i - \bar{Y})^2}
\]

Again: split by UserID groups.

---

# Stage C — ISI-grade scientific checks (must-have)

## C1) Ablation study (prove exposure correction matters)

Compare two pipelines:

1. **Naïve**: target \( \log(1+L+T) \) directly (no exposure correction)
2. **Exposure-aware (ours)**: Stage A → \(\widehat{R}\), Stage B → \(Y\)

You should observe:

- better generalization (especially under UserID-group split)
- more stable and interpretable coefficients

This justifies your design decisions academically.

---

## C2) Coefficient stability via bootstrap (publishable rigor)

To show weights are reliable:

1. Resample users (UserID groups) with replacement
2. Retrain Stage B each time
3. Store \(\beta_1\) and \(\beta_2\)
4. Report 95% confidence intervals

\[
CI*{95\%}(\beta_j) = [q*{0.025}(\beta*j^\*),\ q*{0.975}(\beta_j^\*)]
\]

---

# End-to-end pipeline (what you will actually do)

1. Preprocess data (types, missing values, encoding).
2. **Stage A**: train exposure model:
   - target: \(Z=\log(1+Reach)\)
   - inputs: Klout, Hour, Weekday, Day, Lang, IsReshare, LocationID
   - output: \(\widehat{R}=\exp(\hat{Z})-1\)
3. Construct debiased popularity:
   - \(Y = \log(1+Likes+Retweets) - \log(1+\widehat{R})\)
4. **Stage B**: train ElasticNet to predict \(Y\) using Likes/Retweets + controls + text features.
5. Add interactions to obtain **dynamic weights**.
6. Evaluate with group split, ablation, and bootstrap stability.

---

# What you will report in the paper

- Stage A (Reach estimation): MAE/RMSE/R² on log-Reach
- Stage B (debiased popularity): MAE/RMSE/R² on \(Y\)
- Learned static weights: \(\beta_1\) (Like), \(\beta_2\) (Retweet)
- Dynamic weight functions: \(w*{like}(Hour)\), \(w*{rt}(Hour)\), \(w\_{like}(Klout)\), ...
- Ablation results (naïve vs exposure-aware)
- Bootstrap confidence intervals for key coefficients
