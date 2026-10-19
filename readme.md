# 🛡️ WikiGuard Live: Real-Time Wikipedia Revert Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-L1_Logistic_Regression-F7931E?style=for-the-badge&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE_Streaming-009688?style=for-the-badge&logo=fastapi)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-Glassmorphism_UI-38B2AC?style=for-the-badge&logo=tailwind-css)

**WikiGuard Live** is an end-to-end streaming machine learning pipeline that taps into Wikimedia's live Server-Sent Events (`recentchange`) feed, extracts real-time revision diffs, and uses an **L1-regularized (Lasso) Logistic Regression** model to predict the probability that a human edit will be reverted by moderators within 24 hours.

Instead of functioning as a black-box classifier, WikiGuard decomposes every live prediction into exact **log-odds contributions ($\beta_i x_i$)**, surfacing human-readable explanations for why an edit was flagged in sub-millisecond inference time.

---

## ✨ Key Engineering Highlights

* **The "Time-Travel" Auto-Labeling Pipeline:** Avoids the 24-hour labeling bottleneck by querying historical revisions from $T - 72\text{ hours}$ via the MediaWiki API and inspecting native moderation tags (`mw-reverted`, `mw-undo`, `mw-rollback`) to construct a ground-truth dataset automatically.
* **Hybrid Dense + Sparse Feature Architecture:** Combines standardized behavioral telemetry (`log_bytes_added`, `log_bytes_deleted`, temporary/anonymous account flags) with high-dimensional lexical diff signals (`TfidfVectorizer` word and character n-grams on newly inserted HTML diff lines).
* **L1 (Lasso) Sparsity as a Vandalism Dictionary:** Applies an $L_1$ penalty (`solver='liblinear'`) across $2,000+$ TF-IDF features, driving the weights of neutral encyclopedic tokens to exact zero (`0.0000`) while isolating high-risk vandalism markers and structural deletions.
* **Zero-Flicker FastAPI + SSE Glassmorphism Dashboard:** Streams live predictions to the browser over Server-Sent Events with interactive decision-threshold tuning, live KPI telemetry, and per-edit log-odds attribution chips.

---

## 🧠 Mathematical Formulation & Interpretability

Logistic regression models the log-odds of an edit being reverted ($y = 1$) as a linear combination of standardized behavioral features $x_{\text{dense}}$ and sparse TF-IDF diff tokens $x_{\text{text}}$:

$$\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \sum_{i=1}^{k} \beta_i x_i \quad \Longrightarrow \quad p(\text{revert}) = \frac{1}{1 + e^{-z}}$$

During training, we minimize the $L_1$-penalized negative log-likelihood:

$$\min_{\beta_0, \beta} \left[ -\frac{1}{N} \sum_{j=1}^{N} \left( y_j \ln p_j + (1 - y_j) \ln(1 - p_j) \right) + \frac{1}{C} \sum_{i=1}^{k} \vert{}\beta_i\vert{} \right]$$

### Sample Learned Weights from Training Run
| Feature | Type | Learned Weight ($\beta_i$) | Odds Multiplier ($e^{\beta_i}$) | Behavioral Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| `log_bytes_deleted` | Dense Behavioral | `+0.410` | `1.51x` | Large text blanking / section removals strongly trigger reverts |
| `log_bytes_added` | Dense Behavioral | `+0.095` | `1.10x` | Unsolicited text dumps carry mild positive revert risk |
| `category` | Sparse TF-IDF | `-3.212` | `0.04x` | Standard encyclopedic categorization (`[[Category:...]]`) signals safe maintenance |
| `2026` | Sparse TF-IDF | `-2.197` | `0.11x` | Up-to-date citation timestamps strongly predict edit survival |

---

## 🏗️ System Architecture

```text
[PHASE 1 & 2: HISTORICAL AUTO-LABELING]
MediaWiki RecentChanges API (T - 72h) ──► MediaWiki Compare API (HTML Diffs)
                                                  │
                                                  ▼
                                        training_data.csv
                                                  │
[PHASE 3: MODEL TRAINING]                         ▼
ColumnTransformer(StandardScaler + TfidfVectorizer) ──► L1 LogisticRegression ──► wiki_revert_model.joblib
                                                                                           │
[PHASE 4 & 5: REAL-TIME STREAMING INFERENCE]                                               ▼
Wikimedia Live SSE Stream (recentchange) ──► Live Diff Fetcher ──► FastAPI SSE Server + Rich CLI Dashboard







