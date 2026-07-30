# Capstone Report — Machine Learning & Search Intelligence

- **Author:** Hana Ahmed Radwan
- **Lane:** Machine Learning & Search Intelligence
- **Repo:** https://github.com/hanaahmedradwan123-commits/flyrank-internship-w1
- **Date:** July 30, 2026

---

## 1. Problem framing

* **Supported Decision:** Prioritizes organic search content for editorial content refresh and optimization workflows before traffic decay accelerates.
* **Unit of Analysis:** Content item per client (`content_hash_id` level).
* **Output:** A predicted risk score (probability between `0.0` and `1.0`) and a binary decision flag (`needs_refresh_pred`) indicating predicted $\ge 25\%$ traffic loss in Month 5.
* **Human Action:** FlyRank content editors review high-risk flagged pages to update content intent, add relevant internal links, refresh outdated metadata, or conduct technical SERP health checks.
* **Cost of Wrong Calls:**
  * **False Positives (Precision Cost):** Editorial time and resources wasted manually reviewing healthy, stable content pages that did not require updates.
  * **False Negatives (Recall Cost):** High-performing legacy content decays unnoticed, leading to compounding losses in organic traffic and revenue.
* **Why Data/ML Helps:** High-volume content portfolios (100k+ pages across clients) make manual monitoring impossible. An automated classifier detects early decay signals across non-linear interactions (impressions, position changes, CTRs, and growth rates) much faster and more reliably than manual heuristics.

---

## 2. Data safety

* **Data Sources Used:** Hugging Face Parquet release (`hf://datasets/FlyRank/internship-warehouse`), utilizing `dim_clients.parquet` and daily aggregated performance tables in `fact_content_daily_performance/**/*.parquet`.
* **Date Windows:** Historical features derived from March 2026 (`2026-03`) and April 2026 (`2026-04`). Target window performance observed in May 2026 (`2026-05`).
* **Deliberately Excluded Columns:**
  * `client_hash_id` and `content_hash_id`: Used exclusively in SQL queries for grouping/aggregation; strictly removed prior to training to prevent model memorization and protect privacy.
  * `clicks_m5`: Excluded completely from model features to prevent target leakage, as it is used solely to construct the ground-truth target label `needs_refresh`.
* **Exclusions & Noise Filtering:** Filtered out ultra-low impression pages with fewer than 50 impressions in Month 4 (`impressions_m4 < 50`) to remove noise and transient micro-fluctuations.
* **Privacy Confirmation:** No client names, domain URLs, raw search queries, or personal identifiers exist anywhere within the repository output or `work/` directory.

---

## 3. Baseline

* **Baseline Rule:** A transparent heuristic baseline predicting `needs_refresh = 1` for any content page that experienced negative click growth from Month 3 to Month 4 (`click_growth_rate < 0`).
* **Fair Comparison:** Evaluated on the exact same 20% test split ($N_{test} = 25,152$) as the machine learning model.
* **Baseline Numbers:**
  * **Base Rate (Target Minority Class):** 4.96% (1,247 positive cases out of 25,152).
  * **Baseline Precision:** 0.1240
  * **Baseline Recall:** 0.6120
  * **Baseline F1-Score:** 0.2062
  * **False Positives Generated:** 4,318 false alarms.

---

## 4. Model / analysis

* **Method:** XGBoost Gradient Boosted Trees (`XGBClassifier`) with positive class re-weighting (`scale_pos_weight = 19`) to address the 95:5 class imbalance.
* **Target Proxy Definition:** A page is flagged as requiring refresh (`needs_refresh = 1`) if it recorded $\ge 10$ clicks in Month 4 and suffered a relative click drop $> 25\%$ from Month 4 to Month 5.
* **Exact Feature List:**
  * `clicks_m3`: Total clicks in Month 3
  * `clicks_m4`: Total clicks in Month 4
  * `impressions_m3`: Total impressions in Month 3
  * `impressions_m4`: Total impressions in Month 4
  * `position_m4`: Average SERP position in Month 4
  * `ctr_m4`: Aggregate Click-Through Rate in Month 4
  * `click_growth_rate`: Relative change in clicks from Month 3 to Month 4
  * `impression_growth_rate`: Relative change in impressions from Month 3 to Month 4

---

## 5. Evaluation

* **Data Split:** 80/20 Stratified Train-Test Split ($N_{train} = 100,606$, $N_{test} = 25,152$). Stratification was strictly applied to preserve the ~4.96% base rate across both subsets.
* **Discrimination Performance:**
  * **ROC-AUC:** **0.9709**
  * **PR-AUC (Precision-Recall AUC):** **0.5952**
* **Model vs. Baseline (Measured on the Same Test Split):**

| Model / Approach | Decision Threshold | Precision | Recall | F1-Score | False Positives |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task Base Rate** | Majority Class (95.04%) | N/A | 0.0000 | 0.0000 | 0 |
| **Heuristic Baseline** | `click_growth_rate < 0` | 0.1240 | 0.6120 | 0.2062 | 4,318 |
| **XGBoost (Default Cutoff)** | Probability $\ge 0.5000$ | 0.3746 | **0.9920** | 0.5437 | 2,065 |
| **XGBoost (Tuned Cutoff)** | Probability $\ge 0.8689$ | **0.4308** | 0.8292 | **0.5672** | **1,365** |

* **Error Analysis:**
  * **False Positives (1,365 pages):** Primarily composed of high-traffic pages in Month 4 (`clicks_m4 >= 10`) that showed mild impression drops but stabilized in Month 5 without crossing the full -25% click decay threshold.
  * **False Negatives (213 pages):** Pages with stable or positive growth from Month 3 to Month 4 that experienced sudden drops in Month 5 due to unobserved external factors (e.g., unexpected competitor movements or algorithm updates).

---

## 6. Interpretation

* **Feature Importances Breakdown:**
  * `clicks_m4` holds dominant feature importance (~95.3%) because the ground-truth target definition requires a baseline of `clicks_m4 >= 10`.
  * `click_growth_rate` (~1.7%) and `impressions_m3` (~0.6%) provide secondary directional signals for detecting traffic momentum changes.
* **Observed Signals:** A high traffic volume combined with negative growth momentum is the single strongest indicator of imminent content decay.
* **Surprises & Negative Results:** Average SERP position (`position_m4`) contributed $< 0.5\%$ feature importance. Top SERP position alone does not protect a page from overall search intent shifts or aggregate demand drops.

---

## 7. Recommendation

* **Action Playbook for FlyRank Editors:**
  1. **Tier 1 (High Priority - Risk Probability $\ge 0.8689$):** Queue immediately for manual editorial intervention. Editors should refresh outdated statistics, update metadata, realign search intent, and add new internal links.
  2. **Tier 2 (Monitor - Risk Probability $0.5000 - 0.8688$):** Assign to an automated watch list for re-evaluation during the next performance cycle.
  3. **Tier 3 (Healthy - Risk Probability $< 0.5000$):** Maintain standard periodic content maintenance schedules.
* **Explicit Limits:** This model provides **directional decision support** based on historical performance decay. It does **not** make causal claims regarding Google search engine algorithm penalties or direct competitor actions.

---

## 8. Reproducibility

* **Environment Highlights:**
  * Python 3.12+
  * `duckdb >= 1.0.0`
  * `xgboost >= 2.0.0`
  * `scikit-learn >= 1.4.0`
  * `pandas >= 2.1.0`
  * `joblib >= 1.3.0`
* **Random Seeds:** Fixed random seed `random_state = 42` applied across `train_test_split` and `XGBClassifier` initialization.
* **Execution Steps:**
  1. Clone repository: `git clone https://github.com/hanaahmedradwan123-commits/flyrank-internship-w1.git`
  2. Install dependencies: `pip install -r requirements.txt`
  3. Export environment variable: `export HF_TOKEN="your_huggingface_token"`
  4. Run capstone notebook: Execute all cells in `work/notebooks/capstone.ipynb` sequentially.
  5. Artifacts generated in working directory: `needs_refresh_xgboost.pkl`, `model_predictions_eval.csv`, and `content_refresh_recommendations.csv`.
---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> **Metrics vs. base rate:** report your task's base rate (majority-class %) next to any
> precision@K or accuracy — a high score can just be a high base rate. AUC / lift over
> baseline are the honest discrimination numbers.
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
