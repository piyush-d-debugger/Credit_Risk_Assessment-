# Credit Risk Assessment

An end-to-end credit risk project that goes beyond a binary "will they default"
prediction — it diagnoses *why* risk patterns exist in the data, quantifies
the dollar impact of model decisions, and ships as an interactive Streamlit
app that gives applicants a loan offer while keeping a separate internal risk
view for lending managers.

**Live demo:** _[ [Streamlit link](https://creditassessment.streamlit.app/) ]_

---

## Overview

Given an applicant's profile (income, employment history, home ownership,
credit history, prior defaults) and a requested loan amount, this project:

1. Predicts the probability of default using a CatBoost model
2. Prices an interest rate specific to the requested loan
3. Calculates a stable borrowing capacity independent of the specific request
4. Surfaces the underlying risk grade and probabilities for internal
   (lending manager) review only — applicants see just their offer

The project is built around a Kaggle credit risk dataset and follows a full
workflow: EDA → hypothesis testing → feature engineering → model comparison
→ calibration → business-impact framing → an interactive web app.

## Dataset

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
(Kaggle) — ~32,500 records with the following fields:

| Column | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_home_ownership` | RENT / MORTGAGE / OWN / OTHER |
| `person_emp_length` | Years employed |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Lender-assigned risk grade (A–G) |
| `loan_amnt` | Requested loan amount |
| `loan_int_rate` | Interest rate charged |
| `loan_status` | Target — 1 = default, 0 = no default |
| `loan_percent_income` | Loan amount as a fraction of income |
| `cb_person_default_on_file` | Prior default on record (Y/N) |
| `cb_person_cred_hist_length` | Credit history length (years) |

## Tools

- **Environment:** Python, managed with [`uv`](https://github.com/astral-sh/uv) (`pyproject.toml` + `uv.lock`)
- **Core stack:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Modeling:** CatBoost, imbalanced-learn
- **App:** Streamlit
- **Notebook:** Jupyter

---

## Workflow

### 1. EDA & hypothesis testing

Before touching the data, hypotheses were written down for each feature's
expected relationship with default (e.g. "higher loan-to-income ratio →
higher default risk"), then validated with crosstabs, boxplots, and
chi-square tests. Key findings:

| Hypothesis | Verdict |
|---|---|
| Mortgage holders default more than renters/owners | Confirmed |
| Longer employment length → fewer defaults | Confirmed |
| Prior default on file → more defaults | Confirmed |
| Longer credit history → fewer defaults | Confirmed |
| Personal loans default most, venture loans least | Confirmed |
| Lower loan grade → more defaults | Confirmed (strongest predictor) |
| Higher loan amount → higher default rate | Confirmed |
| Higher interest rate → higher default rate | Confirmed |

**Notable discovery:** binning `loan_percent_income` against actual observed
default rates revealed a sharp real threshold around **30% loan-to-income
ratio** — default rate roughly triples (24% → 68%) crossing that line. This
mirrors common real-world debt-to-income lending guidelines and became a key
consideration later in the modeling process (see "Lessons learned" below).

### 2. Feature engineering & class imbalance

The dataset is imbalanced (~22% default rate). Missing values in
`person_emp_length` and `loan_int_rate` were imputed with the median
(explicitly reassigning columns rather than using pandas' `inplace=True`,
which silently fails under pandas 2.x's copy-on-write behavior). Numerical
features were scaled with `MinMaxScaler`; categorical features one-hot
encoded.

### 3. Model comparison

Logistic Regression, Random Forest, XGBoost, and CatBoost were compared on
the full feature set (including `loan_grade`/`loan_int_rate`):

| Model | False Negatives ↓ (missed defaulters) | Notes |
|---|---|---|
| Logistic Regression | 328 (best recall on defaulters) | Highest false positives |
| Random Forest | 401 | |
| XGBoost | 418 | |
| CatBoost | 438 (best overall separation) | Highest overall accuracy/AUC |

This highlights the precision/recall trade-off central to the business
framing below: the model with the *best overall metrics* isn't always the
one that catches the most actual defaulters.

### 4. Business framing: cost-asymmetric errors

A false negative (approving a loan that defaults) is far more costly than a
false positive (rejecting a safe borrower). Rather than optimizing for
accuracy alone, model selection considered this asymmetry directly.

### 5. Expected loss quantification

```
Expected Loss = Probability of Default × Loss Given Default × Exposure at Default × Loan Amount
```

Predicted probabilities were calibrated (`CalibratedClassifierCV`) before
being used in this formula — an uncalibrated model can rank borrowers
correctly while still producing probability values that don't reflect true
likelihood, which would make the resulting dollar figure unreliable.

---

## A leakage catch: the application-time model

The full model above uses `loan_grade` and `loan_int_rate` as predictive
features. Both are **assigned by the lender after deciding how risky someone
is** — a brand-new applicant doesn't have them yet. Using them as inputs to
assess a first-time applicant is circular.

**Fix:** a second, leaner CatBoost model was trained using only fields a
genuinely new applicant provides (age, income, employment length, home
ownership, loan intent, requested amount, credit history, prior default).
This model has measurably lower ROC-AUC than the full model — expected,
since `loan_grade` was the single strongest predictor — but it's the only
one that can honestly be used to assess someone applying for the first time.

## Diagnosing and fixing a model stability issue

After deploying the application-time model, a sensitivity sweep (holding an
applicant profile fixed and varying only the requested loan amount) revealed
an unrealistic cliff: probability of default jumped from ~5% to ~99.9%
across a $500 change in requested amount.

Checking the **raw data** (not the model) confirmed a real threshold exists
around 30% loan-to-income ratio, but the actual jump in that data is far
smaller (24% → 68%). The model's exaggerated ~100% jump was traced to
**SMOTETomek's synthetic oversampling** creating an artificially dense
cluster right at that threshold. Switching to CatBoost's native
`auto_class_weights='Balanced'` (fitting on the original imbalanced data
instead of synthetic samples) brought the model's transition back in line
with the real, observed default rates.

## Capacity vs. pricing: a deliberate design split

Early versions of the applicant-facing app had a flaw: the "maximum approved
amount" was computed using the model's prediction on the *requested* loan
amount — meaning asking for more money made the applicant look riskier,
which lowered their own ceiling. That's circular, and doesn't match how
lending actually works.

The fix separates two distinct questions:

- **Borrowing capacity** — "how much *could* this person borrow?" Computed
  from a transparent, rule-based profile score (income, employment
  stability, credit history, home ownership, prior default), independent of
  what they're requesting today. This stays fixed as the requested amount
  changes.
- **Loan pricing** — "what does *this specific* loan cost?" Computed from
  the CatBoost model using the actual requested amount — it's reasonable for
  a bigger ask relative to income to carry a higher rate, but it shouldn't
  shrink the ceiling itself.

Capacity is capped at $35,000, since the model was never trained on loans
above that amount.

---

## The app

Built with Streamlit. The applicant fills out a short form and sees:

- Their offered interest rate
- Their maximum approved amount
- Whether their specific request is approved or exceeds their limit

A collapsed **"For lending manager use only"** section holds the internal
risk grade and both underlying probabilities (intrinsic profile risk and
request-specific risk) — kept separate from the customer-facing view.

### Running locally

```bash
uv sync
uv run streamlit run app.py
```

### Project structure

```
credit_risk/
├── app.py                    # Streamlit frontend
├── pipeline.py                # preprocessing + risk assessment logic
├── requirements.txt            # for Streamlit Community Cloud
├── pyproject.toml / uv.lock    # local dependency management
├── data/                      # raw + cleaned datasets
├── model/
│   └── model_application_time.pkl
├── parameter/
│   ├── scaler_application_time.pkl
│   ├── train_columns_application_time.pkl
│   └── grade_mapping.pkl
└── Credit_Risk.ipynb          # full analysis notebook
```

---

## Lessons learned & limitations

- Grades D and E are nearly statistically indistinguishable under the
  application-time model (avg. predicted probability within 0.01 of each
  other) — some grade-separating signal was necessarily lost when
  `loan_grade` was removed to fix the leakage issue.
- The model's probability curve still transitions somewhat faster than the
  raw data around the ~30% DTI threshold, even after the class-weighting
  fix — a reasonable trade-off, not a perfect match, and worth further
  tuning in a production setting.
- Capacity is governed by hand-set business rules (income multipliers by
  loan purpose, profile scoring weights) rather than learned from data —
  a deliberate choice for stability and explainability, but an area that
  could be replaced with a proper affordability model given more data.
- The model was trained on loans between $500–$35,000; recommendations
  outside that range would be extrapolation, so a hard ceiling is enforced.

## Future improvements

- Replace hand-set capacity weights with a learned affordability model
- Add SHAP-based explanations to the lending manager view
- Expand the loan-purpose multiplier table with real underwriting data
- A/B test discrete-grade vs. continuous-rate pricing with real users

---

## Credits

- Dataset: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) on Kaggle
- Project structure loosely inspired by [brunokatekawa/credit_risk](https://github.com/brunokatekawa/credit_risk) (methodology reference only — this implementation, its models, and its business logic were built independently)
