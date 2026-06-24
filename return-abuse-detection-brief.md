# Return Abuse Detection — Project Brief & Build Guide

**Title:** ReturnShield — Explainable Return Abuse Detection and Risk Scoring for E-Commerce

This document is the full specification and reasoning for the project. Read all of it before writing code — several decisions below look optional but are the difference between a project that survives technical questioning and one that quietly falls apart under it. Sections 2 (the core challenge) and 8 (precautions) are the two that matter most and are the easiest to skip.

> ## The one rule above all others
> **Optimize for realism, not accuracy.** A model scoring 78–85% on overlapping behavioral profiles is more believable and more valuable than one scoring 99% on artificially separable data. High accuracy here is a *red flag*, not a success — it means the labels leaked into the features and the "model" is a rule engine in disguise. If you find yourself tuning for a higher number, stop and check that the data still has genuine overlap.

---

## 1. The goal

Build an end-to-end system that scores e-commerce customers on how likely they are to be abusing the returns/refund policy, explains *why* each customer is flagged, and lets a manager simulate the business impact of changing the return policy. This mirrors what a real fraud / trust-and-safety team at Amazon, Flipkart, or Walmart would build.

The deliverable is **not** a binary "fraud / not fraud" classifier. It is a **0–100 risk score** with action bands (safe / monitor / high risk), plus explainability and a policy simulator. That framing is what makes it read as an industry project rather than a homework classifier.

---

## 2. The core challenge — read this first

**There is no public dataset with real "return abuser" labels.** That data is proprietary to fraud teams and never released. So the labels have to be *created*, and this is where almost every version of this project goes wrong.

The trap, stated plainly:

```python
# DO NOT DO THIS
label = 1 if return_ratio > 0.6 else 0     # label is a function of a feature
# ...then train a model on return_ratio to predict that label
```

If the label is a deterministic function of the features, the model isn't learning anything — it's reverse-engineering the rule you already wrote. SHAP will "discover" that `return_ratio` drives the prediction... because `return_ratio` *defined* it. The whole pipeline becomes an elaborate way of recomputing an `if` statement. An interviewer will ask *"what did the ML add over the rule?"* and there is no good answer.

**The fix — this is the single most important idea in the document:**

> The label must be a hidden cause, and the features must be a noisy, overlapping consequence of it.

Concretely:

- Assign each synthetic customer a **latent type** (normal / impulse / serial returner / fraudster). That hidden type is the ground-truth label.
- Generate each customer's behavior by **sampling from type-conditional distributions, with real variance and overlap.** A fraudster *tends* to a high return rate but the realized value is stochastic; some honest customers randomly look bad; each fraudster shows only *some* red flags, not all of them.
- The model's job is to **recover the hidden type from the noisy observable features.** Because the classes overlap, no single threshold separates them — the model only wins by combining many weak signals nonlinearly. That is genuine, defensible ML.

**Litmus test:** if your model gets ~100% accuracy, your data is circular and worthless. You *want* irreducible error from the overlap. That's the proof the task is real.

---

## 3. Dataset strategy

**Base:** the Olist Brazilian e-commerce dataset (real customers, orders, payments, reviews, timestamps). Olist has no returns table, so returns/abuse are simulated *on top of* the real transaction scaffold. Using real base transactions is more honest than a fully synthetic dataset.

**Optional upgrade:** Online Retail II (UCI) contains genuine cancellations (invoices prefixed with `C`). You can blend real cancellation behavior with simulated abuse intensity if you want at least one real return-like signal underneath. Not required; Olist alone is fine.

### How to generate labels the right way (generative, not circular)

```python
import numpy as np

# Each archetype has a HIDDEN type and a distribution over behaviors.
# Note the OVERLAP between adjacent types — that's intentional.
ARCHETYPES = {
    "normal":  {"weight": 0.70, "return_rate": (0.05, 0.15)},
    "impulse": {"weight": 0.18, "return_rate": (0.10, 0.35)},
    "serial":  {"weight": 0.09, "return_rate": (0.30, 0.70)},
    "fraud":   {"weight": 0.03, "return_rate": (0.50, 0.95)},
}

def generate_customer(rng):
    t = rng.choice(list(ARCHETYPES), p=[a["weight"] for a in ARCHETYPES.values()])

    # Behavior is SAMPLED with noise — not a fixed value per type.
    lo, hi = ARCHETYPES[t]["return_rate"]
    return_rate = np.clip(rng.normal((lo + hi) / 2, (hi - lo) / 3), 0, 1)

    # Each red flag fires PROBABILISTICALLY. A fraudster does not show every flag.
    high_value_returns = rng.random() < {"normal":.05,"impulse":.15,"serial":.4,"fraud":.7}[t]
    coupon_heavy       = rng.random() < {"normal":.1, "impulse":.2, "serial":.5,"fraud":.8}[t]
    short_account      = rng.random() < {"normal":.1, "impulse":.2, "serial":.3,"fraud":.6}[t]
    # ...more behaviors, each sampled, each overlapping across types

    features = assemble_features(return_rate, high_value_returns, coupon_heavy, short_account, ...)
    latent_type = t                # the HIDDEN type — never a threshold on a feature
    return features, latent_type

# Add ~3–5% label noise: a small fraction of customers behave off-type.
```

**Model target — pick one and be explicit (do NOT leave this ambiguous).** Keep the latent type around for evaluation, but the model predicts a **binary target** in Version 1:

```text
label = 1 if latent_type == "fraud" else 0     # fraudster vs everyone else
```

Binary is the right V1 choice: it aligns cleanly with the 0–100 risk-score story and with PR-AUC, it's simpler to evaluate, and the serial/impulse types act as hard negatives the model must learn to separate from true fraud. (A 4-class version — normal / impulse / serial / fraud — is a reasonable later extension, but don't start there.)

**Two rules that keep this honest:**
1. **Spread the signal across several weak features.** Don't let one variable both define the type and sit in the feature set as the sole separator. If `return_ratio` alone explains 90% of the SHAP plot, a reviewer will (correctly) assume it defined the label.
2. **The professional report wording is fine, but only describes work you actually did.** You may write *"generative simulation with industry-derived return-rate distributions"* — but only because you actually modeled overlapping noisy distributions. The phrasing is a description, not a substitute.

### The four training archetypes (target distributions)

| Type | Share | Return rate | Notes |
|---|---|---|---|
| Normal buyer | ~70% | 5–15% | Baseline |
| Impulse buyer | ~18% | 10–35% | Overlaps normal at the low end |
| Serial returner | ~9% | 30–70% | Overlaps fraud |
| Fraudster | ~3% | 50–95% | High-value returns, coupon stacking, short account age |

Adjacent bands must overlap. That overlap is the whole point.

### The hidden 5th archetype (held out for evaluation only — never in training)

This one is reserved entirely for the evaluation experiment in Section 7. It exists to test whether the anomaly detectors can catch a pattern the supervised model was never trained on.

**Coupon Abuser** — characteristics:
- Extremely high coupon usage
- **Moderate** return rate (not high)
- **Long** account age (not short)
- Normal order frequency

The critical design constraint: **it must NOT look like the trained fraudster.** The fraudster's signature is short account + high return + high-value items. If the coupon abuser shares that signature, the supervised model will catch it and the experiment proves nothing. The whole point is that its abuse expresses through a *different* combination of behaviors, so the labeled model misses it and the unsupervised detectors flag it as "unusual."

---

## 4. Feature engineering (~15–20 features)

**Behavioral:** `total_orders`, `total_returns`, `return_ratio`, `high_value_return_rate`, `coupon_usage_rate`, `avg_days_to_return`, `payment_type_mix` (e.g. COD ratio).

**Temporal:** `returns_last_7_days`, `returns_last_30_days`, `returns_last_90_days`, `order_frequency`, `days_since_last_order`, a return-burst indicator.

**Customer:** `account_age`, `average_order_value`, `customer_lifetime_value`, `product_category_entropy`, `unique_categories_purchased`.

Keep it to roughly 15–25 engineered features. No PCA — with this few features it hurts interpretability and clashes with SHAP.

---

## 5. Architecture (locked — do not redesign)

```
Olist base transactions
      |
Latent customer archetype            (normal / impulse / serial / fraud)
      |
Stochastic behavior generator        (type-conditional, OVERLAPPING, noisy)
      |
Observable features  +  a held-out NOVEL archetype reserved for evaluation
      |
Feature engineering                  (~15-20: behavioral / temporal / customer)
      |
   +--------------------+--------------------+
   |                                         |
   v                                         v
Isolation Forest                        Autoencoder
   |  if_score                              |  reconstruction_error
   |                                        |  (fit on predominantly-normal
   |                                        |   bulk population — see 8.2)
   +--------------------+--------------------+
                        |
                        v
        Enhanced feature vector            (features + if_score + reconstruction_error)
                        |
                        v
        XGBoost                            (class weights, NOT SMOTE;
                        |                   trained on FEATURES, not the rules)
                        v
        Probability calibration            (CalibratedClassifierCV; to simulated population)
                        |
                        v
        Risk score 0-100                   (safe 0-30 / monitor 31-70 / high 71-100)
                        |
                        v
        SHAP explainability                (per-customer attribution)
                        |
                        v
        Dashboard + Policy Simulator

  K-Means cohort view = PARALLEL analytics only, never a model input
  Evaluation = PR-AUC / ROC-AUC + Precision@K / Recall@K
               + held-out archetype recovery
               + qualitative manual review
               + score stability analysis
```

Key shape points:
- Isolation Forest and the autoencoder are **parallel** — both consume the same engineered features. The autoencoder does **not** depend on Isolation Forest.
- Their two anomaly scores are appended to the feature vector that feeds **one** supervised model.
- K-Means is for a "this customer behaves like cohort X" dashboard section and offline analysis — it is **not** fed back into the model.

---

## 6. Component decisions and why

**One supervised model, not a stack.** Stacking XGBoost + LightGBM + CatBoost + an LR meta-learner gives near-zero gain on noisy labels and reads as padding. Use a single well-tuned XGBoost (or compare LR baseline → one tree model → one GBM and pick the winner — that's a fine, honest "I compared models" story). If you want a *second* signal, the anomaly scores already provide genuinely different information; that's the blend.

**Break the circularity.** Use the latent type as the label, train the supervised model on **features only** (plus the IF/AE scores) — so it learns to generalize patterns rather than echo a rule. Interview answer: *"The rules are rigid single-feature thresholds; the model learns the joint nonlinear signature across behaviors and generalizes to customers no single rule catches."*

**Class weights, not SMOTE.** Use `scale_pos_weight` / class weighting for imbalance. SMOTE on tabular fraud data is increasingly treated as a red flag in fraud interviews; class weighting is cleaner and more defensible. Be ready to say why.

**Calibration — with an honest caveat.** `CalibratedClassifierCV` makes the 0–100 score meaningful (raw boosted-tree probabilities are not well-calibrated). But in a synthetic setup you're calibrating to the *simulated* population's base rate, not real-world prevalence. Say so: "calibrated to the simulated population."

**Autoencoder — what it's for.** In a supervised synthetic world, IF/AE look redundant. Their real job is catching abuse patterns *not in the labeled training set*. Demonstrate this with the held-out coupon-abuser archetype (Section 7). That's why fraud teams run unsupervised + supervised together.

---

## 7. Evaluation methodology (the real differentiator)

Because the data is generated, you already hold the true latent labels — so evaluation is done against those, not against hand-labeled data. (An earlier draft mentioned a "hand-labeled holdout"; ignore that — it's redundant when you already know every customer's true type.) This section is what separates a thoughtful project from a naive one.

**Quantitative, against the latent labels:**
- **PR-AUC** — the headline metric. The positive (abuser) class is rare, so precision-recall area is far more informative than accuracy.
- **ROC-AUC** — secondary, for completeness.
- **Precision@k and Recall@k on the latent fraud class** — how good are the top-ranked customers, and how many true fraudsters does the top-N catch. (Optional secondary analysis: recovery rate of serial returners among top-ranked customers — they're negatives in the binary target, so this is a diagnostic, not a headline metric.)

**The held-out archetype experiment (strongest single result):**
- Train the supervised model on the four base archetypes only.
- Score the held-out **coupon abuser** population (never seen in training).
- Show that XGBoost misses many of them while Isolation Forest and the autoencoder flag a meaningful share. This is the concrete proof of *why anomaly detection exists* in the pipeline. Report it explicitly with numbers.

**Qualitative:**
- Manual review of the top 20 highest-risk customers — do they look obviously suspicious? Screenshot two for the writeup.
- Stability — do scores stay consistent across retrains / time windows?

Target line for the report and interviews:
> *"No labeled return-abuse data exists publicly, so I built a generative simulation with overlapping behavioral profiles and validated the pipeline against the known latent labels — including recovery of a held-out abuse pattern the model was never trained on."*

That sentence preempts the obvious objection ("your model only works on your own simulation") by showing you already know it's the main limitation and tested against it.

---

## 8. Precautions — DO NOT SKIP

These are the things an interviewer probes and a code reviewer catches. Getting the architecture right but these wrong invalidates the results.

1. **Anomaly detectors must not leak.** Fit Isolation Forest and the autoencoder on the **training split only**, then apply them to compute scores on the test split. Fitting on the full dataset before splitting means the scores have seen the test customers — inflated, invalid numbers. Same rule for any scaler.

2. **Train the autoencoder on the predominantly-normal bulk population — and be precise about which.** The autoencoder must learn "normal" behavior so that abusive patterns reconstruct *badly*. The cleanest signal comes from training it on normal-like customers only. But there's a subtlety to get right and to be able to defend: in real deployment you don't *know* who is normal — that's what you're trying to find. So the most defensible setup is to train on the **unlabeled bulk majority population** (which is ~90% normal with mild contamination), justified by the realistic assumption that most customers are legitimate. Training strictly on the known-"normal" archetype gives a cleaner signal but quietly uses label knowledge you wouldn't have in production — if you do that for the writeup, *say so explicitly*. Either choice is fine; the unforgivable thing is not knowing which one you made or why.

3. **Split at the customer level.** A single customer must never appear in both train and test. If you use time-windowed features (`returns_last_7_days`, etc.), make sure the feature window and the label don't let the label peek into the future.

4. **Don't let one feature dominate SHAP.** If a single variable explains the vast majority of attributions, it almost certainly defined the label. Spread the generative signal across multiple weak features.

5. **You want < 100% accuracy.** Perfect separation = circular data. Build in overlap and ~3–5% label noise on purpose. (See the rule at the top of this document.)

6. **Calibrate to the simulated population and say so.** Don't claim real-world calibration you can't have.

---

## 9. Dashboard + Policy Simulator

Streamlit is the pragmatic choice (fast to build, fine for a portfolio). Use React only if you specifically want to signal full-stack skill.

Panels:
- **User risk profile** — search a customer ID, show the 0–100 score and band.
- **SHAP waterfall** — top contributing factors for that customer.
- **Cohort explorer** — which K-Means behavioral cluster they fall in.
- **Policy simulator (the standout feature)** — let a manager change the return window (e.g. 30 days → 15 days) and estimate expected abuse reduction, expected legitimate-customer impact, and expected cost savings. This is what turns a model into a business decision tool, and it's the thing to talk about that no one else has. Keep it front and center.

---

## 10. Deliverables

The spec describes a pipeline; these are the concrete artifacts the implementer must produce so the project isn't considered "done" after merely training a model:

- Reproducible data-generation pipeline (seeded, re-runnable end to end)
- Trained, calibrated model (serialized)
- Evaluation report (PR-AUC, ROC-AUC, precision@k/recall@k, the held-out archetype result, manual-review screenshots)
- Streamlit dashboard
- Policy simulator
- Architecture diagram
- Project documentation / README (how to run, design rationale, limitations)

---

## 11. Build order

You are past the design phase. The next thing that improves the project is the first commit, not another box.

**Build the simplest end-to-end skeleton first, then add layers.** The most common way projects like this die is trying to build every layer at once. Get a thin vertical slice working — generate archetypes → engineer features → train XGBoost → SHAP — and confirm the story holds before adding Isolation Forest, the autoencoder, calibration, the dashboard, and the policy simulator on top. Suggested sequence:

1. Olist ingestion → aggregate to one row per customer.
2. Generative simulation (latent types → noisy overlapping behaviors → features + labels). Validate the overlap visually (plot return-rate distributions per type — they should overlap).
3. Feature engineering pipeline.
4. Leakage-safe split + fit IF and AE on the right subsets; produce anomaly scores.
5. XGBoost on enhanced features, class-weighted; calibrate.
6. Evaluation harness — including the held-out coupon-abuser experiment. This is where the credibility is.
7. SHAP integration.
8. Streamlit dashboard, policy simulator last.

---

## 12. Why this is a strong project

It demonstrates, in one coherent system: correct generative synthetic-data design, anomaly detection, supervised learning, explainability, risk scoring, business analytics, dashboarding, and a real evaluation methodology — with honest handling of the no-labels problem that most candidates fumble. That's well above the typical MNIST-CNN / movie-recommender / sentiment-analysis tier.

**Resume bullet:**
> Built an end-to-end return-abuse risk scoring system on real e-commerce transaction data, using a generative simulation framework, dual anomaly detectors (Isolation Forest + autoencoder), a calibrated gradient-boosted classifier, SHAP explainability, and a policy-impact simulator — validated with PR-AUC and recovery of held-out abuse patterns the model was never trained on.
