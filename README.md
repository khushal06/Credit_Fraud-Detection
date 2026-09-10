# Credit Card Fraud Detection

A machine learning project that looks at a credit card transaction and predicts whether it's **fraud** or **legit** — and, just as importantly, explains *why* it made that call.

Built on real anonymized data: **284,807 transactions**, of which only **492 are fraud (0.17%)**.

---

## Why I built this (the problem)

Credit card companies process millions of transactions a day. A tiny fraction are fraud — someone using a stolen card. Humans can't check them all, so companies need a model that flags the suspicious ones automatically.

The catch is that there are two very different ways to be wrong, and they cost different things:

- **Missing a fraud** → the bank eats the loss. Expensive.
- **Falsely flagging a real customer** → you decline someone's legit purchase, they get frustrated, and support has to deal with it. Annoying and costly at scale.

So the real goal isn't "be right as often as possible." It's **catch as much fraud as you can without blocking too many real customers.** That tradeoff is the whole point of the project.

---

## What it is (the project)

An end-to-end pipeline that:
1. Loads and cleans the transaction data
2. Trains five models — four supervised, one unsupervised — and compares them
3. Explains which signals drove each fraud prediction
4. Serves it all through a clickable app where you can score a transaction and adjust how strict the model is

---

## How it works (the approach)

**The data.** Each transaction has 30 features. Most (V1–V28) are anonymized (the bank hid the real details for privacy), plus `Time` and `Amount`. The label `Class` is 1 for fraud, 0 for legit.

**The imbalance problem.** Only 0.17% of transactions are fraud. This breaks a beginner mistake called the **accuracy trap**: a model that just guesses "legit" every single time is 99.83% "accurate" — and catches *zero* fraud. Useless. So accuracy is the wrong scoreboard here.

![Class balance](reports/figures/class_balance.png)

**Better scoreboard.** Instead of accuracy, I measure:
- **Precision** — of the transactions I flagged as fraud, how many *actually* were? (high precision = few false alarms)
- **Recall** — of all the real frauds, how many did I catch? (high recall = few misses)
- **PR-AUC** — a single score that summarizes the precision/recall tradeoff. This is the honest headline number for imbalanced problems like this.

**The models.**
- **Logistic Regression** — a simple baseline. I told it to weight the rare fraud class more heavily so it wouldn't ignore it.
- **XGBoost** — a stronger model. I used a setting called `scale_pos_weight` to handle the imbalance (it makes each fraud example "count" ~580x more during training, since there are ~580 legit transactions for every fraud).
- **Random Forest** — another tree-based model, also given `class_weight="balanced"`, as a second supervised comparison point.
- **LightGBM** — a faster gradient-boosting alternative to XGBoost, using `is_unbalance=True` for the same reason.
- **Isolation Forest** — the odd one out: it's **unsupervised**. It never sees the fraud labels during training, only the transaction features, and instead learns what "normal" looks like and flags whatever sits furthest from it as anomalous. This matters in practice because in real fraud systems you often *don't* have labeled examples of brand-new fraud patterns — an unsupervised model like this is what you'd lean on until labels catch up.

**Explainability.** I used **SHAP** to show which features pushed a transaction toward "fraud." This matters because a real fraud team won't trust a black box — they need to see the reasoning.

![SHAP summary](reports/figures/shap_summary.png)

---

## What I found (the results)

Sorted by PR-AUC — the honest headline number under this much imbalance:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Fraud caught / missed | False alarms |
|-------|-----------|--------|----|---------|--------|----------------------|--------------|
| **XGBoost** | **0.872** | 0.837 | **0.854** | **0.979** | **0.873** | 82 / 16 | **12** |
| Random Forest | 0.895 | 0.786 | 0.837 | 0.962 | 0.847 | 77 / 21 | 9 |
| Logistic Regression | 0.061 | 0.918 | 0.114 | 0.972 | 0.716 | 90 / 8 | 1,386 |
| Isolation Forest *(unsupervised)* | 0.280 | 0.306 | 0.293 | 0.954 | 0.172 | 30 / 68 | 77 |
| LightGBM | 0.046 | 0.857 | 0.088 | 0.913 | 0.040 | 84 / 14 | 1,725 |

Isolation Forest is the one model here that learns no labels at all — it's included as a useful comparison because in real fraud systems you often don't have labeled examples of brand-new fraud patterns, so this is roughly what you'd fall back on. The other four are all supervised.

**In plain English:**

- The **Logistic Regression** model caught almost all the fraud (91.8%) — but to do it, it screamed "fraud!" at **1,386 legitimate customers**. In real life that's a lot of angry people getting their cards declined. Not workable.
- **XGBoost** caught a bit less fraud (83.7%) but with only **12 false alarms** instead of 1,386. That's a massively better balance, and the best PR-AUC of the group.
- **Random Forest** landed close behind XGBoost — fewer false alarms (9), but it also missed more fraud (21 vs. 16).
- **Isolation Forest**, working with zero labels, still beat random guessing by a wide margin (PR-AUC 0.172 vs. a ~0.0017 baseline) — but it's nowhere near the supervised models. That gap *is* the point: it's the cost of not having labels.
- **LightGBM** is the surprise of the table: strong ROC-AUC but the worst PR-AUC by far, because at the default 0.5 decision threshold it drowned in false alarms (1,725). This is a good reminder that a model isn't "bad" or "good" in the abstract — its default threshold might just be poorly calibrated for a problem this imbalanced, and PR-AUC (which looks across all thresholds) is what catches that.

This is a textbook example of the **precision/recall tradeoff**: you can crank one up, but usually at the cost of the other. The whole skill is choosing the right balance for the situation.

**Confusion matrix — fraud caught vs. missed, side by side:**

![Confusion matrices](reports/figures/confusion_matrices.png)

**ROC curve and Precision-Recall curve** (PR-AUC is the one that matters here — ROC can look deceptively good under this much imbalance):

![ROC and PR curves](reports/figures/roc_pr_curves.png)

**All five metrics side by side:**

![Metrics comparison](reports/figures/metrics_comparison.png)

---

## Which model I'd actually ship

**XGBoost.** The small drop in fraud caught is more than worth it — going from 1,386 false alarms down to 12 is the difference between a system a business can actually run and one that buries the support team. Catching slightly less fraud is a fair price for not alienating over a thousand real customers.

And because the app has an adjustable **threshold slider**, the business can dial this in themselves: lower the threshold to catch more fraud (accepting more false alarms), or raise it to bother fewer customers (accepting a few more misses). The tradeoff becomes a *business decision*, not a fixed one.

---

## If this were running in production

- **Retrain regularly** — fraud tactics change over time, so a model trained today gets stale. Retrain on a rolling window (e.g. monthly).
- **Watch for drift** — alert if the incoming transactions start looking different from what the model was trained on.
- **Treat the threshold as a lever** — tune it to the real cost of a missed fraud vs. a false alarm.
- **Monitor live** — track fraud caught vs. missed and the false-alarm rate on real traffic, not just the test set.

---

## How to run it

```bash
pip install -r requirements.txt

# download creditcard.csv from Kaggle and put it in data/
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

python src/eda.py        # explore the data + save charts
python src/train.py      # train all five models + save results
python src/explain.py    # SHAP explanations
streamlit run src/app.py # launch the interactive app
```

## Tech used

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · Streamlit
