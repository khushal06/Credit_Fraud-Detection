import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.config import FIGURES_DIR, TARGET
    from src.data import load_dataset
except ModuleNotFoundError:  # pragma: no cover
    from config import FIGURES_DIR, TARGET
    from data import load_dataset


def summarize_dataset(df):
    fraud_count = int(df[TARGET].sum())
    row_count = len(df)
    fraud_rate = fraud_count / row_count
    imbalance_ratio = (df[TARGET] == 0).sum() / fraud_count

    print(f"Row count: {row_count}")
    print(f"Fraud count: {fraud_count}")
    print(f"Fraud rate: {fraud_rate:.6%}")
    print(f"Imbalance ratio (non-fraud/fraud): {imbalance_ratio:.2f}")

    return {
        "row_count": row_count,
        "fraud_count": fraud_count,
        "fraud_rate": float(fraud_rate),
        "imbalance_ratio": float(imbalance_ratio),
    }


def save_balance_chart(df, output_path: Path):
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Legit", "Fraud"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=["steelblue", "crimson"])
    plt.title("Class Balance")
    plt.ylabel("Transactions")
    for idx, value in enumerate(values):
        plt.text(idx, value + max(values) * 0.01, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_amount_histogram(df, output_path: Path):
    plt.figure(figsize=(9, 5))
    for label, color in [(0, "steelblue"), (1, "crimson")]:
        subset = df.loc[df[TARGET] == label, "Amount"]
        plt.hist(subset, bins=60, alpha=0.7, label=f"Class {label}", color=color)
    plt.title("Transaction Amount by Class")
    plt.xlabel("Amount")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    df = load_dataset()
    summary = summarize_dataset(df)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_balance_chart(df, FIGURES_DIR / "class_balance.png")
    save_amount_histogram(df, FIGURES_DIR / "amount_by_class.png")
    with (FIGURES_DIR.parent / "eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved class balance chart to {FIGURES_DIR / 'class_balance.png'}")
    print(f"Saved amount histogram to {FIGURES_DIR / 'amount_by_class.png'}")


if __name__ == "__main__":
    main()
