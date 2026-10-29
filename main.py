"""Main pipeline script for ATMOS 41 temperature forecasting.

Executes data preprocessing, exploratory data analysis, dataset windowing,
persistence baseline evaluation, and recurrent neural network training.
"""

import json
import sys
from pathlib import Path

# Fix console encoding on Windows platforms if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from src.baseline import evaluate_persistence_baseline
from src.config import RANDOM_SEED, RESULTS_DIR
from src.data_prep import load_and_clean_raw_data, resample_to_hourly
from src.dataset import (
    create_balanced_day_night_dataset,
    create_sliding_windows,
    scale_features,
    split_chronologically,
)
from src.eda import calculate_correlations, generate_eda_plots
from src.train import (
    evaluate_recurrent_model,
    plot_model_comparison_bars,
    train_recurrent_model,
)


def main() -> None:
    """Run full experimental pipeline and print benchmark results."""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data Loading and Quality Audit
    print("1. Data loading and quality audit...")
    clean_df, data_meta = load_and_clean_raw_data()
    hourly_df, hourly_meta = resample_to_hourly(clean_df)
    print(f"   Raw records: {data_meta['raw_count']}")
    print(f"   Duplicates removed: {data_meta['duplicate_count']} ({data_meta['duplicate_percentage']}%)")
    print(f"   Clean hourly records: {hourly_meta['hourly_count']} hours")

    # 2. Exploratory Data Analysis
    print("2. Generating exploratory data analysis plots...")
    pearson_corr, _ = calculate_correlations(hourly_df)
    generate_eda_plots(hourly_df, RESULTS_DIR)

    # 3. Chronological Splitting and Sizing
    print("3. Chronological dataset partitioning and scaling...")
    train_df, val_df, test_df = split_chronologically(hourly_df)

    # Note: Scaler is strictly fit only on train_df to prevent data leakage
    x_train, y_train, x_val, y_val, x_test, y_test, scaler_y = scale_features(
        train_df, val_df, test_df
    )

    x_train_s, y_train_s, day_train_s = create_sliding_windows(x_train, y_train, train_df)
    x_val_s, y_val_s, _ = create_sliding_windows(x_val, y_val, val_df)
    x_test_s, y_test_s, day_test_s = create_sliding_windows(x_test, y_test, test_df)

    x_train_bal, y_train_bal = create_balanced_day_night_dataset(
        x_train_s, y_train_s, day_train_s
    )

    # 4. Persistence Baseline Evaluation
    print("4. Evaluating persistence baseline...")
    metrics_baseline = evaluate_persistence_baseline(test_df, day_test_s)

    # 5. Model Training (LSTM & GRU)
    print("5. Training LSTM and GRU models (Natural vs Balanced)...")
    model_lstm_nat, time_lstm_nat = train_recurrent_model("LSTM", x_train_s, y_train_s, x_val_s, y_val_s)
    model_lstm_bal, time_lstm_bal = train_recurrent_model("LSTM", x_train_bal, y_train_bal, x_val_s, y_val_s)
    model_gru_nat, time_gru_nat = train_recurrent_model("GRU", x_train_s, y_train_s, x_val_s, y_val_s)
    model_gru_bal, time_gru_bal = train_recurrent_model("GRU", x_train_bal, y_train_bal, x_val_s, y_val_s)

    # 6. Evaluation on Held-Out Test Set
    print("6. Evaluating on test set...")
    metrics_lstm_nat = evaluate_recurrent_model(model_lstm_nat, x_test_s, y_test_s, day_test_s, scaler_y)
    metrics_lstm_bal = evaluate_recurrent_model(model_lstm_bal, x_test_s, y_test_s, day_test_s, scaler_y)
    metrics_gru_nat = evaluate_recurrent_model(model_gru_nat, x_test_s, y_test_s, day_test_s, scaler_y)
    metrics_gru_bal = evaluate_recurrent_model(model_gru_bal, x_test_s, y_test_s, day_test_s, scaler_y)

    metrics_lstm_nat["time_sec"] = round(time_lstm_nat, 2)
    metrics_lstm_bal["time_sec"] = round(time_lstm_bal, 2)
    metrics_gru_nat["time_sec"] = round(time_gru_nat, 2)
    metrics_gru_bal["time_sec"] = round(time_gru_bal, 2)

    results = {
        "Persistence_Baseline": metrics_baseline,
        "LSTM_Natural": metrics_lstm_nat,
        "LSTM_Balanced": metrics_lstm_bal,
        "GRU_Natural": metrics_gru_nat,
        "GRU_Balanced": metrics_gru_bal,
    }

    # Save metrics JSON and comparison plot
    metrics_file = RESULTS_DIR / "benchmark_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    plot_model_comparison_bars(results, RESULTS_DIR / "step5_model_comparison_bars.png")

    # 7. Print Output Table
    print("\nBenchmark Results (Held-Out Test Set, 3141 Windows):")
    print(f"{'Model / Strategy':<32} {'MAE (°C)':<10} {'RMSE (°C)':<10} {'R²':<8} {'Day MAE':<10} {'Night MAE':<10} {'Time (s)':<8}")
    print("-" * 88)
    for model_name, m in results.items():
        time_str = f"{m['time_sec']:.2f}s" if m["time_sec"] > 0 else "-"
        print(f"{model_name:<32} {m['mae']:<10.4f} {m['rmse']:<10.4f} {m['r2']:<8.4f} {m['day_mae']:<10.4f} {m['night_mae']:<10.4f} {time_str:<8}")
    print("-" * 88)
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
