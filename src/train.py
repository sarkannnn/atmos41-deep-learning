"""Model training routines, evaluation metrics, and comparison visualizations."""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED
from src.models import RecurrentTemperatureModel


def train_recurrent_model(
    cell_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> tuple[RecurrentTemperatureModel, float]:
    """Train a PyTorch LSTM or GRU model."""
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = RecurrentTemperatureModel(cell_type=cell_type)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    training_time = time.time() - start_time
    return model, training_time


def evaluate_recurrent_model(
    model: RecurrentTemperatureModel,
    x_test: np.ndarray,
    y_test: np.ndarray,
    is_day_test: np.ndarray,
    scaler_y: MinMaxScaler,
) -> dict[str, float]:
    """Compute overall, daytime, and nighttime MAE and RMSE metrics on held-out test data."""
    model.eval()
    with torch.no_grad():
        scaled_predictions = model(torch.tensor(x_test, dtype=torch.float32)).numpy()

    predictions = scaler_y.inverse_transform(scaled_predictions).flatten()
    actuals = scaler_y.inverse_transform(y_test).flatten()

    mae_all = mean_absolute_error(actuals, predictions)
    rmse_all = np.sqrt(mean_squared_error(actuals, predictions))
    r2_all = r2_score(actuals, predictions)

    mae_day = mean_absolute_error(actuals[is_day_test], predictions[is_day_test])
    rmse_day = np.sqrt(mean_squared_error(actuals[is_day_test], predictions[is_day_test]))

    mae_night = mean_absolute_error(actuals[~is_day_test], predictions[~is_day_test])
    rmse_night = np.sqrt(mean_squared_error(actuals[~is_day_test], predictions[~is_day_test]))

    return {
        "mae": float(mae_all),
        "rmse": float(rmse_all),
        "r2": float(r2_all),
        "day_mae": float(mae_day),
        "day_rmse": float(rmse_day),
        "night_mae": float(mae_night),
        "night_rmse": float(rmse_night),
    }


def plot_model_comparison_bars(results: dict[str, dict[str, float]], output_path: Path) -> Path:
    """Generate bar chart comparing MAE and RMSE across evaluated models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["LSTM (Doğal)", "LSTM (Dengeli)", "GRU (Doğal)", "GRU (Dengeli)"]
    keys = ["LSTM_Natural", "LSTM_Balanced", "GRU_Natural", "GRU_Balanced"]

    maes = [results[k]["mae"] for k in keys]
    rmses = [results[k]["rmse"] for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    rects1 = ax.bar(x - width / 2, maes, width, label="Genel MAE (°C)", color="#3498db")
    rects2 = ax.bar(x + width / 2, rmses, width, label="Genel RMSE (°C)", color="#e74c3c")

    ax.set_ylabel("Hata Değeri (°C)", fontweight="bold")
    ax.set_title("Model Performans Karşılaştırması", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close()
    return output_path
