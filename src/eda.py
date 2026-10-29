"""Exploratory data analysis, correlation calculation, and visualization."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FEATURES, RESULTS_DIR


def calculate_correlations(hourly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices for the meteorological features."""
    pearson_corr = hourly_df[FEATURES].corr(method="pearson").round(3)
    spearman_corr = hourly_df[FEATURES].corr(method="spearman").round(3)
    return pearson_corr, spearman_corr


def generate_eda_plots(hourly_df: pd.DataFrame, output_dir: Path = RESULTS_DIR) -> dict[str, Path]:
    """Generate correlation heatmap, scatter relations, and diurnal cycle plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pearson_corr, _ = calculate_correlations(hourly_df)

    # 1. Combined EDA figure (Correlation Matrix + Scatter relations)
    fig = plt.figure(figsize=(13, 3.8))

    ax1 = plt.subplot(1, 3, 1)
    cax = ax1.matshow(pearson_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
    labels = ["Sıcaklık", "Basınç", "Nem", "Radyasyon"]
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(labels, rotation=25, ha="left", fontsize=8)
    ax1.set_yticklabels(labels, fontsize=8)
    for (i, j), val in np.ndenumerate(pearson_corr.values):
        ax1.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            color="white" if abs(val) > 0.5 else "black",
            fontweight="bold",
            fontsize=9,
        )
    ax1.set_title("(a) Korelasyon Matrisi", pad=15, fontweight="bold", fontsize=10)

    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(hourly_df["solar_rad_wm2"], hourly_df["temp_c"], alpha=0.15, color="#e67e22", s=8)
    ax2.set_title("(b) Radyasyon vs Sıcaklık (r=+0.57)", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Güneş Radyasyonu (W/m²)", fontsize=9)
    ax2.set_ylabel("Hava Sıcaklığı (°C)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    ax3 = plt.subplot(1, 3, 3)
    ax3.scatter(hourly_df["humidity_pct"], hourly_df["temp_c"], alpha=0.15, color="#2980b9", s=8)
    ax3.set_title("(c) Bağıl Nem vs Sıcaklık (r=-0.42)", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Bağıl Nem (%)", fontsize=9)
    ax3.set_ylabel("Hava Sıcaklığı (°C)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    combined_path = output_dir / "fig_combined_eda.png"
    fig.savefig(combined_path, dpi=200)
    plt.close()

    # 2. 24-hour diurnal cycle profile
    hourly_df_copy = hourly_df.copy()
    hourly_df_copy["hour"] = hourly_df_copy["datetime"].dt.hour
    diurnal = hourly_df_copy.groupby("hour")[FEATURES].mean()

    fig_d, ax_temp = plt.subplots(figsize=(8, 4))
    ax_rad = ax_temp.twinx()
    ax_temp.plot(diurnal.index, diurnal["temp_c"], "r-o", lw=2, label="Sıcaklık (°C)")
    ax_rad.plot(diurnal.index, diurnal["solar_rad_wm2"], "y--s", lw=1.8, label="Güneş Radyasyonu (W/m²)")
    ax_temp.plot(diurnal.index, diurnal["humidity_pct"], "b:^", lw=1.5, label="Bağıl Nem (%)")

    ax_temp.set_xlabel("Günün Saati (00:00 - 23:00)", fontweight="bold", fontsize=9)
    ax_temp.set_ylabel("Sıcaklık (°C) / Nem (%)", fontweight="bold", fontsize=9)
    ax_rad.set_ylabel("Güneş Radyasyonu (W/m²)", color="goldenrod", fontweight="bold", fontsize=9)
    ax_temp.grid(True, linestyle=":", alpha=0.5)
    lines1, labels1 = ax_temp.get_legend_handles_labels()
    lines2, labels2 = ax_rad.get_legend_handles_labels()
    ax_temp.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout()
    diurnal_path = output_dir / "step2_diurnal_cycle.png"
    fig_d.savefig(diurnal_path, dpi=200)
    plt.close()

    return {"combined": combined_path, "diurnal": diurnal_path}
