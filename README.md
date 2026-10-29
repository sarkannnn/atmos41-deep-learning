# ATMOS 41 Weather Station Temperature Forecasting

A deep learning project implementing Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks to predict 1-hour ahead air temperature ($T_{t+1}$) using METER Group ATMOS 41 all-in-one weather station data. The study compares a natural time series distribution against a day/night balanced sampling strategy.

## Dataset

- **Source:** Historical meteorological data collected by an ATMOS 41 weather station located on a green roof in Maia, Portugal ([D4fa Repository](https://github.com/pedroccpimenta/D4fa)).
- **Timeframe:** November 25, 2021 – April 22, 2024 (879 days, 82,610 raw 15-minute observations).
- **Auto-download:** `main.py` automatically downloads `data/atmos41_raw.csv` from the official repository if not present locally.

### Column Descriptions

| Column Name | Metric | Description | Unit |
| :--- | :--- | :--- | :---: |
| `datetime` | Timestamp | Observation timestamp (15-minute nominal frequency) | YYYY-MM-DD HH:MM:SS |
| `temp_c` | Air Temperature | Ambient air temperature measured by micro-thermistor | °C |
| `pressure_kpa` | Barometric Pressure | Absolute barometric pressure measured by MEMS sensor | kPa |
| `humidity_pct` | Relative Humidity | Relative humidity measured by capacitive polymer | % |
| `solar_rad_wm2` | Solar Radiation | Total solar irradiance measured by pyranometer | W/m² |

*Note on Data Quality: 5,550 duplicate records (6.72%) caused by transmission retries were eliminated, and data was resampled to 21,096 regular 1-hour intervals using time-based linear interpolation.*

## Installation

Ensure Python 3.10+ is installed. Clone the repository and install required dependencies:

```bash
git clone https://github.com/sarkannnn/atmos41-deep-learning.git
cd atmos41-deep-learning
pip install -r requirements.txt
```

## Running the Pipeline

Execute the end-to-end pipeline (data preparation, EDA, baseline, LSTM/GRU training, and evaluation):

```bash
python main.py
```

*Reproducibility: Randomness is fixed across all modules using seed `42` (`src/config.py`).*

## Benchmark Results

Evaluated on the held-out test split (3,141 independent sliding windows, December 12, 2023 – April 22, 2024). Feature normalization (`MinMaxScaler`) is fit strictly on the training set to prevent data leakage.

| Model / Strategy | Overall MAE (°C) | Overall RMSE (°C) | $R^2$ Score | Day MAE (°C) | Night MAE (°C) | Train Time (s) | Baseline Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline ($T_{t+1}=T_t$)** | 0.6407 | 0.9014 | 0.9618 | 0.8575 | 0.4742 | — | *Reference* |
| **LSTM (Natural Distribution)** | 0.5244 | 0.7162 | 0.9759 | 0.7104 | 0.3816 | 56.33 s | 18.2% |
| **LSTM (Day/Night Balanced)** | **0.4710** | **0.6464** | **0.9804** | **0.5742** | 0.3918 | **57.34 s** | **26.5%** |
| **GRU (Natural Distribution)** | 0.4462 | 0.6148 | 0.9822 | 0.5424 | 0.3723 | 75.86 s | 30.4% |
| **GRU (Day/Night Balanced)** | **0.4308** | **0.6122** | **0.9824** | 0.5460 | **0.3424** | **55.38 s** | **32.8%** |

## Repository Structure

```text
atmos41-deep-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py         # Hyperparameters, constants, and paths
│   ├── data_prep.py      # Cleaning, duplicate filtering, hourly resampling
│   ├── eda.py            # Correlation matrices and EDA visualizations
│   ├── dataset.py        # Sliding windows, chronological split, balanced sampling
│   ├── models.py         # PyTorch LSTM and GRU architectures
│   ├── train.py          # Training and evaluation routines
│   └── baseline.py       # Persistence baseline model
├── results/              # Output tables, metrics JSON, and generated plots
│   ├── benchmark_metrics.json
│   ├── fig_combined_eda.png
│   ├── step2_diurnal_cycle.png
│   └── step5_model_comparison_bars.png
└── main.py               # Main sequential execution script
```

## License

This project is licensed under the MIT License.
