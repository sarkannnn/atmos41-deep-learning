# ATMOS 41 - Deep Learning Temperature Forecasting

This repository contains my deep learning project for forecasting hourly air temperature ($T_{t+1}$) using real-world weather station data collected from a METER Group ATMOS 41 all-in-one station.

I built and evaluated two recurrent architectures in PyTorch (LSTM and GRU) to predict temperature one hour ahead using a 24-hour lookback window of multi-sensor meteorological data. In addition to standard training, I tested a day/night balanced sampling strategy to see if it helps models better capture daytime temperature peaks.

---

## Dataset & Preprocessing Notes

The data was collected by an ATMOS 41 unit on a green roof in Maia, Portugal ([source repo: pedroccpimenta/D4fa](https://github.com/pedroccpimenta/D4fa)), covering roughly 2.4 years (Nov 25, 2021 – Apr 22, 2024).

When inspecting and cleaning the raw data, I ran into a few specific quirks:
- **Duplicate timestamps:** There were 5,550 duplicate timestamps (~6.7% of raw records), likely caused by LoRaWAN transmission retries. I dropped duplicate timestamps before resampling.
- **Sensor unit bug:** The raw column `pressão_Pa` is labeled "Pa" in the original file, but readings are around ~100.6. That clearly indicates barometric pressure in **kPa**, not Pa. I renamed it to `pressure_kpa` to keep the units physically correct.
- **Resampling to 1 hour:** The raw station logged every ~15 minutes with some missing intervals. I resampled the timeline to regular 1-hour steps using time-based linear interpolation, resulting in 21,096 continuous hourly records.
- **Split & Normalization:** I split the data chronologically (70% train, 15% validation, 15% test). The `MinMaxScaler` is fit *only* on the training split to make sure no future test statistics leak into the pipeline.

### Input Features Used

| Feature | Unit | Sensor type in ATMOS 41 |
| :--- | :---: | :--- |
| `temp_c` | °C | Micro-thermistor in anemometer cavity |
| `pressure_kpa` | kPa | Piezoresistive MEMS pressure sensor |
| `humidity_pct` | % | Capacitive polymer sensor |
| `solar_rad_wm2` | W/m² | Silicon pyranometer |

---

## Experiments & Model Setup

Each model takes 24 hours of input (`temp_c`, `pressure_kpa`, `humidity_pct`, `solar_rad_wm2`) to predict the next hour's temperature.

I compared:
1. **Persistence Baseline:** $\hat{T}_{t+1} = T_t$ (what the error would be if we simply guess that the next hour will be the same as right now).
2. **LSTM:** 2-layer stacked LSTM (hidden size 64, dropout 0.2).
3. **GRU:** 2-layer stacked GRU (hidden size 64, dropout 0.2).

I trained both networks under two training regimes:
- **Natural:** Standard sequential sliding windows sampled as they occur in nature.
- **Day/Night Balanced:** Since nighttime temperatures change slowly and make up roughly half the dataset, models can become lazy during daytime swings. Here, nighttime windows (solar radiation < 10 W/m²) are balanced against daytime windows so daytime thermal dynamics get equal gradient emphasis.

---

## Test Results

Evaluated on the 3,141 held-out test windows (Dec 12, 2023 – Apr 22, 2024):

| Model & Strategy | Overall MAE (°C) | Overall RMSE (°C) | $R^2$ Score | Day MAE (°C) | Night MAE (°C) | Training Time | vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline ($T_{t+1}=T_t$)** | 0.6407 | 0.9014 | 0.9618 | 0.8575 | 0.4742 | — | *reference* |
| **LSTM (Natural)** | 0.5244 | 0.7162 | 0.9759 | 0.7104 | 0.3816 | ~56 s | +18.2% |
| **LSTM (Day/Night Balanced)** | **0.4710** | **0.6464** | **0.9804** | **0.5742** | 0.3918 | ~57 s | **+26.5%** |
| **GRU (Natural)** | 0.4462 | 0.6148 | 0.9822 | 0.5424 | 0.3723 | ~76 s | +30.4% |
| **GRU (Day/Night Balanced)** | **0.4308** | **0.6122** | **0.9824** | 0.5460 | **0.3424** | ~55 s | **+32.8%** |

### Key Observations:
- Both deep learning models easily beat persistence. GRU balanced achieved the lowest overall MAE (0.4308 °C).
- Balancing day/night windows noticeably helped daytime accuracy: in LSTM, daytime MAE dropped from **0.7104 °C** to **0.5742 °C** without degrading nighttime performance.
- Results are based on a fixed random seed (`42`) across runs for direct comparability.

---

## How to Run

1. **Clone and setup environment:**
   ```bash
   git clone https://github.com/sarkannnn/atmos41-deep-learning.git
   cd atmos41-deep-learning
   pip install -r requirements.txt
   ```

2. **Run the full pipeline:**
   ```bash
   python main.py
   ```
   *Note: If `data/atmos41_raw.csv` is not in your `data/` folder, the script will automatically download it from the source repo on the first run.*

---

## Project Structure

```text
atmos41-deep-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md         # Data source info and download instructions
├── src/
│   ├── config.py         # Hyperparameters and paths
│   ├── data_prep.py      # Duplicate removal, 1h resampling, interpolation
│   ├── eda.py            # Correlation matrices and diurnal cycle plots
│   ├── dataset.py        # 24h windowing, chronological split, balanced sampling
│   ├── models.py         # PyTorch LSTM and GRU implementations
│   ├── train.py          # Train loop, validation early stopping, evaluation
│   └── baseline.py       # Persistence baseline calculation
├── results/              # Generated plots and benchmark JSON
│   ├── benchmark_metrics.json
│   ├── fig_combined_eda.png
│   ├── step2_diurnal_cycle.png
│   └── step5_model_comparison_bars.png
└── main.py               # Runs everything from start to finish
```

## License

MIT License.
