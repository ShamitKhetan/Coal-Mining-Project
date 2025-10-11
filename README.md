# Coal Mine Sensor Simulator

A modular Python system for generating realistic coal mine sensor data with configurable noise patterns and data quality issues.

## Features

- **12 Sensor Types**: CH4, CO, CO2, H2S, SO2, NH3, NO, NO2, PM2.5, PM10, Temperature, Humidity
- **Realistic Noise Models**:
  - Gaussian noise
  - Systematic bias
  - Sensor drift (random walk, sinusoidal, linear)
  - Quantization effects
  - Outliers
  - Missing data (MCAR, MAR, MNAR mechanisms)
  - Stuck values (global sensor failures)
  - Invalid sensor readings (string values)
- **Configurable**: JSON-based configuration for easy customization
- **Visualization**: Built-in plotting for data analysis

## Directory Structure

```
coal_mine_sensor_simulator/
│
├── config/
│   ├── features.json              # Sensor feature definitions
│   └── noise_config.json          # Noise configuration (auto-generated)
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── config_loader.py           # Configuration management
│   ├── noise_functions.py         # Noise application functions
│   ├── dataset_generator.py       # Dataset generation logic
│   └── visualization.py           # Plotting functions
│
├── data/
│   ├── clean/                     # Clean datasets
│   └── noisy/                     # Noisy datasets
│
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. Clone or download the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create the `config/features.json` file (see Configuration section)

## Quick Start

Run the simulator:

```bash
python main.py
```

This will:
1. Load configurations from `config/`
2. Generate 100,000 samples of clean data
3. Generate 100,000 samples of noisy data
4. Save datasets to `data/clean/` and `data/noisy/`
5. Display statistics and visualizations

## Configuration

### Feature Configuration (`config/features.json`)

Defines sensor types, safe/unsafe ranges, and units:

```json
{
    "CH4": {
        "safe": [0, 1.25],
        "unsafe": [1.25, 5],
        "unit": "%vol"
    },
    ...
}
```

### Noise Configuration (`config/noise_config.json`)

Auto-generated on first run. Customize noise parameters for each sensor:

```json
{
    "CH4": {
        "gaussian_noise": {"enabled": true, "std_dev": 0.05},
        "bias": {"enabled": true, "value": 0.02},
        "drift": {"enabled": true, "rate": 0.01, "type": "random_walk"},
        "missing_data": {
            "enabled": true,
            "probability": 0.005,
            "mechanism": "MCAR",
            "auxiliary_feature": null
        },
        ...
    },
    "global": {
        "stuck_values": {
            "enabled": true,
            "probability": 0.002,
            "duration": 5
        }
    }
}
```

## Missing Data Mechanisms

- **MCAR (Missing Completely At Random)**: Random missingness
- **MAR (Missing At Random)**: Depends on another feature (e.g., high temperature increases missingness)
- **MNAR (Missing Not At Random)**: Depends on the value itself (e.g., extreme values more likely missing)

## Usage Examples

### Basic Generation

```python
from src import load_features_config, load_noise_config, generate_dataset

features = load_features_config()
noise_config = load_noise_config()

# Generate noisy dataset
df_noisy = generate_dataset(
    features=features,
    n_samples=10000,
    noise_config=noise_config,
    apply_noise=True
)
```

### Custom Visualization

```python
from src import plot_comparison, plot_time_series

# Compare clean vs noisy
plot_comparison(df_clean, df_noisy, feature="CH4")

# Plot time series
plot_time_series(df_noisy, feature="Temperature", n_samples=1000)
```

### Analysis

```python
from src import count_string_values

# Count invalid sensor readings
invalid_counts = count_string_values(df_noisy)
print(invalid_counts)
```

## Output Files

- `data/clean/coal_mine_data_clean.csv` - Clean sensor data
- `data/noisy/coal_mine_data_noisy.csv` - Noisy sensor data with quality issues

## Customization

1. **Modify Features**: Edit `config/features.json` to add/remove sensors
2. **Adjust Noise**: Edit `config/noise_config.json` to change noise parameters
3. **Sample Size**: Change `n_samples` in `main.py`
4. **Random Seed**: Change `random_state` for reproducibility

## Dependencies

- numpy: Numerical operations
- pandas: Data manipulation
- matplotlib: Basic plotting
- seaborn: Statistical visualizations
