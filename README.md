# Coal Mine Sensor Simulator

A modular Python system for generating realistic coal mine sensor data with configurable noise patterns and data quality issues.

## Features

- **12 Sensor Types**: CH4, CO, CO2, H2S, SO2, NH3, NO, NO2, PM2.5, PM10, Temperature, Humidity
- **Realistic Noise Models**:
  - Gaussian noise with configurable standard deviation
  - Systematic bias with adjustable offset values
  - Sensor drift (random walk, sinusoidal, linear) with **bounded drift limits**
  - Quantization effects with configurable resolution
  - Outliers with probability and magnitude controls
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
│   └── noise_config.json          # Noise configuration (generated using config_loader)
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── config_loader.py           # Configuration management
│   ├── noise_functions.py         # Noise application functions
│   ├── dataset_generator.py       # Dataset generation logic
│   ├── visualization.py           # Plotting functions
│   └── streaming_simulator.py     # Stateful streaming simulator
│
├── data/
│   ├── clean/                     # Clean datasets
│   └── noisy/                     # Noisy datasets
│
├── batch_generator.py             # Batch generator entry point
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
python batch_generator.py
```

This will:
1. Load configurations from `config/`
2. Generate 100,000 samples of clean data
3. Generate 100,000 samples of noisy data
4. Save datasets to `data/clean/` and `data/noisy/`
5. Display statistics and visualizations

## Streaming Generator

Stateful continuous generator that maintain drift and stuck state across ticks.

```bash
python stream_to_csv.py \
  --features config/features.json \
  --noise-config config/noise_config.json \
  --output data/stream/coal_mine_stream.csv \
  --interval 1.0 \
  --batch-size 1 \
  --random-state 42 \
  --duration -1
```

Options:
- `--max-bytes` rotate file when exceeding size (default 0 = disabled)

The script supports Ctrl+C to stop gracefully.

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
        "gaussian_noise": {"enabled": true, "std_dev": 0.01},
        "bias": {"enabled": true, "value": 0.01},
        "drift": {"enabled": true, "rate": 0.002, "type": "random_walk", "max_drift": 0.5},
        "quantization": {"enabled": true, "resolution": 0.01},
        "outliers": {"enabled": true, "probability": 0.005, "magnitude": 4.0},
        "missing_data": {
            "enabled": true,
            "probability": 0.003,
            "mechanism": "MCAR",
            "auxiliary_feature": null
        },
        "invalid_values": {"enabled": true, "probability": 0.001}
    },
    "global": {
        "stuck_values": {
            "enabled": true,
            "probability": 0.001,
            "duration": 5
        }
    }
}
```

**Noise Parameters:**
- `gaussian_noise`: Adds random Gaussian noise with specified standard deviation
- `bias`: Adds systematic bias offset to sensor readings
- `drift`: Applies sensor drift over time (random_walk, sinusoidal, linear) with optional `max_drift` clipping
- `quantization`: Simulates sensor resolution limits by rounding to specified resolution
- `outliers`: Adds extreme values with specified probability and magnitude
- `missing_data`: Creates missing values using MCAR, MAR, or MNAR mechanisms
- `invalid_values`: Converts some readings to string format to simulate sensor errors
- `stuck_values`: Simulates sensor failures where all readings freeze for a duration

## Missing Data Mechanisms

- **MCAR (Missing Completely At Random)**: Random missingness
- **MAR (Missing At Random)**: Depends on another feature (e.g., high temperature increases missingness)
- **MNAR (Missing Not At Random)**: Depends on the value itself (e.g., extreme values more likely missing)

## API Reference

### Configuration Functions (`src.config_loader`)

#### `load_features_config(filepath="config/features.json")`
Load feature configuration from JSON file defining sensor types, safe/unsafe ranges, and units.

**Returns:** Dictionary with feature configurations

#### `load_noise_config(filepath="config/noise_config.json")`
Load noise configuration from JSON file. Creates default config if file doesn't exist.

**Returns:** Dictionary with noise parameters for each feature

#### `create_default_noise_config(filepath="config/noise_config.json")`
Create default noise configuration file with predefined parameters for all sensors.

**Parameters:**
- `filepath`: Path where to save the configuration file

#### `validate_noise_config(noise_config, features)`
Validate that noise configuration matches available features and warn about mismatches.

**Parameters:**
- `noise_config`: Dictionary with noise configurations
- `features`: Dictionary with feature configurations

**Returns:** Boolean indicating validation success

### Dataset Generation Functions (`src.dataset_generator`)

#### `generate_dataset(features, n_samples=1000, random_state=None, noise_config=None, apply_noise=True)`
Generate coal mine sensor dataset with optional noise application.

**Parameters:**
- `features`: Feature configuration dictionary
- `n_samples`: Number of samples to generate (default: 1000)
- `random_state`: Random seed for reproducibility
- `noise_config`: Noise configuration dictionary
- `apply_noise`: Whether to apply noise to the dataset (default: True)

**Returns:** Pandas DataFrame with generated sensor data

#### `save_dataset(df, filepath, preserve_strings=False)`
Save dataset to CSV file with optional string value preservation.

**Parameters:**
- `df`: DataFrame to save
- `filepath`: Output file path
- `preserve_strings`: If True, use quoting to preserve string values (default: False)

#### `print_dataset_summary(df, dataset_name="Dataset")`
Print comprehensive summary statistics for a dataset including shape, statistics, missing values, and invalid readings.

**Parameters:**
- `df`: DataFrame to summarize
- `dataset_name`: Name to display in summary (default: "Dataset")

#### `count_string_values(df)`
Count string values (invalid sensor readings) per feature.

**Parameters:**
- `df`: DataFrame to analyze

**Returns:** Pandas Series with count of string values per column

### Noise Application Functions (`src.noise_functions`)

#### `apply_gaussian_noise(values, std_dev)`
Add Gaussian noise to sensor values.

**Parameters:**
- `values`: Array of sensor values
- `std_dev`: Standard deviation of noise

**Returns:** Array with noise applied

#### `apply_bias(values, bias_value)`
Add systematic bias to sensor values.

**Parameters:**
- `values`: Array of sensor values
- `bias_value`: Bias amount to add

**Returns:** Array with bias applied

#### `apply_drift(values, drift_rate, drift_type="random_walk", max_drift=None)`
Apply sensor drift over time with configurable clipping.

**Parameters:**
- `values`: Array of sensor values
- `drift_rate`: Rate of drift
- `drift_type`: Type of drift ("random_walk", "sinusoidal", "linear")
- `max_drift`: Maximum absolute drift value to clip to (None for no clipping)

**Returns:** Array with drift applied

#### `apply_quantization(values, resolution)`
Apply quantization to simulate sensor resolution limits.

**Parameters:**
- `values`: Array of sensor values
- `resolution`: Quantization resolution

**Returns:** Quantized values

#### `apply_outliers(values, probability, magnitude, std_dev)`
Add outliers to sensor data.

**Parameters:**
- `values`: Array of sensor values
- `probability`: Probability of outlier occurrence
- `magnitude`: Multiplier for outlier size (relative to std_dev)
- `std_dev`: Standard deviation for outlier generation

**Returns:** Array with outliers applied

#### `apply_missing_data(values, probability, mechanism="MCAR", auxiliary_data=None)`
Apply missing data based on mechanism type.

**Parameters:**
- `values`: Array of sensor values
- `probability`: Base probability of missing data
- `mechanism`: Missing data mechanism ("MCAR", "MAR", "MNAR")
- `auxiliary_data`: Auxiliary feature data for MAR mechanism

**Returns:** Array with missing values (NaN) applied

#### `apply_global_stuck_values(df, probability, duration)`
Apply stuck values to entire rows (all sensors stuck at once).

**Parameters:**
- `df`: DataFrame with sensor data
- `probability`: Probability of stuck value occurrence
- `duration`: Number of consecutive rows to keep stuck

**Returns:** DataFrame with stuck values applied

#### `apply_string_invalid_values(df, invalid_indices_dict)`
Convert numeric values to string format for invalid entries.

**Parameters:**
- `df`: DataFrame with sensor data
- `invalid_indices_dict`: Dictionary mapping features to boolean masks

**Returns:** DataFrame with invalid values converted to strings

### Visualization Functions (`src.visualization`)

#### `plot_dataset(df, title="Dataset Distribution", figsize=(16, 12))`
Plot histograms for all features in the dataset.

**Parameters:**
- `df`: DataFrame with sensor data
- `title`: Title for the plot (default: "Dataset Distribution")
- `figsize`: Figure size as (width, height) (default: (16, 12))

#### `plot_comparison(df_clean, df_noisy, feature, figsize=(14, 5))`
Plot comparison between clean and noisy data for a single feature.

**Parameters:**
- `df_clean`: DataFrame with clean data
- `df_noisy`: DataFrame with noisy data
- `feature`: Feature name to compare
- `figsize`: Figure size as (width, height) (default: (14, 5))

#### `plot_time_series(df, feature, n_samples=500, figsize=(14, 5))`
Plot time series for a specific feature.

**Parameters:**
- `df`: DataFrame with sensor data
- `feature`: Feature name to plot
- `n_samples`: Number of samples to plot (default: 500)
- `figsize`: Figure size as (width, height) (default: (14, 5))

#### `plot_correlation_matrix(df, figsize=(12, 10))`
Plot correlation matrix heatmap for all numeric features.

**Parameters:**
- `df`: DataFrame with sensor data
- `figsize`: Figure size as (width, height) (default: (12, 10))

#### `plot_missing_data_pattern(df, figsize=(14, 6))`
Visualize missing data patterns across features.

**Parameters:**
- `df`: DataFrame with sensor data
- `figsize`: Figure size as (width, height) (default: (14, 6))

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
from src import plot_comparison, plot_time_series, plot_correlation_matrix

# Compare clean vs noisy
plot_comparison(df_clean, df_noisy, feature="CH4")

# Plot time series
plot_time_series(df_noisy, feature="Temperature", n_samples=1000)

# Analyze correlations
plot_correlation_matrix(df_noisy)
```

### Analysis and Data Quality

```python
from src import count_string_values, print_dataset_summary

# Count invalid sensor readings
invalid_counts = count_string_values(df_noisy)
print(invalid_counts)

# Print comprehensive dataset summary
print_dataset_summary(df_noisy, "Noisy Dataset Analysis")
```

### Advanced Noise Configuration

```python
from src import create_default_noise_config, validate_noise_config

# Create custom noise configuration
create_default_noise_config("config/custom_noise.json")

# Validate configuration
features = load_features_config()
noise_config = load_noise_config("config/custom_noise.json")
validate_noise_config(noise_config, features)
```

## Output Files

- `data/clean/coal_mine_data_clean.csv` - Clean sensor data
- `data/noisy/coal_mine_data_noisy.csv` - Noisy sensor data with quality issues

## Customization

1. **Modify Features**: Edit `config/features.json` to add/remove sensors
2. **Adjust Noise**: Edit `config/noise_config.json` to change noise parameters
3. **Sample Size**: Change `n_samples` in `batch_generator.py`
4. **Random Seed**: Change `random_state` for reproducibility

## Dependencies

- numpy: Numerical operations
- pandas: Data manipulation
- matplotlib: Basic plotting
- seaborn: Statistical visualizations
