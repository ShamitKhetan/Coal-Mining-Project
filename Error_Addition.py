# Imports & Constants

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

FEATURES = {
    "CH4": {"safe": (0, 1.25), "unsafe": (1.25, 5), "unit": "%vol"},      
    "CO": {"safe": (0, 50), "unsafe": (50, 200), "unit": "ppm"},          
    "CO2": {"safe": (0, 5000), "unsafe": (5000, 20000), "unit": "ppm"},   
    "H2S": {"safe": (0, 10), "unsafe": (10, 100), "unit": "ppm"},         
    "SO2": {"safe": (0, 2), "unsafe": (2, 20), "unit": "ppm"},            
    "NH3": {"safe": (0, 25), "unsafe": (25, 100), "unit": "ppm"},         
    "NO": {"safe": (0, 25), "unsafe": (25, 100), "unit": "ppm"},          
    "NO2": {"safe": (0, 3), "unsafe": (3, 20), "unit": "ppm"},            
    "PM2.5": {"safe": (0, 60), "unsafe": (60, 500), "unit": "µg/m³"},     
    "PM10": {"safe": (0, 100), "unsafe": (100, 1000), "unit": "µg/m³"},   
    "Temperature": {"safe": (20, 33.5), "unsafe": (33.5, 50), "unit": "°C"},
    "Humidity": {"safe": (30, 80), "unsafe": (80, 100), "unit": "%RH"}    
}

# Default noise configuration

DEFAULT_NOISE_CONFIG = {
    "CH4": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.05},
        "bias": {"enabled": True, "value": 0.02},
        "drift": {"enabled": True, "rate": 0.01, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 0.01},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "CO": {
        "gaussian_noise": {"enabled": True, "std_dev": 2.0},
        "bias": {"enabled": True, "value": 1.5},
        "drift": {"enabled": True, "rate": 0.5, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 1.0},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "CO2": {
        "gaussian_noise": {"enabled": True, "std_dev": 50.0},
        "bias": {"enabled": True, "value": 20.0},
        "drift": {"enabled": True, "rate": 10.0, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 10.0},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "H2S": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.5},
        "bias": {"enabled": True, "value": 0.3},
        "drift": {"enabled": True, "rate": 0.1, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 0.1},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "SO2": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.2},
        "bias": {"enabled": True, "value": 0.1},
        "drift": {"enabled": True, "rate": 0.05, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 0.1},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "NH3": {
        "gaussian_noise": {"enabled": True, "std_dev": 1.0},
        "bias": {"enabled": True, "value": 0.5},
        "drift": {"enabled": True, "rate": 0.2, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 0.5},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "NO": {
        "gaussian_noise": {"enabled": True, "std_dev": 1.0},
        "bias": {"enabled": True, "value": 0.5},
        "drift": {"enabled": True, "rate": 0.2, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 0.5},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "NO2": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.3},
        "bias": {"enabled": True, "value": 0.2},
        "drift": {"enabled": True, "rate": 0.08, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 0.1},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "PM2.5": {
        "gaussian_noise": {"enabled": True, "std_dev": 5.0},
        "bias": {"enabled": True, "value": 2.0},
        "drift": {"enabled": True, "rate": 1.0, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 1.0},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "PM10": {
        "gaussian_noise": {"enabled": True, "std_dev": 10.0},
        "bias": {"enabled": True, "value": 5.0},
        "drift": {"enabled": True, "rate": 2.0, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 1.0},
        "outliers": {"enabled": True, "probability": 0.01, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.005},
        "stuck_values": {"enabled": True, "probability": 0.002, "duration": 5}
    },
    "Temperature": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.5},
        "bias": {"enabled": True, "value": 0.3},
        "drift": {"enabled": True, "rate": 0.1, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 0.1},
        "outliers": {"enabled": True, "probability": 0.005, "magnitude": 2.0},
        "missing_data": {"enabled": True, "probability": 0.003},
        "stuck_values": {"enabled": True, "probability": 0.001, "duration": 10}
    },
    "Humidity": {
        "gaussian_noise": {"enabled": True, "std_dev": 2.0},
        "bias": {"enabled": True, "value": 1.0},
        "drift": {"enabled": True, "rate": 0.5, "type": "random_walk"},
        "quantization": {"enabled": True, "resolution": 1.0},
        "outliers": {"enabled": True, "probability": 0.005, "magnitude": 2.0},
        "missing_data": {"enabled": True, "probability": 0.003},
        "stuck_values": {"enabled": True, "probability": 0.001, "duration": 10}
    }
}

# Noise Configuration Helpers

def create_default_noise_config(filepath="noise_config.json"):
    with open(filepath, 'w') as f:
        json.dump(DEFAULT_NOISE_CONFIG, f, indent=4)
    print(f"Default noise configuration saved to {filepath}")

def load_noise_config(filepath="noise_config.json"):
    if not Path(filepath).exists():
        print(f"Config file not found. Creating default config at {filepath}")
        create_default_noise_config(filepath)
    with open(filepath, 'r') as f:
        return json.load(f)

# Noise Application Functions

def apply_gaussian_noise(values, std_dev):
    noise = np.random.normal(0, std_dev, len(values))
    return values + noise

def apply_bias(values, bias_value):
    return values + bias_value

def apply_drift(values, drift_rate, drift_type="random_walk"):
    if drift_type == "random_walk":
        drift = np.cumsum(np.random.normal(0, drift_rate, len(values)))
    elif drift_type == "sinusoidal":
        drift = drift_rate * np.sin(np.linspace(0, 4*np.pi, len(values)))
    else:
        drift = drift_rate * np.linspace(0, 1, len(values))
    return values + drift

def apply_quantization(values, resolution):
    return np.round(values / resolution) * resolution

def apply_outliers(values, probability, magnitude, std_dev):
    outlier_mask = np.random.random(len(values)) < probability
    outlier_noise = np.random.normal(0, magnitude * std_dev, len(values))
    values[outlier_mask] += outlier_noise[outlier_mask]
    return values

def apply_missing_data(values, probability):
    missing_mask = np.random.random(len(values)) < probability
    values[missing_mask] = np.nan
    return values

def apply_stuck_values(values, probability, duration):
    i = 0
    while i < len(values):
        if np.random.random() < probability and not np.isnan(values[i]):
            stuck_value = values[i]
            stuck_length = min(duration, len(values) - i)
            values[i:i+stuck_length] = stuck_value
            i += stuck_length
        else:
            i += 1
    return values

# Apply Noise to Feature

def apply_noise_to_feature(values, feature_name, noise_config):
    cfg = noise_config.get(feature_name, {})
    ranges = FEATURES[feature_name]
    min_val = min(ranges["safe"][0], ranges["unsafe"][0])
    max_val = max(ranges["safe"][1], ranges["unsafe"][1])

    if cfg.get("gaussian_noise", {}).get("enabled", False):
        values = apply_gaussian_noise(values, cfg["gaussian_noise"]["std_dev"])
    if cfg.get("bias", {}).get("enabled", False):
        values = apply_bias(values, cfg["bias"]["value"])
    if cfg.get("drift", {}).get("enabled", False):
        values = apply_drift(values, cfg["drift"]["rate"], cfg["drift"].get("type", "random_walk"))

    values = np.clip(values, min_val, max_val)

    if cfg.get("quantization", {}).get("enabled", False):
        values = apply_quantization(values, cfg["quantization"]["resolution"])
    if cfg.get("outliers", {}).get("enabled", False):
        values = apply_outliers(values.copy(), cfg["outliers"]["probability"], cfg["outliers"]["magnitude"], cfg.get("gaussian_noise", {}).get("std_dev", 1.0))
        values = np.clip(values, min_val, max_val)
    if cfg.get("stuck_values", {}).get("enabled", False):
        values = apply_stuck_values(values.copy(), cfg["stuck_values"]["probability"], cfg["stuck_values"]["duration"])
    if cfg.get("missing_data", {}).get("enabled", False):
        values = apply_missing_data(values.copy(), cfg["missing_data"]["probability"])
    
    return values

# Dataset Generator

def generate_dataset(n_samples=1000, random_state=None, noise_config_path="noise_config.json", apply_noise=True):
    if random_state is not None:
        np.random.seed(random_state)
    noise_config = load_noise_config(noise_config_path) if apply_noise else {}
    data = {}

    for feature, ranges in FEATURES.items():
        n_in = int(0.8 * n_samples)
        n_out = n_samples - n_in
        safe_vals = np.random.uniform(ranges["safe"][0], ranges["safe"][1], n_in)
        unsafe_vals = np.random.uniform(ranges["unsafe"][0], ranges["unsafe"][1], n_out)
        values = np.concatenate([safe_vals, unsafe_vals])
        np.random.shuffle(values)
        if apply_noise:
            values = apply_noise_to_feature(values, feature, noise_config)
        data[feature] = values

    return pd.DataFrame(data)

# Run Example

create_default_noise_config("noise_config.json")
df_noisy = generate_dataset(n_samples=100000, random_state=42, apply_noise=True)
df_clean = generate_dataset(n_samples=100000, random_state=42, apply_noise=False)

print("Dataset with noise:")
print(df_noisy.head())
print("\nDataset statistics:")
print(df_noisy.describe())
print(f"\nMissing values per feature:\n{df_noisy.isnull().sum()}")

df_noisy.to_csv("coal_mine_data_noisy.csv", index=False)
df_clean.to_csv("coal_mine_data_clean.csv", index=False)
print("\nDatasets saved to CSV files.")

# Plotting the results

def plot_dataset(df):
    features = df.columns
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(features, 1):
        plt.subplot(4, 3, i)
        sns.histplot(df[feature], bins=30, kde=True, color="steelblue")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()

plot_dataset(df_clean)

plot_dataset(df_noisy)
