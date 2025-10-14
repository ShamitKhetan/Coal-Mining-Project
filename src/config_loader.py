"""
Configuration loader for coal mine sensor simulator.
Handles loading and validation of feature and noise configurations.
"""

import json
from pathlib import Path

# Default noise configuration template
DEFAULT_NOISE_CONFIG = {
    "CH4": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.01},
        "bias": {"enabled": True, "value": 0.01},
        "drift": {"enabled": True, "rate": 0.002, "type": "random_walk", "max_drift": 0.5},
        "quantization": {"enabled": True, "resolution": 0.01},
        "outliers": {"enabled": True, "probability": 0.005, "magnitude": 4.0},
        "missing_data": {"enabled": True, "probability": 0.003, "mechanism": "MCAR", "auxiliary_feature": None},
        "invalid_values": {"enabled": True, "probability": 0.001}
    },
    "CO": {
        "gaussian_noise": {"enabled": True, "std_dev": 1.0},
        "bias": {"enabled": True, "value": 0.5},
        "drift": {"enabled": True, "rate": 0.01, "type": "random_walk", "max_drift": 5.0},
        "quantization": {"enabled": True, "resolution": 1.0},
        "outliers": {"enabled": True, "probability": 0.005, "magnitude": 4.0},
        "missing_data": {"enabled": True, "probability": 0.01, "mechanism": "MAR", "auxiliary_feature": "Temperature"},
        "invalid_values": {"enabled": True, "probability": 0.0015}
    },
    "CO2": {
        "gaussian_noise": {"enabled": True, "std_dev": 30.0},
        "bias": {"enabled": True, "value": 10.0},
        "drift": {"enabled": True, "rate": 0.5, "type": "random_walk", "max_drift": 50.0},
        "quantization": {"enabled": True, "resolution": 1.0},
        "outliers": {"enabled": True, "probability": 0.004, "magnitude": 3.0},
        "missing_data": {"enabled": True, "probability": 0.003, "mechanism": "MNAR", "auxiliary_feature": None},
        "invalid_values": {"enabled": True, "probability": 0.001}
    },
    "H2S": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.2},
        "bias": {"enabled": True, "value": 0.1},
        "drift": {"enabled": True, "rate": 0.02, "type": "random_walk", "max_drift": 0.5},
        "quantization": {"enabled": True, "resolution": 0.1},
        "outliers": {"enabled": True, "probability": 0.004, "magnitude": 4.0},
        "missing_data": {"enabled": True, "probability": 0.003, "mechanism": "MCAR", "auxiliary_feature": None},
        "invalid_values": {"enabled": True, "probability": 0.001}
    },
    "SO2": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.08},
        "bias": {"enabled": True, "value": 0.05},
        "drift": {"enabled": True, "rate": 0.01, "type": "random_walk", "max_drift": 0.05},
        "quantization": {"enabled": True, "resolution": 0.1},
        "outliers": {"enabled": True, "probability": 0.004, "magnitude": 4.0},
        "missing_data": {"enabled": True, "probability": 0.003, "mechanism": "MAR", "auxiliary_feature": "Humidity"},
        "invalid_values": {"enabled": True, "probability": 0.001}
    },
    "NH3": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.6},
        "bias": {"enabled": True, "value": 0.25},
        "drift": {"enabled": True, "rate": 0.02, "type": "random_walk", "max_drift": 1.0},
        "quantization": {"enabled": True, "resolution": 0.5},
        "outliers": {"enabled": True, "probability": 0.004, "magnitude": 4.0},
        "missing_data": {"enabled": True, "probability": 0.003, "mechanism": "MNAR", "auxiliary_feature": None},
        "invalid_values": {"enabled": True, "probability": 0.001}
    },
    "NO": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.6},
        "bias": {"enabled": True, "value": 0.25},
        "drift": {"enabled": True, "rate": 0.02, "type": "random_walk", "max_drift": 1.0},
        "quantization": {"enabled": True, "resolution": 0.5},
        "outliers": {"enabled": True, "probability": 0.004, "magnitude": 4.0},
        "missing_data": {"enabled": True, "probability": 0.003, "mechanism": "MCAR", "auxiliary_feature": None},
        "invalid_values": {"enabled": True, "probability": 0.001}
    },
    "NO2": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.12},
        "bias": {"enabled": True, "value": 0.08},
        "drift": {"enabled": True, "rate": 0.01, "type": "random_walk", "max_drift": 0.3},
        "quantization": {"enabled": True, "resolution": 0.05},
        "outliers": {"enabled": True, "probability": 0.004, "magnitude": 4.0},
        "missing_data": {"enabled": True, "probability": 0.005, "mechanism": "MAR", "auxiliary_feature": "Temperature"},
        "invalid_values": {"enabled": True, "probability": 0.001}
    },
    "PM2.5": {
        "gaussian_noise": {"enabled": True, "std_dev": 3.0},
        "bias": {"enabled": True, "value": 1.0},
        "drift": {"enabled": True, "rate": 0.3, "type": "random_walk", "max_drift": 6.0},
        "quantization": {"enabled": True, "resolution": 1.0},
        "outliers": {"enabled": True, "probability": 0.02, "magnitude": 10.0},
        "missing_data": {"enabled": True, "probability": 0.01, "mechanism": "MNAR", "auxiliary_feature": None},
        "invalid_values": {"enabled": True, "probability": 0.002}
    },
    "PM10": {
        "gaussian_noise": {"enabled": True, "std_dev": 6.0},
        "bias": {"enabled": True, "value": 2.0},
        "drift": {"enabled": True, "rate": 0.5, "type": "random_walk", "max_drift": 10.0},
        "quantization": {"enabled": True, "resolution": 1.0},
        "outliers": {"enabled": True, "probability": 0.02, "magnitude": 10.0},
        "missing_data": {"enabled": True, "probability": 0.01, "mechanism": "MCAR", "auxiliary_feature": None},
        "invalid_values": {"enabled": True, "probability": 0.002}
    },
    "Temperature": {
        "gaussian_noise": {"enabled": True, "std_dev": 0.2},
        "bias": {"enabled": True, "value": 0.1},
        "drift": {"enabled": True, "rate": 0.01, "type": "random_walk", "max_drift": 0.8},
        "quantization": {"enabled": True, "resolution": 0.1},
        "outliers": {"enabled": True, "probability": 0.002, "magnitude": 2.0},
        "missing_data": {"enabled": True, "probability": 0.003, "mechanism": "MAR", "auxiliary_feature": "Humidity"},
        "invalid_values": {"enabled": True, "probability": 0.001}
    },
    "Humidity": {
        "gaussian_noise": {"enabled": True, "std_dev": 1.0},
        "bias": {"enabled": True, "value": 0.5},
        "drift": {"enabled": True, "rate": 0.02, "type": "random_walk", "max_drift": 1.0},
        "quantization": {"enabled": True, "resolution": 0.1},
        "outliers": {"enabled": True, "probability": 0.002, "magnitude": 2.0},
        "missing_data": {"enabled": True, "probability": 0.003, "mechanism": "MNAR", "auxiliary_feature": None},
        "invalid_values": {"enabled": True, "probability": 0.001}
    },
    "global": {
        "stuck_values": {
            "enabled": True,
            "probability": 0.001,
            "duration": 5
        }
    }
}


def load_features_config(filepath="config/features.json"):
    """Load feature configuration from JSON file."""
    with open(filepath, 'r') as f:
        features = json.load(f)
    
    # Convert lists back to tuples for consistency with original code
    for feature, config in features.items():
        config["safe"] = tuple(config["safe"])
        config["unsafe"] = tuple(config["unsafe"])
    
    return features


def create_default_noise_config(filepath="config/noise_config.json"):
    """Create default noise configuration file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(DEFAULT_NOISE_CONFIG, f, indent=4)
    print(f"Default noise configuration saved to {filepath}")


def load_noise_config(filepath="config/noise_config.json"):
    """Load noise configuration from JSON file."""
    if not Path(filepath).exists():
        print(f"Config file not found. Creating default config at {filepath}")
        create_default_noise_config(filepath)
    
    with open(filepath, 'r') as f:
        return json.load(f)


def validate_noise_config(noise_config, features):
    """Validate that noise config matches available features."""
    for feature in features.keys():
        if feature not in noise_config:
            print(f"Warning: No noise configuration found for feature '{feature}'")
    
    for feature in noise_config.keys():
        if feature != "global" and feature not in features:
            print(f"Warning: Noise configuration exists for unknown feature '{feature}'")
    
    return True