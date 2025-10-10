"""
Visualization functions for coal mine sensor simulator.
Provides plotting and analysis capabilities for sensor data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_dataset(df, title="Dataset Distribution", figsize=(16, 12)):
    """
    Plot histograms for all features in the dataset.
    
    Args:
        df: DataFrame with sensor data
        title: Title for the plot
        figsize: Figure size as (width, height)
    """
    # Convert to numeric for plotting, coercing strings to NaN
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    features = df_numeric.columns
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, y=0.995)
    
    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df_numeric[feature], bins=30, kde=True, color="steelblue")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()


def plot_comparison(df_clean, df_noisy, feature, figsize=(14, 5)):
    """
    Plot comparison between clean and noisy data for a single feature.
    
    Args:
        df_clean: DataFrame with clean data
        df_noisy: DataFrame with noisy data
        feature: Feature name to compare
        figsize: Figure size as (width, height)
    """
    # Convert to numeric
    clean_numeric = pd.to_numeric(df_clean[feature], errors='coerce')
    noisy_numeric = pd.to_numeric(df_noisy[feature], errors='coerce')
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Clean data
    axes[0].hist(clean_numeric.dropna(), bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0].set_title(f'{feature} - Clean Data')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    
    # Noisy data
    axes[1].hist(noisy_numeric.dropna(), bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[1].set_title(f'{feature} - Noisy Data')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


def plot_time_series(df, feature, n_samples=500, figsize=(14, 5)):
    """
    Plot time series for a specific feature.
    
    Args:
        df: DataFrame with sensor data
        feature: Feature name to plot
        n_samples: Number of samples to plot
        figsize: Figure size as (width, height)
    """
    # Convert to numeric
    values = pd.to_numeric(df[feature].head(n_samples), errors='coerce')
    
    plt.figure(figsize=figsize)
    plt.plot(values, linewidth=1, alpha=0.8)
    plt.title(f'Time Series: {feature} (First {n_samples} samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, figsize=(12, 10)):
    """
    Plot correlation matrix heatmap for all numeric features.
    
    Args:
        df: DataFrame with sensor data
        figsize: Figure size as (width, height)
    """
    # Convert to numeric
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Calculate correlation
    corr = df_numeric.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_missing_data_pattern(df, figsize=(14, 6)):
    """
    Visualize missing data patterns across features.
    
    Args:
        df: DataFrame with sensor data
        figsize: Figure size as (width, height)
    """
    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    plt.figure(figsize=figsize)
    missing_pct.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Missing Data Percentage by Feature')
    plt.xlabel('Feature')
    plt.ylabel('Missing Data (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()