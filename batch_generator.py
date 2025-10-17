"""
Main entry point for coal mine sensor simulator.
Generates clean and noisy datasets with visualization.
"""

from src import (
    load_features_config,
    load_noise_config,
    create_default_noise_config,
    generate_dataset,
    save_dataset,
    print_dataset_summary,
    plot_dataset
)


def main():
    """Main execution function."""
    
    # Configuration
    n_samples = 100000
    random_state = 42
    
    print("Coal Mine Sensor Data Generator")
    print("=" * 60)
    
    # Load configurations
    print("\nLoading configurations...")
    features = load_features_config("config/features.json")
    print(f"✓ Loaded {len(features)} features")
    
    # Create default noise config if it doesn't exist
    create_default_noise_config("config/noise_config.json")
    noise_config = load_noise_config("config/noise_config.json")
    print("✓ Noise configuration loaded")
    
    # Generate clean dataset
    print("\nGenerating clean dataset...")
    df_clean = generate_dataset(
        features=features,
        n_samples=n_samples,
        random_state=random_state,
        apply_noise=False
    )
    print(f"✓ Generated {len(df_clean)} clean samples")
    
    # Generate noisy dataset
    print("\nGenerating noisy dataset...")
    df_noisy = generate_dataset(
        features=features,
        n_samples=n_samples,
        random_state=random_state,
        noise_config=noise_config,
        apply_noise=True
    )
    print(f"✓ Generated {len(df_noisy)} noisy samples")
    
    # Save datasets
    print("\nSaving datasets...")
    save_dataset(df_clean, "data/clean/coal_mine_data_clean.csv")
    save_dataset(df_noisy, "data/noisy/coal_mine_data_noisy.csv", preserve_strings=True)
    print("✓ Datasets saved successfully")
    
    # Print summaries
    print_dataset_summary(df_clean, "Clean Dataset")
    print_dataset_summary(df_noisy, "Noisy Dataset")
    
    # Visualize datasets
    print("\nGenerating visualizations...")
    print("Plotting clean dataset distribution...")
    plot_dataset(df_clean, title="Clean Dataset Distribution")
    
    print("Plotting noisy dataset distribution...")
    plot_dataset(df_noisy, title="Noisy Dataset Distribution")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


