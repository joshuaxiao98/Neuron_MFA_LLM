"""
Calculate emergence metrics from multifractal analysis results.
"""

import numpy as np
from pathlib import Path
import os
from math import log


def calculate_emergence(model_name, threshold=150):
    """
    Calculate the degree of emergence for a model based on multifractal analysis results.

    Args:
        model_name: Name of the model to process
        threshold: Threshold used for multifractal analysis

    Returns:
        Dictionary containing:
        - epochs_npy: Numpy array of epochs
        - E_npy: Numpy array of emergence values
    """
    # Setup paths
    emergence_dir = Path('./results/emergence')
    emergence_dir.mkdir(parents=True, exist_ok=True)

    # Get epoch list based on model name
    if model_name == 'pythia-31m':
        epochs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
                  63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 140000]
    else:
        epochs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
                  63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 143000]

    epochs_npy = np.array(epochs)

    # Load alpha_0 and width data
    alpha_0_path = emergence_dir / 'npy_stored' / model_name / f'threshold{threshold}' / f'{model_name}_alpha_0_mean.npy'
    width_path = emergence_dir / 'npy_stored' / model_name / f'threshold{threshold}' / f'{model_name}_width_mean.npy'

    if not os.path.exists(alpha_0_path) or not os.path.exists(width_path):
        # If the data doesn't exist, run the multifractal analysis
        from analysis.multifractal import process_network
        process_network(model_name, threshold=threshold)

    alpha_0_npy = np.load(alpha_0_path)
    width_npy = np.load(width_path)

    # Get initial values
    alpha0_0 = alpha_0_npy[0]
    width_0 = width_npy[0]

    # Calculate emergence metric
    E = []
    for i in range(len(alpha_0_npy)):
        # Metric: width_ratio * log(alpha_ratio)
        # This captures both heterogeneity and regularity changes
        cur = width_npy[i] / width_0 * log(alpha0_0 / alpha_0_npy[i])
        E.append(cur)

    E_npy = np.array(E)

    # Save the emergence metric
    np.save(emergence_dir / f'E_{model_name}.npy', E_npy)

    return {
        "epochs_npy": epochs_npy,
        "E_npy": E_npy
    }


def calculate_emergence_for_all_models(model_names, threshold=150):
    """
    Calculate emergence metrics for multiple models.

    Args:
        model_names: List of model names to process
        threshold: Threshold used for multifractal analysis

    Returns:
        Dictionary mapping model names to their emergence data
    """
    results = {}

    for model_name in model_names:
        print(f"Calculating emergence for {model_name}")
        result = calculate_emergence(model_name, threshold)
        results[model_name] = result

    return results


if __name__ == "__main__":
    # Example usage
    models_to_process = [
        "pythia-70m-deduped",
        "pythia-160m-deduped",
        "pythia-410m-deduped",
        "pythia-1b-deduped",
        "pythia-1.4b-deduped",
        "pythia-2.8b-deduped"
    ]

    calculate_emergence_for_all_models(models_to_process)