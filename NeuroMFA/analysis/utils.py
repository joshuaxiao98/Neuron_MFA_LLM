"""
Utility functions for multifractal analysis and emergence calculations.
"""

import numpy as np
import os
from pathlib import Path
import re


def filter_data_by_mode(alpha_list, f_alpha_list, filter_mode=0):
    """
    Filter multifractal spectrum data based on filtering mode.

    Args:
        alpha_list: List of alpha values
        f_alpha_list: List of f(alpha) values
        filter_mode: 0: no filtering, 1: f_alpha <= 0,
                    2: (max(alpha) - alpha[0]) < 0.01, 3: both

    Returns:
        Tuple of (alpha_list, f_alpha_list) after filtering
    """
    # If no filtering, return the original data
    if filter_mode == 0:
        return alpha_list, f_alpha_list

    filtered_alpha = []
    filtered_f_alpha = []

    for i in range(len(alpha_list)):
        a = alpha_list[i]
        f = f_alpha_list[i]

        # Check filter conditions
        if filter_mode == 1:
            if any(val > 0 for val in f):
                continue
        elif filter_mode == 2:
            if (max(a) - a[0]) > 0.01:
                continue
        elif filter_mode == 3:
            if any(val > 0 for val in f) or (max(a) - a[0]) > 0.01:
                continue

        filtered_alpha.append(a)
        filtered_f_alpha.append(f)

    return filtered_alpha, filtered_f_alpha


def get_model_size(model_name):
    """
    Extract model size in millions of parameters from model name.

    Args:
        model_name: Name of the model (e.g., "pythia-70m-deduped")

    Returns:
        Model size in millions of parameters
    """
    # Extract size part from model name
    size_part = model_name.split('-')[1]

    # Convert to numeric value
    if size_part.endswith('b'):
        # Convert billions to millions
        return float(size_part[:-1]) * 1000
    elif size_part.endswith('m'):
        # Already in millions
        return float(size_part[:-1])
    else:
        # Try to convert as is
        try:
            return float(size_part)
        except ValueError:
            # Return None if can't determine size
            return None


def calculate_statistics(data_list):
    """
    Calculate basic statistics for a list of values.

    Args:
        data_list: List of numeric values

    Returns:
        Dictionary containing statistics
    """
    data_array = np.array(data_list)

    # Handle empty arrays
    if len(data_array) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan
        }

    # Calculate statistics
    return {
        "mean": np.mean(data_array),
        "std": np.std(data_array),
        "min": np.min(data_array),
        "max": np.max(data_array),
        "median": np.median(data_array)
    }


def get_available_models(sample_num=5):
    """
    Get list of available models from network directory.

    Args:
        sample_num: Sample number to look for

    Returns:
        List of available model names
    """
    network_dir = Path(f'./data/networks/sample_{sample_num}')

    if not network_dir.exists():
        return []

    # Get directories (each represents a model)
    return [d.name for d in network_dir.iterdir() if d.is_dir()]


def get_training_steps(model_name):
    """
    Get training steps available for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        List of training steps as integers
    """
    if model_name == 'pythia-31m':
        return [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
                63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 140000]
    else:
        return [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
                63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 143000]


def compare_spectra(model_name1, model_name2, threshold=150, step=143000, sample_idx=0):
    """
    Compare multifractal spectra of two models at a specific training step.

    Args:
        model_name1: First model name
        model_name2: Second model name
        threshold: Threshold used in multifractal analysis
        step: Training step to compare
        sample_idx: Sample index to use

    Returns:
        Dictionary with comparison metrics
    """
    from analysis.multifractal import calculate_multifractal_spectrum

    # Load network data for both models
    network_dir = Path(f'./data/networks/sample_{5}')

    file1 = network_dir / model_name1 / f"{model_name1}_step{step}_sampled_{sample_idx}.pkl"
    file2 = network_dir / model_name2 / f"{model_name2}_step{step}_sampled_{sample_idx}.pkl"

    if not file1.exists() or not file2.exists():
        return {"error": "Model files not found"}

    # Load data
    import pickle
    with open(file1, 'rb') as f:
        net1 = pickle.load(f)

    with open(file2, 'rb') as f:
        net2 = pickle.load(f)

    # Compute multifractal analysis
    from analysis.multifractal import wnfd_llm

    Q = [q / 100 for q in range(-300, 301, 10)]

    tau1, _, _, _ = wnfd_llm(net1, Q, threshold=threshold)
    tau2, _, _, _ = wnfd_llm(net2, Q, threshold=threshold)

    # Calculate spectrum
    alpha0_1, width1, alpha1, f_alpha1 = calculate_multifractal_spectrum(tau1, Q)
    alpha0_2, width2, alpha2, f_alpha2 = calculate_multifractal_spectrum(tau2, Q)

    # Calculate differences
    alpha0_diff = alpha0_1 - alpha0_2
    width_diff = width1 - width2

    # Calculate spectrum area (approximate)
    area1 = np.trapz(f_alpha1, alpha1)
    area2 = np.trapz(f_alpha2, alpha2)
    area_diff = area1 - area2

    return {
        "model1": model_name1,
        "model2": model_name2,
        "step": step,
        "alpha0_1": alpha0_1,
        "alpha0_2": alpha0_2,
        "alpha0_diff": alpha0_diff,
        "width1": width1,
        "width2": width2,
        "width_diff": width_diff,
        "area1": area1,
        "area2": area2,
        "area_diff": area_diff
    }


if __name__ == "__main__":
    # Example usage
    print(get_model_size("pythia-70m-deduped"))
    print(get_model_size("pythia-1.4b-deduped"))

    models = get_available_models()
    print(f"Available models: {models}")

    if len(models) >= 2:
        comparison = compare_spectra(models[0], models[1])
        print("Spectrum comparison:")
        for key, value in comparison.items():
            print(f"  {key}: {value}")