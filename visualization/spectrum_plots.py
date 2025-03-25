"""
Functions for visualizing multifractal spectra.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Configure plot styles
plt.rcParams.update({'font.size': 30})
sns.set_theme(style="whitegrid")


def load_spectrum_data(model_name, threshold=150, filter_mode=0):
    """
    Load multifractal spectrum data for a model.

    Args:
        model_name: Name of the model
        threshold: Threshold used in multifractal analysis
        filter_mode: Filter mode used in analysis

    Returns:
        Dictionary containing alpha and f(alpha) data
    """
    # Setup paths
    fig_dir = Path(f'./results/figures/{model_name}/threshold{threshold}')
    os.makedirs(fig_dir, exist_ok=True)

    # Run multifractal analysis if data doesn't exist
    ntauls_list_stored_path = Path(f'./results/emergence/npy_stored/{model_name}/threshold{threshold}/ntauls_list.npy')

    if not os.path.exists(ntauls_list_stored_path):
        from analysis.multifractal import process_network
        process_network(model_name, threshold=threshold)

    # Load tau values
    ntauls_list = np.load(ntauls_list_stored_path)
    Q = [q / 100 for q in range(-300, 301, 10)]

    # Process tau values to get alpha and f(alpha)
    from analysis.multifractal import calculate_multifractal_spectrum

    alpha, f_alpha = [], []
    repeat_n = 10
    count_num = 0
    repeat_n_list = []

    for i in range(len(ntauls_list) + 1):
        if i % repeat_n == 0 and i > 0:
            repeat_n_list.append(count_num)
            count_num = 0
            if i == len(ntauls_list):
                break

        if i < len(ntauls_list):
            _, _, a, f = calculate_multifractal_spectrum(ntauls_list[i], Q)

            # Apply filters based on filter_mode
            if filter_mode == 1:
                if any(n > 0 for n in f):
                    continue
            elif filter_mode == 2:
                if (max(a) - a[0]) > 0.01:
                    continue
            elif filter_mode == 3:
                if any(n > 0 for n in f) or (max(a) - a[0]) > 0.01:
                    continue

            alpha.append(a)
            f_alpha.append(f)
            count_num += 1

    # Calculate mean alpha and f(alpha) for each epoch
    alpha_mean, f_alpha_mean = [], []
    alpha_std, f_alpha_std = [], []
    start_index, end_index = 0, 0

    if model_name == 'pythia-31m':
        name = ['Step ' + str(i) for i in [1, 8, 64, 512, 1000, 5000, 10000, 35000, 71000, 107000, 140000]]
    else:
        name = ['Step ' + str(i) for i in [1, 8, 64, 512, 1000, 5000, 10000, 35000, 71000, 107000, 143000]]

    for i in range(len(name)):
        if i < len(repeat_n_list):
            end_index = end_index + repeat_n_list[i]

            if start_index < len(alpha) and end_index <= len(alpha) and start_index < end_index:
                alpha_mean.append(np.mean(alpha[start_index:end_index], axis=0))
                f_alpha_mean.append(np.mean(f_alpha[start_index:end_index], axis=0))
                alpha_std.append(np.std(alpha[start_index:end_index], axis=0))
                f_alpha_std.append(np.std(f_alpha[start_index:end_index], axis=0))
            else:
                # Handle case where indices are out of bounds
                if len(alpha) > 0:
                    alpha_mean.append(np.zeros_like(alpha[0]))
                    f_alpha_mean.append(np.zeros_like(f_alpha[0]))
                    alpha_std.append(np.zeros_like(alpha[0]))
                    f_alpha_std.append(np.zeros_like(f_alpha[0]))

            start_index = start_index + repeat_n_list[i]

    return {
        "alpha_mean": alpha_mean,
        "f_alpha_mean": f_alpha_mean,
        "alpha_std": alpha_std,
        "f_alpha_std": f_alpha_std,
        "step_names": name
    }


def plot_multifractal_spectrum(model_name, threshold=150, filter_mode=0, save_fig=True):
    """
    Plot multifractal spectrum for a model.

    Args:
        model_name: Name of the model
        threshold: Threshold used in multifractal analysis
        filter_mode: Filter mode used in analysis
        save_fig: Whether to save the figure to disk

    Returns:
        Figure object
    """
    # Load spectrum data
    data = load_spectrum_data(model_name, threshold, filter_mode)
    alpha_mean = data["alpha_mean"]
    f_alpha_mean = data["f_alpha_mean"]
    alpha_std = data["alpha_std"]
    f_alpha_std = data["f_alpha_std"]
    step_names = data["step_names"]

    # Create colormap for steps
    colors = sns.color_palette("coolwarm", len(step_names))

    # Create figure
    plt.figure(figsize=(10, 10))

    # Plot spectra for each step
    for i in range(len(step_names)):
        mid = np.argmax(f_alpha_mean[i])

        # Plot the mean spectrum line
        plt.plot(alpha_mean[i], f_alpha_mean[i], color=colors[i], label=step_names[i], linewidth=3)

        # Add standard deviation shading
        plt.fill_between(alpha_mean[i], f_alpha_mean[i] - f_alpha_std[i], f_alpha_mean[i], alpha=0.8, color=colors[i])
        plt.fill_between(alpha_mean[i], f_alpha_mean[i], f_alpha_mean[i] + f_alpha_std[i], alpha=0.8, color=colors[i])

        # Add standard deviation for x-axis (alpha) values
        plt.fill_betweenx(f_alpha_mean[i][:mid], alpha_mean[i][:mid] - alpha_std[i][:mid],
                          alpha_mean[i][:mid] + alpha_std[i][:mid], alpha=0.55, color=colors[i], interpolate=True)
        plt.fill_betweenx(f_alpha_mean[i][mid:], alpha_mean[i][mid:] - alpha_std[i][mid:],
                          alpha_mean[i][mid:] + alpha_std[i][mid:], alpha=0.55, color=colors[i], interpolate=True)

    # Set axis labels
    plt.xlabel('Lipschitz-Holder exponent, 'r'$\alpha$')
    plt.ylabel('Multi-fractal spectrum, 'r'$f(\alpha)$')
    plt.grid(False)

    # Add legend with smaller font
    plt.legend(fontsize=15, loc='best')

    # Save figure if requested
    if save_fig:
        fig_dir = Path(f'./results/figures/{model_name}/threshold{threshold}')
        os.makedirs(fig_dir, exist_ok=True)
        save_path = fig_dir / f'{model_name}_mean_f{filter_mode}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')

    return plt.gcf()


def plot_all_model_spectra(models, threshold=150, filter_mode=0):
    """
    Plot multifractal spectra for multiple models.

    Args:
        models: List of model names
        threshold: Threshold used in multifractal analysis
        filter_mode: Filter mode used in analysis

    Returns:
        None
    """
    for model_name in models:
        print(f"Plotting spectrum for {model_name}")
        plot_multifractal_spectrum(model_name, threshold, filter_mode)
        plt.close()


if __name__ == "__main__":
    # Example usage
    models_to_plot = [
        "pythia-70m-deduped",
        "pythia-160m-deduped",
        "pythia-410m-deduped",
        "pythia-1b-deduped",
        "pythia-1.4b-deduped",
        "pythia-2.8b-deduped"
    ]

    plot_all_model_spectra(models_to_plot)