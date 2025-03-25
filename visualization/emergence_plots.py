"""
Functions for visualizing emergence metrics and correlating with model performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def load_emergence_data(model_name):
    """
    Load emergence data for a model.

    Args:
        model_name: Name of the model

    Returns:
        Tuple of (epochs_npy, E_npy) or None if data doesn't exist
    """
    emergence_dir = Path('./results/emergence')
    emergence_file = emergence_dir / f'E_{model_name}.npy'

    if not os.path.exists(emergence_file):
        # Calculate emergence if it doesn't exist
        from analysis.emergence import calculate_emergence
        result = calculate_emergence(model_name)
        return result["epochs_npy"], result["E_npy"]

    # Load emergence data
    E_npy = np.load(emergence_file)

    # Define epochs based on model
    if model_name == 'pythia-31m':
        epochs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
                  63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 140000]
    else:
        epochs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
                  63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 143000]

    epochs_npy = np.array(epochs)

    return epochs_npy, E_npy


def load_metrics_data(model_name, metrics):
    """
    Load performance metrics data for a model.

    Args:
        model_name: Name of the model
        metrics: List of metric names to load

    Returns:
        Dictionary mapping metric names to arrays of scores
    """
    from utils.io_utils import get_model_scores

    # Load epochs first
    epochs_npy, _ = load_emergence_data(model_name)

    # Get metric scores for each epoch
    metrics_data = {}

    for metric in metrics:
        scores = []
        for epoch in epochs_npy:
            score_dict = get_model_scores(model_name, int(epoch), [metric])
            scores.append(score_dict.get(metric, 0))

        metrics_data[metric] = np.array(scores)

    return metrics_data


def plot_emergence_vs_metric(model_name, metric, save_fig=True):
    """
    Plot emergence vs. a specific performance metric.

    Args:
        model_name: Name of the model
        metric: Metric name to plot
        save_fig: Whether to save the figure

    Returns:
        Figure object
    """
    # Set up colors
    color_blue = sns.color_palette()[0]  # Blue for emergence
    color_red = sns.color_palette()[3]  # Red for accuracy

    # Load data
    epochs_npy, E_npy = load_emergence_data(model_name)
    metrics_data = load_metrics_data(model_name, [metric])
    y_ori = metrics_data[metric]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Plot emergence
    ax1.scatter(epochs_npy, E_npy, color=color_blue)
    ax1.plot(epochs_npy, E_npy, color=color_blue, linewidth=3)
    ax1.set_xlabel('Epochs', fontsize=18)
    ax1.set_ylabel('Emergence', color=color_blue, fontsize=18)
    ax1.tick_params(axis='y', labelcolor=color_blue, labelsize=16)
    ax1.tick_params(axis='x', which='major', labelsize=16)

    # Set emergence y-axis limits
    y1_min = (min(E_npy) - 0.025)
    y1_max = (max(E_npy) + 0.025)
    ax1.set_ylim([y1_min, y1_max])

    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.scatter(epochs_npy, y_ori, color=color_red)
    ax2.plot(epochs_npy, y_ori, color=color_red, linewidth=3)
    ax2.set_ylabel('Accuracy', color=color_red, fontsize=18)
    ax2.tick_params(axis='y', labelcolor=color_red, labelsize=16)
    ax2.tick_params(axis='x', which='major', labelsize=16)

    # Set accuracy y-axis limits
    y2_min = (min(y_ori) - 0.025)
    y2_max = (max(y_ori) + 0.025)
    ax2.set_ylim([y2_min, y2_max])

    # Title
    plt.title(f'{model_name} - {metric}', fontsize=16)

    # Save figure if requested
    if save_fig:
        save_dir = Path(f'./results/emergence/figures_metrics')
        im_dir = save_dir / model_name / metric
        os.makedirs(im_dir, exist_ok=True)
        save_path = im_dir / f'{model_name}_{metric}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')

    return fig


def plot_radar_chart(model_name, metrics, steps, save_fig=True):
    """
    Plot radar chart showing model performance and emergence across different training steps.

    Args:
        model_name: Name of the model
        metrics: List of metric names to include
        steps: List of training steps to include
        save_fig: Whether to save the figure

    Returns:
        Figure object
    """
    # Load data
    epochs_npy, E_npy = load_emergence_data(model_name)
    metrics_data = load_metrics_data(model_name, metrics)

    # Find indices for the requested steps
    step_indices = []
    for step in steps:
        idx = np.where(epochs_npy == step)[0]
        if len(idx) > 0:
            step_indices.append(idx[0])

    # Prepare data for radar chart
    categories = ['Emergence'] + metrics
    num_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Colors for different steps
    colors = sns.color_palette("viridis", len(step_indices))

    # Plot each step
    for i, idx in enumerate(step_indices):
        # Normalize data between 0 and 1
        values = [E_npy[idx] / max(E_npy)]
        for metric in metrics:
            values.append(metrics_data[metric][idx])

        # Close the loop
        values += values[:1]

        # Plot values
        ax.plot(angles, values, linewidth=2, color=colors[i], label=f'Step {epochs_npy[idx]}')
        ax.fill(angles, values, color=colors[i], alpha=0.25)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Save figure if requested
    if save_fig:
        save_dir = Path(f'./results/emergence/figures_radar')
        os.makedirs(save_dir, exist_ok=True)
        steps_str = '_'.join([str(s) for s in steps])
        save_path = save_dir / f'{model_name}_radar_{steps_str}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')

    return fig


def plot_emergence_vs_model_size(model_sizes, model_names, epoch=53000, save_fig=True):
    """
    Plot emergence vs. model size for a specific training epoch.

    Args:
        model_sizes: List of model sizes (in millions of parameters)
        model_names: Corresponding list of model names
        epoch: Training epoch to compare
        save_fig: Whether to save the figure

    Returns:
        Figure object
    """
    # Load emergence data for each model
    emergence_values = []

    for model_name in model_names:
        epochs_npy, E_npy = load_emergence_data(model_name)

        # Find the index for the requested epoch
        idx = np.where(epochs_npy == epoch)[0]
        if len(idx) > 0:
            emergence_values.append(E_npy[idx[0]])
        else:
            # Use the closest epoch if exact match is not found
            closest_idx = np.argmin(np.abs(epochs_npy - epoch))
            emergence_values.append(E_npy[closest_idx])

    # Convert model sizes to log scale
    log_sizes = np.log(np.array(model_sizes))

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot scatter with connecting line
    plt.scatter(log_sizes, emergence_values, s=100, color='blue')
    plt.plot(log_sizes, emergence_values, color='blue', linewidth=2)

    # Set axis labels
    plt.xlabel('ln(Model Size)', fontsize=16)
    plt.ylabel('Degree of Emergence', fontsize=16)

    # Add title
    plt.title(f'Degree of Emergence vs. Model Size at Epoch {epoch}', fontsize=18)

    # Set x-tick labels to original model sizes
    plt.xticks(log_sizes, [f'{size}M' for size in model_sizes])

    # Add grid
    plt.grid(True, alpha=0.3)

    # Save figure if requested
    if save_fig:
        save_dir = Path('./results/emergence/figures_size')
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f'emergence_vs_size_epoch{epoch}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')

    return plt.gcf()


if __name__ == "__main__":
    # Example usage
    # Plot emergence vs. specific metrics for a model
    model_name = "pythia-1.4b-deduped"
    metrics = ["lambada_openai", "piqa", "arc_easy", "arc_challenge"]

    for metric in metrics:
        plot_emergence_vs_metric(model_name, metric)
        plt.close()

    # Plot radar chart for specific steps
    steps = [512, 5000, 35000, 107000, 143000]
    plot_radar_chart(model_name, metrics, steps)
    plt.close()

    # Plot emergence vs. model size
    model_sizes = [70, 160, 410, 1000, 1400, 2800]
    model_names = [
        "pythia-70m-deduped",
        "pythia-160m-deduped",
        "pythia-410m-deduped",
        "pythia-1b-deduped",
        "pythia-1.4b-deduped",
        "pythia-2.8b-deduped"
    ]
    plot_emergence_vs_model_size(model_sizes, model_names)
    plt.close()