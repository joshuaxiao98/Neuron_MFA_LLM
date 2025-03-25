"""
Main execution script for the NeuroMFA (Neuron-based Multifractal Analysis) framework.
This script coordinates the workflow from network extraction to visualization.
"""

import argparse
import os
from pathlib import Path
import time

from config import MODELS_NAME, TRAINING_STEPS, create_directories
from data_processing.network_extraction import extract_networks
from data_processing.data_preprocessing import preprocess_all_networks
from analysis.multifractal import process_network
from analysis.emergence import calculate_emergence_for_all_models
from visualization.spectrum_plots import plot_all_model_spectra
from visualization.emergence_plots import (
    plot_emergence_vs_metric,
    plot_radar_chart,
    plot_emergence_vs_model_size
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NeuroMFA: Neuron-based Multifractal Analysis')

    parser.add_argument('--stage', type=str, choices=['all', 'extract', 'preprocess', 'analyze', 'visualize'],
                        default='all', help='Processing stage to run')

    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Specific models to process (default: all models)')

    parser.add_argument('--threshold', type=int, default=150,
                        help='Threshold for multifractal analysis')

    parser.add_argument('--sample_num', type=int, default=5,
                        help='Sample number for network extraction')

    parser.add_argument('--metrics', nargs='+', type=str,
                        default=['lambada_openai', 'piqa', 'arc_easy', 'arc_challenge'],
                        help='Metrics to visualize')

    parser.add_argument('--filter_mode', type=int, default=0,
                        help='Filter mode for multifractal analysis')

    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip processing of existing files')

    return parser.parse_args()


def run_extraction(args):
    """Run network extraction stage."""
    print("\n===== Stage 1: Network Extraction =====")

    models_to_process = args.models or [m for m in MODELS_NAME if "deduped" in m]
    print(f"Processing models: {models_to_process}")

    extract_networks(
        models_to_process=models_to_process,
        indices_range=(0, 10),
        sample_num=args.sample_num
    )

    print("Network extraction completed.")


def run_preprocessing(args):
    """Run data preprocessing stage."""
    print("\n===== Stage 2: Data Preprocessing =====")

    models_to_process = args.models or [m for m in MODELS_NAME if "deduped" in m]
    print(f"Processing models: {models_to_process}")

    preprocess_all_networks(
        models_to_process=models_to_process,
        sample_num=args.sample_num
    )

    print("Data preprocessing completed.")


def run_analysis(args):
    """Run multifractal analysis and emergence calculation stage."""
    print("\n===== Stage 3: Multifractal Analysis =====")

    models_to_process = args.models or [m for m in MODELS_NAME if "deduped" in m]
    print(f"Processing models: {models_to_process}")

    # Process each model
    for model_name in models_to_process:
        print(f"Processing {model_name}")
        process_network(
            model_name=model_name,
            repeat_n=10,
            threshold=args.threshold,
            filter_mode=args.filter_mode
        )

    # Calculate emergence metrics
    calculate_emergence_for_all_models(
        model_names=models_to_process,
        threshold=args.threshold
    )

    print("Multifractal analysis completed.")


def run_visualization(args):
    """Run visualization stage."""
    print("\n===== Stage 4: Visualization =====")

    models_to_process = args.models or [m for m in MODELS_NAME if "deduped" in m]
    print(f"Processing models: {models_to_process}")

    # Plot multifractal spectra
    plot_all_model_spectra(
        models=models_to_process,
        threshold=args.threshold,
        filter_mode=args.filter_mode
    )

    # Plot emergence vs metrics
    for model_name in models_to_process:
        for metric in args.metrics:
            plot_emergence_vs_metric(model_name, metric)

    # Plot radar charts
    steps = [512, 5000, 35000, 107000, 143000]
    for model_name in models_to_process:
        plot_radar_chart(model_name, args.metrics, steps)

    # Plot emergence vs model size
    if len(models_to_process) >= 3:
        # Extract sizes and sort by size
        sizes_and_models = [(int(m.split('-')[1].replace('m', '').replace('b', '000')), m)
                            for m in models_to_process]
        sizes_and_models.sort()

        # Convert sizes to millions and prepare model names list
        model_sizes = [size / 1000 if size >= 1000 else size for size, _ in sizes_and_models]
        model_names = [m for _, m in sizes_and_models]

        # Plot for different epochs
        for epoch in [512, 5000, 35000, 107000, 143000]:
            plot_emergence_vs_model_size(model_sizes, model_names, epoch=epoch)

    print("Visualization completed.")


def main():
    """Main function to run the entire workflow."""
    start_time = time.time()

    # Parse arguments
    args = parse_args()

    # Create necessary directories
    create_directories()

    # Run requested stages
    if args.stage in ['all', 'extract']:
        run_extraction(args)

    if args.stage in ['all', 'preprocess']:
        run_preprocessing(args)

    if args.stage in ['all', 'analyze']:
        run_analysis(args)

    if args.stage in ['all', 'visualize']:
        run_visualization(args)

    # Print execution time
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTotal execution time: {duration:.2f} seconds ({duration / 60:.2f} minutes)")


if __name__ == "__main__":
    main()