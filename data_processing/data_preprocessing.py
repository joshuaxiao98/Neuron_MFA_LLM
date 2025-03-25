"""
Preprocess raw network data for multifractal analysis.
Converts raw edge lists to weighted network representations.
"""

import os
import re
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


def preprocess_network(model_name, sample_num=5, training_steps=None):
    """
    Preprocess network data for a specific model.

    Args:
        model_name: Name of the model to process
        sample_num: Sample number to process
        training_steps: Specific training steps to process (if None, process all)

    Returns:
        None (saves processed data to disk)
    """
    # Setup paths
    net_dir = Path(f'./data/models/sample_{sample_num}')
    pkl_dir = Path(f'./data/networks/sample_{sample_num}')

    # Ensure the output directory exists
    os.makedirs(os.path.join(pkl_dir, model_name), exist_ok=True)

    # Get all network files for this model
    model_dir = net_dir / model_name
    files = os.listdir(model_dir)
    models_npy = [file for file in files if file.endswith('.npy')]

    print(f"Processing {len(models_npy)} network files for {model_name}")

    for model_npy in tqdm(models_npy, desc=f"Processing {model_name}"):
        # Extract step number
        step_num = int(re.search(r"step(\d+)_sampled", model_npy).group(1))

        # Skip if not in requested training steps
        if training_steps is not None and step_num not in training_steps:
            continue

        # Extract sample number
        sample_num = int(re.search(r"sampled_(\d+)\.npy", model_npy).group(1))

        # Load network data
        npy_path = os.path.join(model_dir, model_npy)
        net = np.load(npy_path)

        # Process network data
        collections = {}

        for row in net:
            key = row[0]
            value = row[2]
            if key not in collections:
                collections[key] = []
            # Inverse of absolute weight value
            collections[key].append(1 / np.abs(value))

        # Sort keys for consistency
        sorted_keys = sorted(collections.keys())
        nets = [collections[key] for key in sorted_keys]

        # Save processed network as pickle
        model_npy = Path(model_npy)
        model_pkl = model_npy.with_suffix('.pkl')
        pkl_path = os.path.join(pkl_dir, model_name, model_pkl)

        with open(pkl_path, 'wb') as file:
            pickle.dump(nets, file)

        print(f"{model_pkl} successfully saved")


def preprocess_all_networks(models_to_process=None, sample_num=5):
    """
    Preprocess all network data for multiple models.

    Args:
        models_to_process: List of model names to process (if None, use models in the directory)
        sample_num: Sample number to process

    Returns:
        None
    """
    from config import TRAINING_STEPS

    # Get available models if not specified
    if models_to_process is None:
        net_dir = Path(f'./data/models/sample_{sample_num}')
        models_to_process = [entry for entry in os.listdir(net_dir)
                             if os.path.isdir(os.path.join(net_dir, entry))]

    for model_name in models_to_process:
        print(f"Preprocessing networks for {model_name}")
        preprocess_network(model_name, sample_num, TRAINING_STEPS)


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

    preprocess_all_networks(models_to_process, sample_num=5)