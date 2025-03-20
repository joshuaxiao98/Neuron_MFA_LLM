"""
Utility functions for file I/O operations.
"""

import os
import pickle
import numpy as np
import json
from pathlib import Path


def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Directory path as string or Path object
    """
    if isinstance(directory, str):
        directory = Path(directory)

    directory.mkdir(parents=True, exist_ok=True)


def save_pickle(data, filepath):
    """
    Save data to a pickle file.

    Args:
        data: Data to save
        filepath: Path to save the file
    """
    ensure_dir(os.path.dirname(filepath))

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    """
    Load data from a pickle file.

    Args:
        filepath: Path to the pickle file

    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_numpy(data, filepath):
    """
    Save numpy array to a file.

    Args:
        data: Numpy array to save
        filepath: Path to save the file
    """
    ensure_dir(os.path.dirname(filepath))
    np.save(filepath, data)


def load_numpy(filepath):
    """
    Load numpy array from a file.

    Args:
        filepath: Path to the numpy file

    Returns:
        Loaded numpy array
    """
    return np.load(filepath)


def save_json(data, filepath):
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        filepath: Path to save the file
    """
    ensure_dir(os.path.dirname(filepath))

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Load data from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_model_scores(model_name, step, metrics):
    """
    Get evaluation scores for a model at a specific training step.

    Args:
        model_name: Name of the model
        step: Training step
        metrics: List of metric names to extract

    Returns:
        Dictionary mapping metric names to scores
    """
    # Adjust path based on model name
    if model_name == 'pythia-1b':
        metric_path = f"./pythia/evals/pythia-v1/{model_name}-bf16/zero-shot/"
    else:
        metric_path = f"./pythia/evals/pythia-v1/{model_name}/zero-shot/"

    sub_name = model_name.split("pythia-")[1]
    metric_path += sub_name

    # Adjust file path based on model name
    if model_name == 'pythia-1b':
        file_path = f"{metric_path}-bf16_step{step}.json"
    else:
        file_path = f"{metric_path}_step{step}.json"

    # Load and extract scores
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        scores = {}
        for metric in metrics:
            if metric in data["results"] and "acc" in data["results"][metric]:
                scores[metric] = data["results"][metric]["acc"]
            else:
                scores[metric] = None

        return scores
    except FileNotFoundError:
        print(f"Score file not found: {file_path}")
        return {metric: None for metric in metrics}
    except json.JSONDecodeError:
        print(f"Error parsing JSON: {file_path}")
        return {metric: None for metric in metrics}


if __name__ == "__main__":
    # Example usage
    model_name = "pythia-1.4b-deduped"
    step = 1000
    metrics = ["lambada_openai", "piqa", "arc_easy"]

    scores = get_model_scores(model_name, step, metrics)
    print(f"Scores for {model_name} at step {step}:")
    for metric, score in scores.items():
        print(f"  {metric}: {score}")