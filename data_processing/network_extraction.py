"""
Extract neuron interaction networks from Pythia models.
This module converts PyTorch model weights into network representations.
"""

import os
import re
import numpy as np
import torch
from transformers import GPTNeoXForCausalLM
from pathlib import Path


def rand_sample(sample_num, total_num):
    """
    Randomly sample indices without replacement.

    Args:
        sample_num: Number of samples to draw
        total_num: Total population size to sample from

    Returns:
        Array of sampled indices
    """
    nums = np.arange(0, total_num)
    samples = np.random.choice(nums, sample_num, replace=False)
    return samples


def net_sample_nums(block_num, layer_sample, feature_num):
    """
    Generate network samples for each block of the model.

    Args:
        block_num: Number of blocks in the model
        layer_sample: Sample size for each layer
        feature_num: Number of features per layer

    Returns:
        Dictionary containing sampled indices for each network component
    """
    net_samples = {}

    for i in range(block_num):
        # Sampling for different parts of the network
        nums1 = rand_sample(layer_sample, feature_num)
        nums2_tmp = rand_sample(layer_sample, feature_num)
        nums2 = np.concatenate((nums2_tmp, nums2_tmp + feature_num, nums2_tmp + feature_num * 2))
        nums3 = nums2_tmp
        nums4 = rand_sample(layer_sample, feature_num)
        nums5 = rand_sample(layer_sample * 2, feature_num * 4)

        # Store sampled indices
        net_samples[f'block{i}_nums1'] = nums1
        net_samples[f'block{i}_nums2'] = nums2
        net_samples[f'block{i}_nums3'] = nums3
        net_samples[f'block{i}_nums4'] = nums4
        net_samples[f'block{i}_nums5'] = nums5

        # Add extra sample for the last block
        if i == block_num - 1:
            nums6 = rand_sample(layer_sample, feature_num)
            net_samples[f'block{i}_nums6'] = nums6

    return net_samples


def edge(x_s, y_s, param, x_sample, y_sample):
    """
    Create edges between sampled nodes with their weights.

    Args:
        x_s: x-coordinate start index
        y_s: y-coordinate start index
        param: Parameter matrix (transposed)
        x_sample: Sampled x indices
        y_sample: Sampled y indices

    Returns:
        List of edges with weights and edge count
    """
    edge_ls = []
    for i in x_sample:
        for j in y_sample:
            if param[i, j] != 0:
                temp = [x_s + i, y_s + j, param[i, j]]
                edge_ls.append(temp)
    return edge_ls, len(edge_ls)


def make_edge(model, model_name, step, index, net_samples, blocks_num, feature_num, sample_num):
    """
    Extract edges from model weights to create a network representation.

    Args:
        model: PyTorch model
        model_name: Name of the model
        step: Training step
        index: Sample index
        net_samples: Sampled network indices
        blocks_num: Number of blocks in the model
        feature_num: Feature dimension
        sample_num: Sample number

    Returns:
        None (saves the network to disk)
    """
    x_index_start, y_index_start = 0, feature_num
    param_edge = []

    for name, param in model.named_parameters():
        # Skip irrelevant parameters
        if name[9] != 'l':
            continue
        if 'bias' in name:
            continue
        if 'layernorm' in name:
            continue

        param_mux = param.to('cpu').detach().numpy()
        block_num = int(re.findall(r'\d+', name)[0])

        # Get corresponding samples
        nums1 = net_samples[f'block{block_num}_nums1']
        nums2 = net_samples[f'block{block_num}_nums2']
        nums3 = net_samples[f'block{block_num}_nums3']
        nums4 = net_samples[f'block{block_num}_nums4']
        nums5 = net_samples[f'block{block_num}_nums5']

        if block_num == blocks_num - 1:
            nums6 = net_samples[f'block{block_num}_nums6']
        else:
            nums6 = net_samples[f'block{block_num + 1}_nums1']

        print(f'Processing: {name}')

        x_sample = None
        y_sample = None

        # Process query-key-value weights
        if 'attention.query_key_value.weight' in name:
            x_sample = nums1
            y_sample = nums2

        # Process attention dense weights
        elif 'attention.dense.weight' in name:
            x_sample = nums3
            y_sample = nums4

        # Process MLP dense h to 4h weights
        elif 'mlp.dense_h_to_4h.weight' in name:
            x_sample = nums4
            y_sample = nums5

        # Process MLP dense 4h to h weights
        elif 'mlp.dense_4h_to_h.weight' in name:
            x_sample = nums5
            y_sample = nums6

        # Skip if no samples are assigned
        if x_sample is None or y_sample is None:
            continue

        # Get shape and indices
        shape0, shape1 = param_mux.shape[0], param_mux.shape[1]
        y_index_end = y_index_start + shape0
        x_index_end = x_index_start + shape1

        # Create edges
        edges, num = edge(x_index_start, y_index_start, param_mux.T, x_sample, y_sample)
        param_edge += edges
        y_index_start = y_index_end
        x_index_start = x_index_end

        # Special handling for attention.query_key_value.weight
        if 'attention.query_key_value.weight' in name:
            adj_1 = np.eye(shape1)
            adj_2 = np.ones((int(shape0 / 3 * 2), shape1))
            adj = np.vstack((adj_1, adj_2))

            y_index_end = y_index_end + int(shape0 / 3)
            x_index_end = x_index_end + shape0
            edges, num = edge(x_index_start, y_index_start, adj, nums2, nums3)
            param_edge += edges
            y_index_start = y_index_end
            x_index_start = x_index_end

    # Create save directory if it doesn't exist
    npy_dir = os.path.join('./data/models', f'sample_{sample_num}', model_name)
    os.makedirs(npy_dir, exist_ok=True)

    # Save the network as a numpy array
    save_path = os.path.join(npy_dir, f'{model_name}_step{step}_sampled_{index}.npy')
    np.save(save_path, np.array(param_edge))

    param_num = len(param_edge)
    print(f'{model_name}_step{step}_sampled_{index} made successfully')
    print(f'Edges count: {param_num}')


def extract_networks(models_to_process=None, indices_range=(0, 10), sample_num=5):
    """
    Main function to extract networks from multiple models and training steps.

    Args:
        models_to_process: List of model names to process (if None, use all)
        indices_range: Range of sample indices to generate
        sample_num: Number of samples per model

    Returns:
        None
    """
    from config import MODELS_NAME, TRAINING_STEPS, FEATURE_NUMS, BLOCK_NUMS, LAYER_SAMPLES

    print(f"Training steps to process: {TRAINING_STEPS}")

    if models_to_process is None:
        models_to_process = MODELS_NAME

    start_idx, end_idx = indices_range

    for i, model_name in enumerate(models_to_process):
        for index in range(start_idx, end_idx):
            np.random.seed(index)
            net_samples = net_sample_nums(BLOCK_NUMS[i], LAYER_SAMPLES[i], FEATURE_NUMS[i])

            for step in TRAINING_STEPS:
                # Handle special case for pythia-31m model
                if model_name == "pythia-31m" and step == 143000:
                    step = 140000

                print(f"Processing {model_name} at step {step}, sample {index}")

                # Load the model
                model = GPTNeoXForCausalLM.from_pretrained(
                    f"EleutherAI/{model_name}",
                    revision=f"step{step}",
                    cache_dir=f"./cache/{model_name}/step{step}",
                )

                # Extract network
                make_edge(
                    model, model_name, step, index, net_samples,
                    BLOCK_NUMS[i], FEATURE_NUMS[i], sample_num
                )


if __name__ == "__main__":
    # Example usage: extract networks for specific models
    models_to_process = [
        "pythia-70m-deduped",
        "pythia-160m-deduped",
        "pythia-410m-deduped",
        "pythia-1b-deduped",
        "pythia-1.4b-deduped",
        "pythia-2.8b-deduped"
    ]

    extract_networks(models_to_process, indices_range=(0, 10), sample_num=5)