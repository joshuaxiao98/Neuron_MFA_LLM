"""
Multifractal analysis algorithms for neuron interaction networks.
"""

import numpy as np
import scipy.stats as stats
from collections import Counter
from tqdm import tqdm
import os
import pickle
from pathlib import Path
import re


def wnfd_llm(mat, Q, draw=False, fdigi=0, threshold=150):
    """
    Weighted Network Fractal Dimension calculation for LLM networks.

    Args:
        mat: Weighted network matrix
        Q: List of q values for multifractal analysis
        draw: Whether to draw the plots (default: False)
        fdigi: Number of digits for rounding (default: 0)
        threshold: Maximum weight threshold (default: 150)

    Returns:
        tau_list: List of tau values (mass exponents)
        r_value: R-value of the linear regression
        p_value: P-value of the linear regression
        stderr: Standard error of the linear regression
    """
    N_list = []  # Store the selected nodes
    r_g_all_set = set()  # Find radius

    # Collect weights and update radius set
    for i in range(len(mat)):
        # Store the non-zero weights within threshold
        grow = [m for m in mat[i] if 1 < m <= threshold]
        if len(grow) > 10:
            num = Counter(np.round(grow, fdigi))
            r_g_all_set.update(num.keys())
            N_list.append(num)

    if not N_list:
        print("Error: No node selected")
        return [], 0, 0, 0

    # Sort radius values
    r_g_all = np.array(sorted(list(r_g_all_set)))

    # Initialize Num_r matrix: column: node, row: radius
    Nw_mat = np.ones((len(N_list), len(r_g_all)))

    # Fill the matrix
    for i, num in enumerate(N_list):
        for j, r in enumerate(r_g_all):
            Nw_mat[i, j] += sum(count for radius, count in num.items() if radius <= r)

    # Get maximum diameter
    diameter = r_g_all[-1]

    # Calculate the partition function for each q
    Zq_list = []
    for q in Q:
        Zq_mat = np.power(Nw_mat / Nw_mat[:, -1, None], q)
        Zq_list.append(np.sum(Zq_mat, axis=0))

    # Calculate tau (slope)
    tau_list = []

    if draw:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        for idx, (q, Zq) in enumerate(zip(Q, Zq_list)):
            x = np.log(r_g_all / diameter)
            y = np.log(Zq)
            slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
            tau_list.append(slope)
            plt.plot(x, y, '*', label=f'q={q:.0f}')
            plt.xlabel('ln(r/d)')
            plt.ylabel('ln(Partition function)')
        plt.show()
        return tau_list, r_value, p_value, stderr
    else:
        for Zq in Zq_list:
            x = np.log(r_g_all / diameter)
            y = np.log(Zq)
            slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
            tau_list.append(slope)

        return tau_list, r_value, p_value, stderr


def calculate_multifractal_spectrum(tau_list, q_list):
    """
    Calculate multifractal spectrum from tau values.

    Args:
        tau_list: List of tau values (mass exponents)
        q_list: List of q values

    Returns:
        alpha_0: Alpha value at maximum f(alpha)
        width: Width of the multifractal spectrum
        alpha_list: List of alpha values
        f_alpha_list: List of f(alpha) values
    """
    alpha_list = []
    f_alpha_list = []

    # Calculate alpha values
    for i in range(1, len(q_list)):
        alpha = (tau_list[i] - tau_list[i - 1]) / (q_list[i] - q_list[i - 1])
        alpha_list.append(alpha)

    # Calculate f(alpha) values
    for j in range(len(q_list) - 1):
        f_alpha = q_list[j] * alpha_list[j] - tau_list[j]
        f_alpha_list.append(f_alpha)

    # Find alpha_0 (alpha at maximum f(alpha))
    alpha_0 = alpha_list[np.argmax(f_alpha_list)]

    # Calculate spectrum width
    width = np.max(alpha_list) - np.min(alpha_list)

    return alpha_0, width, alpha_list, f_alpha_list


def process_network(model_name, repeat_n=10, threshold=150, q_values=None, filter_mode=0):
    """
    Process a network and perform multifractal analysis.

    Args:
        model_name: Name of the model to process
        repeat_n: Number of repetitions for averaging
        threshold: Maximum weight threshold
        q_values: List of q values for multifractal analysis
        filter_mode: Filtering mode (0: no filtering, 1: f_alpha<=0,
                    2: (alpha[0]-max(alpha))<0.01, 3: both)

    Returns:
        Dictionary containing processed data:
        - ntauls_list: List of tau values for each matrix
        - alpha_0_mean_list: List of average alpha_0 values for each epoch
        - width_mean_list: List of average width values for each epoch
        - epochs: List of epochs
    """
    # Set default q_values if not provided
    if q_values is None:
        q_values = [q / 100 for q in range(-300, 301, 10)]

    # Import here to avoid circular imports
    from config import TRAINING_STEPS

    # Setup paths
    pkl_dir = Path(f'./data/networks/sample_{repeat_n}')
    emergence_dir = Path(f'./results/emergence')

    # Create output directories
    os.makedirs(emergence_dir / 'npy_stored' / model_name / f'threshold{threshold}', exist_ok=True)

    # Paths for cached computations
    ntauls_list_stored_path = emergence_dir / 'npy_stored' / model_name / f'threshold{threshold}' / 'ntauls_list.npy'

    # Check if we have precomputed values
    if os.path.exists(ntauls_list_stored_path):
        print(f"Loading precomputed tau values for {model_name}")
        ntauls_list = np.load(ntauls_list_stored_path)
    else:
        print(f"Computing tau values for {model_name}")

        # Load network data
        model_dir = pkl_dir / model_name
        files = os.listdir(model_dir)

        # Define epochs based on model
        if model_name == 'pythia-31m':
            epochs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
                      63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 140000]
        else:
            epochs = TRAINING_STEPS

        # Store matrices for each sample and epoch
        mat_lists = []
        steps = len(epochs)

        # Load matrices for each sample
        for sample_num in range(repeat_n):
            models_pkl = [file for file in files if file.endswith(f'{sample_num}.pkl')]
            sorted_models_pkl = sorted(models_pkl, key=lambda x: int(re.search('step(\d+)_', x).group(1)))

            mat_list = []

            for model_pkl in sorted_models_pkl:
                step_num = int(re.search(r'step(\d+)_sampled', model_pkl).group(1))
                if step_num not in epochs:
                    continue

                pkl_path = os.path.join(model_dir, model_pkl)

                with open(pkl_path, 'rb') as file:
                    net = pickle.load(file)

                mat_list.append(net)

            mat_lists.append(mat_list)

        # Combine matrices across all samples for each epoch
        mat_average_list = []
        for i in range(steps):
            for j in range(repeat_n):
                mat_average_list.append(mat_lists[j][i])

        # Calculate tau values for each matrix
        Q = q_values
        ntauls_list = []
        r_value_list, p_value_list, stderr_list = [], [], []

        for i in tqdm(range(len(mat_average_list)), desc=f"Processing {model_name}"):
            ntau, r_value, p_value, stderr = wnfd_llm(mat_average_list[i], Q, draw=False, threshold=threshold)
            ntauls_list.append(ntau)
            r_value_list.append(r_value)
            p_value_list.append(p_value)
            stderr_list.append(stderr)

        # Save the computed values
        np.save(ntauls_list_stored_path, np.array(ntauls_list))
        np.save(emergence_dir / 'npy_stored' / model_name / f'threshold{threshold}' / 'r_value_list.npy',
                np.array(r_value_list))
        np.save(emergence_dir / 'npy_stored' / model_name / f'threshold{threshold}' / 'p_value_list.npy',
                np.array(p_value_list))
        np.save(emergence_dir / 'npy_stored' / model_name / f'threshold{threshold}' / 'stderr_list.npy',
                np.array(stderr_list))

    # Process tau values to get multifractal spectra
    alpha_0_list = []
    width_list = []
    all_al_list = []
    all_fal_list = []
    count_num = 0
    repeat_n_list = []

    # First pass to calculate all spectra
    for i in range(len(ntauls_list)):
        alpha_0, width, al_list, fal_list = calculate_multifractal_spectrum(ntauls_list[i], q_values)
        all_al_list.append(al_list)
        all_fal_list.append(fal_list)
        alpha_0_list.append(alpha_0)
        width_list.append(width)

    # Second pass for filtering if needed
    filtered_alpha_0_list = []
    filtered_width_list = []

    for i in range(len(ntauls_list) + 1):
        if i % repeat_n == 0 and i > 0:
            repeat_n_list.append(count_num)
            count_num = 0
            if i == len(ntauls_list):
                break

        if i < len(ntauls_list):  # Make sure we don't go out of bounds
            alpha_0, width, al_list, fal_list = calculate_multifractal_spectrum(ntauls_list[i], q_values)

            # Apply filters if specified
            if filter_mode == 1:
                if any(n > 0 for n in fal_list):
                    continue
            elif filter_mode == 2:
                if (max(al_list) - al_list[0]) > 0.01:
                    continue
            elif filter_mode == 3:
                if any(n > 0 for n in fal_list) or (max(al_list) - al_list[0]) > 0.01:
                    continue

            filtered_alpha_0_list.append(alpha_0)
            filtered_width_list.append(width)
            count_num = count_num + 1

    # Calculate average metrics for each epoch
    alpha_0_mean_list = []
    width_mean_list = []
    start_index, end_index = 0, 0

    if model_name == 'pythia-31m':
        epochs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
                  63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 140000]
    else:
        epochs = TRAINING_STEPS

    for i in range(len(epochs)):
        if i < len(repeat_n_list):  # Check if we have enough data
            end_index = end_index + repeat_n_list[i]

            if start_index < end_index and start_index < len(filtered_alpha_0_list):
                end_idx = min(end_index, len(filtered_alpha_0_list))
                alpha_0_mean_list.append(np.mean(filtered_alpha_0_list[start_index:end_idx]))
                width_mean_list.append(np.mean(filtered_width_list[start_index:end_idx]))
            else:
                # Handle case where we don't have enough data
                alpha_0_mean_list.append(np.nan)
                width_mean_list.append(np.nan)

            start_index = start_index + repeat_n_list[i]

    # Save computed metrics
    alpha_0_mean_npy = np.array(alpha_0_mean_list)
    width_mean_npy = np.array(width_mean_list)

    np.save(emergence_dir / 'npy_stored' / model_name / f'threshold{threshold}' / f'{model_name}_alpha_0_mean.npy',
            alpha_0_mean_npy)
    np.save(emergence_dir / 'npy_stored' / model_name / f'threshold{threshold}' / f'{model_name}_width_mean.npy',
            width_mean_npy)

    return {
        "ntauls_list": ntauls_list,
        "alpha_0_mean_list": alpha_0_mean_list,
        "width_mean_list": width_mean_list,
        "epochs": epochs
    }


if __name__ == "__main__":
    # Example usage
    model_name = "pythia-1b-deduped"
    results = process_network(model_name, repeat_n=10, threshold=150)
    print(f"Processed {model_name}:")
    print(f"  Alpha_0 means: {results['alpha_0_mean_list']}")
    print(f"  Width means: {results['width_mean_list']}")