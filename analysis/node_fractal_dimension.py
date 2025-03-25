"""
Node Fractal Dimension (NFD) analysis for neuron interaction networks.

This module provides functions to calculate the Node Fractal Dimension for each node
in a network derived from LLM weights, offering insights into the local structural 
complexity around individual neurons.
"""

import os
import re
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from collections import Counter
from tqdm import tqdm
import pickle
from pathlib import Path


def create_network_from_pkl(pkl_file):
    """
    Create a NetworkX graph from a preprocessed PKL file.
    
    Args:
        pkl_file: Path to the pickle file containing network data
        
    Returns:
        NetworkX Graph object
    """
    print("Loading network data...")
    with open(pkl_file, 'rb') as f:
        net_data = pickle.load(f)
    
    print("Creating network...")
    G = nx.Graph()
    
    # Add nodes
    node_ids = set()
    
    for idx, weights in enumerate(net_data):
        node_ids.add(idx)
        
    G.add_nodes_from(node_ids)
    
    # Add edges with weights
    for i, weights_list in enumerate(net_data):
        if not weights_list:
            continue
            
        for j, weight in enumerate(weights_list):
            # Use inverse of weight for better NFD calculation
            # We already have inverse weights stored from preprocess_networks function
            G.add_edge(i, j, weight=weight)
    
    return G


def node_dimension(G, weight=True, fdigi=6):
    """
    Calculate the Node Fractal Dimension (NFD) for each node in the network.
    
    Args:
        G: NetworkX graph
        weight: Whether to use edge weights (default: True)
        fdigi: Number of digits for rounding (default: 6)
        
    Returns:
        Dictionary mapping node IDs to their NFD values
    """
    node_dimension = {}
    
    for node in tqdm(G.nodes(), desc="Calculating node dimensions"):
        # Calculate shortest paths from this node to all others
        distances = nx.single_source_dijkstra_path_length(G, node, weight='weight' if weight else None)
        
        # Process distances
        grow = sorted(list(distances.values()))
        grow = [round(d, fdigi) for d in grow]
        grow = grow[1:]  # Remove self-distance (0)
        
        r_g = []
        num_g = []
        num_nodes = 0
        
        # Count nodes at each distance
        for i, count in Counter(grow).items():
            num_nodes += count
            if i > 0:  # Avoid log(0)
                r_g.append(i)
                num_g.append(num_nodes)
        
        # Calculate NFD (slope of log-log plot)
        if len(r_g) > 1:
            x = np.log(r_g)
            y = np.log(num_g)
            slope, _, _, _, _ = stats.linregress(x, y)
            node_dimension[node] = slope
        else:
            node_dimension[node] = 0
            
    return node_dimension


def calculate_and_save_nfd(model_name, step, sample_idx=0, sample_num=5, output_dir=None):
    """
    Calculate and save NFD values for a specific model network.
    
    Args:
        model_name: Name of the model
        step: Training step
        sample_idx: Sample index (default: 0)
        sample_num: Sample number (default: 5)
        output_dir: Output directory (default: None, will use './results/nfd/{model_name}')
        
    Returns:
        DataFrame containing node IDs and their NFD values
    """
    # Setup paths
    network_dir = Path(f'./data/networks/sample_{sample_num}')
    pkl_file = network_dir / model_name / f"{model_name}_step{step}_sampled_{sample_idx}.pkl"
    
    if output_dir is None:
        output_dir = Path(f'./results/nfd/{model_name}')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"nfd_{model_name}_step{step}_sample{sample_idx}.csv"
    
    print(f"Processing network: {model_name}, step {step}, sample {sample_idx}")
    
    # Create network
    G = create_network_from_pkl(pkl_file)
    
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate NFD values
    print("Calculating NFD values...")
    nfd_dict = node_dimension(G)
    
    # Create DataFrame
    nfd_df = pd.DataFrame({
        'node_id': list(nfd_dict.keys()),
        'nfd_value': list(nfd_dict.values())
    })
    
    # Save results
    nfd_df.to_csv(output_file, index=False)
    print(f"NFD metrics have been saved to {output_file}")
    
    return nfd_df


def process_all_networks(model_name, sample_num=5, output_dir=None):
    """
    Process all networks for a specific model across all training steps.
    
    Args:
        model_name: Name of the model
        sample_num: Sample number (default: 5)
        output_dir: Output directory (default: None)
        
    Returns:
        None
    """
    # Setup paths
    network_dir = Path(f'./data/networks/sample_{sample_num}')
    model_dir = network_dir / model_name
    
    if not model_dir.exists():
        print(f"Error: Directory for model {model_name} not found at {model_dir}")
        return
    
    # Get all network files
    pkl_files = list(model_dir.glob(f"{model_name}_step*_sampled_*.pkl"))
    print(f"Found {len(pkl_files)} networks to process for {model_name}")
    
    # Get unique steps and sample indices
    steps = set()
    for file in pkl_files:
        step_match = re.search(r"step(\d+)_sampled", file.name)
        if step_match:
            steps.add(int(step_match.group(1)))
    
    steps = sorted(list(steps))
    
    # Process each step
    if output_dir is None:
        output_dir = Path(f'./results/nfd/{model_name}')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for step in steps:
        for sample_idx in range(sample_num):
            pkl_file = model_dir / f"{model_name}_step{step}_sampled_{sample_idx}.pkl"
            if pkl_file.exists():
                try:
                    calculate_and_save_nfd(model_name, step, sample_idx, sample_num, output_dir)
                except Exception as e:
                    print(f"Error processing {pkl_file}: {str(e)}")


def analyze_nfd_evolution(model_name, sample_idx=0, sample_num=5):
    """
    Analyze how NFD values evolve throughout training for a specific model.
    
    Args:
        model_name: Name of the model
        sample_idx: Sample index to analyze (default: 0)
        sample_num: Sample number (default: 5)
        
    Returns:
        DataFrame containing summary statistics for each step
    """
    # Setup paths
    nfd_dir = Path(f'./results/nfd/{model_name}')
    
    if not nfd_dir.exists():
        print(f"Error: NFD directory for model {model_name} not found")
        return None
    
    # Get all NFD files for the specified sample
    nfd_files = list(nfd_dir.glob(f"nfd_{model_name}_step*_sample{sample_idx}.csv"))
    
    if not nfd_files:
        print(f"No NFD files found for {model_name} with sample index {sample_idx}")
        return None
    
    # Extract steps and sort
    steps = []
    for file in nfd_files:
        step_match = re.search(f"nfd_{model_name}_step(\d+)_sample{sample_idx}", file.name)
        if step_match:
            steps.append(int(step_match.group(1)))
    
    step_file_pairs = sorted(zip(steps, nfd_files))
    
    # Calculate statistics for each step
    results = []
    
    for step, file in step_file_pairs:
        df = pd.read_csv(file)
        nfd_values = df['nfd_value']
        
        stats_dict = {
            'step': step,
            'mean': nfd_values.mean(),
            'median': nfd_values.median(),
            'std': nfd_values.std(),
            'min': nfd_values.min(),
            'max': nfd_values.max(),
            'range': nfd_values.max() - nfd_values.min(),
            'count': len(nfd_values)
        }
        
        results.append(stats_dict)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Save summary
    summary_file = nfd_dir / f"nfd_evolution_{model_name}_sample{sample_idx}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"NFD evolution summary saved to {summary_file}")
    
    return summary_df


def compare_nfd_across_models(models, step, sample_idx=0, sample_num=5):
    """
    Compare NFD values across different models at a specific training step.
    
    Args:
        models: List of model names to compare
        step: Training step to compare
        sample_idx: Sample index (default: 0)
        sample_num: Sample number (default: 5)
        
    Returns:
        DataFrame containing NFD statistics for each model
    """
    results = []
    
    for model_name in models:
        nfd_dir = Path(f'./results/nfd/{model_name}')
        nfd_file = nfd_dir / f"nfd_{model_name}_step{step}_sample{sample_idx}.csv"
        
        if not nfd_file.exists():
            print(f"Warning: NFD file not found for {model_name} at step {step}")
            continue
        
        df = pd.read_csv(nfd_file)
        nfd_values = df['nfd_value']
        
        stats_dict = {
            'model': model_name,
            'mean': nfd_values.mean(),
            'median': nfd_values.median(),
            'std': nfd_values.std(),
            'min': nfd_values.min(),
            'max': nfd_values.max(),
            'range': nfd_values.max() - nfd_values.min(),
            'count': len(nfd_values)
        }
        
        results.append(stats_dict)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Sort by model size
    from analysis.utils import get_model_size
    model_sizes = {model: get_model_size(model) for model in models}
    comparison_df['model_size'] = comparison_df['model'].map(model_sizes)
    comparison_df = comparison_df.sort_values('model_size')
    
    # Save comparison
    output_dir = Path('./results/nfd/comparisons')
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_file = output_dir / f"nfd_comparison_step{step}_sample{sample_idx}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"NFD comparison saved to {comparison_file}")
    
    return comparison_df


def visualize_nfd_distribution(model_name, step, sample_idx=0, sample_num=5, save_fig=True):
    """
    Visualize the distribution of NFD values for a specific network.
    
    Args:
        model_name: Name of the model
        step: Training step
        sample_idx: Sample index (default: 0)
        sample_num: Sample number (default: 5)
        save_fig: Whether to save the figure (default: True)
        
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Setup paths
    nfd_dir = Path(f'./results/nfd/{model_name}')
    nfd_file = nfd_dir / f"nfd_{model_name}_step{step}_sample{sample_idx}.csv"
    
    if not nfd_file.exists():
        print(f"Error: NFD file not found at {nfd_file}")
        return None
    
    # Load data
    df = pd.read_csv(nfd_file)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot distribution
    sns.histplot(df['nfd_value'], kde=True)
    
    # Add statistics to plot
    stats_text = (
        f"Mean: {df['nfd_value'].mean():.3f}\n"
        f"Median: {df['nfd_value'].median():.3f}\n"
        f"Std Dev: {df['nfd_value'].std():.3f}\n"
        f"Min: {df['nfd_value'].min():.3f}\n"
        f"Max: {df['nfd_value'].max():.3f}"
    )
    
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    plt.xlabel('Node Fractal Dimension (NFD)')
    plt.ylabel('Count')
    plt.title(f'NFD Distribution for {model_name} at Step {step}')
    
    # Save figure if requested
    if save_fig:
        fig_dir = nfd_dir / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"nfd_dist_{model_name}_step{step}_sample{sample_idx}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
    
    return plt.gcf()


if __name__ == "__main__":
    
    # Example usage
    model_name = "pythia-70m-deduped"
    step = 1000
    sample_idx = 0
    
    # Calculate NFD for a specific network
    calculate_and_save_nfd(model_name, step, sample_idx)
    
    # Visualize the NFD distribution
    visualize_nfd_distribution(model_name, step, sample_idx)
    
    # Process all networks for a model
    # process_all_networks(model_name)
    
    # Analyze NFD evolution throughout training
    # analyze_nfd_evolution(model_name)
    
    # Compare NFD across models
    # models = ["pythia-70m-deduped", "pythia-160m-deduped", "pythia-410m-deduped"]
    # compare_nfd_across_models(models, step)
