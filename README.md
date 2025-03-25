# NeuroMFA: Neuron-based Multifractal Analysis for Large Language Models

This repository contains code for analyzing the structural properties of large language models using Neuron-based Multifractal Analysis (NeuroMFA), as described in the paper "Neuron-based Multifractal Analysis of Neuron Interaction Dynamics in Large Models".

## Overview

NeuroMFA is a framework for quantitatively analyzing the structural dynamics of large language models (LLMs) during training. It provides a new perspective for investigating "emergent abilities" in large models by examining the structural self-organization of neuron interactions.

The framework consists of four main components:
1. **Network Extraction**: Converting LLM weights into neuron interaction networks
2. **Multifractal Analysis**: Analyzing the fractal properties of neuron interactions
3. **Emergence Metrics**: Quantifying the degree of self-organization and emergence
4. **Node Fractal Dimension (NFD)**: Calculating individual neuron complexity metrics

## Project Structure

```
neuromfa/
├── config.py              # Configuration parameters
├── data_processing/
│   ├── network_extraction.py   # Extract networks from LLMs
│   └── data_preprocessing.py   # Process raw network data
├── analysis/
│   ├── multifractal.py    # Multifractal analysis algorithms
│   ├── emergence.py       # Calculate emergence metrics
│   ├── node_fractal_dimension.py  # Node-level fractal dimension analysis
│   └── utils.py           # Utility functions for analysis
├── visualization/
│   ├── spectrum_plots.py  # Plot multifractal spectra
│   └── emergence_plots.py # Plot emergence metrics
├── utils/
│   └── io_utils.py        # File I/O utilities
└── main.py                # Main execution scripts
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neuromfa.git
cd neuromfa

# Install dependencies
pip install -r requirements.txt
```

## Usage

The main script (`main.py`) provides a command-line interface to run the entire pipeline or specific stages:

```bash
# Run all stages
python main.py --stage all

# Run specific stage
python main.py --stage extract --models pythia-70m-deduped pythia-1.4b-deduped

# Run with custom parameters
python main.py --threshold 200 --filter_mode 1 --metrics lambada_openai piqa
```

### Command Line Arguments

- `--stage`: Processing stage to run (`extract`, `preprocess`, `analyze`, `visualize`, or `all`)
- `--models`: Specific models to process (default: all available models)
- `--threshold`: Threshold for multifractal analysis (default: 150)
- `--sample_num`: Sample number for network extraction (default: 5)
- `--metrics`: Metrics to visualize (default: lambada_openai, piqa, arc_easy, arc_challenge)
- `--filter_mode`: Filter mode for multifractal analysis (default: 0)
- `--skip_existing`: Skip processing of existing files

## Results

The analysis results will be saved in the `results/` directory:
- `results/emergence/`: Emergence metric data
- `results/figures/`: Multifractal spectrum plots
- `results/emergence/figures_metrics/`: Emergence vs. metrics plots
- `results/emergence/figures_radar/`: Radar charts
- `results/emergence/figures_size/`: Emergence vs. model size plots
- `results/nfd/`: Node Fractal Dimension analysis results

## Node Fractal Dimension Analysis

The Node Fractal Dimension (NFD) functionality allows for calculating fractal dimensions at the individual neuron level. This provides a fine-grained view of how the local complexity around each neuron evolves during training.

To use this functionality:

```python
# Calculate NFD for a specific network
from analysis.node_fractal_dimension import calculate_and_save_nfd

model_name = "pythia-70m-deduped"
step = 1000  # Training step
sample_idx = 0  # Sample index

# Calculate and save NFD values
calculate_and_save_nfd(model_name, step, sample_idx)

# Process all networks for a model across all training steps
from analysis.node_fractal_dimension import process_all_networks
process_all_networks(model_name)

# Analyze NFD evolution throughout training
from analysis.node_fractal_dimension import analyze_nfd_evolution
evolution_df = analyze_nfd_evolution(model_name)

# Compare NFD across models
from analysis.node_fractal_dimension import compare_nfd_across_models
models = ["pythia-70m-deduped", "pythia-160m-deduped", "pythia-410m-deduped"]
comparison_df = compare_nfd_across_models(models, step=1000)

# Visualize NFD distribution
from analysis.node_fractal_dimension import visualize_nfd_distribution
visualize_nfd_distribution(model_name, step=1000)
```

The NFD results are saved in the `results/nfd/` directory with the following structure:
- `results/nfd/{model_name}/`: NFD values for each node in the network
- `results/nfd/{model_name}/figures/`: Visualizations of NFD distributions
- `results/nfd/comparisons/`: Comparisons across different models

## Citation

If you use this code in your research, please cite:

```
@inproceedings{
  xiao2025neuromfa,
  title={Neuron-based Multifractal Analysis of Neuron Interaction Dynamics in Large Models},
  author={Xiongye Xiao and Heng Ping and Chenyu Zhou and Defu Cao and Yaxing Li and Yi-Zhuo Zhou and Shixuan Li and Nikos Kanakaris and Paul Bogdan},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
