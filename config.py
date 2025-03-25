"""
Global configuration parameters for the NeuroMFA analysis.
"""

import os
from pathlib import Path

# Base directory paths
BASE_DIR = Path("./neuromfa")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Network extraction parameters
MODELS_NAME = [
    "pythia-14m", "pythia-31m", "pythia-70m-deduped", "pythia-160m-deduped",
    "pythia-410m-deduped", "pythia-1b-deduped", "pythia-1.4b-deduped",
    "pythia-2.8b-deduped", "pythia-6.9b-deduped", "pythia-12b-deduped"
]

# Training steps to analyze
TRAINING_STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000,
    43000, 53000, 63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 143000
]

# Model architecture parameters
FEATURE_NUMS = [128, 256, 512, 768, 1024, 2048, 2048, 2560, 4096, 5120]
BLOCK_NUMS = [6, 6, 6, 12, 24, 16, 24, 32, 32, 36]
LAYER_SAMPLES = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]

# Network sampling parameters
SAMPLE_NUM = 5
REPEAT_N = 10

# Multifractal analysis parameters
THRESHOLD = 150
Q_VALUES = [q / 100 for q in range(-300, 301, 10)]  # q values for multifractal analysis

# Visualization parameters
METRICS_NAME = ['wsc', 'sciq', 'logiqa', 'arc_easy', 'arc_challenge', 'lambada_openai', 'piqa', 'winogrande']


# Directory paths
def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        DATA_DIR,
        DATA_DIR / "models",
        DATA_DIR / f"sample_{SAMPLE_NUM}",
        DATA_DIR / "networks",
        DATA_DIR / f"networks/sample_{SAMPLE_NUM}",
        RESULTS_DIR,
        RESULTS_DIR / "figures",
        RESULTS_DIR / "emergence",
        RESULTS_DIR / "spectra",
        RESULTS_DIR / "nfd",
        RESULTS_DIR / "nfd/comparisons"
    ]

    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

    # Create model-specific directories
    for model_name in MODELS_NAME:
        model_dir = DATA_DIR / f"sample_{SAMPLE_NUM}" / model_name
        os.makedirs(model_dir, exist_ok=True)

        network_dir = DATA_DIR / f"networks/sample_{SAMPLE_NUM}" / model_name
        os.makedirs(network_dir, exist_ok=True)

        result_dir = RESULTS_DIR / "emergence" / model_name / f"threshold{THRESHOLD}"
        os.makedirs(result_dir, exist_ok=True)

        figure_dir = RESULTS_DIR / "figures" / model_name / f"threshold{THRESHOLD}"
        os.makedirs(figure_dir, exist_ok=True)
        
        # Create NFD directories
        nfd_dir = RESULTS_DIR / "nfd" / model_name
        os.makedirs(nfd_dir, exist_ok=True)
        os.makedirs(nfd_dir / "figures", exist_ok=True)


if __name__ == "__main__":
    create_directories()
