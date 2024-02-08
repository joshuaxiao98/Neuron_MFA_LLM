import numpy as np
from pathlib import Path
import os
import pickle
import re


# transform the sampled network into the input data of NMFA analysis
net_dir = Path('./models')
pkl_dir = Path('./networks')
entries = os.listdir(net_dir)

models_name = [entry for entry in entries if os.path.isdir(os.path.join(net_dir, entry))]

for model_name in models_name:

    model_dir = os.path.join(net_dir, model_name)

    files = os.listdir(model_dir)

    models_npy = [file for file in files if file.endswith('.npy')]

    for model_npy in models_npy:

        step_num = int(re.search(r"step(\d+)_sampled", model_npy).group(1))

        sample_num = int(re.search(r"sampled_(\d+)\.npy", model_npy).group(1))

        npy_path = os.path.join(model_dir, model_npy)

        net = np.load(npy_path)

        collections = {}

        for row in net:
            key = row[0]
            value = row[2]
            if key not in collections:
                collections[key] = []
            collections[key].append(1/np.abs(value))

        sorted_keys = sorted(collections.keys())

        nets = [collections[key] for key in sorted_keys]

        model_npy = Path(model_npy)
        model_pkl = model_npy.with_suffix('.pkl')
        if not os.path.exists(os.path.join(pkl_dir, model_name)):
            os.makedirs(os.path.join(pkl_dir, model_name))

        pkl_path = os.path.join(pkl_dir, model_name, model_pkl)

        with open(pkl_path, 'wb') as file:
            pickle.dump(nets, file)

        print(f"{model_pkl} successfully saved")