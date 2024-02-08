import os
import json
from math import log
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
color_blue = color[0]
color_red = color[3]

def draw_graph_E(epochs_npy, alpha_0_npy, save_dir, model_name):
    global count
    if "14m" in model_name:
        return
    metric_path = "./pythia/evals/pythia-v1/" + model_name + "/zero-shot/"
    sub_name = model_name.split("pythia-")[1]
    metric_path += sub_name

    new_path = metric_path + f"_step0.json"
    with open(new_path, "r") as f:
        data = json.load(f)
        metric_names = []
        for key, value in data["results"].items():
            if "acc" in value:
                metric_names.append(key)

    for metric_name in metric_names:

        y_ori = []
        x_ori = [0] + [2 ** i for i in range(9)]
        x_ori += [1000, 103000, 113000, 123000, 13000, 133000, 143000, 23000, 3000, 33000, 43000, 53000, 63000, 73000,
                  83000, 93000]
        x_ori.sort()
        for i in x_ori:
            new_path = metric_path + f"_step{i}.json"
            with open(new_path, "r") as f:
                data = json.load(f)
                #             print(acc_data)
                cur = data["results"][metric_name]["acc"]
                y_ori.append(cur)

        x_ori = np.array(x_ori)
        y_ori = np.array(y_ori)

        x = epochs_npy
        y = alpha_0_npy


        xy_values = []
        for i in range(len(x)):
            xy_values.append((x[i], y[i]))

        title = f'{model_name}_{metric_name}'

        fig, ax1 = plt.subplots(figsize=(10, 4))

        ax1.scatter(x, y, color=color_blue)
        ax1.plot(x, y, color=color_blue, linewidth=3)
        ax1.set_xlabel('Epochs', fontsize=18)
        ax1.set_ylabel('Emergence', color=color_blue, fontsize=18)
        ax1.tick_params(axis='y', labelcolor=color_blue, labelsize=16)
        ax1.tick_params(axis='x', which='major', labelsize=16)

        y1_min = (min(y) - 0.025)
        y1_max = (max(y) + 0.025)
        ax1.set_ylim([y1_min, y1_max])

        ax2 = ax1.twinx()
        ax2.scatter(x_ori, y_ori, color=color_red)
        ax2.plot(x_ori, y_ori, color=color_red, linewidth=3)
        ax2.set_ylabel('Accuracy', color=color_red, fontsize=18)
        ax2.tick_params(axis='y', labelcolor=color_red, labelsize=16)
        ax2.tick_params(axis='x', which='major', labelsize=16)

        y2_min = (min(y_ori) - 0.025)
        y2_max = (max(y_ori) + 0.025)
        ax2.set_ylim([y2_min, y2_max])


        im_dir = os.path.join(save_dir, metric_name)
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        save_path = os.path.join(im_dir, f'{title}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'figure {title} saved successfully')

if __name__ == '__main__':

    models_name = ["pythia-70m-deduped", "pythia-160m-deduped", "pythia-410m-deduped", "pythia-1b-deduped",
                   "pythia-1.4b-deduped", "pythia-2.8b-deduped"]

    emergence_dir = Path('./emergence')

    save_dir = Path('./emergence/figures_metrics')

    fig_dir = Path('./emergence/figures_metrics')

    threshold = 150

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    E_scale_epoch = []

    epochs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
              63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 143000]

    for model_name in models_name:

        epochs_npy = np.array(epochs)
        alpha_0_npy = np.load(os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}',
                                           f'{model_name}_alpha_0_mean.npy'))
        width_npy = np.load(
            os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}', f'{model_name}_width_mean.npy'))

        alpha0_0 = alpha_0_npy[0]
        width_0 = width_npy[0]

        E = []

        for i in range(len(alpha_0_npy)):
            cur = width_npy[i] / width_npy[0] * log(alpha0_0 / alpha_0_npy[i])

            E.append(cur)

        E_npy = np.array(E)

        E_scale_epoch.append(E)

        model_save_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        #     Make sure draw_graph functions are updated to use the save_dir correctly
        draw_graph_E(epochs_npy, E_npy, save_dir, model_name)

