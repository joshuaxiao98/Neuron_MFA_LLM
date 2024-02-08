import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as stats
from tqdm import tqdm
from pathlib import Path
import pickle
import os
import re


#4 NFD for distance list
def wnfd_llm(mat,Q,draw=False,fdigi=0, threshold=150): # Choose an appopriate fidgi

    N_list = [] # Store the selected nodes
    r_g_all_set = set() # Find radius

    for i in range(len(mat)):
        grow = [m for m in mat[i] if 1 < m <= threshold] # Store the non-zero weights
        if len(grow) > 10:
            num = Counter(np.round(grow, fdigi))
            r_g_all_set.update(num.keys())
            N_list.append(num)

    if not N_list:
        print("Error: No node selected")

    r_g_all = np.array(sorted(list(r_g_all_set)))
    Nw_mat = np.ones((len(N_list), len(r_g_all))) # Num_r matrix: column: node, row: radius

    for i, num in enumerate(N_list):
        for j, r in enumerate(r_g_all):
            Nw_mat[i, j] += sum(count for radius, count in num.items() if radius <= r)

    diameter = r_g_all[-1]

    Zq_list = []

    for q in Q:
        Zq_mat = np.power(Nw_mat / Nw_mat[:, -1, None], q)
        Zq_list.append(np.sum(Zq_mat, axis=0))

    tau_list = [] # Get tau(slope)

    if draw:
        plt.figure(figsize=(7, 7))
        for idx, (q, Zq) in enumerate(zip(Q, Zq_list)):
            x = np.log(r_g_all / diameter)
            y = np.log(Zq)
            slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
            tau_list.append(slope)
            plt.plot(x, y, '*', label=f'q={q:.0f}')
            plt.xlabel('ln(r/d)')
            plt.ylabel('ln(Partition function)')
        # plt.legend(fontsize=10)
        plt.show()
    else:
        for Zq in Zq_list:
            x = np.log(r_g_all / diameter)
            y = np.log(Zq)
            slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
            tau_list.append(slope)

    return tau_list, r_value, p_value, stderr



def nspectrum(tau_list,q_list,k):
    al_list = []
    fal_list = []
    for i in range(1,len(q_list)):
        al=(tau_list[i]-tau_list[i-1])/(q_list[i]-q_list[i-1])
        al_list.append(al)
    for j in range(len(q_list)-1):
        fal=q_list[j]*al_list[j]-tau_list[j]
        fal_list.append(fal)
    alpha_0= al_list[np.argmax(fal_list)]
    width = np.max(al_list) - np.min(al_list)
    return alpha_0, width, al_list, fal_list


if __name__ == '__main__':

    repeat_n = 10
    steps = 27
    threshold = 200
    filter_mode = 0
    # sns.color_palette("coolwarm", as_cmap=True)

    pkl_dir = Path('./networks')

    emergence_dir = Path('./emergence')

    entries = os.listdir(pkl_dir)

    models_name = [entry for entry in entries if os.path.isdir(os.path.join(pkl_dir, entry))]

    for model_name in models_name:
        print(f'{model_name}')

        if model_name == 'pythia-31m':
            epochs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
             63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 140000]
        else:
            epochs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 13000, 23000, 33000, 43000, 53000,
             63000, 73000, 83000, 93000, 103000, 113000, 123000, 133000, 143000]

        model_dir = os.path.join(pkl_dir, model_name)
        files = os.listdir(model_dir)

        ntauls_list_stored = False
        mat_average_list_stored = False

        ntauls_list_stored_path = os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}',
                                               f'ntauls_list.npy')
        mat_average_list_stored_path = os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}',
                                                    f'mat_average_list.pkl')

        if os.path.exists(mat_average_list_stored_path):
            mat_average_list_stored = True

        if os.path.exists(ntauls_list_stored_path):
            ntauls_list_stored = True

        if ntauls_list_stored == True:
            ntauls_list = np.load(ntauls_list_stored_path)
            Q = [q / 100 for q in range(-300, 301, 10)]

        else:

            if mat_average_list_stored == True:
                with open(mat_average_list_stored_path, 'rb') as file:
                    mat_average_list = pickle.load(file)

            else:
                mat_lists = []
                for sample_num in tqdm(range(repeat_n)):
                    models_pkl = [file for file in files if file.endswith(f'{sample_num}.pkl')]
                    sorted_models_pkl = sorted(models_pkl, key=lambda x: int(re.search('step(\d+)_', x).group(1)))
                    # print(sorted_models_pkl)
                    mat_list = []
                    for model_pkl in sorted_models_pkl:
                        step_num = int(re.search(r'step(\d+)_sampled', model_pkl).group(1))
                        # print(step_num)
                        if step_num not in epochs:
                            print(f'{step_num} continue')
                            continue

                        pkl_path = os.path.join(model_dir, model_pkl)
                        with open(pkl_path, 'rb') as file:
                            net = pickle.load(file)
                        mat_list.append(net)

                    mat_lists.append(mat_list)

                mat_average_list = []

                for i in range(steps):
                    for j in range(repeat_n):
                        mat_average_list.append(mat_lists[j][i])

            Q = [q / 100 for q in range(-300, 301, 10)]
            G_list = mat_average_list
            ntauls_list = []
            r_value_list, p_value_list, stderr_list = [], [], []

            for i in tqdm(range(len(G_list))):
                ntau, r_value, p_value, stderr = wnfd_llm(G_list[i], Q, draw=False, threshold=threshold)
                ntauls_list.append(ntau)
                r_value_list.append(r_value)
                p_value_list.append(p_value)
                stderr_list.append(stderr)

            if not os.path.exists(os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}')):
                os.makedirs(os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}'))

            np.save(os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}', f'ntauls_list.npy'),
                    np.array(ntauls_list))
            np.save(os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}', f'r_value_list.npy'),
                    np.array(r_value_list))
            np.save(os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}', f'p_value_list.npy'),
                    np.array(p_value_list))
            np.save(os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}', f'stderr_list.npy'),
                    np.array(stderr_list))


        alpha_0_list = []
        width_list = []
        all_al_list = []
        all_fal_list = []
        count_num = 0
        repeat_n_list = []

        for i in range(len(ntauls_list)):
            alpha_0, width, al_list, fal_list = nspectrum(ntauls_list[i], Q, i)
            all_al_list.append(al_list)
            all_fal_list.append(fal_list)
            alpha_0_list.append(alpha_0)
            width_list.append(width)

        for i in range(len(ntauls_list) + 1):
            if i % repeat_n == 0 and i > 0:
                repeat_n_list.append(count_num)
                count_num = 0
                if i == len(ntauls_list):
                    break

            alpha_0, width, al_list, fal_list = nspectrum(ntauls_list[i], Q, i)

            if filter_mode == 1:
                if any(n > 0 for n in fal_list):
                    continue
            elif filter_mode == 2:
                # print(f'alpha {a}')
                if (max(al_list) - al_list[0]) > 0.01:
                    print(f'alpha {al_list}')
                    continue
            elif filter_mode == 3:
                if any(n > 0 for n in fal_list) or (max(al_list) - al_list[0]) > 0.01:
                    continue

            alpha_0_list.append(alpha_0)
            width_list.append(width)

            count_num = count_num + 1

        alpha_0_mean_list = []
        width_mean_list = []
        start_index, end_index = 0, 0

        for i in range(len(epochs)):
            end_index = end_index + repeat_n_list[i]

            alpha_0_mean_list.append(np.mean(alpha_0_list[start_index: end_index], axis=0))
            width_mean_list.append(np.mean(width_list[start_index: end_index], axis=0))

            start_index = start_index + repeat_n_list[i]

        alpha_0_mean_npy = np.array(alpha_0_mean_list)
        width_mean_npy = np.array(width_mean_list)

        np.save(os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}',
                             f'{model_name}_alpha_0_mean.npy'), alpha_0_mean_npy)
        np.save(os.path.join(emergence_dir, 'npy_stored', model_name, f'threshold{threshold}',
                             f'{model_name}_width_mean.npy'), width_mean_npy)
