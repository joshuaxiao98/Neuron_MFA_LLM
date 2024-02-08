import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as stats
from tqdm import tqdm
from pathlib import Path
import pickle
import os
import re
import seaborn as sns


#4 NFD for distance list
def wnfd_llm(mat,Q,draw=False,fdigi=0, threshold=100): # Choose an appopriate fidgi

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
            slope, intercept, _, _, _ = stats.linregress(x, y)
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
            slope, intercept, _, _, _ = stats.linregress(x, y)
            tau_list.append(slope)

    return tau_list


def nspectrum(tau_list,q_list,k):
    al_list = []
    fal_list = []
    for i in range(1,len(q_list)):
        al=(tau_list[i]-tau_list[i-1])/(q_list[i]-q_list[i-1])
        al_list.append(al)
    for j in range(len(q_list)-1):
        fal=q_list[j]*al_list[j]-tau_list[j]
        fal_list.append(fal)
    return [al_list,fal_list]


if __name__ == '__main__':

    modes = ['mean', 'std']
    repeat_n = 10
    steps = 11
    threshold = 150
    filter_mode = 2

    sns.palplot(sns.color_palette("coolwarm", steps))
    color = sns.color_palette("coolwarm", steps)

    pkl_dir = Path('./networks')

    fig_dir = Path('.figures')

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    entries = os.listdir(pkl_dir)

    models_name = [entry for entry in entries if os.path.isdir(os.path.join(pkl_dir, entry))]

    for num, mode in enumerate(modes):

        for model_name in models_name:
            print(model_name)

            model_dir = os.path.join(pkl_dir, model_name)

            files = os.listdir(model_dir)

            if model_name == 'pythia-31m':
                name = ['Step ' + str(i) for i in [1, 8, 64, 512, 1000, 5000, 10000, 35000, 71000, 107000, 140000]]
                epochs = [1, 8, 64, 512, 1000, 5000, 10000, 35000, 71000, 107000, 140000]
            else:
                name = ['Step ' + str(i) for i in [1, 8, 64, 512, 1000, 5000, 10000, 35000, 71000, 107000, 143000]]
                epochs = [1, 8, 64, 512, 1000, 5000, 10000, 35000, 71000, 107000, 143000]

            ntauls_list_stored = False

            ntauls_list_stored_path = os.path.join(fig_dir, model_name, f'threshold{threshold}', f'ntauls_list.npy')

            if os.path.exists(ntauls_list_stored_path):
                ntauls_list_stored = True

            if ntauls_list_stored == True:
                ntauls_list = np.load(ntauls_list_stored_path)
                Q = [q / 100 for q in range(-300, 301, 10)]

            else:

                mat_lists = []

                for sample_num in range(repeat_n):
                    models_pkl = [file for file in files if file.endswith(f'{sample_num}.pkl')]
                    sorted_models_pkl = sorted(models_pkl, key=lambda x: int(re.search('step(\d+)_', x).group(1)))

                    mat_list = []

                    for model_pkl in sorted_models_pkl:

                        step_num = int(re.search(r'step(\d+)_sampled', model_pkl).group(1))
                        if step_num not in epochs:
                            continue
                        # print(step_num)
                        pkl_path = os.path.join(model_dir, model_pkl)

                        with open(pkl_path, 'rb') as file:
                            net = pickle.load(file)

                        mat_list.append(net)

                    mat_lists.append(mat_list)

                mat_list_average = []

                for i in range(steps):
                    for j in range(repeat_n):
                        mat_list_average.append(mat_lists[j][i])


                plt.rcParams.update({'font.size': 30})
                plt.figure(figsize=(10, 10))

                Q = [q / 100 for q in range(-300, 301, 10)]
                G_list = mat_list_average
                ntauls_list = []
                for i in tqdm(range(len(G_list))):
                    ntau = wnfd_llm(G_list[i], Q, draw=False, threshold=threshold)
                    ntauls_list.append(ntau)

                if not os.path.exists(os.path.join(fig_dir, model_name, f'threshold{threshold}')):
                    os.makedirs(os.path.join(fig_dir, model_name, f'threshold{threshold}'))

                np.save(os.path.join(fig_dir, model_name, f'threshold{threshold}',f'ntauls_list.npy'), np.array(ntauls_list))


            alpha, f_alpha = [], []
            repeat_n_list = []
            count_num = 0

            for i in range(len(ntauls_list)+1):
                if i % repeat_n == 0 and i>0:
                    repeat_n_list.append(count_num)
                    count_num = 0
                    if i == len(ntauls_list):
                        break

                [a, f] = nspectrum(ntauls_list[i], Q, i)

                if filter_mode == 1:
                    if any(n > 0 for n in f):
                        continue
                elif filter_mode == 2:
                    # print(f'alpha {a}')
                    if (max(a) - a[0])>0.01:
                        print(f'alpha {a}')
                        continue
                elif filter_mode == 3:
                    if any(n > 0 for n in f) or (max(a) - a[0])>0.01:
                        continue
                alpha.append(a)
                f_alpha.append(f)
                count_num = count_num + 1


            alpha_mean, f_alpha_mean = [], []
            alpha_std, f_alpha_std = [], []
            alpha_max, f_alpha_max = [], []
            alpha_min, f_alpha_min = [], []
            start_index, end_index = 0, 0

            for i in range(len(name)):
                end_index = end_index + repeat_n_list[i]
                alpha_mean.append(np.mean(alpha[start_index: end_index], axis=0))
                f_alpha_mean.append(np.mean(f_alpha[start_index: end_index], axis=0))

                alpha_std.append(np.std(alpha[start_index: end_index], axis=0))
                f_alpha_std.append(np.std(f_alpha[start_index: end_index], axis=0))

                start_index = start_index + repeat_n_list[i]


            alpha_std = np.array(alpha_std)
            f_alpha_std = np.array(f_alpha_std)

            alpha_std = alpha_std.tolist()
            f_alpha_std = f_alpha_std.tolist()

            plt.rcParams.update({'font.size': 30})
            plt.figure(figsize=(10, 10))


            for i in range(len(name)):

                mid = np.argmax(f_alpha_mean[i])

                if num == 0:
                    plt.plot(alpha_mean[i],f_alpha_mean[i],color=color[0+i],label=name[i],linewidth=3)
                else:
                    plt.plot(alpha_mean[i], f_alpha_mean[i], color=color[0 + i], label=name[i], linewidth=3)


                    plt.fill_between(alpha_mean[i], f_alpha_mean[i] - f_alpha_std[i], f_alpha_mean[i], alpha=0.8,
                                     color=color[0 + i])
                    plt.fill_between(alpha_mean[i], f_alpha_mean[i], f_alpha_mean[i] + f_alpha_std[i], alpha=0.8,
                                     color=color[0 + i])
                    plt.fill_betweenx(f_alpha_mean[i][:mid], alpha_mean[i][:mid] - alpha_std[i][:mid],
                                      alpha_mean[i][:mid] + alpha_std[i][:mid], label=name[i], alpha=0.55, color=color[0 + i],
                                      interpolate=True)
                    plt.fill_betweenx(f_alpha_mean[i][mid:], alpha_mean[i][mid:] - alpha_std[i][mid:],
                                      alpha_mean[i][mid:] + alpha_std[i][mid:], alpha=0.55, color=color[0 + i],
                                      interpolate=True)
                plt.xlabel('Lipschitz-Holder exponent, 'r'$\alpha$')
                plt.ylabel('Multi-fractal spectrum, 'r'$f(\alpha)$')
            ax = plt.gca()
            plt.grid(False)

            title = model_name.split(".pkl")[0]
            plt.title(title)

            if not os.path.exists(os.path.join(fig_dir, model_name, f'threshold{threshold}')):
                os.makedirs(os.path.join(fig_dir, model_name, f'threshold{threshold}'))

            save_path = os.path.join(fig_dir, model_name, f'threshold{threshold}', f'{title}_{mode}_f{filter_mode}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

            print(f'figure {title}_threshold{threshold}_{mode} saved successfully')
            plt.close()

