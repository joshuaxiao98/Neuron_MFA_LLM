import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import os
import re


# generate randomly sampled numbers
def rand_sample(sample_num, total_num):
    nums = np.arange(0, total_num)
    samples = np.random.choice(nums, sample_num, replace=False)
    return samples

# generate randomly sampled numbers of every layer in a LLM model
def net_sample_nums(block_num, layer_sample, feature_num):
    net_samples = {}

    for i in range(block_num):
        nums1 = rand_sample(layer_sample, feature_num)
        nums2_tmp = rand_sample(layer_sample, feature_num)
        nums2 = np.concatenate((nums2_tmp, nums2_tmp + feature_num, nums2_tmp + feature_num * 2))
        nums3 = nums2_tmp
        nums4 = rand_sample(layer_sample, feature_num)
        nums5 = rand_sample(layer_sample * 2, feature_num * 4)

        net_samples['block' + str(i) + '_nums1'] = nums1
        net_samples['block' + str(i) + '_nums2'] = nums2
        net_samples['block' + str(i) + '_nums3'] = nums3
        net_samples['block' + str(i) + '_nums4'] = nums4
        net_samples['block' + str(i) + '_nums5'] = nums5
        if i == block_num - 1:
            nums6 = rand_sample(layer_sample, feature_num)
            net_samples['block' + str(i) + '_nums6'] = nums6

    return net_samples

# transform weight matrix to a list
# every element of the list represents an edge in the network of LLM
def edge(x_s, y_s, param, x_sample, y_sample):
    edge_ls = []
    for i in x_sample:
        for j in y_sample:
            if param[i, j] != 0:
                temp = [x_s + i, y_s + j, param[i, j]]
                edge_ls.append(temp)
    return edge_ls, len(edge_ls)

# the main function which converts a LLM model to a complex network
def make_edge(model, model_name, step, index, net_samples, blocks_num, feature_num):
    x_index_start, y_index_start = 0, feature_num
    param_edge = []

    for name, param in model.named_parameters():

        if name[9] != 'l':
            continue
        if 'bias' in name:
            continue
        if 'layernorm' in name:
            continue

        param_mux = param.to('cpu').detach().numpy()

        block_num = int(re.findall(r'\d+', name)[0])

        nums1 = net_samples['block' + str(block_num) + '_nums1']
        nums2 = net_samples['block' + str(block_num) + '_nums2']
        nums3 = net_samples['block' + str(block_num) + '_nums3']
        nums4 = net_samples['block' + str(block_num) + '_nums4']
        nums5 = net_samples['block' + str(block_num) + '_nums5']
        if block_num == blocks_num - 1:
            nums6 = net_samples['block' + str(block_num) + '_nums6']
        else:
            nums6 = net_samples['block' + str(block_num + 1) + '_nums1']

        print('name :{}'.format(name))

        x_sample = None
        y_sample = None

        if ('attention.query_key_value.weight' in name):
            x_sample = nums1
            y_sample = nums2

        # process attention.dense.weight
        elif ('attention.dense.weight' in name):
            x_sample = nums3
            y_sample = nums4

        # process mlp.dense_h_to_4h.weight
        elif ('mlp.dense_h_to_4h.weight' in name):
            x_sample = nums4
            y_sample = nums5

        # process mlp.dense_4h_to_h.weight
        elif ('mlp.dense_4h_to_h.weight' in name):
            x_sample = nums5
            y_sample = nums6


        shape0, shape1 = param_mux.shape[0], param_mux.shape[1]
        y_index_end = y_index_start + shape0
        x_index_end = x_index_start + shape1
        edges, num = edge(x_index_start, y_index_start, param_mux.T, x_sample, y_sample)
        param_edge += edges
        y_index_start = y_index_end
        x_index_start = x_index_end

        # process attention.query_key_value.weight
        if ('attention.query_key_value.weight' in name):

            adj_1 = np.eye(shape1)
            adj_2 = np.ones((int(shape0 / 3 * 2), shape1))
            adj = np.vstack((adj_1, adj_2))

            y_index_end = y_index_end + int(shape0 / 3)
            x_index_end = x_index_end + shape0
            edges, num = edge(x_index_start, y_index_start, adj, nums2, nums3)
            param_edge += edges
            y_index_start = y_index_end
            x_index_start = x_index_end

    npy_dir = os.path.join('./models', model_name)
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    np.save(f'./models/{model_name}/{model_name}_step{step}_sampled_' + str(index) + '.npy', np.array(param_edge))
    param_num = len(param_edge)
    print(f'{model_name}_step{step}_sampled_' + str(index) + ' made successfully')
    print('edges num :{}'.format(param_num))


if __name__ == '__main__':

    sample_repeat_times = 10

    steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 5000, 10000, 13000, 23000, 33000, 35000, 43000, 53000,
             63000, 71000, 73000, 83000, 93000, 103000, 107000, 113000, 123000, 133000, 143000]


    feature_nums = [128, 256, 512, 768, 1024, 2048, 2048, 2560]

    block_nums = [6, 6, 6, 12, 24, 16, 24, 32, 32, 36]

    layer_samples = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]

    models_name = ["pythia-14m", "pythia-31m", "pythia-70m-deduped", "pythia-160m-deduped", "pythia-410m-deduped",
                   "pythia-1b-deduped", "pythia-1.4b-deduped", "pythia-2.8b-deduped"]

    for i, model_name in enumerate(models_name):

        for index in range(0, sample_repeat_times):
            np.random.seed(index)
            net_samples = net_sample_nums(block_nums[i], layer_samples[i], feature_nums[i])

            for step in steps:

                # model "pythia-31m" doesn't have the training step 143000
                if model_name == "pythia-31m" and step == 143000:
                    step = 140000

                model = GPTNeoXForCausalLM.from_pretrained(
                    f"EleutherAI/{model_name}",
                    revision=f"step{step}",
                    cache_dir=f"./{model_name}/step{step}",
                )
                print(model_name)

                make_edge(model, model_name, step, index, net_samples,  block_nums[i], feature_nums[i])

