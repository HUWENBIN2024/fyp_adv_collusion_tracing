import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import argparse

from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config
from watermark import Watermark
from tqdm import tqdm

import numpy as np

from bibdcalc import BIBD, BIBDParams

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
parser.add_argument('--dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB'])
parser.add_argument('-k', '--num_collusion', help = 'The number of attackers (k).', type = int, default = 2)
parser.add_argument('-n', '--num_samples', help = 'The number of generated collusive samples.', type = int, default = 1000)
parser.add_argument('--attack_name', help = 'Which black-box attack', choices = [ "Bandit", "NES", "HopSkipJump", "SignOPT", "SimBA-px"])
parser.add_argument('--collusion_attack', help = 'collusion methods', choices = [ "mean", "max", "min", "median", "negative", "negative_prob"])
parser.add_argument('-M', '--num_models', help = 'The number of models used.', type = int, default = 100)
args = parser.parse_args()

def and_logic(a_, b_):
    c_ = []
    for i in range(len(a_)):
        if a_[i] == 1 and b_[i] == 1:
            c_.append(1)
        else:
            c_.append(0)
    return c_

def get_head_code_dict(v=3*32*32, k=2, b=100):
    lambda_ = 1  # Lambda value

    # Create BIBDParams object with the specified parameters
    params = BIBDParams(None, v, k, lambda_)

    # Generate the blocks for the BIBD
    blocks = []
    for i in (range(b)):
        block = set((i * k + j) % v for j in range(k))
        blocks.append(block)

    # Create the BIBD with the specified blocks and parameters
    bibd = BIBD(blocks, params)

    # Get the incidence matrix
    matrix = bibd.get_incidency_matrix()

    a = []
    for k in range(len(matrix)):
        a.append([(1 - int(i)) for i in matrix[k]])
        # a.append([(int(i)) for i in matrix[k]])


    dic = {}
    and_logic_and_code = []
    k_ = 0
    for i in range(len(a)):
        for j in range(i):
            dic[k_] = [i, j]
            k_ += 1
            and_logic_and_code.append(and_logic(a[i], a[j]))

    and_logic_and_code = np.array(and_logic_and_code)

    and_logic_and_code = 1 - and_logic_and_code
    
    return dic, and_logic_and_code

    

model_dir = f'saved_models/{args.model_name}-{args.dataset_name}'

total = 0
success_num = 0

prob = 0.8
a = np.load(f"saved_collusion_adv_examples/{args.model_name}-{args.dataset_name}/{args.num_collusion}_attackers/{args.attack_name}_{args.num_samples}_num_of_samples.npz", allow_pickle=True)

img = a['X'] # shape: n, 3, 32, 32
img_adv = a['X_attacked_k'] # shape: k, n, 3, 32, 32
label = a['y']
head_index = a['head']


adv_perturb = img_adv - img

if args.collusion_attack =='mean':
    collusion_perturb = np.mean(adv_perturb, axis=0) # (n, 3, 32, 32)
elif args.collusion_attack =='max':
    collusion_perturb = np.max(adv_perturb, axis=0) 
elif args.collusion_attack =='min':
    collusion_perturb = np.min(adv_perturb, axis=0) 
elif args.collusion_attack =='median':
    collusion_perturb = np.median(adv_perturb, axis=0) 
elif args.collusion_attack =='negative':
    collusion_perturb = np.max(adv_perturb, axis=0) + np.min(adv_perturb, axis=0) - np.median(adv_perturb, axis=0)
elif args.collusion_attack =='negative_prob':
    rand_mask = np.random.choice([1, 0], size=img.shape, p=[prob, 1-prob])
    collusion_perturb = np.max(adv_perturb, axis=0) * rand_mask + np.min(adv_perturb, axis=0) * (1 - rand_mask)
    collusion_perturb = collusion_perturb.astype(np.float32)

img_collusion = img + collusion_perturb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img =  torch.from_numpy(img).to(device)
img_adv = torch.from_numpy(img_adv).to(device)
img_collusion = torch.from_numpy(img_collusion).to(device)
label = torch.from_numpy(label).to(device)


dataset = eval(f'config.{args.dataset_name}()')
training_set, testing_set = dataset.training_set, dataset.testing_set
num_classes = dataset.num_classes
means, stds = dataset.means, dataset.stds


dic, code = get_head_code_dict()

threshold = 0.0003
success_count = 0
for i in tqdm(range(len(img_collusion))):
    code_vec = (collusion_perturb[i] <= threshold).astype(np.int32).reshape(3*32*32,)
    index = ((code_vec - code) ** 2).sum(axis=0).argmin()
    # with np.printoptions(threshold=np.inf):
    #     print(code_vec)
    #     print(code[0])
    #     print(collusion_perturb[0].reshape(3*32*32,)[head_index[i][0]*2])
    #     print(collusion_perturb[0].reshape(3*32*32,)[head_index[i][0]*2+1])
    #     print(dic[index])
    #     print(head_index[i])
    head_pred = dic[index] # format: [99,98]
    if head_pred[0] == head_index[i][0] and head_pred[1] == head_index[i][1] or \
        head_pred[0] == head_index[i][1] and head_pred[1] == head_index[i][0]:
        success_count += 1

trace_acc = success_count / len(img_collusion)
print('the tracing accuracy is: ', trace_acc)

# for i in tqdm(range(args.num_models)):

#     head_dir = model_dir + f"/head_{i}/state_dict"
#     tail_dir = model_dir + "/base_tail_state_dict"
#     wm_dir = model_dir + f'/head_{i}/watermark.npy'

#     # mask for adv samples generated by i-th model
#     mask = torch.ones_like(label).to(torch.bool)
#     for j, h_idx in enumerate(head_index):
#         # print(h_idx)
#         if i in h_idx:
#             mask[j] = False

#     # Create the model and the dataset
    
#     Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')
#     normalizer = transforms.Normalize(means, stds)

#     # Load the tail of the model
#     tail = Tail(num_classes)
#     tail.load_state_dict(torch.load(tail_dir))

#     head = Head()
#     head.load_state_dict(torch.load(head_dir))

#     wm = Watermark.load(wm_dir)

#     model = nn.Sequential(normalizer, wm, head, tail).eval().to(device)

#     # y_pred = model(img).argmax(dim=1)
#     y_pred_adv = model(img_collusion[mask]).argmax(dim=1)

#     success_num += (y_pred_adv != label[mask]).sum().item()

#     # print(y_pred_adv.shape, label[mask].shape)
#     # print(mask)
#     total += len(label[mask])


# print("success rate of tranfered adv: ", success_num / total, ", the num of successful and total samples: ", success_num, total)

# os.makedirs('evl', exist_ok=True)
# f = open("col_evl.txt", "a") 
# f.write(f"{args.attack_name} attack, {args.collusion_attack} collusion: {success_num / total}\n")
# f.close()        