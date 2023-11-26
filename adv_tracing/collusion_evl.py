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

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
parser.add_argument('--dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB'])
parser.add_argument('--attack_name', help = 'Which black-box attack', choices = [ "Bandit", "NES", "HopSkipJump", "SignOPT", "SimBA-px"])
parser.add_argument('--collusion_attack', help = 'collusion methods', choices = [ "mean", "max", "min", "negative", "negative_prob"])
parser.add_argument('-M', '--num_models', help = 'The number of models used.', type = int, default = 100)
args = parser.parse_args()

model_dir = f'saved_models/{args.model_name}-{args.dataset_name}'

total = 0
success_num = 0

prob = 0.8
a = np.load(f"saved_collusion_adv_examples/{args.model_name}-{args.dataset_name}/{args.num_collusion}_attackers/{args.attack_name}.npz", allow_pickle=True)

img = torch.from_numpy(a['X']) # shape: n, 3, 32, 32
img_adv = torch.from_numpy(a['X_attacked']) # shape: k, n, 3, 32, 32
label = torch.from_numpy(a['y'])

adv_perturb = img_adv - img
if args.collusion_attack =='mean':
    collusion_perturb = np.mean(adv_perturb, axis=0) # linear combinition, (n, 3, 32, 32)
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

img_collusion = img + collusion_perturb

for i in tqdm(range(args.num_models)):

    head_dir = model_dir + f"/head_{i}/state_dict"
    tail_dir = model_dir + "/base_tail_state_dict"
    wm_dir = model_dir + f'/head_{i}/watermark.npy'


    
    C, H, W = 3, 32, 32

    # Create the model and the dataset
    dataset = eval(f'config.{args.dataset_name}()')
    training_set, testing_set = dataset.training_set, dataset.testing_set
    num_classes = dataset.num_classes
    means, stds = dataset.means, dataset.stds
    Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')
    normalizer = transforms.Normalize(means, stds)
    # training_loader = torch.utils.data.DataLoader(training_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    # testing_loader = torch.utils.data.DataLoader(testing_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Place to save the trained model
    # save_dir = f'saved_models/{args.model_name}-{args.dataset_name}'
    # os.makedirs(save_dir, exist_ok = True)

    # Load the tail of the model
    tail = Tail(num_classes)
    tail.load_state_dict(torch.load(tail_dir))

    head = Head()
    head.load_state_dict(torch.load(head_dir))

    wm = Watermark.load(wm_dir)

    model = nn.Sequential(normalizer, wm, head, tail).eval()

    # y_pred = model(img).argmax(dim=1)
    y_pred_adv = model(img_collusion).argmax(dim=1)

    success_num += (y_pred_adv != label).sum().item()
    total += len(label)

    

print("success rate of tranfered adv: ", success_num / total)