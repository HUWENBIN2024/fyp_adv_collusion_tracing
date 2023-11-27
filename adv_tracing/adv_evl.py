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
parser.add_argument('-M', '--num_models', help = 'The number of models used.', type = int, default = 100)
args = parser.parse_args()

model_dir = f'saved_models/{args.model_name}-{args.dataset_name}'

total = 0
success_num = 0

dataset = eval(f'config.{args.dataset_name}()')
training_set, testing_set = dataset.training_set, dataset.testing_set
num_classes = dataset.num_classes
means, stds = dataset.means, dataset.stds

for i in tqdm(range(args.num_models)):
    
    head_dir = model_dir + f"/head_{i}/state_dict"
    tail_dir = model_dir + "/base_tail_state_dict"
    wm_dir = model_dir + f'/head_{i}/watermark.npy'

    # Create the model and the dataset
    
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

    for j in range(100):
        if i == j:
            continue

        a = np.load(f"saved_adv_examples/{args.model_name}-{args.dataset_name}/head_{j}/{args.attack_name}.npz", allow_pickle=True)

        img = torch.from_numpy(a['X'])
        img_adv = torch.from_numpy(a['X_attacked'])
        label = torch.from_numpy(a['y'])
        
    
        # y_pred = model(img).argmax(dim=1)
        y_pred_adv = model(img_adv).argmax(dim=1)

        success_num += (y_pred_adv != label).sum().item()
        total += len(label)

    

print("success rate of tranfered adv: ", success_num / total)