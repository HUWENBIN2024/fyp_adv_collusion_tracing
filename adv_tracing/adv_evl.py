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

model_dir = 'saved_models/ResNet18-CIFAR10'

total = 0
success_num = 0

for i in tqdm(range(100)):
    

    

    head_dir = model_dir + f"/head_{i}/state_dict"
    tail_dir = model_dir + "/base_tail_state_dict"
    wm_dir = model_dir + f'/head_{i}/watermark.npy'


    
    C, H, W = 3, 32, 32

    # Create the model and the dataset
    dataset = eval(f'config.CIFAR10()')
    training_set, testing_set = dataset.training_set, dataset.testing_set
    num_classes = dataset.num_classes
    means, stds = dataset.means, dataset.stds
    Head, Tail = eval(f'ResNet18Head'), eval(f'ResNet18Tail')
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

        a = np.load(f"/ssddata/whuak/fyp_adv_collusion_tracing/adv_tracing/saved_adv_examples/ResNet18-CIFAR10/head_{j}/NES.npz", allow_pickle=True)

        img = torch.from_numpy(a['X'])
        img_adv = torch.from_numpy(a['X_attacked'])
        label = torch.from_numpy(a['y'])
        
    
        # y_pred = model(img).argmax(dim=1)
        y_pred_adv = model(img_adv).argmax(dim=1)

        success_num += (y_pred_adv != label).sum().item()
        total += len(label)

    

print("success rate of tranfered adv: ", success_num / total)