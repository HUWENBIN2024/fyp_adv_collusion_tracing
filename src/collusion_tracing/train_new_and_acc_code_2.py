import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import argparse

from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config
from watermark import Watermark

from bibdcalc import BIBD, BIBDParams
from tqdm import tqdm
import numpy as np
import random
random.seed(3407)


'''
Train the multi-head-one-tail model.
'''

# def get_collusion_watermark_location(v=3*32*32, k=2, b=100):
#     '''
#     func: get location of the masked pixels, which will be used to generate watermark for collusion tracing.

#     return: a list of np arrays. 
#     '''
#     lambda_ = 1  # Lambda value

#     # Create BIBDParams object with the specified parameters
#     params = BIBDParams(None, v, k, lambda_)

#     # Generate the blocks for the BIBD
#     blocks_set = []
#     locations = []
#     for i in tqdm(range(b)):
#         block = set((i * k + j) % v for j in range(k))
#         blocks_set.append(block)
#         block_ = np.array(list(block))
#         locations.append(np.stack([block_ // (H * W), (block_ // W) % H, block_ % W], axis = -1))


#     # Create the BIBD with the specified blocks and parameters
#     bibd = BIBD(blocks_set, params)
#     print("bibd design generated!!!")

#     # # Get the incidence matrix
#     matrix = bibd.get_incidency_matrix()
#     # print('matrix generated!!!')
#     return locations

def generate_and_acc_code_book(n, v, K):
    code_book = []
    for _ in range(n):
        # vector = [random.choice([0 , 0 ,0 ,0 ,0, 0, 0, 0, 0, 1]) for _ in range(v)]
        # vector = [random.choice([0, 1, 1, 1]) for _ in range(v)]
        vector = [random.choice([0, 1]) for _ in range(v)]

        code_book.append(vector)
        while not is_and_acc(code_book, K):
            vector = [random.choice([0, 1]) for _ in range(v)]
            code_book[-1] = vector
    code_book = np.array(code_book)
    
    
    return code_book


def is_and_acc(code_book, K):
    for i in range(len(code_book)):
        for j in range(i + 1, len(code_book)):
            subset_i = code_book[:i+1]
            subset_j = code_book[:j+1]
            if len(subset_i) <= K and len(subset_j) <= K:
                if all(a & b == a for a, b in zip(subset_i[-1], subset_j[-1])):
                    return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('--dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB', 'TINY'])
    parser.add_argument('--num_workers', help = 'Number of workers', type = int, default = 0)
    parser.add_argument('-N', '--num_heads', help = 'Number of heads.', type = int, default = 100)
    parser.add_argument('-b', '--batch_size', help = 'Batch size.', type = int, default = 128)
    parser.add_argument('-e', '--num_epochs', help = 'Number of epochs.', type = int, default = 10)
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate.', type = float, default = 1e-3)
    parser.add_argument('--head_id', type = int, default = 0)

    # parser.add_argument('-md', '--masked_dims', help = 'Number of masked dimensions', type = int, default = 100)

    # collusion
    # parser.add_argument('-nc', '--num_collusion', help = 'number of collusive attacks.', type = int, default = 2)
    # parser.add_argument('-np', '--num_pixel_watermarked', help = 'number of pixel watermarked.', type = int, default = 300)


    
    args = parser.parse_args()
    
    if args.dataset_name == 'CIFAR10' or args.dataset_name == 'GTSRB':
        C, H, W = 3, 32, 32
    elif args.dataset_name == 'tiny':
        C, H, W = 3, 64, 64

    # Create the model and the dataset
    dataset = eval(f'config.{args.dataset_name}()')
    training_set, testing_set = dataset.training_set, dataset.testing_set
    num_classes = dataset.num_classes
    means, stds = dataset.means, dataset.stds
    Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')
    normalizer = transforms.Normalize(means, stds)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Place to save the trained model
    save_dir = f'saved_models_50_masked_pixel/{args.model_name}-{args.dataset_name}'
    os.makedirs(save_dir, exist_ok = True)

    # Load the tail of the model
    tail = Tail(num_classes)
    tail.load_state_dict(torch.load(f'{save_dir}/base_tail_state_dict'))
    
    tail.to(device)

    # location array for watermark
    code_book = generate_and_acc_code_book(n=args.num_heads, v=3*32*32, K=5)
    # (100,3072)
    
    # training
    
    # for i in range(args.num_heads):
    i = args.head_id
    os.makedirs(f'{save_dir}/head_{i}', exist_ok = True)
    
    # head = nn.Sequential(Watermark.random(args.masked_dims, C, H, W), Head())
    pos = np.where(code_book[i].reshape(3,32,32)==1)
    head = nn.Sequential(Watermark(np.array(pos).T), Head())
    
    head.to(device)
    head[0].save(f'{save_dir}/head_{i}/watermark.npy')
    head[1].load_state_dict(torch.load(f'{save_dir}/base_head_state_dict'))
    optimizer = torch.optim.Adam(head.parameters(), lr = args.learning_rate)
    Loss = nn.CrossEntropyLoss()
    best_accuracy = 0.

    for n in range(args.num_epochs):
        head.train()
        epoch_mask_grad_norm, epoch_mask_grad_norm_inverse = 0., 0.
        epoch_loss = 0.0
        for X, y in tqdm(training_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out_clean = tail(head(normalizer(X)))
            clean_loss = Loss(out_clean, y)
            loss = clean_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y) / len(training_set)

        # testing
        head.eval()
        tail.eval()
        
        accuracy = 0.0
        with torch.no_grad():
            for X, y in testing_loader:
                X, y = X.to(device), y.to(device)
                _, pred = tail(head(normalizer(X))).max(axis = -1)
                accuracy += (pred == y).sum().item() / len(testing_set)

        print(f'Head {i}, epoch {n}, loss {epoch_loss:.3f}, accuracy = {accuracy:.4f}')

        f = open(f"{save_dir}/{args.model_name}-{args.dataset_name}.txt", "a")
        f.write(f'{args.num_heads}-{args.num_epochs}-{accuracy}\n')
        f.close()

        # save the best result
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(head[1].state_dict(), f'{save_dir}/head_{i}/state_dict')

        print(f'Completed the training for head {i}, accuracy = {best_accuracy:.4f}.')
    print(f'Completed the training of {args.num_heads} heads, {args.model_name}-{args.dataset_name}.')
