import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import argparse
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SimBA

from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config
from watermark import Watermark
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('--dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB'])
    parser.add_argument('-M', '--num_models', help = 'The number of models used.', type = int, default = 100)
    
    parser.add_argument('-k', '--num_collusion', help = 'The number of attackers (k).', type = int, default = 2)
    parser.add_argument('-n', '--num_samples', help = 'number of adv sample you want to generate.', type = int, default = 200)
    
    parser.add_argument('-c', '--cont', help = 'Continue from the stopped point last time.', action = 'store_true')
    parser.add_argument('-d', '--domain', help = 'Choose the domain of the attack.', choices = ['dct', 'px'], default = 'px')
    parser.add_argument('-b', '--batch_size', help = 'The batch size used for attacks.', type = int, default = 10)
    parser.add_argument('-v', '--verbose', help = 'Verbose when attacking.', action = 'store_true')
    args = parser.parse_args()
    
    # renaming
    dataset = eval(f'config.{args.dataset_name}()')
    training_set, testing_set = dataset.training_set, dataset.testing_set
    num_classes = dataset.num_classes
    means, stds = dataset.means, dataset.stds
    C, H, W = dataset.C, dataset.H, dataset.W
    Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size = args.batch_size, shuffle = True, num_workers = 2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_dir = f'saved_models/{args.model_name}-{args.dataset_name}'

    save_dir = f'saved_collusion_adv_examples/{args.model_name}-{args.dataset_name}'

    
    # load the tail of the model
    normalizer = transforms.Normalize(means, stds)
    tail = Tail(num_classes)
    tail.load_state_dict(torch.load(f'{model_dir}/base_tail_state_dict'))
    tail.to(device)

    # load the classifiers
    classifiers = []
    for i in range(args.num_models):
        head = Head()
        head.to(device)
        head.load_state_dict(torch.load(f'{model_dir}/head_{i}/state_dict'))
        watermark = Watermark.load(f'{model_dir}/head_{i}/watermark.npy')
        
        classifier = PyTorchClassifier(
            model = nn.Sequential(normalizer, watermark, head, tail, nn.Softmax(dim = -1)).eval(),
            loss = None, # dummy
            optimizer = None, # dummy
            clip_values = (0, 1),
            input_shape=(C, H, W),
            nb_classes=num_classes,
            device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
        )
        classifiers.append(classifier)
    classifiers = np.array(classifiers)

    original_images, attacked_images, labels, head = [], [], [], []
    count_success = 0
    success_num = 0

    np.random.seed(3407)

    k = args.num_collusion

    # attacking
    for X, y in tqdm(testing_loader):
        with torch.no_grad():
            
            model_index = np.random.choice(np.arange(args.num_models), k, replace=False).astype(int)
            mask_k = np.ones_like(y.numpy())
            X_attacked_k = []
            X, y = X.numpy(), y.numpy()
            for model_id in model_index:
                c = classifiers[model_id]
                
                pred = c.predict(X)
                correct_mask = pred.argmax(axis = 1) == y

                a = SimBA(c, attack = args.domain, verbose = args.verbose)

                X_attacked = a.generate(X)
                attacked_preds = np.vectorize(lambda z: z.predict(X_attacked), signature = '()->(m,n)')(classifiers) # (num_model, batch_size, num_class)
                success_mask = attacked_preds.argmax(axis = -1) != y 
                success_mask = np.logical_and(success_mask[model_id], success_mask.sum(axis=0) >= 2)
                
                mask = np.logical_and(correct_mask, success_mask)
                mask_k = np.logical_and(mask_k, mask)

                X_attacked_k.append(X_attacked)
                    
            X_attacked_k = np.stack(X_attacked_k)
                
            if mask_k.sum()> 0:
                original_images.append(X[mask_k])
                attacked_images.append(X_attacked_k[:,mask_k])
                
                labels.append(y[mask_k])
                for _ in range(mask_k.sum().item()):
                    head.append(model_index)
            
                count_success += mask_k.sum()

                if count_success >= args.num_samples:
                    print(f'SimBA, {count_success} out of {args.num_samples} generated, done!')
                    break
                else:
                    print(f'SimBA, {count_success} out of {args.num_samples} generated...')

            else:
                print('not generated!')

    original_images = np.concatenate(original_images)
    attacked_images = np.concatenate(attacked_images, axis=1)
    labels = np.concatenate(labels)
    os.makedirs(f'{save_dir}/{k}_attackers', exist_ok = True)
    np.savez(f'{save_dir}/{k}_attackers/SimBA-px_{args.num_samples}_num_of_samples.npz', X = original_images, X_attacked_k = attacked_images, y = labels, head=head)
