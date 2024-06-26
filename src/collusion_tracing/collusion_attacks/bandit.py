import torch
import torch.nn as nn
from torch.nn import Upsample
from torchvision import transforms
import numpy as np
import os
import argparse
from art.estimators.classification import PyTorchClassifier

from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config
from watermark import Watermark
from collusion_attacks.score import ScoreBlackBoxAttack
from collusion_attacks import *

from tqdm import tqdm


Loss = nn.CrossEntropyLoss(reduction = 'none')

class BanditAttack(ScoreBlackBoxAttack):
    """
    Bandit Attack
    """

    def __init__(self,
                 max_loss_queries,
                 epsilon, p,
                 fd_eta, lr,
                 prior_exploration, prior_size, data_size, prior_lr,
                 lb, ub, batch_size, name):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lr: learning rate of NES step
        :param prior_exploration: exploration noise
        :param prior_size: prior height/width (this is applicable only to images), you can disable it by setting it to
            None (it is assumed to prior_size = prior_height == prior_width)
        :param data_size: data height/width (applicable to images of the from `c x h x w`, you can ignore it
            by setting it to none, it is assumed that data_size = height = width
        :param prior_lr: learning rate in the prior space
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_extra_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size= batch_size,
                         name = "Bandit")
        # other algorithmic parameters
        self.fd_eta = fd_eta
        # learning rate
        self.lr = lr
        # data size
        self.data_size = data_size

        # prior setup:
        # 1. step function
        if self.p == '2':
            self.prior_step = step
        elif self.p == 'inf':
            self.prior_step = eg_step
        else:
            raise Exception("Invalid p for l-p constraint")
        # 2. prior placeholder
        self.prior = None
        # prior size
        self.prior_size = prior_size
        # prior exploration
        self.prior_exploration = prior_exploration
        # 3. prior upsampler
        self.prior_upsample_fct = None if self.prior_size is None else upsample_maker(data_size, data_size)
        self.prior_lr = prior_lr

    def _perturb(self, xs_t, loss_fct):
        """
        The core of the bandit algorithm
        since this is compute intensive, it is implemented with torch support to push ops into gpu (if available)
        however, the input / output are numpys
        :param xs: numpy
        :return new_xs: returns a torch tensor
        """

        _shape = list(xs_t.shape)
        eff_shape = list(xs_t.shape)
        # since the upsampling assumes xs_t is batch_size x c x h x w. This is not the case for mnist,
        # which is batch_size x dim, let's take care of that below
        
        if self.prior_size is None:
            prior_shape = eff_shape
        else:
            prior_shape = eff_shape[:-2] + [self.prior_size] * 2
        # reset the prior if xs  is a new batch
        if self.is_new_batch:
            self.prior = torch.zeros(prior_shape, device = xs_t.device)
        # create noise for exploration, estimate the gradient, and take a PGD step
        # exp_noise = torch.randn(prior_shape) / (np.prod(prior_shape[1:]) ** 0.5)  # according to the paper
        exp_noise = torch.randn(prior_shape, device = xs_t.device)
        # Query deltas for finite difference estimator
        if self.prior_size is None:
            q1 = step(self.prior, exp_noise, self.prior_exploration)
            q2 = step(self.prior, exp_noise, - self.prior_exploration)
        else:
            q1 = self.prior_upsample_fct(step(self.prior, exp_noise, self.prior_exploration))
            q2 = self.prior_upsample_fct(step(self.prior, exp_noise, - self.prior_exploration))
        # Loss points for finite difference estimator
        l1 = loss_fct(l2_step(xs_t, q1.view(_shape), self.fd_eta))
        l2 = loss_fct(l2_step(xs_t, q2.view(_shape), self.fd_eta))
        # finite differences estimate of directional derivative
        est_deriv = (l1 - l2) / (self.fd_eta * self.prior_exploration)
        # 2-query gradient estimate
        # Note: Ilyas' implementation multiply the below by self.prior_exploration (different from pseudocode)
        # This should not affect the result as the `self.prior_lr` can be adjusted accordingly
        est_grad = est_deriv.view(-1, *[1] * len(prior_shape[1:]))* exp_noise
        # update prior with the estimated gradient:
        self.prior = self.prior_step(self.prior, est_grad, self.prior_lr)
        # gradient step in the data space
        if self.prior_size is None:
            gs = self.prior.clone()
        else:
            gs = self.prior_upsample_fct(self.prior)
        # perform the step
        new_xs = lp_step(xs_t, gs.view(_shape), self.lr, self.p)
        return new_xs, 2 * torch.ones(_shape[0], device = xs_t.device)

    def _config(self):
        return {
            "name": self.name,
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_extra_queries": "inf" if np.isinf(self.max_extra_queries) else self.max_extra_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "lr": self.lr,
            "prior_lr": self.prior_lr,
            "prior_exploration": self.prior_exploration,
            "prior_size": self.prior_size,
            "data_size": self.data_size,
            "fd_eta": self.fd_eta,
            "attack_name": self.__class__.__name__
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('--dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB'])

    parser.add_argument('-k', '--num_collusion', help = 'The number of attackers (k).', type = int, default = 2)
    parser.add_argument('-n', '--num_samples', help = 'number of adv sample you want to generate.', type = int, default = 200)


    parser.add_argument('-M', '--num_models', help = 'The number of models used.', type = int, default = 100)
    parser.add_argument('-c', '--cont', help = 'Continue from the stopped point last time.', action = 'store_true')
    parser.add_argument('-b', '--batch_size', help = 'The batch size used for attacks.', type = int, default = 10)
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
    
    # load the classifiers
    classifiers = []
    models = []
    tail = Tail(num_classes)
    tail.load_state_dict(torch.load(f'{model_dir}/base_tail_state_dict'))
    tail.to(device)
    for i in range(args.num_models):
        

        head = Head()
        head.to(device)
        head.load_state_dict(torch.load(f'{model_dir}/head_{i}/state_dict'))
        watermark = Watermark.load(f'{model_dir}/head_{i}/watermark.npy')
        models.append(nn.Sequential(normalizer, watermark, head, tail).eval())
        models[-1].to(device)
        
        classifier = PyTorchClassifier(
            model = models[-1],
            loss = None, 
            optimizer = None, 
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



    for X, y in tqdm(testing_loader):
        with torch.no_grad():
            model_index = np.random.choice(np.arange(args.num_models), k, replace=False) 
            mask_k = np.ones_like(y.numpy())
            X_attacked_k = []
            for model_id in model_index:
                model, c = models[model_id], classifiers[model_id]
                pred = c.predict(X.numpy())
                correct_mask = pred.argmax(axis = -1) == y.numpy()

                X_device, y_device = X.to(device), y.to(device)
                def loss_fct(xs, es = False):
                    logits = model(xs)
                    loss = Loss(logits.to(device), y_device)
                    if es:
                        return torch.argmax(logits, axis= -1) != y_device, loss
                    else: 
                        return loss

                def early_stop_crit_fct(xs):
                    logits = model(xs)
                    return logits.argmax(axis = -1) != y_device

                a = BanditAttack(max_loss_queries = 10000, epsilon = 1.0, p = '2', lb = 0.0, ub = 1.0, batch_size = args.batch_size, name = 'Bandit',
                       fd_eta = 0.01, lr = 0.01, prior_exploration = 0.1, prior_size = 20, data_size = 32, prior_lr = 0.1)

                X_attacked = a.run(X_device, loss_fct, early_stop_crit_fct).cpu().numpy()

                attacked_preds = np.vectorize(lambda z: z.predict(X_attacked), signature = '()->(m,n)')(classifiers)
                
                success_mask = attacked_preds.argmax(axis = -1) != y.numpy()
                success_mask = np.logical_and(success_mask[model_id], success_mask.sum(axis=0) >= 2)

                mask = np.logical_and(correct_mask, success_mask)
                mask_k = np.logical_and(mask_k, mask)

                X_attacked_k.append(X_attacked)
                
            X_attacked_k = np.stack(X_attacked_k)
            
            if mask_k.sum() >= 4:
                original_images.append(X[mask_k])
                attacked_images.append(X_attacked_k[:,mask_k])
                
                labels.append(y[mask_k])
                for _ in range(mask_k.sum().item()):
                    head.append(model_index)
            
                count_success += mask_k.sum()

                if count_success >= args.num_samples:
                    print(f'Bandit, {count_success} out of {args.num_samples} generated, done!')
                    break
                else:
                    print(f'Bandit, {count_success} out of {args.num_samples} generated...')

            else:
                print('not generated!')    
    
    original_images = np.concatenate(original_images)
    attacked_images = np.concatenate(attacked_images, axis=1)
    labels = np.concatenate(labels)
    os.makedirs(f'{save_dir}/{k}_attackers', exist_ok = True)
    np.savez(f'{save_dir}/{k}_attackers/Bandit_{args.num_samples}_num_of_samples.npz', X = original_images, X_attacked_k = attacked_images, y = labels, head=head)
