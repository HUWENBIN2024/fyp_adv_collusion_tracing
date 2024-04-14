import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import argparse
from scipy.linalg import qr
from art.estimators.classification import PyTorchClassifier

from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config
from watermark import Watermark
from collusion_attacks.decision import DecisionBlackBoxAttack

from tqdm import tqdm

if torch.cuda.is_available():
    t = lambda z: torch.tensor(data = z).cuda()
else:
     t = lambda z: torch.tensor(data = z)

start_learning_rate = 1.0

def quad_solver(Q, b):
    """
    Solve min_a  0.5*aQa + b^T a s.t. a>=0
    """
    K = Q.shape[0]
    alpha = torch.zeros((K,))
    g = b
    Qdiag = torch.diag(Q)
    for _ in range(20000):
        delta = torch.maximum(alpha - g/Qdiag,0) - alpha
        idx = torch.argmax(torch.abs(delta))
        val = delta[idx]
        if abs(val) < 1e-7: 
            break
        g = g + val*Q[:,idx]
        alpha[idx] += val
    return alpha

def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = torch.sign(y)
    y_sign[y_sign==0] = 1
    return y_sign


class SignOPTAttack(DecisionBlackBoxAttack):
    """
    Sign_OPT
    """

    def __init__(self, epsilon, p, alpha, beta, svm, momentum, max_queries, k, lb, ub, batch_size, sigma):
        super().__init__(max_queries = max_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size = batch_size)
        self.alpha = alpha
        self.beta = beta
        self.svm = svm
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.query_count = 0


    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001):
        """ 
        Attack the original image and return adversarial example
        """

        y0 = y0[0]
        self.query_count = 0

        # Calculate a good starting point.
        num_directions = 10
        best_theta, g_theta = None, float('inf')

        for i in range(num_directions):
            self.query_count += 1
            theta = torch.randn_like(x0)
            if self.predict_label(x0+theta)!=y0:
                initial_lbd = torch.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(x0, y0, theta, initial_lbd, g_theta)
                self.query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
      
        if g_theta == float('inf'):    
            return x0, self.query_count

        # Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        vg = torch.zeros_like(xg)
        
        assert not self.svm
        for i in range(1500):
            sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)
            self.query_count += grad_queries
            # Line search
            min_theta = xg
            min_g2 = gg
            min_vg = vg
            for _ in range(15):
                if self.momentum > 0:
                    new_vg = self.momentum*vg - alpha*sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta /= torch.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                self.query_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    if self.momentum > 0:
                        min_vg = new_vg
                else:
                    break
            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    if self.momentum > 0:
                        new_vg = self.momentum*vg - alpha*sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    self.query_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        if self.momentum > 0:
                            min_vg = new_vg
                        break
            if alpha < 1e-4:
                alpha = 1.0
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            xg, gg = min_theta, min_g2
            vg = min_vg
            

            if self.query_count > self.max_queries:
               break

            dist = self.distance(gg*xg)
            if dist < self.epsilon:
                break

        dist = self.distance(gg*xg)
        return x0 + gg*xg, self.query_count

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = torch.zeros_like(theta)
        queries = 0
        for _ in range(K):          
            u = torch.randn_like(theta)
            u /= torch.norm(u)
            
            sign = 1
            new_theta = theta + h*u
            new_theta /= torch.norm(new_theta)
            
            # Targeted case.
            if (target is not None and 
                self.predict_label(x0+initial_lbd*new_theta) == target):
                sign = -1
                
            # Untargeted case
            if (target is None and
                self.predict_label(x0+t(initial_lbd*new_theta)) != y0):
                sign = -1
            queries += 1
            sign_grad += u*sign
        
        sign_grad /= K    
        
        return sign_grad, queries

    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        if self.predict_label(x0+lbd*theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.predict_label(x0+lbd_hi*theta) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.predict_label(x0+lbd_lo*theta) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
                if nquery + self.query_count> self.max_queries:
                    break

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if nquery + self.query_count> self.max_queries:
                break
            if self.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if self.predict_label(x0+t(current_best*theta)) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if nquery + self.query_count> self.max_queries:
                break
            if self.predict_label(x0 + t(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search_local_targeted(self, x0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.predict_label(x0 + t(lbd*theta)) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.predict_label(x0 + t(lbd_hi*theta)) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100: 
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.predict_label(x0 + t(lbd_lo*theta)) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.predict_label(x0 + t(lbd_mid*theta)) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, x0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if self.predict_label(x0 + t(current_best*theta)) != t:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.predict_label(x0 + t(lbd_mid*theta)) != t:
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
        return lbd_hi, nquery


    def _perturb(self, xs_t, ys):
        if self.targeted:
            adv, q = self.attack_targeted(xs_t, ys, self.alpha, self.beta)
        else:
            adv, q = self.attack_untargeted(xs_t, ys, self.alpha, self.beta)

        return adv, q

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
    tail = Tail(num_classes)
    tail.load_state_dict(torch.load(f'{model_dir}/base_tail_state_dict'))
    tail.to(device)

    # load the classifiers
    classifiers = []
    models = []
    for i in range(args.num_models):
        head = Head()
        head.to(device)
        head.load_state_dict(torch.load(f'{model_dir}/head_{i}/state_dict'))
        watermark = Watermark.load(f'{model_dir}/head_{i}/watermark.npy')

        models.append(nn.Sequential(normalizer, watermark, head, tail).eval())
        models[-1].to(device)
        
        classifier = PyTorchClassifier(
            model = models[-1],
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

    for X, y in tqdm(testing_loader):
        with torch.no_grad():
            model_index = np.random.choice(np.arange(args.num_models), k, replace=False).astype(int)
            mask_k = np.ones_like(y.numpy())
            X_attacked_k = []
            for model_id in model_index:
                model, c = models[model_id], classifiers[model_id]
                pred = c.predict(X.numpy())
                correct_mask = pred.argmax(axis = -1) == y.numpy()

                X_device, y_device = X.to(device), y.to(device)

                a = SignOPTAttack(epsilon = 1, p = '2', alpha = 0.2, beta = 0.001, svm = False, momentum = 0, max_queries = 10000, k = 200, lb = 0, ub = 1, batch_size = 1, sigma = 0)
                X_attacked = a.run(X_device, y_device, model, False, None).cpu().numpy()

                attacked_preds = np.vectorize(lambda z: z.predict(X_attacked), signature = '()->(m,n)')(classifiers)
                
                success_mask = attacked_preds.argmax(axis = -1) != y.numpy()
                success_mask = np.logical_and(success_mask[model_id], success_mask.sum(axis=0) >= 2)

                mask = np.logical_and(correct_mask, success_mask)
                mask_k = np.logical_and(mask_k, mask)
                if mask_k.sum() < 4:
                    print("run continue due to lack of successful attacks")
                    continue
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
                    print(f'SignOPT, {count_success} out of {args.num_samples} generated, done!')
                    break
                else:
                    print(f'SignOPT, {count_success} out of {args.num_samples} generated...')

            else:
                print('not generated!')
    original_images = np.concatenate(original_images)
    attacked_images = np.concatenate(attacked_images, axis=1)
    labels = np.concatenate(labels)
    os.makedirs(f'{save_dir}/{k}_attackers', exist_ok = True)
    np.savez(f'{save_dir}/{k}_attackers/SignOPT_{args.num_samples}_num_of_samples.npz', X = original_images, X_attacked_k = attacked_images, y = labels, head=head)
