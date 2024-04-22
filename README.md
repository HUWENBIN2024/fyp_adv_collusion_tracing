# fyp_adv_collusion_tracing
This is the repo for our final year project.

## Baseline in ICML paper
To conduct baseline experiments, cd to `src/adv_tracing` and follow the instruction in the README file.  

To evaluate the transferability of adv attacks:
```
python adv_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px
```

## Collusive Adv Attack (Our FYP project)
Firstly, cd to `src/collusion_tracing`.   
To train a base model:  
```
python train_base_model.py --model_name ResNet18 --dataset_name CIFAR10
```
To train different model copies: (run this command for each head. we need 10 head for eache setting; argument needed to change: model_name, dataset_name, head_id)
```
CUDA_VISIBLE_DEVICES=0 python train_new_and_acc_code_one_head.py --model_name ResNet18 --dataset_name CIFAR10 --num_epochs 100 --batch_size 1024 --head_id 0
``` 
To generate collusive adversarial samples: (argument needed to change: model_name, dataset_name)
```
python -m collusion_attacks.nes --model_name ResNet18 --dataset_name CIFAR10 -k 2 --num_samples 500 -b 64 --num_models 10

```
To evaluate the transferability of the collusive attacks: 
```
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack mean
```
To calculate the tracing back accuracy:  
- Run `tracing_new.ipynb`
Arguments we need to modify in class *Agrs* are: {model_name, dataset_name, collusion_attack, num_obtained_adv, attack_name}
