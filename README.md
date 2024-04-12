# fyp_adv_collusion_tracing
This is the repo for the final year project.

## Baseline in ICML paper
To conduct baseline experiments, cd to src/adv_tracing and follow the instruction in the README file.  

To evaluate the transferability of adv attacks:
```
python adv_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px
```

## Collusive Adv Attack
Firstly, cd to src/collusion_tracing.   
To train a base model:  
```
python train_base_model.py --model_name ResNet18 --dataset_name CIFAR10
```
To train different model copies:  
```
CUDA_VISIBLE_DEVICES=0 python train.py --model_name ResNet18 --dataset_name CIFAR10 --num_pixel_watermarked 300
``` 
To generate collusive adversarial samples:
```
python -m collusion_attacks.nes --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 1000 -b 128
```
To evaluate the transferability of the collusive attacks:
```
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack mean
```
To calculate the tracing back accuracy:
```
python collusion_trace_acc.py --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 1000 --attack_name NES --collusion_attack mean -M 100
```

