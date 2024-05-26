# fyp_adv_collusion_tracing
This is the repo for our final year project on "
Identification of Collusive Adversaries from a Single Adversarial Example via Machine Learning Watermarks", supervised by Professor Cheng Minhao.

## Baseline in paper "Identification of the Adversary from a Single Adversarial Example"
To conduct baseline experiments, cd to `src/adv_tracing` and follow the instructions in the README file.  

To evaluate the transferability of adv attacks:
```
python adv_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px
```

## Collusive Adv Attack (Our FYP project)
Firstly, cd to `src/collusion_tracing`.   
To train a base model, run: 
```
python train_base_model.py --model_name ResNet18 --dataset_name CIFAR10
```
To train different model copies, run: (You may change: model_name, dataset_name, head_id, we need 10 heads for each setting)
```
python train_new_and_acc_code_one_head.py --model_name ResNet18 --dataset_name CIFAR10 --num_epochs 100 --batch_size 1024 --head_id 0
``` 
To generate collusive adversarial samples, run: (You may change model_name and dataset_name)
```
python -m collusion_attacks.nes --model_name ResNet18 --dataset_name CIFAR10 -k 2 --num_samples 500 -b 64 --num_models 10

```
To evaluate the transferability of the collusive attacks, run:
```
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack mean
```
To calculate the collusive tracing accuracy， run `tracing_new.ipynb`
You may modify the arguments in class *Agrs* with your {model_name, dataset_name, collusion_attack, num_obtained_adv, attack_name}
