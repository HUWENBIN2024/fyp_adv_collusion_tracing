# fyp_adv_collusion_tracing
This is the repo for the final year project CMH1.

Firstly, change direction to adv_tracing.
```
cd adv_tracing
```

## Baseline in ICML paper
Evaluate transferability of adv attacks:
```
python adv_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px
```

## Collusion Adv Attack
Generate collusion adversarial samples:
```
python -m collusion.nes --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 200 -b 128
```

Collusion evalution:
```
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack mean
```
For details of the arguments, please go to see the desciption in related py flies.