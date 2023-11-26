# fyp_adv_collusion_tracing
This is the repo for the final year project CMH1.

Evaluate transferability of adv attacks:
```
python adv_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px
```


Generate collusion adversarial samples:
```
python -m collusion.nes --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 200
```
