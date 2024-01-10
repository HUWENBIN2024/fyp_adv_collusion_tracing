# CUDA_VISIBLE_DEVICES=3 python -m collusion.nes --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 100 -b 10 &
# CUDA_VISIBLE_DEVICES=4 python -m collusion.hsj --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 100 -b 10 &
# CUDA_VISIBLE_DEVICES=5 python -m collusion.bandit --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 100 -b 10 &
# CUDA_VISIBLE_DEVICES=6 python -m collusion.signopt --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 100 -b 10 &
# CUDA_VISIBLE_DEVICES=7 python -m collusion.simba --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 100 -b 10

python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack mean 
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack max
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack min
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack median
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack negative
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name NES --collusion_attack negative_prob

python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name Bandit --collusion_attack mean
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name Bandit --collusion_attack max
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name Bandit --collusion_attack min
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name Bandit --collusion_attack median
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name Bandit --collusion_attack negative
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name Bandit --collusion_attack negative_prob

python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SignOPT --collusion_attack mean
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SignOPT --collusion_attack max
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SignOPT --collusion_attack min
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SignOPT --collusion_attack median
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SignOPT --collusion_attack negative
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SignOPT --collusion_attack negative_prob

python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px --collusion_attack mean
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px --collusion_attack max
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px --collusion_attack min
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px --collusion_attack median
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px --collusion_attack negative
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name SimBA-px --collusion_attack negative_prob

python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name HopSkipJump --collusion_attack mean
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name HopSkipJump --collusion_attack max
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name HopSkipJump --collusion_attack min
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name HopSkipJump --collusion_attack median
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name HopSkipJump --collusion_attack negative
python collusion_evl.py --model_name ResNet18 --dataset_name CIFAR10 --attack_name HopSkipJump --collusion_attack negative_prob
