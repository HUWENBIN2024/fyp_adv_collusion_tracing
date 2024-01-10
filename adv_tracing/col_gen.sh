CUDA_VISIBLE_DEVICES=0 python -m collusion.nes --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=1 python -m collusion.hsj --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=2 python -m collusion.bandit --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=3 python -m collusion.signopt --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=4 python -m collusion.simba --model_name ResNet18 --dataset_name CIFAR10 -k 2 -n 1000 -b 10 &

CUDA_VISIBLE_DEVICES=5 python -m collusion.nes --model_name ResNet18 --dataset_name CIFAR10 -k 3 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=6 python -m collusion.hsj --model_name ResNet18 --dataset_name CIFAR10 -k 3 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=7 python -m collusion.bandit --model_name ResNet18 --dataset_name CIFAR10 -k 3 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=0 python -m collusion.signopt --model_name ResNet18 --dataset_name CIFAR10 -k 3 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=1 python -m collusion.simba --model_name ResNet18 --dataset_name CIFAR10 -k 3 -n 1000 -b 10 &

CUDA_VISIBLE_DEVICES=2 python -m collusion.nes --model_name ResNet18 --dataset_name CIFAR10 -k 4 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=3 python -m collusion.hsj --model_name ResNet18 --dataset_name CIFAR10 -k 4 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=4 python -m collusion.bandit --model_name ResNet18 --dataset_name CIFAR10 -k 4 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=5 python -m collusion.signopt --model_name ResNet18 --dataset_name CIFAR10 -k 4 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=6 python -m collusion.simba --model_name ResNet18 --dataset_name CIFAR10 -k 4 -n 1000 -b 10 &

CUDA_VISIBLE_DEVICES=7 python -m collusion.nes --model_name ResNet18 --dataset_name CIFAR10 -k 5 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=0 python -m collusion.hsj --model_name ResNet18 --dataset_name CIFAR10 -k 5 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=1 python -m collusion.bandit --model_name ResNet18 --dataset_name CIFAR10 -k 5 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=2 python -m collusion.signopt --model_name ResNet18 --dataset_name CIFAR10 -k 5 -n 1000 -b 10 &
CUDA_VISIBLE_DEVICES=3 python -m collusion.simba --model_name ResNet18 --dataset_name CIFAR10 -k 5 -n 1000 -b 10 &