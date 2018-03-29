#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export PYTHONPATH=/home/tusimple/Hongbin/incubator-mxnet/python

# you'd better change setting with your own --data-dir, --depth, --batch-size, --gpus.
# train cifar10

#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 12 --num-examples 50000 --gpus=0,1,2 2>>resnet-164-3GPU-local.log


#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 12 --num-examples 50000 --gpus=0 2>>resnet-164-1GPU.log


#python -u train_resnet.py --global-BN True --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 6 --num-examples 50000 --gpus=1 2>>./log/resnet-20-6-1GPU.log


#python -u train_resnet.py --global-BN True --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 6 --num-examples 50000 --gpus=0,1,2 2>>./log/resnet-20-6-3GPU-global.log


#python -u train_resnet.py  --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 6 --num-examples 50000 --gpus=0,1,2 2>>./log/resnet-20-6-3GPU-local.log


#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 16 --num-examples 50000 --gpus=0,1,2,3 2>>./log/resnet-20-16-4GPU.log
#python -u train_resnet.py --data-dir /mnt/truenas/scratch/tianqi.tang/data  --data-type imagenet --depth 50 --batch-size 32 --num-examples 50000 --gpus=0

python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 256 --num-examples 50000 --gpus=0  2>>./log/resnet-20-256-1GPU.log
python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 256 --num-examples 50000 --gpus=0,1,2,3 --num-gpus=4 --global-bn 2>>./log/resnet-20-256-4GPU-global.log
python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 256 --num-examples 50000 --gpus=0,1,2,3,4,5,6,7 --num-gpus=8 --global-bn 2>>./log/resnet-20-256-8GPU-global.log
python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 256 --num-examples 50000 --gpus=0,1,2,3 2>>./log/resnet-20-256-4GPU-local.log
python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 256 --num-examples 50000 --gpus=0,1,2,3,4,5,6,7 2>>./log/resnet-20-256-8GPU-local.log


#python -u train_resnet.py  --data-dir data/cifar10 --data-type cifar10 --depth 20 --batch-size 3 --num-examples 50000 --gpus=0,1,2 2>>./log/resnet-20-3-3GPU-local.log










#python -u train_resnet.py --global-BN True --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 12 --num-examples 50000 --gpus=0,1,2 2>>resnet-164-12-3GPU-global.log


#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 128 --num-examples 50000 --gpus=2,3,4,5,6,7

## train resnet-50
#python -u train_resnet.py --data-dir data/imagenet --data-type imagenet --depth 50 --batch-size 256 --gpus=0,1,2,3
