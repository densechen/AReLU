# AFS
# MNIST
export CUDA_VISIBLE_DEVICES=0
python main.py --batch_size 128 --lr 1e-5 --epochs 20 --times 5 --data_root data --dataset MNIST --num_workers 2 --net ConvMNIST --af all --optim SGD --exname AFS
python main.py --batch_size 128 --lr 1e-4 --epochs 20 --times 5 --data_root data --dataset MNIST --num_workers 2 --net ConvMNIST --af all --optim SGD --exname AFS
# SVHN
python main.py --batch_size 128 --lr 1e-5 --epochs 20 --times 5 --data_root data --dataset SVHN --num_workers 2 --net ConvMNIST --af all --optim SGD --exname AFS
python main.py --batch_size 128 --lr 1e-4 --epochs 20 --times 5 --data_root data --dataset SVHN --num_workers 2 --net ConvMNIST --af all --optim SGD --exname AFS

# Transfer Learning
# MNIST -> SVHN
python main.py --batch_size 128 --lr 1e-2 --lr_aux 1e-5 --epochs 20 --epochs_aux 200 --times 5 --data_root data --dataset MNIST --dataset_aux SVHN --num_workers 2 --net ConvMNIST --af all --optim SGD --exname TransferLearning
python main.py --batch_size 128 --lr 1e-2 --lr_aux 1e-5 --epochs 20 --epochs_aux 200 --times 5 --data_root data --dataset SVHN --dataset_aux MNIST --num_workers 2 --net ConvMNIST --af all --optim SGD --exname TransferLearning
