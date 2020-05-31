# SGD
## ConvMNIST
export CUDA_VISIBLE_DEVICES=0
python main.py --batch_size 128 --lr 1e-5 --epochs 50 --times 5 --data_root data --num_workers 8 --net ConvMNIST --af all --optim SGD --cuda --exname SGD
python main.py --batch_size 128 --lr 1e-4 --epochs 50 --times 5 --data_root data --num_workers 8 --net ConvMNIST --af all --optim SGD --cuda --exname SGD
python main.py --batch_size 128 --lr 1e-3 --epochs 50 --times 5 --data_root data --num_workers 8 --net ConvMNIST --af all --optim SGD --cuda --exname SGD
python main.py --batch_size 128 --lr 1e-2 --epochs 50 --times 5 --data_root data --num_workers 8 --net ConvMNIST --af all --optim SGD --cuda --exname SGD

## LinearMNIST
python main.py --batch_size 128 --lr 1e-5 --epochs 50 --times 5 --data_root data --num_workers 8 --net LinearMNIST --af all --optim SGD --cuda --exname SGD
python main.py --batch_size 128 --lr 1e-4 --epochs 50 --times 5 --data_root data --num_workers 8 --net LinearMNIST --af all --optim SGD --cuda --exname SGD
python main.py --batch_size 128 --lr 1e-3 --epochs 50 --times 5 --data_root data --num_workers 8 --net LinearMNIST --af all --optim SGD --cuda --exname SGD
python main.py --batch_size 128 --lr 1e-2 --epochs 50 --times 5 --data_root data --num_workers 8 --net LinearMNIST --af all --optim SGD --cuda --exname SGD

# ADAM
## ConvMNIST
# export CUDA_VISIBLE_DEVICES=1
python main.py --batch_size 128 --lr 1e-5 --epochs 50 --times 5 --data_root data --num_workers 8 --net ConvMNIST --af all --optim Adam --cuda --exname Adam
python main.py --batch_size 128 --lr 1e-4 --epochs 50 --times 5 --data_root data --num_workers 8 --net ConvMNIST --af all --optim Adam --cuda --exname Adam
python main.py --batch_size 128 --lr 1e-3 --epochs 50 --times 5 --data_root data --num_workers 8 --net ConvMNIST --af all --optim Adam --cuda --exname Adam
python main.py --batch_size 128 --lr 1e-2 --epochs 50 --times 5 --data_root data --num_workers 8 --net ConvMNIST --af all --optim Adam --cuda --exname Adam

## LinearMNIST
python main.py --batch_size 128 --lr 1e-5 --epochs 50 --times 5 --data_root data --num_workers 8 --net LinearMNIST --af all --optim Adam --cuda --exname Adam
python main.py --batch_size 128 --lr 1e-4 --epochs 50 --times 5 --data_root data --num_workers 8 --net LinearMNIST --af all --optim Adam --cuda --exname Adam
python main.py --batch_size 128 --lr 1e-3 --epochs 50 --times 5 --data_root data --num_workers 8 --net LinearMNIST --af all --optim Adam --cuda --exname Adam
python main.py --batch_size 128 --lr 1e-2 --epochs 50 --times 5 --data_root data --num_workers 8 --net LinearMNIST --af all --optim Adam --cuda --exname Adam