# SGD
## ConvMNIST
export CUDA_VISIBLE_DEVICES=0
python main.py --batch_size 128 --lr 1e-5 --epochs 20 --times 5 --data_root data --num_workers 4 --net ConvMNIST --af all --optim SGD --cuda --exname AFP
python main.py --batch_size 128 --lr 1e-4 --epochs 20 --times 5 --data_root data --num_workers 4 --net ConvMNIST --af all --optim SGD --cuda --exname AFP
python main.py --batch_size 128 --lr 1e-3 --epochs 20 --times 5 --data_root data --num_workers 4 --net ConvMNIST --af all --optim SGD --cuda --exname AFP
python main.py --batch_size 128 --lr 1e-2 --epochs 20 --times 5 --data_root data --num_workers 4 --net ConvMNIST --af all --optim SGD --cuda --exname AFP

# # ADAM
# ## ConvMNIST
# export CUDA_VISIBLE_DEVICES=1
# python main.py --batch_size 128 --lr 1e-5 --epochs 20 --times 5 --data_root data --num_workers 4 --net ConvMNIST --af all --optim Adam --cuda --exname AFP
# python main.py --batch_size 128 --lr 1e-4 --epochs 20 --times 5 --data_root data --num_workers 4 --net ConvMNIST --af all --optim Adam --cuda --exname AFP
# python main.py --batch_size 128 --lr 1e-3 --epochs 20 --times 5 --data_root data --num_workers 4 --net ConvMNIST --af all --optim Adam --cuda --exname AFP
# python main.py --batch_size 128 --lr 1e-2 --epochs 20 --times 5 --data_root data --num_workers 4 --net ConvMNIST --af all --optim Adam --cuda --exname AFP