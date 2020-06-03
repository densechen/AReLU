# DO PRETRAIN
python main.py --batch_size 128 --lr 0.01 --epochs 1 --times 1 --data_root data --dataset MNIST --num_workers 4 --net ConvMNIST --af all --optim SGD --cuda --exname TransferLearningPretrain

# FINETUNE
python main.py --batch_size 128 --lr 0.00001 --epochs 1 --times 1 --data_root data --dataset SVHN --num_workers 4 --net ConvMNIST --af all --optim SGD --cuda --exname TransferLearningFinetune --resume pathtopretrain
