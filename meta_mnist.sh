###
 # @Descripttion: densechen@foxmail.com
 # @version: 0.0
 # @Author: Dense Chen
 # @Date: 1970-01-01 08:00:00
 # @LastEditors: Dense Chen
 # @LastEditTime: 2020-09-26 19:47:35
### 
mkdir logs51
export CUDA_VISIBLE_DEVICES=0
for act in "APL" "AReLU" "GELU" "Maxout" "Mixture" "SLAF" "Swish" "ReLU" "ReLU6" "Sigmoid" "LeakyReLU" "ELU" "PReLU" "SELU" "Tanh" "RReLU" "CELU" "Softplus" "PAU"; do
    echo $act
    python meta_mnist.py --afs $act --iterations 100 --waygs 5 --shots 1 > logs51/$act.log
done 

mkdir logs55
export CUDA_VISIBLE_DEVICES=0
for act in "APL" "AReLU" "GELU" "Maxout" "Mixture" "SLAF" "Swish" "ReLU" "ReLU6" "Sigmoid" "LeakyReLU" "ELU" "PReLU" "SELU" "Tanh" "RReLU" "CELU" "Softplus" "PAU"; do
    echo $act
    python meta_mnist.py --afs $act --iterations 100 --ways 5 --shots 5 > logs55/$act.log
done 