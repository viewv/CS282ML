
We modified the code from https://github.com/juntang-zhuang/Adabelief-Optimizer/tree/update_0.2.0/PyTorch_Experiments/classification_cifar10

we simply modified the code to reappearance:
(a): SGD, Adam, and AdaBelief with ResNet34 on Cifar10
(b): Adam, and AdaBelief with DenseNet121  on Cifar10

### Dependencies
python 3.7
pytorch 1.10.0      % here is different with the author's code 
jupyter notebook
AdaBound  (Please instal by "pip install adabound")



### Visualization of pre-trained curves
Please use the jupyter notebook "visualization.ipynb" to visualize the training and test curves of different optimizers. We provide logs for pre-trained models in the folder "curve".



### Training and evaluation code

(1) train network with
CUDA_VISIBLE_DEVICES=0 python main.py --optim adabelief --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9

--optim: name of optimizers, choices include ['sgd', 'adam',  'adabelief',]
--lr: learning rate
--eps: epsilon value used for optimizers. Note that Yogi uses a default of 1e-03, other optimizers typically uses 1e-08
--beta1, --beta2: beta values in adaptive optimizers
--momentum: momentum used for SGD.s

(2) visualize using the notebook "visualization.ipynb"
