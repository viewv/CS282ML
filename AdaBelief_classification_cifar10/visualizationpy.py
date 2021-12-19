#!/usr/bin/env python
# coding: utf-8

# # Visualization of training and test curves with different optimizers
# This notebook is modified from https://github.com/Luolc/AdaBound/blob/master/demos/cifar10/visualization.ipynb.
# We compare the performace of AdaBelief optimizer and 8 other optimizers (SGDM, AdaBound, Yogi, Adam, MSVAG, RAdam, AdamW, Fromage).
# The training setting is the same as the official implementation of AdaBound: https://github.com/Luolc/AdaBound,
# hence we exactly reproduce the results of AdaBound.
# AdaBound is claimed to achieve "fast convergence and good generalization", and in this project we will show that AdaBelief outperforms AdaBound and other optimizers.

# In[1]:


import os
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import torch
import numpy as np

params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
         }
plt.rcParams.update(params)


# In[2]:


def get_data(names):
    folder_path = './curve1'
    paths = [os.path.join(folder_path, name) for name in names]
    return {name: torch.load(fp) for name, fp in zip(names, paths)}

def plot(names, curve_type='train', labels = None, ylim=(80,101), loc = 'upper left'):
    plt.figure()
    plt.ylim(ylim)# if curve_type == 'train' else 96)
    curve_data = get_data(names)
    for i, label in zip(curve_data.keys(),labels):
        acc = np.array(curve_data[i]['{}_acc'.format(curve_type.lower())])
        if label == 'AdaBelief':
            plt.plot(acc, '-', label=label)
        else:
            plt.plot(acc, '--',label = label)
    
    plt.grid()
    plt.legend(fontsize=14, loc=loc)
    plt.title('{} accuracy ~ Training epoch'.format(curve_type))
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy')
    plt.show()


# # ResNet
# Plot the training and test curves for all optimizers in one plot. "names" is a list containing the log files in "/curve" folder, "labels" is the corresponding legends for different optimizers. Note that "names" and "labels" must match (log for the i-th element in "labels" in the i-th element in "names")

# In[3]:


names = ['resnet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse',
         'resnet-sgd-lr0.1-momentum0.9-wdecay0.0005-run0-resetFalse',
         'resnet-adabound-lr0.001-betas0.9-0.999-final_lr0.1-gamma0.001-wdecay0.0005-run0-resetFalse',
         'resnet-yogi-lr0.001-betas0.9-0.999-eps0.001-wdecay0.0005-run0-resetFalse',
         'resnet-adam-lr0.001-betas0.9-0.999-wdecay0.0005-eps1e-08-run0-resetFalse',
         'resnet-msvag-lr0.1-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse',
         'resnet-radam-lr0.001-betas0.9-0.999-wdecay0.0005-eps1e-08-run0-resetFalse',
         'resnet-adamw-lr0.001-betas0.9-0.999-wdecay0.01-eps1e-08-run0-resetFalse',
         'resnet-fromage-lr0.01-betas0.9-0.999-wdecay0.0005-eps1e-08-run0-resetFalse',
]
labels = ['AdaBelief',
          'SGD',
          #'AdaBound',
         # 'Yogi',
          'Adam',
         # 'MSVAG',
         # 'RAdam',
         # 'AdamW',
         # 'Fromage',
        ]
plot(names, 'Train', labels)
plot(names, 'Test', labels, ylim = (88,96))     


# # DenseNet

# In[ ]:


names = ['densenet-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse',
         'densenet-sgd-lr0.1-momentum0.9-wdecay0.0005-run0-resetFalse',
         'densenet-adabound-lr0.001-betas0.9-0.999-final_lr0.1-gamma0.001-wdecay0.0005-run0-resetFalse',
         'densenet-yogi-lr0.001-betas0.9-0.999-eps0.001-wdecay0.0005-run0-resetFalse',
         'densenet-adam-lr0.001-betas0.9-0.999-wdecay0.0005-eps1e-08-run0-resetFalse',
         'densenet-msvag-lr0.1-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse',
         'densenet-radam-lr0.001-betas0.9-0.999-wdecay0.0005-eps1e-08-run0-resetFalse',
         'densenet-adamw-lr0.001-betas0.9-0.999-wdecay0.01-eps1e-08-run0-resetFalse',
         'densenet-fromage-lr0.01-betas0.9-0.999-wdecay0.0005-eps1e-08-run0-resetFalse',
]
labels = ['AdaBelief',
          'SGD',
          'AdaBound',
          'Yogi',
          'Adam',
          'MSVAG',
          'RAdam',
          'AdamW',
          'Fromage',
        ]
plot(names, 'Train', labels)
plot(names, 'Test', labels, ylim = (88,96))     


# ## VGG Network

# In[ ]:


names = ['vgg-adabelief-lr0.001-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse',
         'vgg-sgd-lr0.1-momentum0.9-wdecay0.0005-run0-resetFalse',
         'vgg-adabound-lr0.001-betas0.9-0.999-final_lr0.1-gamma0.001-wdecay0.0005-run0-resetFalse',
         'vgg-yogi-lr0.001-betas0.9-0.999-eps0.001-wdecay0.0005-run0-resetFalse',
         'vgg-adam-lr0.001-betas0.9-0.999-wdecay0.0005-eps1e-08-run0-resetFalse',
         'vgg-msvag-lr0.1-betas0.9-0.999-eps1e-08-wdecay0.0005-run0-resetFalse',
         'vgg-radam-lr0.001-betas0.9-0.999-wdecay0.0005-eps1e-08-run0-resetFalse',
         'vgg-adamw-lr0.001-betas0.9-0.999-wdecay0.01-eps1e-08-run0-resetFalse',
         'vgg-fromage-lr0.01-betas0.9-0.999-wdecay0.0005-eps1e-08-run0-resetFalse',
]
labels = ['AdaBelief',
          'SGD',
          'AdaBound',
          'Yogi',
          'Adam',
          'MSVAG',
          'RAdam',
          'AdamW',
          'Fromage',
        ]
plot(names, 'Train', labels)
plot(names, 'Test', labels, ylim = (84,92))     

