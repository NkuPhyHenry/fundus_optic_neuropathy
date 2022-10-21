# -*- coding: utf-8 -*-
'''
main.py is the key, and you can change the parameters of it to test what you want.
'''

import numpy as np
import torch as tc
from tqdm import tqdm
import os
import argparse
from dataset import AugDataset, OriginDataset  # 自己的增强数据
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from eval import eval_net  # 调用测试网络，计算准确率
from utils import anal_net, load_net, change_classifier, FocalLoss, interp_net
from train import *
from dataset import PlainDataset
from train import *

os.environ['TORCH_HOME']=r'D:\anaconda\models'

if __name__ == '__main__':

    n_classes = 3
    class_name=['Normal', 'Atrophy', 'Edema']
    model_name = 'vgg16'
    '''
     ['vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn',
                  'resnet34', 'resnet50', 'resnet101', 'resnet152',
                  'inception_v3',
                  'densenet161', 'densenet201',
                  'resnext50', 'resnext101',
                  'wide_resnet50', 'wide_resnet101']
    '''
    loss_mode = 'CE'
    '''CE-->Cross Entropy | FC-->Focal Loss, Default is Cross Entropy'''
    aug = 3
    '''aug[1,2,3,4]'''
    lr = 1e-4
    '''[1e-3,1e-4,1e-5]'''
    opt = 'Adam'
    '''Adam | SGD, Default is Adam'''
    epoch = 100
    batch_size = 4
    sch_mode = False


    data_dir = f'./data/data1012'
    save_dir = f'./saved_models/{model_name}_{loss_mode}_aug{aug}_{opt}'


    if model_name == 'inception_v3':
        size = 299
    else:
        size = 224

    print("Loading the pretrained model")
    net = load_net(model_name, pretrain=True)
    net = change_classifier(net, model_name, n_classes=n_classes)
    device = tc.device('cuda')
    net.to(device)

    train_net(net=net,
              device=device,
              data_dir=data_dir,
              save_dir=save_dir,
              epochs=epoch,
              batch_size=batch_size,
              size=size,
              aug=aug,
              loss_mode=loss_mode,
              opt_mode=opt,
              lr=lr,
              sch_mode=sch_mode)

    print("Loading best model")
    net = load_net(model_name, pretrain=True)
    net = change_classifier(net, model_name, n_classes=n_classes)
    net.load_state_dict(tc.load(os.path.join(save_dir, 'best.pth')))
    net.to(device)

    test_net(net=net,
             device=device,
             data_dir=data_dir,
             plot_dir=save_dir,
             class_names=['Normal', 'Atrophy', 'Edema'])

    print('generating cams')
    test_name=[1,2,3]
    for i in test_name:
        data_dir_test = os.path.join(data_dir, f'test/{i}')
        save_pic=os.path.join('cams',save_dir,f'{i}')
        save_pic2 = os.path.join('ggcams', save_dir,f'{i}')
        dataset = PlainDataset(data_dir_test, size=size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        interp_net(net,dataloader, class_name,save_pic,save_pic2,device)


