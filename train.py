# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:28:24 2020

@author: dell
"""

import numpy as np
import torch as tc
from tqdm import tqdm
import os

from dataset import AugDataset#自己的增强数据
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from eval import eval_net#调用测试网络，计算准确率
from utils import anal_net, load_net, change_classifier, FocalLoss

import argparse


def train_net(net,
              device,
              data_dir,
              save_dir,
              epochs=100,
              batch_size=5,
              lr=1e-5,
              size=224,
              aug=3,
              opt_mode='Adam',
              sch_mode=False,
              loss_mode='CE'):
    train_dataset = AugDataset(os.path.join(data_dir, 'train'), size=size, aug=aug)
    test_dataset = AugDataset(os.path.join(data_dir, 'test'), val=True, size=size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_{save_dir}')
    global_step = 0
    
    print(f'''Start training:
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {lr}
        Training size: {len(train_dataset)}
        Validation size: {len(test_dataset)}
        Loss function: {loss_mode}
        Augmentation: {aug}
        Saved model: {save_dir}
        Device:{device}''')
    
    if opt_mode == 'Adam':
        opt = tc.optim.Adam(net.parameters(), lr=lr)
    elif opt_mode == 'SGD':
        opt = tc.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise RuntimeError('Optimizer not defined')

    if sch_mode:
        lr_lambda = lambda epoch : np.power(0.5, int(epoch/(epochs//4)))
        scheduler = tc.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    else:
        scheduler = None

    if loss_mode == 'CE':
        criterion = tc.nn.CrossEntropyLoss()
    elif loss_mode == 'FC':
        criterion = FocalLoss()
    else:
        raise RuntimeError('Criterion not defined')

    best_acc = 0
    for epoch in range(epochs):
        net.train()
        running_loss = 0
        if (scheduler is not None) and (epoch > 0):
            scheduler.step()
        with tqdm(total=len(train_dataset), desc=f'{epoch + 1}/epoch', unit='i') as pbar:
            for batch in train_loader:
                imgs, tars = batch
                
                imgs = imgs.to(device, dtype=tc.float32)
                tars = tars.to(device, dtype=tc.long)
                
                # features = net.features(imgs)
                # features = features.view(features.size(0), -1)
                outs = net(imgs)
                if not tc.is_tensor(outs):
                    loss1 = criterion(outs[0], tars)
                    loss2 = criterion(outs[1], tars)
                    loss = loss1 + 0.4 * loss2
                else:
                    loss = criterion(outs, tars)
                
                running_loss += loss.item()
                writer.add_scalar('Loss(step)/train', loss.item(), global_step)
                
                pbar.set_postfix(**{'ls': loss.item()})
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                pbar.update(imgs.shape[0])
                global_step += 1
        if epoch % 1 == 0:
            acc, val_loss = eval_net(net, test_loader, criterion, device, len(test_dataset))
            print(f'Validation accuracy {acc.item():.4f}')
            writer.add_scalar('Acc/val', acc.item(), epoch)
            writer.add_scalars('Loss(epoch)/tra&val', {'train': running_loss / (len(train_dataset) // batch_size),
                                                       'val': val_loss.item() / (len(test_dataset) // batch_size)},
                                                        epoch)
            if acc.item() > best_acc:
                print('save best model')
                try:
                    os.mkdir(save_dir)
                    print(f'Save folder {save_dir} created')
                except OSError:
                    pass
                tc.save(net.state_dict(), os.path.join(save_dir, f'best.pth'))
                best_acc = acc.item()
    writer.close()


def test_net(net,
             device,
             data_dir,
             plot_dir,
             class_names,
             batch_size=32,
             size=224):
    test_dataset = AugDataset(os.path.join(data_dir, 'test'), val=True, size=size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    anal_net(net, test_loader, class_names, plot_dir, device, len(test_dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('-df', dest='data_from', type=str, help='data folder',default='vgg16')
    #parser.add_argument('-st', dest='save_to', type=str, help='save_folder')
    parser.add_argument('-c', dest='classes', type=int, default=3,
                        help='Default is 3')
    parser.add_argument('-m', dest='model_name', type=str, default='vgg16',
                        help='Default is VGG16')
    parser.add_argument('-l', dest='loss_mode', type=str, default='CE',
                        help='CE-->Cross Entropy | FC-->Focal Loss, Default is Cross Entropy')
    parser.add_argument('-b', dest='batch_size', type=int, default=4,
                        help='Default is 32')
    parser.add_argument('-a', dest='augmenter', type=int, default=1,
                        help='Augmentation Index')
    parser.add_argument('-lr', dest='learning_rate', type=float, default=1e-4,
                        help='Default is 1e-4')
    parser.add_argument('-e', dest='epoch', type=int, default=100,
                        help='Default is 100')
    parser.add_argument('-o', dest='optimizer', type=str, default='Adam',
                        help='Adam | SGD, Default is Adam')
    parser.add_argument('-sc', dest='scheduler', action='store_true',
                        help='If use scheduler')
    args = parser.parse_args()
    
    data_from = args.data_from
    n_classes = args.classes
    model_name = args.model_name
    loss_mode = args.loss_mode
    aug = args.augmenter
    lr = args.learning_rate
    opt = args.optimizer
    sch_mode = args.scheduler
    
    if model_name == 'inception_v3':
        size = 299
    else:
        size = 224
    
    data_dir = f'./data'
    save_dir = f'./saved_models/{data_from}_{model_name}_{loss_mode}_aug{aug}_{opt}'
    device = tc.device('cuda')
    
    net = load_net(model_name, pretrain=True)
    net = change_classifier(net, model_name, n_classes=n_classes)
    #迁移学习模型的末端修改
    net.to(device)
    print("finish building network")
    
    train_net(net=net,
              device=device,
              data_dir=data_dir,
              save_dir=save_dir,
              epochs=args.epoch,
              batch_size=args.batch_size,
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

