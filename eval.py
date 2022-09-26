# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:35:38 2020

@author: dell
"""

import torch as tc
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def eval_net(net, loader, criterion, device, n_val):
    net.eval()
    targets = []
    predictions = []
    with tc.no_grad():
        correct = 0
        loss = 0
    
        with tqdm(total=n_val, desc='Val', unit='i', leave=False) as pbar:
            for batch in loader:
                imgs, tars = batch
                
                imgs = imgs.to(device, dtype=tc.float32)
                tars = tars.to(device, dtype=tc.long)
                
                # features = net.features(imgs)
                # features=features.view(features.size(0),-1)
                outs = net(imgs)
                if not tc.is_tensor(outs):
                    outs = outs[0]
                preds = tc.argmax(outs, dim=1)
                
                loss += criterion(outs, tars)
                correct += (preds == tars).sum()
                
                preds = preds.cpu().data.numpy().tolist()
                predictions += preds
                tars = tars.cpu().data.numpy().tolist()
                targets += tars
                
                pbar.update(imgs.shape[0])
    print(confusion_matrix(targets, predictions))
    return 100 * correct / n_val, loss