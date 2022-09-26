# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:13:58 2020

@author: dell
"""

from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import classification_report, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

import seaborn as sn
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import matplotlib
from time import time
import numpy as np
from torchvision.models.resnet import resnext50_32x4d
from tqdm import tqdm
import torch as tc
import torch.nn.functional as F
import torchvision.models as models
from itertools import cycle
import os
from PIL import Image, ImageDraw, ImageFont
matplotlib.rcParams['font.family'] = 'SimHei'

from grad_CAM import GradCAM
from guided_bp import GuidedBackPropagation


class FocalLoss(tc.nn.Module):
    def __init__(self, degree=1, alpha=None):
        super(FocalLoss, self).__init__()
        self.degree = degree
        self.alpha = alpha

    def forward(self, p, y):
        if not tc.is_tensor(y):
            raise ValueError("Target is not a tensor object")
        if not tc.is_tensor(p):
            raise ValueError("Prediction is not a tensor object")
        if not y.shape[0] == p.shape[0]:
            raise ValueError("Target and Prediction have different number of samples")
        if y.device != p.device:
            raise ValueError("Target and Prediction have different devices")
        y = y.long().view(-1)
        p = p.float()
        device = y.device

        if self.alpha is None:
            self.alpha = tc.ones(p.shape[1],).float().to(device=device)
        else:
            self.alpha = tc.tensor(self.alpha).float().view(-1).to(device=device)

        log_p = F.log_softmax(p, dim=1)
        p = tc.exp(log_p)
        
        return F.nll_loss(
            ((1 - p) ** self.degree) * log_p, target=y,
            weight=self.alpha,
            reduction = 'mean')


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, tc.nn.Conv2d):
            layer_name = name
    return layer_name


def anal_net(net, loader, names, save_dir, device, n_test):
    net.eval()#测试样本
    targets = []
    logits = []
    predictions = []
    with tc.no_grad():
        with tqdm(total=n_test, desc='Test', unit='i', leave=False) as pbar:
            t0 = time()
            for batch in loader:
                imgs, tars = batch
                
                imgs = imgs.to(device, dtype=tc.float32)
                tars = tars.to(device, dtype=tc.long)
                
                outs = net(imgs)
                if not tc.is_tensor(outs):
                    outs = outs[0]
                preds = tc.argmax(outs, dim=1)
                
                targets += tars.cpu().numpy().tolist()
                logits += outs.cpu().numpy().tolist()
                predictions += preds.cpu().numpy().tolist()
                
                pbar.update(imgs.shape[0])
            print(f'Done in {(time() - t0):.3f}s')
    cm = confusion_matrix(targets, predictions)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    report_txt = open(os.path.join(save_dir, 'report.txt'), 'w')
    #print(targets, predictions)
    report_txt.writelines(classification_report(targets, predictions,
                                                target_names=names))
    logits = np.array(logits)
    # print(f'logits shape:{logits.shape}\n')
    report_txt.writelines(str(cm))
    print('report text saved')
    report_txt.close()
    plot_cm(cm, names, save=save_dir)
    plot_roc(targets, logits, names, save=save_dir)
    plot_pr(targets, logits, names, save=save_dir)
    return True


def pred_net(net, loader, names, save_pic, device):
    if not os.path.exists(save_pic):
        os.makedirs(save_pic)
    #if not os.path.exists(save_log):
        #os.makedirs(save_log)
    net.eval()
    predictions = []
    with tc.no_grad():
        for batch in tqdm(loader):
            img, raw, fn = batch
            
            img = img.to(device, dtype=tc.float32)
            out = net(img)
            if not tc.is_tensor(out):
                out = out[0]
            pred = tc.argmax(out, dim=1).squeeze().cpu().numpy()
            raw = raw.squeeze(0).numpy().astype(np.uint8)
            image = Image.fromarray(raw)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", max(int(raw.shape[0]*0.1), 20), encoding="utf-8")
            draw.text((0, 0), names[pred], (255, 0, 0), font=font)
            image.save(os.path.join(save_pic, fn[0]))
            predictions.append(pred)
    #np.save(os.path.join(save_log, 'logits.npy'), np.array(predictions))
    return True


def interp_net(net, loader, names, save_pic, save_pic2, device):
    net.eval()
    if not os.path.exists(save_pic):
        os.makedirs(save_pic)
    if not os.path.exists(save_pic2):
        os.makedirs(save_pic2)
    for batch in tqdm(loader):
        # read images
        img, raw, fn = batch
        img = img.to(device, dtype=tc.float32)
        
        # Grad CAM
        grad_cam = GradCAM(net, get_last_conv_name(net))
        cam, pred = grad_cam(img)
        cam = np.uint8(cam * 255)
        raw = raw.squeeze(0).numpy().astype(np.uint8)
        height, width, _ = raw.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        result = 0.3 * heatmap + 0.5 * raw
        image = Image.fromarray(np.uint8(result))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", max(int(raw.shape[0]*0.1), 20))
        draw.text((0, 0), names[pred], (255, 0, 0), font=font)
        image.save(os.path.join(save_pic, fn[0]))
        
        # Guided Grad CAM
        gbp = GuidedBackPropagation(net)
        guided_bp = gbp(img)
        ggcam = cv2.resize(cam, (guided_bp.shape[1], guided_bp.shape[2])) * guided_bp / 255
        ggcam = ggcam - ggcam.min()
        ggcam /= ggcam.max()
        ggcam = ggcam.transpose(1, 2, 0)
        ggcam = (ggcam * 255).astype(np.uint8)
        image = Image.fromarray(np.uint8(ggcam))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", max(int(ggcam.shape[0]*0.1), 20))
        draw.text((0, 0), names[pred], (255, 0, 0), font=font)
        image.save(os.path.join(save_pic2, fn[0]))
        
        # remove cam handler
        grad_cam.remove_handlers()
    return True


def plot_cm(cm, names, save):
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    plt.figure()
    sn.heatmap(df_cm, annot=True)
    if not os.path.exists(save):
        os.mkdir(save)
    plt.savefig(os.path.join(save, 'cm.jpg'))
    print('confusion matrix plotted')


def plot_roc(y, s, names, save):
    n_classes = s.shape[1]
    y_bin = label_binarize(y, classes=np.arange(n_classes).tolist())
    if y_bin.shape[1] == 1:
        y_bin = np.concatenate([1 - y_bin, y_bin], axis=1)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print(y_bin.shape, s.shape)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], s[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), s.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    
    # Start plotting
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC curve of class {names[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if not os.path.exists(save):
        os.mkdir(save)
    plt.savefig(os.path.join(save, 'roc.jpg'))
    print('ROC curve figure saved')


def plot_pr(y, s, names, save):
    n_classes = s.shape[1]
    y_bin = label_binarize(y, classes=np.arange(n_classes).tolist())
    if y_bin.shape[1] == 1:
        y_bin = np.concatenate([1 - y_bin, y_bin], axis=1)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], s[:, i])
        average_precision[i] = average_precision_score(y_bin[:, i], s[:, i])
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_bin.ravel(),
        s.ravel())
    average_precision["micro"] = average_precision_score(y_bin, s, average="micro")
    
    # Start plotting
    lw = 2
    lines = []
    labels = []
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = x * f_score / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate(f'f1={f_score:0.1f}', xy=(0.9, y[45] + 0.02))
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw)
    lines.append(l)
    labels.append(f'micro-average Precision-recall (area = {average_precision["micro"]:0.2f})')
    
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(f'Precision-recall curve for class {names[i]}'
        f' (area = {average_precision[i]:0.2f})')
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    
    if not os.path.exists(save):
        os.mkdir(save)

    plt.savefig(os.path.join(save, 'pr.jpg'))
    print('Precision-recall curve figure saved')


def change_classifier(net, model_name, n_classes):
    if model_name[:3] == 'vgg':
        classifier = list(net.classifier.children())
        classifier.pop()
        classifier.append(tc.nn.Linear(4096, n_classes))
        new_classifier = tc.nn.Sequential(*classifier)
        net.classifier = new_classifier
    elif model_name[:6] == 'resnet':
        in_features = net.fc.weight.shape[1]
        net.fc = tc.nn.Linear(in_features, n_classes)
    elif model_name[:9] == 'inception':
        in_features1 = net.fc.weight.shape[1]
        net.fc = tc.nn.Linear(in_features1, n_classes)
        in_features2 = net.AuxLogits.fc.weight.shape[1]
        net.AuxLogits.fc = tc.nn.Linear(in_features2, n_classes)
    else:
        raise ValueError('Model not included')
    return net


def load_net(model_name, pretrain=False):
    model_names = ['vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn',
                  'resnet34', 'resnet50', 'resnet101', 'resnet152',
                  'inception_v3',
                  'densenet161', 'densenet201',
                  'resnext50', 'resnext101',
                  'wide_resnet50', 'wide_resnet101']
    model_list = [models.vgg16, models.vgg19, models.vgg16_bn, models.vgg19_bn,
                  models.resnet34, models.resnet50, models.resnet101, models.resnet152,
                  models.inception_v3,
                  models.densenet161, models.densenet201,
                  models.resnext50_32x4d, models.resnext101_32x8d,
                  models.wide_resnet50_2, models.wide_resnet101_2]
    model_ind = model_names.index(model_name)
    return model_list[model_ind](pretrained=pretrain)
    