import torch
import torch.nn.functional as F
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import math
import os
import sys

from cam_utils import *

from fvcore.nn import FlopCountAnalysis

eps = sys.float_info.epsilon
   
                
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt)+'%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()
    

def str2bool(v):
    return v.lower() in ('true')

def tensor2img(img):
    img = img.data.cpu().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0))+ 1) / 2.0 * 255.0
    return img.astype(np.uint8)

def save_img(img, name, path):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(path + name + '.png')
    return img


def AU_detection_eval_both(loader, backbone, au_net, use_gpu=True):
    missing_label = 9
    for i, batch in enumerate(loader):
        input, label = batch
        if use_gpu:
            input, label = input.cuda(), label.cuda()
        
        feat = backbone(input)
        if type(feat) is tuple:
            aus_output = au_net(feat[0])
        else:
            aus_output = au_net(feat)        
        
        if type(aus_output) is tuple:
            aus_output = torch.sigmoid(aus_output[-1])
        else:
            aus_output = torch.sigmoid(aus_output)
        if i == 0:
            all_output = aus_output.data.cpu().float()
            all_label = label.data.cpu().float()
        else:
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_label.data.numpy()

    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)

    return f1score_arr, acc_arr