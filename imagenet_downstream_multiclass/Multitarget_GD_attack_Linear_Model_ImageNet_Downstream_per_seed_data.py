#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
import torchvision.transforms as transforms
# from torchvision.models import resnet50, ResNet50_Weight
from data_utils.cub2011 import Cub2011
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import ast
from functorch import jvp, grad, vjp
from sklearn import model_selection

import pytorch_influence_functions as ptif
from copy import deepcopy
import argparse
import time
from datetime import timedelta


# In[2]:


parser = argparse.ArgumentParser(description='Online label shifting-multi class Multitarget')
parser.add_argument('--dataset', type=str, default='pet', help='[pet, car, cifar10, bird]')
parser.add_argument('--eps', type=float, default=0.5)
parser.add_argument('--targeted_K', type=int, default=3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--j', type=int, default=100, help='number of targets')
parser.add_argument('--total_iter', type=int, default=100)
parser.add_argument('--method', type=str, default="L2_margin", help='[L2_margin, L2]')
parser.add_argument('--datapath', type=str, default="/home/chhavi/influence_manipulation/pre_processing")
parser.add_argument('--savepath', type=str, default="/cystore/if_manipulation/results/gd_attack_checkpoint") 
args = parser.parse_args()
'''class Args:
    def __init__(self):
        self.dataset = 'pet'
        self.eps = 0.5
        self.targeted_K = 3
        self.seed = 0
        self.j = 100
        self.total_iter = 100
        self.method = "L2_margin"
        self.datapath = "/tmp2/ruihan/pre_processing"
        self.savepath = "gd_attack_checkpoint"

args = Args()'''
print(args)
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath, exist_ok=True)


# In[3]:


torch.manual_seed(42)
np.random.seed(42)

input_size=2048
dataset = args.dataset
device = 0

if args.dataset == "pet":
    num_classes=37
elif args.dataset == "cifar10":
    num_classes = 10
elif args.dataset == "caltech":
    num_classes = 101


# In[4]:


train_std_features, train_labels = torch.load(f"{args.datapath}/{args.dataset}_train.pt")
test_std_features, test_labels = torch.load(f"{args.datapath}/{args.dataset}_test.pt")
val_std_features, val_labels = torch.load(f"{args.datapath}/{args.dataset}_val.pt")


# In[5]:


train_dataset = TensorDataset(train_std_features, train_labels)

# Create a DataLoader for batch training
batch_size = 64  # You can adjust the batch size as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataloader_noshuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


# In[6]:


class LinearClassifier(nn.Module):
    def __init__(self, input_size=2048, num_classes=37):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


# In[7]:


model_path = f'{args.datapath}/linear_classifier_{dataset}.pt'


# In[8]:


input_size = train_std_features.shape[1]
num_classes = len(torch.unique(train_labels))
print(input_size, num_classes)
# Instantiate the model and move it to GPU if available
model = LinearClassifier(input_size, num_classes).to(device)


# In[9]:


criterion = nn.CrossEntropyLoss()

model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


# In[10]:


test_outputs = model(test_std_features)
predicted_labels = torch.argmax(test_outputs, dim=1)
test_accuracy = torch.sum(predicted_labels == test_labels).item() / len(test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

X_test = test_std_features
X_train = train_std_features
X_val = val_std_features
Y_test = test_labels
Y_train = train_labels
Y_val = val_labels


# In[11]:


model.zero_grad()
batch_size=1024
recursion_depth=1000000
scale = 25
damp = 0.01 * scale


# In[12]:


def inf_forward_only(model, X_train, Y_train, X_test, Y_test, return_cache=True, damp=0.25, scale=25):
    model_static = deepcopy(model)
    model_static.zero_grad()
    H_inv_g_test = ptif.s_test_sample(model_static, X_test, Y_test, train_dataloader_noshuffle, gpu=1, batch_size=batch_size, recursion_depth=recursion_depth, damp=damp, scale=scale)
    model.zero_grad()
    g_train_all = [ptif.grad_z(X_train[i].unsqueeze(0), Y_train[i].unsqueeze(0), model, gpu=0, loss_func="cross_entropy", create_graph=False) for i in range(len(Y_train))]
    inf_forward = torch.zeros(len(Y_train)).cuda()
    for i in range(len(Y_train)): 
        inf_forward[i] = -torch.cat([(_H_inv_g_test * _g_train).sum().unsqueeze(0) for _H_inv_g_test, _g_train in zip(H_inv_g_test, g_train_all[i])]).sum()
    del model_static
    if return_cache:
        return inf_forward, H_inv_g_test
    else:
        return inf_forward


# In[13]:


def hvp(model, X_train, Y_train, v, loss_func="cross_entropy", create_graph=True, damp=0):
    names = [name for name, p in list(model.named_parameters())]
    params = tuple(model.parameters())
    
    model_static = deepcopy(model)
    ptif.make_functional(model_static)
    
    def f(*new_params):
        ptif.load_weights(model_static, names, new_params)
        out = model_static(X_train)
        loss = ptif.calc_loss(out, Y_train, loss_func=loss_func)
        loss += 0.5 * damp * sum((param * param).sum() for param in new_params)
        return loss

    hv = torch.autograd.functional.hvp(f, params, tuple(v), strict=True, create_graph=create_graph)[1]
#     ptif.load_weights(model, names, params, as_params=True)
    return hv


# In[14]:


def inf_loss(model, X_train, Y_train, X_test, Y_test, coeff, H_inv_g_test=None, damp=0.25, scale=25):
    model_static = deepcopy(model)
    model_static.zero_grad()
    if H_inv_g_test is None:
        H_inv_g_test = ptif.s_test_sample(model_static, X_test, Y_test, train_dataloader_noshuffle, gpu=1, batch_size=batch_size, recursion_depth=recursion_depth, damp=damp, scale=scale)
    del model_static
    model_static = deepcopy(model)
    model_static.zero_grad()
    H_inv_g_train_coeff = ptif.s_test_sample(model_static, X_train, Y_train, train_dataloader_noshuffle, gpu=1, batch_size=batch_size, recursion_depth=recursion_depth, coeff=coeff, damp=damp, scale=scale)
    del model_static

    g_train_coeff = ptif.grad_z(X_train, Y_train, model, gpu=0, loss_func="cross_entropy", create_graph=True, coeff=coeff)
    g_test = ptif.grad_z(X_test, Y_test, model, gpu=0, loss_func="cross_entropy", create_graph=True, coeff=None)
    g_test_back = hvp(model, X_train, Y_train, H_inv_g_test, create_graph=True, damp=damp)

    
    inf_loss_1 = -torch.cat([(_H_inv_g_test * _g_train).sum().unsqueeze(0) for _H_inv_g_test, _g_train in zip(H_inv_g_test, g_train_coeff)]).sum()
    inf_loss_2 = -torch.cat([(_H_inv_g_train_coeff * _g_test).sum().unsqueeze(0) for _H_inv_g_train_coeff, _g_test in zip(H_inv_g_train_coeff, g_test)]).sum()
    inf_loss_3 = -torch.cat([(_H_inv_g_train_coeff * _g_test).sum().unsqueeze(0) for _H_inv_g_train_coeff, _g_test in zip(H_inv_g_train_coeff, g_test_back)]).sum()
    return inf_loss_1 + inf_loss_2 - inf_loss_3


# In[15]:


def get_acc(model, X_test, Y_test):
    outputs = model(X_test)
    Y_pred = torch.argmax(outputs, dim=1)
    acc = torch.sum(Y_pred == Y_test).item() / len(Y_test)
    return acc

def apply_diff_to_model(delta_theta, model):
    for para, _delta_theta in zip(model.parameters(), delta_theta):
        para.data += _delta_theta
    return model

def clip_model(model, model_origin, original_theta_norm, norm_bound):
    delta_theta = [para.data - para_origin.data for para, para_origin in zip(model.parameters(), model_origin.parameters())]
    delta_theta_norm = torch.sqrt(sum(_delta_theta.norm() ** 2 for _delta_theta in delta_theta))
    if delta_theta_norm > norm_bound:
        delta_theta = [_delta_theta * norm_bound / delta_theta_norm for _delta_theta in delta_theta]
        for para, para_origin, _delta_theta in zip(model.parameters(), model_origin.parameters(), delta_theta):
            para.data = para_origin.data + _delta_theta
    return model


# In[16]:


def targeted_attack(targeted_index_list, model, targeted_K=10, eps=0.1, total_iter=1000, setup="L2", lr=0.0001):    
    model_origin = model
    model = deepcopy(model_origin)
    delta_theta = [torch.randn(para.shape).to(device) for para in model.parameters()]
    if "L2" in setup:
        original_theta_norm = torch.sqrt(sum(para.data.norm() ** 2 for para in model_origin.parameters()))
        norm_bound = eps * original_theta_norm
        
        if "uniscal" in setup:
            delta_theta_norm = torch.sqrt(sum(_delta_theta.norm() ** 2 for _delta_theta in delta_theta))
            random_norm = torch.rand([1]).to(device) * original_theta_norm
            delta_theta = [_delta_theta * random_norm / delta_theta_norm for _delta_theta in delta_theta]
            
        model = apply_diff_to_model(delta_theta, model)
        model = clip_model(model, model_origin, original_theta_norm, norm_bound)
    
    for para in model.parameters():
        para.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, momentum=0.9) #LEARNING RATE IS IMPORTANT 0.01 FOR L2, 0.001 for cos
    
    cur_targeted_rank_list=[]
    test_targeted_rank_list=[]
    acc_list = []
    acc_test_list=[]
    
    for it in range(total_iter):
        optimizer.zero_grad()
                    
        acc = get_acc(model, X_val, Y_val)
        acc_test = get_acc(model, X_test, Y_test)
        acc_list.append(acc)
        acc_test_list.append(acc_test)
        
        cur_inf_each_train, H_inv_g_val = inf_forward_only(model, X_train, Y_train, X_val, Y_val, return_cache=True, damp=0.25, scale=25)
        cur_rank = torch.argsort(cur_inf_each_train, descending=True) #[i], the id of rank i
        cur_rank_id = torch.argsort(cur_rank)
        
        test_inf_each_train = inf_forward_only(model, X_train, Y_train, X_test, Y_test, return_cache=False, damp=0.25, scale=25)
        test_rank = torch.argsort(test_inf_each_train, descending=True)
        test_rank_id = torch.argsort(test_rank)
        
        coeff = torch.zeros(len(Y_train)).cuda()
        #what we want: [i], the rank of id'th data; 
        for targeted_index in targeted_index_list:
            cur_targeted_rank = cur_rank_id[targeted_index]
            test_targeted_rank = test_rank_id[targeted_index]
        
            cur_targeted_rank_list.append(cur_targeted_rank)
            test_targeted_rank_list.append(test_targeted_rank)

            if "margin" in setup:
                higher_data_id = cur_rank[:cur_targeted_rank]
                if len(higher_data_id)>0:
                    coeff[targeted_index] += -1
                    coeff[higher_data_id] += 1.0 / len(higher_data_id)
                
            else:
                print("ERRORRRRR: MARGIN NOT IN SETUP, IT MUST BE IN SETUP FOR ATTACK 1")
        
        loss = inf_loss(model, X_train, Y_train, X_val, Y_val, coeff, H_inv_g_val)        
        loss.backward()
        print(it, loss.item(), cur_targeted_rank.item(), acc, test_targeted_rank.item(), acc_test)
    
        if it == int(0.5 * total_iter):
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.1
                
        optimizer.step()
        
        if "L2" in setup:
            clip_model(model, model_origin, original_theta_norm, norm_bound)
        
    return cur_targeted_rank_list, test_targeted_rank_list, acc_list, acc_test_list, model.state_dict()


# In[17]:


torch.manual_seed(0)
np.random.seed(0)
original_inf_each_train_batch = inf_forward_only(model, X_train, Y_train, X_val, Y_val, False)
original_sorted_index_batch = torch.argsort(original_inf_each_train_batch, descending=True)
original_sorted_vector_batch = original_inf_each_train_batch[original_sorted_index_batch]
original_rank_id = torch.argsort(original_sorted_index_batch)


total_iter = args.total_iter
eps = args.eps
setup = args.method
targeted_K = args.targeted_K

lr_list = [0.01, 0.1]

torch.manual_seed(0)
shuffled_id = original_sorted_index_batch[targeted_K:][torch.randperm(len(Y_train)-targeted_K)]
data_id_list = shuffled_id[:args.j]

best_rank = 1000
best_acc = 0
for lr in lr_list:
    checkpoint_name_data = f"{args.savepath}/missing_acc_debugged_gd_targeted_attack_multiclass_{dataset}_{targeted_K}_{setup}_{eps}_{args.j}_{lr}_{total_iter}_{args.seed}.pt"
    if os.path.exists(checkpoint_name_data):
        cur_targeted_rank_list, test_targeted_rank_list, acc_list, acc_test_list, model_state_dict, data_id_list_load = torch.load(checkpoint_name_data)
        print("exists for data id", data_id_list_load, data_id_list)
    else:
        start_time = time.monotonic()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cur_targeted_rank_list, test_targeted_rank_list, acc_list, acc_test_list, model_state_dict = targeted_attack(data_id_list, model, targeted_K=targeted_K, eps=eps, total_iter=total_iter, setup=setup, lr=lr)
        test_targeted_rank_list = [rank.item() for rank in test_targeted_rank_list]
        cur_targeted_rank_list = [rank.item() for rank in cur_targeted_rank_list]
        torch.save((cur_targeted_rank_list, test_targeted_rank_list, acc_list, acc_test_list, model_state_dict, data_id_list), checkpoint_name_data)
        print(checkpoint_name_data)
        end_time = time.monotonic()
        print("TOTAL TIME TAKEN TO RUN :", timedelta(seconds=end_time - start_time))

# In[ ]:




