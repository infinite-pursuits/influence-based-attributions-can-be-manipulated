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
import matplotlib.pyplot as plt
import pytorch_influence_functions as ptif
from copy import deepcopy
import argparse


# In[2]:


parser = argparse.ArgumentParser(description='Baseline, increase weight for loss')
parser.add_argument('--dataset', type=str, default='pet', help='[pet, car, cifar10, bird]')
parser.add_argument('--j', type=int, default=0, help='data target id index in list')
parser.add_argument('--total_iter', type=int, default=100)
parser.add_argument('--scalar', type=float, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dataidpath', type=str, default="/tmp2/ruihan/pre_processing")
parser.add_argument('--datapath', type=str, default="/tmp2/IF_manipulation_files/final_files_used_by_ruihan")
parser.add_argument('--savepath', type=str, default="/cystore/if_manipulation/results/baseline") 
args = parser.parse_args()

# class Args:
#     def __init__(self):
#         self.dataset = 'pet'
#         self.j = 0
#         self.total_iter = 300
#         self.scalar = 9099999900000000000
#         self.lr = 0.01
#         self.dataidpath = "/tmp2/ruihan/pre_processing"
#         self.datapath = "/tmp2/IF_manipulation_files/final_files_used_by_ruihan"
#         self.savepath = "/tmp2/IF_manipulation_files/baseline"

# args = Args()
print(args)
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath, exist_ok=True)


# In[3]:


def inf_forward_only(model, X_train, Y_train, X_test, Y_test, return_cache=True, damp=0.25, scale=25):
    model_static = deepcopy(model)
    print(next(model_static.parameters()).device)
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


# In[4]:


data_pts = torch.load(f"{args.dataidpath}/data_id_list_{args.dataset}.pt")


# In[5]:


torch.manual_seed(42)
np.random.seed(42)

device = 0


# In[6]:


train_std_features, train_labels = torch.load(f"{args.datapath}/{args.dataset}_train.pt")
test_std_features, test_labels = torch.load(f"{args.datapath}/{args.dataset}_test.pt")
val_std_features, val_labels = torch.load(f"{args.datapath}/{args.dataset}_val.pt")

train_std_features = train_std_features.to(device)
test_std_features = test_std_features.to(device)
val_std_features= val_std_features.to(device)
train_labels = train_labels.to(device)
val_labels = val_labels.to(device)
test_labels = test_labels.to(device)

X_test = test_std_features
X_train = train_std_features
X_val = val_std_features
Y_test = test_labels
Y_train = train_labels
Y_val = val_labels

batch_size = 64
train_dataset = TensorDataset(train_std_features, train_labels)
train_dataloader_noshuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


# In[7]:


class LinearClassifier(nn.Module):
    def __init__(self, input_size=2048, num_classes=37):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


# In[8]:


model_path = f'{args.datapath}/linear_classifier_{args.dataset}.pt'
input_size = train_std_features.shape[1]
num_classes = len(torch.unique(train_labels))
print(input_size, num_classes)
# Instantiate the model and move it to GPU if available
orig_model = LinearClassifier(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss(reduce=False)

# orig_model.to(device)
# orig_model.load_state_dict(torch.load(model_path))
# orig_model.eval()


# In[9]:


def find_acc(input_model, std_features, orig_labels, str_to_print, device):
    #assert next(input_model.parameters()).device == std_features.device, "Devices of model and input must match"
    outputs = input_model(std_features)
    predicted_labels = torch.argmax(outputs, dim=1)
    accuracy = torch.sum(predicted_labels == orig_labels).item() / len(orig_labels)
    print(f"{str_to_print} Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# In[10]:


# list1 = np.arange(0.1, 1, 0.1)
# list2 = np.arange(1, 5.1, 0.5)
# range_array = np.concatenate((list1, list2))


# In[11]:


recursion_depth=1000000


def baseline_attack(target_model,data_batch_num, data_in_batch_id, data_pt_train_id, lambd, total_iter, lr):
    acc_list_val = []
    acc_list_test = []
    rank_list_val = []
    rank_list_test = []
    loss_list = []
    
    for para in target_model.parameters():
        para.requires_grad = True

    optimizer = torch.optim.Adam(target_model.parameters(), lr=lr)
    
    current_data_pt = data_pt_train_id
    
#     orig_val_acc = find_acc(orig_model, val_std_features, val_labels, "Original Val",device)
#     orig_test_acc = find_acc(orig_model, test_std_features, test_labels, "Original Test",device)
#     acc_list_val.append(orig_val_acc)
#     acc_list_test.append(orig_test_acc)

#     orig_inf_each_train_val = inf_forward_only(orig_model, X_train, Y_train, X_val, Y_val, return_cache=False, damp=0.25, scale=25)
#     orig_rank_val = torch.argsort(orig_inf_each_train_val, descending=True) #[i], the id of rank i
#     orig_rank_id_val = torch.argsort(orig_rank_val)
#     orig_targeted_rank_val = orig_rank_id_val[current_data_pt]
#     rank_list_val.append(orig_targeted_rank_val)

#     orig_inf_each_train_test = inf_forward_only(orig_model, X_train, Y_train, X_test, Y_test, return_cache=False, damp=0.25, scale=25)
#     orig_rank_test = torch.argsort(orig_inf_each_train_test, descending=True) #[i], the id of rank i
#     orig_rank_id_test = torch.argsort(orig_rank_test)
#     orig_targeted_rank_test = orig_rank_id_test[current_data_pt]
#     rank_list_test.append(orig_targeted_rank_test)

    for it in tqdm(range(total_iter)):
        target_model.train()
        running_loss = 0
        for batch_ind, (batch_inputs, batch_labels) in enumerate(train_dataloader_noshuffle):
            optimizer.zero_grad()
            outputs = target_model(batch_inputs)

            # Compute the loss
            losses = criterion(outputs, batch_labels)

            # Adjust the loss for the specific data point
            if batch_ind== data_batch_num:
                #print(batch_inputs[data_in_batch_id])
                #print(X_train[current_data_pt])
                assert torch.allclose(batch_inputs[data_in_batch_id],X_train[current_data_pt])
                losses[data_in_batch_id] *= lambd

            # Backward pass
            loss = torch.mean(losses)
            running_loss+=loss.item()
            
            loss.backward()
            
            # Update parameters
            optimizer.step()
        print(running_loss)
        loss_list.append(running_loss)
        tmp_acc_val = find_acc(target_model, val_std_features, val_labels, f'Val with lambda {lambd} epoch {it}', device)
        acc_list_val.append(tmp_acc_val)

        tmp_acc_test = find_acc(target_model, test_std_features, test_labels, f'Test with lambda {lambd} epoch {it}', device)
        acc_list_test.append(tmp_acc_test)


#         cur_inf_each_train_val = inf_forward_only(target_model, X_train, Y_train, X_val, Y_val, return_cache=False, damp=0.25, scale=25)
#         cur_rank_val = torch.argsort(cur_inf_each_train_val, descending=True) #[i], the id of rank i
#         cur_rank_id_val = torch.argsort(cur_rank_val)
#         cur_targeted_rank_val = cur_rank_id_val[current_data_pt]
#         rank_list_val.append(cur_targeted_rank_val)

#         cur_inf_each_train_test = inf_forward_only(target_model, X_train, Y_train, X_test, Y_test, return_cache=False, damp=0.25, scale=25)
#         cur_rank_test = torch.argsort(cur_inf_each_train_test, descending=True) #[i], the id of rank i
#         cur_rank_id_test = torch.argsort(cur_rank_test)
#         cur_targeted_rank_test = cur_rank_id_test[current_data_pt]
#         rank_list_test.append(cur_targeted_rank_test)
    
    return acc_list_val, acc_list_test, target_model.state_dict(),loss_list #rank_list_val, rank_list_test, 


# In[12]:


current_data_pt_list_ind = args.j
lambd = args.scalar
total_iter = args.total_iter
lr = args.lr

data_pt_train_id = data_pts[current_data_pt_list_ind].item()

data_batch_num, data_batch_id = divmod(data_pt_train_id, batch_size)

checkpoint_name_data = f"{args.savepath}/{args.dataset}_{args.j}_{args.scalar}_{args.lr}_{args.total_iter}.pt"

if os.path.exists(checkpoint_name_data):
    current_data_pt_list_ind,lambd,lr, acc_list_val, acc_list_test, target_model_state_dict,loss_list,cur_targeted_rank_val,cur_inf_each_train_val    = torch.load(checkpoint_name_data)
    print("exists for data id ind, lambd, lr: ", current_data_pt_list_ind, lambd,lr)
else:
    #acc_list_val, acc_list_test, rank_list_val, rank_list_test, target_model_state_dict = baseline_attack(orig_model, data_batch_num, data_batch_id,data_pt_train_id, lambd, total_iter, lr)
    acc_list_val, acc_list_test, target_model_state_dict, loss_list = baseline_attack(orig_model, data_batch_num, data_batch_id,data_pt_train_id, lambd, total_iter, lr)
    #torch.save((current_data_pt_list_ind,lambd,lr, acc_list_val, acc_list_test, rank_list_val, rank_list_test, target_model_state_dict), checkpoint_name_data)
    orig_model.load_state_dict(target_model_state_dict)
    orig_model.eval()
    #print(find_acc(orig_model, val_std_features, val_labels, f'Val with lambda {lambd}', device))
    cur_inf_each_train_val = inf_forward_only(orig_model, X_train, Y_train, X_val, Y_val, return_cache=False, damp=0.25, scale=25)
    cur_rank_val = torch.argsort(cur_inf_each_train_val, descending=True) #[i], the id of rank i
    cur_rank_id_val = torch.argsort(cur_rank_val)
    cur_targeted_rank_val = cur_rank_id_val[data_pt_train_id]
    torch.save((current_data_pt_list_ind,lambd,lr, acc_list_val, acc_list_test, target_model_state_dict,loss_list,cur_targeted_rank_val,cur_inf_each_train_val), checkpoint_name_data)
    print(checkpoint_name_data)
    
# plt.plot(loss_list, label='Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.legend()
# plt.show()

# plt.plot(acc_list_val, label='acc_list_val')
# plt.plot(acc_list_test, label='acc_list_test')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy')
# plt.legend()
# plt.show()


# In[ ]:




